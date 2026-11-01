# Phase-ingestion contract

This document describes the contract implemented by the efficient ingestion
phase machine: the states it records, how it decides that a phase is stale, how
failures propagate, how it recovers after a restart, how it serialises work,
and what a phase plugin is expected to provide.

It documents the implemented behaviour. Configuration keys, hook signatures and
built-in phase bodies are described in the main [README](../README.md).

## 1. States

Every phase of a source has exactly one of these states in the phase diary:

| State | Meaning |
| --- | --- |
| `pending` | The phase has no diary entry. Absence is the pending state; it is never written explicitly. |
| `processing` | A worker owned the phase and its body has not finished yet. |
| `complete` | The phase body finished without error. |
| `error` | The phase body failed; the entry carries a `reason`. |
| `skipped` | The phase does not apply to this source, decided by its `applies_to` capability; the entry carries a `reason`. |

Completion means the worker finished without error. The `processed` and
`expected` counters are **telemetry only**: they never take part in the
completion decision, and a source is not considered complete because those
numbers match.

A source's status is derived from its phase diary. A source is complete when
every phase is `complete` or `skipped`; it is `error` when any phase is `error`
(or a dependency propagated an error onto it); it is `processing` while a phase
is in flight. An `error` or `skipped` phase is terminal for itself, but it is
not "complete work" and therefore keeps its dependents stale (see section 3).

## 2. Phase tags

A phase's tag is a recursive fingerprint of the configuration that produced its
data:

```
own_tag = sha256(own configuration fingerprint)
tag     = sha256(own_tag | sorted dependency tags)
```

Concretely, for phase `P`:

```
own_tag(P) = sha256(kind + ":" + canonical_json(relevant_settings(P)))
tag(P)     = sha256(own_tag(P) + "|" + "|".join(f"{dep}={tag(dep)}" for dep in sorted(P.depends_on)))
```

The marker reads **configuration only** — it never performs an LLM call. A
phase may narrow the settings that feed its own tag with a `settings_check`
(a key collection, or a possibly-async callable); otherwise the whole settings
value participates.

Dependencies are resolved before their dependents, so a change deep in the
chain invalidates every dependent transitively. A cycle is rejected with
`PhaseTagCycleError` naming the cycle. The legacy flat `settings_version`
field remains readable on old entries but is not part of the staleness
decision.

A phase is stale when any of the following holds:

- its own entry is `pending` (absent) or `processing` with an expired lease;
- a dependency is `pending`, `processing`, `error` or `skipped` — there is no
  completed upstream work to consume;
- its stored `tag` differs from the freshly computed recursive tag, or the
  stored tag is missing (a legacy entry is stale exactly once, until rewritten).

## 3. Propagation

Failure propagates along dependency edges and the body never runs behind a
failed dependency:

- if a dependency is `error`, every dependent is recorded `error` with a
  reason that names the dependency;
- if a dependency is `skipped`, every dependent is recorded `skipped` with a
  reason that names the dependency;
- a dependent is not executed while any upstream is `pending`, `processing`,
  `error` or `skipped`.

An `error` phase is never silently advanced to `complete`, and a source whose
diary contains an `error` is not resurrected to `complete` by a later probe.

## 4. Recovery

Recovery distinguishes "abandoned" from "live" work with a boot identity and
an owner stamp:

- the process exposes `BOOT_EPOCH`, a value unique to the current boot;
- a `processing` entry records its owner as `{pod, pid, boot}`.

On recovery:

- a `processing` entry owned by a **previous boot** is abandoned: the phase's
  `delete()` is called to purge its partial artifacts, the diary entry is
  cleared, and the phase is re-run from zero;
- an entry owned by a live owner (same boot, or a live pod/pid) is never
  touched;
- the lease is only a fallback: ownership is the primary signal, and the
  expiry is consulted when ownership cannot prove the work is live.

The atomic-completion invariant supports this: a phase writes its diary entry
only on successful completion, in a single atomic write, never at phase start.
An interrupted phase therefore has no completed entry and is re-run.

## 5. Locks

Per-source and per-phase work is serialised with two lock levels:

```
ingestion-source:<agent>:<source>
ingestion-phase:<agent>:<source>:<phase>
```

Ordering is strict: the source lock is always acquired **first**, and the
phase lock is acquired **inside** it. The reverse is never done. The delete
path takes only the source lock, held for the **whole operation** (the
`deleting` write, every phase `delete()` and the final row removal), so a
per-source phase sequence and a deletion can never interleave. A dispatching
pass also aborts a source whose status is `deleting`, so an interrupted
deletion is never dispatched against. Lock keys escape the (untrusted) source
and agent identifiers with `escape_glob`, so a source name containing glob
metacharacters cannot collide with another source's key space.

## 6. State/data lockstep

Data and state are written in one order, and every deletion resets the state
that described the data:

- **data first, state last** — a phase records `complete` only after its data
  is durably written;
- a deletion that removes a phase's data also resets the state that described
  it, so there is no window in which a state points at data that no longer
  exists;
- when a phase's `delete()` fails, the entry stays in `processing` and the
  deletion is retried; the failure is not swallowed and no terminal state is
  written over it.

## 7. File deletion

A single-file deletion runs through the phases:

- a file-level `deleting` state is written before any phase delete starts;
- every phase's `delete()` runs in **reverse execution order** (dependents
  first, dependencies last), so a phase is torn down before the data it
  consumes;
- the deletion record is removed **last**, after every phase delete succeeded;
- if a `delete()` fails, the row stays `deleting` and the next pass retries the
  remaining phases (already-deleted phases are not re-run).

The whole path is gated by `phase_delete_active(cat)`, which is true only when
EffING is active and phases are registered. When the gate is false, the
**standard deletion** applies and no `deleting` state is written.

## 8. Plugin surface

A phase plugin provides only this surface:

| Member | Responsibility |
| --- | --- |
| `id` | The phase id, used as the diary key. |
| `depends_on` | Upstream phase ids whose tags are folded into this phase's tag. |
| `process` | The phase body: parse/embed/purge, returning a phase result. |
| `delete` | Purges this phase's artifacts for a source. |
| `tag` | Returns the configuration fingerprint of the settings the phase consumes. |
| `applies_to` | CAPABILITY check: `(True, None)` when the phase applies, `(False, reason)` when it does not (recorded `skipped`). |

The plugin owns **none** of the following: diary state, staleness detection,
locks, retries, or error propagation. Those are owned by the phase machine. A
plugin therefore declares *what* the phase does and *what configuration* it
depends on, and the machine decides *whether, when and in what order* it runs.

## 9. Known limitations

These are current limitations, stated plainly so consumers can plan around
them:

1. **Agent deletion per-file cascade.** The cascade fires the same
   `before_file_manager_file_delete` hook, so every phase's `delete()` runs in
   reverse order before the blanket wipe. The blanket wipe
   (`delete_tenant_points`) still runs afterwards as a redundant catch-all; it
   is not removed because that would require a CAT core change.
2. **Graph purge ordering.** The graph purge walks from the source's
   `Document`s, so it must run **before** any vector wipe. Until entities
   carry their own source locator, removing the vectors first would leave the
   purge with nothing to walk from.
3. **Commit `3d433197` in the cat-efficient-ingestion repository.** That commit carries phase
   3's message but is a real refactor: it extracted the tag resolver. It is
   present on the remote and was not rewritten. Read it as a refactor, not as
   the phase-3 behaviour change its message describes.
