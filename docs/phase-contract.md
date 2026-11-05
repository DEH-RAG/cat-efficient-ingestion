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

### Warnings: ran, but degraded

A warning is **not** a phase state. It is an optional list attached to a
successful (`complete`) diary entry, for a phase that RAN correctly but
produced nothing, or less than it should have. The distinction from a failure
is structural, never a size threshold:

| Situation | Result |
| --- | --- |
| ran, the call succeeded, produced nothing and an empty product is VALID | `done` + one or more warnings |
| ran, the call succeeded, produced nothing and an empty product is INVALID | `error` with a reason (no warning) |
| could not run (timeout, transport error, LLM unavailable, no input) | `not_ready` (retried) then `error`, or the existing skip path |

The **phase owns the decision**: it knows whether it really ran and whether an
empty product is valid. The machine applies no input-size or item-count
threshold. A phase that returns `done` with warnings is COMPLETE: it is not
retried and its dependents are not blocked. Only `error` keeps the propagation
of section 3.

#### Per-phase empty-output semantics

The two built-in phases that PRODUCE the searchable product are strict: when
they ran on a source that carried input and stored nothing, the product does
not exist and the phase is an `error`, never a warning. The phases whose empty
result is legitimate keep the warning channel.

| Phase | Situation | Result |
| --- | --- | --- |
| `parsing_chunking` | ran on a non-empty source and produced 0 chunks | `error`: `parsing produced no chunks from <N> bytes of '<source>'` |
| `embedding` | ran with points to embed and stored 0 points | `error`: `embedding stored no points for <N> point(s) of '<source>'` |
| `catalog_cards` | ran and legitimately found no card | `done` + warning |
| graph phases (e.g. `graph_concepts`, `graph_relations`) | ran and legitimately extracted nothing | `done` + warning |
| `parsing_chunking` / `embedding` | could not run (no bytes, no points to embed, source vanished) | unchanged `not_ready` / skip paths — not an empty-output error |

"No chunks from a non-empty file" and "no points from a non-empty embedding
run" are production failures, not documents without content, so they must not
look complete. The judgement belongs to the phase bodies, which know the input
size and the produced count; the machine still applies no threshold.

Each warning is:

```json
{"code": "empty-extraction", "message": "the parser returned 0 chunks"}
```

Documented codes are `empty-extraction`, `partial-extraction` and `degraded`.
The vocabulary is descriptive, not an enforcement allow-list.

A diary entry gains `warnings: [...]` only when its phase returned `done` with
warnings; a clean (or legacy) phase has no `warnings` key at all. Warnings are
persisted in the diary, so they survive a restart and remain visible after the
fact.

The source-level status entry **aggregates** every phase's warnings and adds:

```json
"warnings": [
  {"phase": "graph_concepts", "code": "empty-extraction", "message": "..."}
],
"warning_count": 1,
"has_warnings": true
```

`status` is unchanged: warnings never change the derived state, so a source
whose phases all completed is `completed`, never `error`. The aggregate is
exposed by `GET /ingestion/status` and backs the file manager's fourth counter
alongside processing/error/processed.

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
`PhaseTagCycleError` naming the cycle. A legacy (v1) entry has no `tag`; it is
compared against the current `marker` / `settings_version` / `deps` (as the
pre-tag machine did), and a current one is backfilled with its tag in place
instead of being re-run.

A phase is stale when any of the following holds:

- its own entry is `pending` (absent) or `processing` with an expired lease;
- a dependency is `pending`, `processing`, `error` or `skipped` — there is no
  completed upstream work to consume;
- its stored `tag` differs from the freshly computed recursive tag;
- a legacy (v1) entry's recorded `marker` / `settings_version` / `deps` no
  longer match the current values.

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
`deleting`/`reindexing` write, every phase `delete()` and the final row
removal when deleting), so a per-source phase sequence and a deletion/re-ingest
can never interleave. A dispatching pass also aborts a source whose status is
`deleting` or `reindexing`, so an interrupted operation is never dispatched
against. Lock keys escape the (untrusted) source
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

## 7. File deletion and re-ingest

Two routines share one artifact wipe:

- `delete_artifacts(source, cat)` removes every artifact of the source —
  derived images first (through the multimodal plugin's
  `delete_source_image_files`), then every phase `delete()` in **reverse
  execution order** (dependents first, dependencies last) — and touches
  neither the source file, the file record, nor the status row. It is
  idempotent and never takes a nested source lock.
- `delete_file(source, cat)` is used only by the `deleting` path:
  `delete_artifacts` → source file → status row **last**.

A single-file deletion therefore writes a file-level `deleting` state before
any phase delete starts and removes the record only after every phase delete
succeeded; a failure leaves the row `deleting` and the next pass retries the
remaining phases. The whole path is gated by `phase_delete_active(cat)`, which
is true only when EffING is active and phases are registered. When the gate is
false, the **standard deletion** applies and no `deleting` state is written.

A re-ingest (`purge_source_for_reingest`) never removes the row. It writes
`reindexing` **before** the first destructive step, calls `delete_artifacts`,
clears the diary and then writes `processing` for the phase machine. A
`reindexing` row left by a crash is re-run from the start (idempotent wipe →
diary clear → `processing`) once its owner is provably gone (`should_abandon`);
a live owner is never touched, and the dispatch loop skips both `deleting` and
`reindexing`.

## 8. Plugin surface

A phase plugin provides only this surface:

| Member | Responsibility |
| --- | --- |
| `id` | The phase id, used as the diary key. |
| `label` | OPTIONAL human-readable display label (`None` defaults to `id`). Display-only: the machine identity is always `id`. |
| `display_label()` | Returns `label` when declared, else `id`. |
| `depends_on` | Upstream phase ids whose tags are folded into this phase's tag. |
| `process` | The phase body: parse/embed/purge, returning a phase result (status plus optional warnings). |
| `delete` | Purges this phase's artifacts for a source. |
| `tag` | Returns the configuration fingerprint of the settings the phase consumes. |
| `applies_to` | CAPABILITY check: `(True, None)` when the phase applies, `(False, reason)` when it does not (recorded `skipped`). |
| `not_ready_budget_seconds` | OPTIONAL wall-clock budget (seconds) this phase may stay `not_ready` before it is marked `error`. `None` (default) uses the generic `CAT_INGESTION_NOT_READY_MAX_SECONDS`. |
| `not_ready_budget_env` | OPTIONAL env var that overrides `not_ready_budget_seconds` when set to a non-negative number. Resolution order: declared env var → declared value → generic budget. |

Built-in labels are `parsing_chunking` → "Parsing and chunking" and
`embedding` → "Computing embeddings". `GET /ingestion/phases` returns them in
dependency order as `{"id", "label", "order", "depends_on"}` rows, plus the
lifecycle states as `{"id", "label"}` rows under `states`. A status entry
carries the matching `phase_label` / `status_label` alongside the unchanged
machine fields `phase` / `status`. Labels are backend English defaults only:
there is no i18n here, and clients localize by `id`.

The plugin owns **none** of the following: diary state, staleness detection,
locks, retries, or error propagation. Those are owned by the phase machine. A
plugin therefore declares *what* the phase does and *what configuration* it
depends on, and the machine decides *whether, when and in what order* it runs.

### Result normalization: small, intentional duplication

Each phase plugin normalizes its own legacy return value into the shared
`PhaseResult` with a private helper (`_to_result` / `_coerce_phase_result`) and
a private count helper (`_as_count`). This duplication is **intentional**:
plugins are independently shippable and never import each other, so EffING,
MyGRAPH and MyALOG cannot share a helper module without coupling their release
cycles. The copies are small; the contract is the semantics below, not one
source location. Every copy MUST agree on:

- a `PhaseResult` instance is returned unchanged;
- a legacy dict whose `status` is exactly one of `done`, `not_ready`, `error`
  or `skipped` is converted with that status and its `reason`;
- any other status becomes `error` with a reason naming the invalid status
  (`phase '<id>' returned invalid status <repr>`);
- `None` (no run implementation) becomes `error` with a reason naming the
  phase; any other non-dict value becomes `error` naming the unexpected value;
- `processed` / `expected` are counts ONLY when the value is an `int` that is
  not a `bool`; every other type (string, float, bool, object) normalizes to
  `None`. Never coerce a numeric string or float into a count;
- counts are telemetry only (section 1) and never change the phase state;
- `warnings`, when present, is normalized to a list of well-formed
  `{"code": <str>, "message": <str>}` rows (malformed rows are dropped) and
  NEVER changes the status: it is only valid on a `done` result.

If this contract changes, every copy must be updated in the same release; the
duplication is the reason there is no single place to edit.

### LLM-backed phases

A phase whose body calls a language model is still an ordinary phase to the
machine. The machine knows only the four outcomes `done`, `not_ready` (retried
until its budget expires), `error` (with a `reason`) and `skipped` (with a
`reason`), plus the optional `warnings` list on `done`. **The phase owns the
mapping**: it inspects the LLM result and translates it into exactly one of
those four outcomes. The machine never sees an LLM.

The mapping a phase must implement is:

| LLM outcome | State |
| --- | --- |
| timeout / 429 / 5xx / connection dropped | `not_ready` with a reason (transient) |
| empty-but-valid response on non-empty text | `not_ready` on the first attempt, then `error` carrying the raw text (the extraction attempt failed) |
| no LLM configured | `error` with a configuration reason — never `skipped` |
| partial output (schema drops items) | `error` for now; the reason carries `dropped=<n>` |
| success, but it legitimately produced nothing / less than expected | `done` + a warning (`empty-extraction` / `partial-extraction`) |
| success with a product | `done` |

These are owner decisions, implemented for the graph concept phase in MyGRAPH
`341e61c`.

`done` means the phase **ran to completion**, not merely that it did not crash.
A phase that produced nothing, and whose empty product is legitimate, is still
`done` but MUST say so with a warning (section 1); `done` with no warning
asserts that the expected product was produced. A phase whose empty product is
NOT valid (the built-in `parsing_chunking` / `embedding` invariants in section
1) reports `error` instead: there is nothing to warn about because the product
does not exist. "Could not run" is different again: a timeout, transport error
or unavailable LLM is still `not_ready` then `error`. (The `processed`/
`expected` counters remain telemetry only — see section 1 — so this is an
obligation on the phase, not a check the machine performs.)

**Budget.** An LLM-backed phase declares `not_ready_budget_seconds` /
`not_ready_budget_env` on the ABC. The graph concept phase points its
`not_ready_budget_env` at `CAT_INGESTION_LLM_NOT_READY_MAX_SECONDS` (default
`300`), so a dead provider releases the source in minutes instead of holding it
for the generic `CAT_INGESTION_NOT_READY_MAX_SECONDS` (`3600`). The resolution
order is the same as the general budget: declared env var → declared value →
generic.

**OPEN DECISION — partial output.** *When a partially successful phase may be
considered valid is not decided.* Until it is, partial output remains an
`error` unless the phase can establish that the partial product is valid, in
which case it reports `done` + a `partial-extraction` warning. Either way the
degradation is recorded, never silent. Note that "partial" has three distinct
meanings and only the last is already handled mechanically:

- **per-item within a phase** — the LLM dropped some items the schema asked
  for (the case above);
- **per-source within a run** — some sources finished and some did not;
- **per-phase within a source** — some phases `complete` and some `error`.
  This one is already covered by normal error propagation (section 3).

Any future definition of "valid partial" must still record the degradation in
the diary — `processed=M of N, dropped=K, reason=...` and/or a warning — so a
partial success is never silent.

**`applies_to` must not mask a missing LLM.** A phase that needs an LLM and
has none reports `error`. `applies_to` is a *capability* check — it decides
whether the phase makes sense for a source at all — and must not be used to
turn a configuration problem into a quiet `skipped`.

## 9. Known limitations

These are current limitations, stated plainly so consumers can plan around
them:

1. **Agent deletion per-file cascade.** An agent teardown uses the standard
   teardown path only: the per-source phase deletes are skipped (the
   `agents:{agent_id}:ingestion:delete` marker short-circuits
   `before_file_manager_file_delete`), because the standard teardown wipes the
   vectors and the graph plugin wipes the whole tenant, so every artifact is
   already covered. A single-file delete keeps using the phase path. If the
   marker cannot be read the handler fails open and runs the phase delete, so
   a genuine single-file delete is never skipped on a read error.
2. **Graph purge ordering.** The graph purge walks from the source's
   `Document`s, so it must run **before** any vector wipe. Until entities
   carry their own source locator, removing the vectors first would leave the
   purge with nothing to walk from.
3. **Commit `3d433197` in the cat-efficient-ingestion repository.** That commit carries phase
   3's message but is a real refactor: it extracted the tag resolver. It is
   present on the remote and was not rewritten. Read it as a refactor, not as
   the phase-3 behaviour change its message describes.
