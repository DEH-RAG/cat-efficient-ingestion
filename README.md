<h1 align="center" id="title">Grinning Cat Efficient Ingestion</h1>

<p id="description">This is a core plugin for the <a href="https://github.com/Grinning-Cat-RAG/grinning-cat-core">Grinning Cat</a> Project which replaces the default ingestion engine with a two-phase, resumable one: it tracks the state of every ingested source, re-embeds the stored chunks instead of re-ingesting the files when the embedder changes, and recovers ingestions interrupted by a crash or a restart.</p>

<h2>🧐 What it does</h2>

The plugin registers `EfficientIngestionEngine` through the `factory_allowed_ingestions` hook, so the core resolves it via the ServiceFactory (`ingestion` category) instead of the default engine. The same engine drives **one** state machine for the whole lifecycle — fresh uploads, recovery, re-embed and re-ingest:

- **`parsing_chunking`**: deletes every artifact of the source (text chunk points, image points, saved image files), then parses, chunks and extracts the images, storing text chunks and image points with **empty vectors** (`vector={}`). No embedding is computed here.
- **`embedding`**: recomputes the vectors of the stored points — `embed_documents` for the text chunks, `embed_images` for the image points when the active embedder is multimodal (otherwise they stay payload-only) — replaces them and marks the source `completed`.

Because the two phases are separate and recorded, an embedder change only needs the `embedding` phase: the stored chunks are reused and the files are not parsed again (with a fallback to the full re-ingest when the chunker changed too). A crash between the two phases resumes from the phase written in the status document.

<h2>🗂️ Ingestion status registry</h2>

Every transition is written as a Redis-JSON document under `agents:{agent_id}:ingestion:{scope}:{sha256(source)}`, where `scope` is `agent` for the agent KB or a conversation `chat_id`.

Lifecycle states: `uploaded → processing → completed | error`, with `downloading → downloaded` inserted for URLs. The document also records the work `phase` (`downloading`, `parsing_chunking`, `embedding`), the `embedder_name` and the `chunker_name` that produced the stored data, plus the timestamps used by the recovery sweep.

Sources are claimed under a **per-source distributed lock**, so different sources of the same agent are ingested concurrently while the same source is never processed twice by two workers. A heartbeat keeps the `processing` row fresh while it is being handled, so live work is never stolen by another replica.

<h2>🧭 Clock-free two-level phase invalidation</h2>

The plugin decides whether a source must be re-run with a **clock-free** model: no timestamps are ever used for correctness. Each status document carries a `completed_phases` diary, keyed by phase id:

```
completed_phases: {phase: {"marker": Any, "settings_version": str|None,
                           "deps": {upstream_phase: upstream_marker}}}
```

- `marker` is the plugin-defined marker of the settings the phase ran with;
- `settings_version` is an **opaque** copy of the settings entry's `updated_at`, used only for equality, never as a time;
- `deps` maps each upstream phase to the marker it ran with.

A phase is stale when any of these hold:

- **(a) dependency**: an upstream in its `depends_on` is missing from the diary, or the recorded `deps[upstream]` differs from the upstream's current diary `marker` (the upstream re-ran, so this phase must re-run too);
- **(b) settings fast-path**: the settings entry for the phase's category was **not** rewritten since the phase ran (`settings_version` equals the current entry's `updated_at`, compared as an opaque token) → FRESH, no marker comparison needed;
- **(c) slow-path**: the plugin-defined marker from `ingestion_phase_settings_marker` differs from the recorded `marker` (the settings entry *was* rewritten; the phase-owning plugin decides whether the change is material).

The built-in phase DAG is `parsing_chunking` (depends on the chunker settings) → `embedding` (depends on the embedder settings **and** the recorded `parsing_chunking` marker). So an embedder-only change re-runs only `embedding`, while a chunker change re-runs `parsing_chunking` first and `embedding` follows on the next probe.

**Atomic-completion invariant**: a phase's diary entry is written **only** on successful completion, in a single atomic write, never at phase start. A phase interrupted by a restart therefore has no entry and is re-run on recovery.

<h2>🔌 Phase hooks</h2>

The phase machine is extensible through five hooks declared in the plugin itself (`hooks.py`, `@hook(priority=0)`), so external plugins can override them with higher priorities — no MyCAT core change is needed. The `completed_phases` argument threaded through the hooks is a `list[dict]` where every entry carries at least a `"phase"` key (MyGRAPH-compatible), plus optional `marker`/`deps` keys.

- **`ingestion_phase_pending(pending, source, completed_phases, cat)`** — accumulator of stale phases for a source. Each registrant appends the `{"phase": <id>, ...}` entries it considers stale and returns the extended list. The default is the identity (nothing pending).
- **`ingestion_phase_run(phase, source, completed_phases, cat)`** — runs one phase with a tri-state contract: return `{"status": "done"}` on success, `{"status": "not_ready", "retry_after": N}` when retriable but not yet runnable (the machine retries after `N` seconds), or raise on permanent failure. The default returns `None`, which the machine treats as fail-hard / unimplemented: a phase with no registrant is an error, never a silent success.
- **`before_ingestion_status_completed(source, cat)`** — final gate before a source is marked COMPLETED. A registrant raises to force the source to ERROR instead; the default is a no-op.
- **`ingestion_phase_settings_marker(phase, cat)`** — the plugin-defined material marker for a phase, compared against the recorded diary `marker` to decide whether the phase's settings changed materially. The default returns `None` ("unknown"), which the machine treats conservatively as stale.
- **`ingestion_phase_specs(specs, cat)`** — accumulator of `PhaseSpec` declarations. Each registrant appends the phases it owns and returns the extended list; EffING merges them into its built-in `PHASES` at probe time (built-ins win on duplicate ids). The default is the identity (no registered phases).

An external phase plugin (e.g. MyGRAPH) registers by declaring its phases through `ingestion_phase_specs`, providing a material marker through `ingestion_phase_settings_marker`, and executing them through `ingestion_phase_run`; the machine dispatches any phase not in its built-in `PHASES` through the run hook and records a real diary entry once it reports `done`. See the phase-registration API section below.

<h2>📦 Phase-registration API</h2>

External plugins can declare their own ingestion phases without touching the EffING machine. A provider registers through the **`ingestion_phase_specs`** accumulator hook: it appends `PhaseSpec` objects and returns the extended list. EffING merges them into its built-in `PHASES` at probe time via `merged_phases(ccat, cat)` — built-ins win on duplicate ids, malformed output (`None`, non-`PhaseSpec` entries) is skipped, and the built-in `PHASES` dict is never mutated.

A `PhaseSpec` declares three things:

- `id` — the phase id (also the diary key);
- `settings_category` — the settings category whose `updated_at` is the opaque fast-path token, or `None` for phases with no settings entry (marker-only invalidation);
- `depends_on` — upstream phase ids whose recorded markers this phase consumed.

A provider only declares a spec + a marker + an execution hook:

- **`ingestion_phase_specs(specs, cat)`** — declares the phase (accumulator);
- **`ingestion_phase_settings_marker(phase, cat)`** — the material marker compared against the recorded diary `marker` (the provider decides whether a settings rewrite is material);
- **`ingestion_phase_run(phase, source, completed_phases, cat)`** — executes the phase (tri-state contract).

EffING computes staleness for ALL merged phases (deps by upstream marker identity + marker comparison), so a provider never re-implements the pending probe. Registered phases get REAL diary entries: `_record_phase` writes the marker from `ingestion_phase_settings_marker`, reads `settings_version` only when `settings_category` is set, and copies `deps` from the current diary upstream markers — the same atomic-completion invariant as the built-ins. A registered phase is therefore restartable and re-runs automatically when its upstreams re-run or its marker changes.

<h2>🔁 Probe-driven dispatcher</h2>

`reembed_sources` is the one probe-driven dispatcher. For each source it reads the `completed_phases` diary (backfilling legacy completed rows), converts it to the hook list shape and threads it into the `ingestion_phase_pending` accumulator. Registrants compare the diary against the current settings markers/versions and report the stale phases:

- **no stale phases** → the source is up to date and is **skipped without claiming**;
- **stale phases** → the row is claimed (per-source lock) and the stale phases run **serially**, one at a time in probe order.

After each phase succeeds, its diary entry (`marker` / `settings_version` / `deps`) is recorded **atomically at completion** — never at phase start — and the probe is re-run against the updated diary. When the re-probe is empty, the terminal COMPLETED status is written (after the `before_ingestion_status_completed` gate), with the work phase cleared. An `error` row is never advanced nor resurrected to COMPLETED.

<h2>🛡️ Hardening</h2>

- **Delete-marker cancellation**: deleting an agent writes a persistent marker at `agents:{agent_id}:ingestion:delete` (inside the plugin's own namespace, so the teardown wipe includes it). Every status write, claim, list and clear checks `ingestion_canceled` (marker present **or** the agent's master key gone) and self-aborts. The heartbeat checks the marker on every tick, so a long embedding self-aborts within one interval instead of letting the delete's quiesce-wait time out.
- **Ghost agents**: an agent whose master key is gone (peripheral keys without a master) is treated as canceled; a `CustomNotFoundException` on resolution is a skip, never a crash.
- **Completed-row revalidation**: the recovery sweep re-runs the `ingestion_phase_pending` probe read-only for every completed row and only claims the rows whose probe comes back non-empty — a genuinely-fresh completed row is never touched.

<h2>🔌 Endpoints</h2>

- `GET /ingestion/status` — the registry, reconciled against the canonical sources (files on disk, URLs in the vector store, existing conversations); pass `?chat_id=<id>` for a conversation scope. In-flight and `error` entries are never purged; only terminal `completed` entries whose source has vanished are removed.
- `DELETE /ingestion/status?source=<name>[&scope=<chat_id>]` — dismiss one terminal row (`error` or `completed`); an in-flight source is refused, remove the file to abandon it.
- `GET /ingestion/settings` — list the available ingestion engines with their schemes and the effective choice (SYSTEM READ).
- `GET /ingestion/settings/{name}` — settings and scheme of one ingestion engine configuration (SYSTEM READ).
- `PUT /ingestion/settings/{name}` — update one engine configuration and select it as the engine to run (SYSTEM WRITE). The `ingestion` category holds a single setting (the active engine), so upserting the config replaces it.

<h2>♻️ Recovery sweep</h2>

On `after_lizard_bootstrap` the plugin schedules a fire-and-forget pass (per agent, never blocking bootstrap) that:

1. hands every stale `uploaded` / `processing` / `error` entry back to the phase machine, re-reading the file from disk or re-downloading the URL (entries whose file is gone are marked `error` with the reason);
2. revalidates `completed` rows through the clock-free probe (see Hardening);
3. purges the status entries whose source is absent from the canonical lists.

The pass is repeated every `CAT_INGESTION_RESUME_INTERVAL_SECONDS`.

<h2>🧵 Ingestion executor lane</h2>

Chunking, embedding and storing are dispatched to a dedicated, low-concurrency `ThreadPoolExecutor` (exposed through the `run_in_ingestion_executor` hook) instead of the shared default one, and its worker threads are de-prioritized with `nice`: a heavy ingestion yields CPU to chat and recall instead of competing with them. With the lane disabled the calls fall back to the default executor. It complements the core-side `CAT_INGESTION_MAX_CONCURRENCY` semaphore: that one bounds how many ingestion tasks run at once, this lane keeps them off the shared pool.

<h2>✂️ Token-budget split</h2>

The plugin also owns the oversized-chunk split (`finalize_oversized_chunks` and `before_rabbithole_stores_documents` hooks): chunks longer than the active embedder's `max_input_tokens` are split into budget-compliant sub-chunks, measured with the embedder's own tokenizer when available, with the original metadata carried forward. With this plugin disabled the core returns to upstream parity (no split).

<h2>⚙️ Plugin settings</h2>

Settings live in the global `system:agent` store, category `ingestion`:

- **Ingestion max concurrency** (`ingestion_max_concurrency`, default `5`): how many agents are re-embedded at the same time during a re-embed pass. The legacy field name `reembed_max_concurrency` is still accepted and migrated automatically.

<h2>🌍 Environment variables</h2>

| Variable | Default | Description |
| --- | --- | --- |
| `CAT_INGESTION_WORKERS` | `2` | Size of the dedicated ingestion thread pool; `<= 0` disables the lane and falls back to the default executor |
| `CAT_INGESTION_NICENESS` | `5` | `nice` applied to the ingestion worker threads; `<= 0` disables the de-prioritization |
| `CAT_INGESTION_HEARTBEAT_SECONDS` | `30` | Interval at which a `processing` row is refreshed while being handled |
| `CAT_INGESTION_RESUME_ON_STARTUP` | `true` | Enable the resume part of the sweep |
| `CAT_INGESTION_STATUS_GC_ON_STARTUP` | `true` | Enable the reconcile (GC) part of the sweep |
| `CAT_INGESTION_RESUME_INTERVAL_SECONDS` | `60` | Period of the recurring sweep; `0` runs the pass only at bootstrap |
| `CAT_INGESTION_RESUME_STALE_SECONDS` | `300` | How old an entry must be to be claimable by the sweep |
| `CAT_INGESTION_RESUME_BOOT_STALE_SECONDS` | `2 x heartbeat` (min `30`) | Staleness used by the pass right after a restart, so rows left behind by a dead worker are recovered quickly |

<h2>🛠️ Installation:</h2>

<p>1. Clone this repo and copy it on cat plugins folder</p>
<p>2. Install from admin panel on the [Grinning Cat Web Admin](https://github.com/matteocacciola/grinning-cat-admin)</p>

<h2>🛡️ License:</h2>
This project is licensed under the GNU GENERAL PUBLIC LICENSE
