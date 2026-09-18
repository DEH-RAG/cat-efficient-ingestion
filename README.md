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

<h2>🔌 Endpoints</h2>

- `GET /ingestion/status` — the registry, reconciled against the canonical sources (files on disk, URLs in the vector store, existing conversations); pass `?chat_id=<id>` for a conversation scope. In-flight and `error` entries are never purged; only terminal `completed` entries whose source has vanished are removed.
- `DELETE /ingestion/status?source=<name>[&scope=<chat_id>]` — dismiss one terminal row (`error` or `completed`); an in-flight source is refused, remove the file to abandon it.

<h2>♻️ Recovery sweep</h2>

On `after_lizard_bootstrap` the plugin schedules a fire-and-forget pass (per agent, never blocking bootstrap) that:

1. hands every stale `uploaded` / `processing` / `error` entry back to the phase machine, re-reading the file from disk or re-downloading the URL (entries whose file is gone are marked `error` with the reason);
2. purges the status entries whose source is absent from the canonical lists.

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
