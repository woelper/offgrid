# TODO

## ModelScope as an alternate model hub (planned, not started)

Add https://www.modelscope.ai as a second source next to Hugging Face for
search, file listing, and download. Default stays Hugging Face.

### Verified against the live API (2026-08-27)

| Operation | Endpoint | Notes |
|---|---|---|
| Search | `PUT /api/v1/dolphin/models`, JSON body `{Name, PageSize, PageNumber, SortBy:"Default", Criterion:[{category:"libraries", predicate:"contains", values:["gguf"]}]}` | Must be **PUT** (POST → 404). Response `Data.Model.Models[]` with `Path` (namespace), `Name`, `Downloads`, `Libraries`, `Tags`; repo id = `Path/Name`. `Data.Model.TotalCount`. Internal website API, undocumented. |
| File list | `GET /api/v1/models/{ns}/{name}/repo/files?Revision=master&Recursive=true` | `Data.Files[]` with `Path`, `Size`, `Type` (`blob`/`tree`), `IsLFS`, `Sha256`. Missing repo → HTTP 404. |
| Download | `GET /api/v1/models/{ns}/{name}/repo?Revision=master&FilePath={path}` | 302 → CDN (`cdn-lfs-ap-1.modelscope.ai`) with a time-limited `auth_key`. CDN honours Range: 206, `Accept-Ranges`, `Content-Range`, ETag → existing resume logic in `hub.rs` works unchanged (it follows redirects manually with Range preserved). |

`modelscope.cn` serves the identical API (users in China).

Errors can arrive as an HTTP status *or* as `Code != 200` inside a 200 body;
handle both.

### Catalog mapping (probed per entry)

Same filenames as HF (`…-Q4_K_M.gguf`) where a mirror exists.

| Catalog entry | ModelScope repo |
|---|---|
| Qwen3 0.6B / 1.7B | `unsloth/Qwen3-0.6B-GGUF`, `unsloth/Qwen3-1.7B-GGUF` (same as HF) |
| Llama 3.2 1B / 3B | `unsloth/Llama-3.2-1B-Instruct-GGUF`, `unsloth/Llama-3.2-3B-Instruct-GGUF` |
| Gemma 3 4B | `unsloth/gemma-3-4b-it-GGUF` |
| Qwen3 4B Instruct 2507 | `unsloth/Qwen3-4B-Instruct-2507-GGUF` |
| Qwen3-Coder-30B-A3B | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` (same as HF) |
| Mistral 7B Instruct v0.3 | **none found** (unsloth / LLM-Research / QuantFactory all 404) → fall back to HF with a status note |

`bartowski/*` repos are not mirrored on ModelScope.

### Design

1. **Backend seam in `hub.rs`.** `pub enum Hub { HuggingFace, ModelScope }`
   (serde; config default HuggingFace). Per-backend URL builders + response
   parsers; `spawn_search`, `spawn_list_files`, `start_download` take a
   `Hub`. `HubEvent`, `RepoResult`, `RepoFile`, `DownloadEvent`,
   `ActiveDownload` unchanged so GUI/TUI list rendering does not change.
   `PartMeta` gains `hub` (serde default HF) so a download interrupted by a
   restart resumes from the right backend. Search is a PUT with a JSON body:
   `ureq` here has no `json` feature, so send the `serde_json` string with a
   `content-type: application/json` header — no new dependency.
2. **Catalog.** `CatalogEntry` gains `ms: Option<(&'static str, &'static str)>`
   (repo, file). `propose` / `/get` use it when the active hub is ModelScope;
   `None` → HF regardless, with "not on ModelScope — fetching from Hugging
   Face".
3. **Config.** `hub: "huggingface" | "modelscope"`,
   `modelscope_domain: "modelscope.ai" | "modelscope.cn"` (default `.ai`).
4. **UI.** Desktop: a Source selector in the Get models group next to the
   search box; result/download rows show the source. TUI: `/hub hf|ms` (bare
   `/hub` shows current); hint line shows it like `web on`. Server: nothing.
5. **Filtering** unchanged: multi-part (`-of-`) shards skipped,
   `models::is_model_file` applied.
6. **Optional:** verify `Sha256` from the listing after a ModelScope download
   (HF path has no equivalent today).
7. **Tests.** Fixture-based parsers for both backends (search + file-list
   JSON), URL-builder tests, one `#[ignore]` live ModelScope search test
   (like `web_search_live`). Loopback resume tests already cover downloads
   backend-agnostically.
8. **README.** Document the switch, `.cn`, and that it is the same privacy
   posture (a query leaves the machine only when you search).

### Risks

- The search endpoint is the website's internal API and may change; it is
  isolated in one function with fixture tests, and failure degrades to a
  clear error. HF stays the default and untouched.
- CDN `auth_key` expires; harmless — every attempt (and resume) starts from
  the API URL and gets a fresh redirect.

### Open decisions

1. Search-only, or also the curated catalog via the mirror table
   (recommended)?
2. Include Sha256 verification for ModelScope downloads, or defer?

Estimated size: ~300–400 lines across `hub.rs`, `models.rs`, `config.rs`,
`app.rs`, `tui.rs`, plus tests.
