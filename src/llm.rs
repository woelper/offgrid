use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

/// Default context window; configurable in Settings. Prompt decoding is
/// chunked, so a large window costs KV-cache memory but not batch memory.
pub const DEFAULT_N_CTX: u32 = 16384;
const N_BATCH: u32 = 2048;

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum LlmCmd {
    /// `n_ctx` is needed at load time, not just at generation time: it sizes
    /// the KV cache, which shares the GPU's memory with the offloaded layers.
    Load {
        path: PathBuf,
        n_ctx: u32,
    },
    Unload,
    /// Generation events (Token/GenDone/Error) are sent to `reply`, so the
    /// chat UI and API server requests can share one worker. The messages are
    /// shared rather than copied: the caller (the agent loop) is usually the
    /// only owner, so it can grow them in place between turns.
    Generate {
        messages: Arc<Vec<ChatMessage>>,
        reply: Sender<LlmEvent>,
        /// Sampling temperature: ~0.7 for chat, lower (~0.25) for agent/tool
        /// use where malformed JSON and sloppy code are costly.
        temp: f32,
        /// Context window size in tokens.
        n_ctx: u32,
    },
}

pub enum LlmEvent {
    Loaded(String),
    Unloaded,
    Token(String),
    /// A human-readable progress note that is not part of the answer — e.g.
    /// web-augmented chat reporting "searching the web…" during its pre-pass,
    /// when no tokens are streaming and the UI would otherwise look frozen.
    Info(String),
    Stats {
        prompt_tokens: usize,
        prompt_secs: f32,
        gen_tokens: usize,
        gen_secs: f32,
    },
    GenDone,
    Error(String),
}

pub struct LlmHandle {
    pub cmd_tx: Sender<LlmCmd>,
    pub event_tx: Sender<LlmEvent>,
    pub event_rx: Receiver<LlmEvent>,
    pub stop: Arc<AtomicBool>,
    /// Where the last loaded model ended up — "29/33 layers on NVIDIA …" —
    /// or empty when nothing is loaded or there is no GPU. Written by the
    /// worker at load time, read by the UI.
    pub accel: Arc<Mutex<String>>,
}

pub fn spawn_worker(n_threads: usize) -> LlmHandle {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
    let (event_tx, event_rx) = std::sync::mpsc::channel::<LlmEvent>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = stop.clone();
    let event_tx_worker = event_tx.clone();
    let accel = Arc::new(Mutex::new(String::new()));
    let accel_worker = accel.clone();
    std::thread::spawn(move || {
        worker(
            cmd_rx,
            event_tx_worker,
            stop_worker,
            n_threads,
            accel_worker,
        )
    });
    LlmHandle {
        cmd_tx,
        event_tx,
        event_rx,
        stop,
        accel,
    }
}

fn worker(
    cmd_rx: Receiver<LlmCmd>,
    tx: Sender<LlmEvent>,
    stop: Arc<AtomicBool>,
    n_threads: usize,
    accel: Arc<Mutex<String>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            let _ = tx.send(LlmEvent::Error(format!("llama backend init failed: {e}")));
            return;
        }
    };
    // A `LlamaContext` borrows its model, so the two cannot live in one struct.
    // Instead, a load runs `run_model` in the model's scope; the command that
    // needs a different model is handed back here to be replayed.
    let mut pending: Option<LlmCmd> = None;
    loop {
        let cmd = match pending.take() {
            Some(cmd) => cmd,
            None => match cmd_rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => return,
            },
        };
        match cmd {
            LlmCmd::Unload => {
                if let Ok(mut slot) = accel.lock() {
                    slot.clear();
                }
                let _ = tx.send(LlmEvent::Unloaded);
            }
            // No model: fail the request rather than silently dropping it.
            LlmCmd::Generate { reply, .. } => {
                let _ = reply.send(LlmEvent::Error("no model loaded".into()));
                let _ = reply.send(LlmEvent::GenDone);
            }
            LlmCmd::Load { path, n_ctx } => {
                // Free the old model before loading the new one.
                let (params, note) = load_params(&path, n_ctx);
                if let Ok(mut slot) = accel.lock() {
                    *slot = note;
                }
                match LlamaModel::load_from_file(&backend, &path, &params) {
                    Ok(model) => {
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let _ = tx.send(LlmEvent::Loaded(name));
                        pending = run_model(&model, &backend, &cmd_rx, &stop, n_threads);
                        if pending.is_none() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(LlmEvent::Error(format!("failed to load model: {e}")));
                    }
                }
            }
        }
    }
}

/// Serve one loaded model until a command needs the model replaced. Returns
/// the command that ended the session (`Load`/`Unload`), or `None` if the
/// command channel closed.
fn run_model(
    model: &LlamaModel,
    backend: &LlamaBackend,
    cmd_rx: &Receiver<LlmCmd>,
    stop: &AtomicBool,
    n_threads: usize,
) -> Option<LlmCmd> {
    let mut session = Session::default();
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            LlmCmd::Load { .. } | LlmCmd::Unload => return Some(cmd),
            LlmCmd::Generate {
                messages,
                reply,
                temp,
                n_ctx,
            } => {
                if let Err(e) = session.generate(
                    model, backend, &messages, &reply, stop, n_threads, temp, n_ctx,
                ) {
                    let _ = reply.send(LlmEvent::Error(e));
                }
                let _ = reply.send(LlmEvent::GenDone);
            }
        }
    }
    None
}

/// Per-model inference state that outlives a single generation: the context
/// and the exact token sequence currently evaluated in its KV cache. A new
/// prompt usually shares a long prefix with the last one (the agent loop only
/// appends turns), so only the tail has to be decoded.
#[derive(Default)]
struct Session<'m> {
    ctx: Option<LlamaContext<'m>>,
    cache: Vec<LlamaToken>,
    n_ctx: u32,
}

impl<'m> Session<'m> {
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        model: &'m LlamaModel,
        backend: &LlamaBackend,
        messages: &[ChatMessage],
        tx: &Sender<LlmEvent>,
        stop: &AtomicBool,
        n_threads: usize,
        temp: f32,
        n_ctx: u32,
    ) -> Result<(), String> {
        let n_ctx = n_ctx.max(2048);
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.as_str().to_string(), m.content.clone()))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        // Old GGUF files (pre-2024) carry no embedded chat template. Fall back
        // to ChatML — llama.cpp resolves the name to its built-in template. Not
        // the format those models were trained on, but a workable degradation.
        let template = match model.chat_template(None) {
            Ok(t) => t,
            Err(_) => LlamaChatTemplate::new("chatml")
                .map_err(|e| format!("chat template fallback failed: {e}"))?,
        };
        let prompt = model
            .apply_chat_template(&template, &chat, true)
            .map_err(|e| e.to_string())?;

        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| e.to_string())?;
        if tokens.is_empty() {
            return Err("prompt tokenised to nothing".into());
        }
        if tokens.len() as u32 >= n_ctx - 256 {
            // Callers match on this prefix to offer their own remedy.
            return Err(format!(
                "context window full ({} tokens, limit {})",
                tokens.len(),
                n_ctx
            ));
        }

        // (Re)create the context only when the requested window changed;
        // otherwise its KV cache is still useful for the prefix reuse below.
        if self.ctx.is_none() || self.n_ctx != n_ctx {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(N_BATCH)
                .with_n_threads(n_threads as i32)
                .with_n_threads_batch(n_threads as i32);
            self.ctx = Some(
                model
                    .new_context(backend, ctx_params)
                    .map_err(|e| e.to_string())?,
            );
            self.n_ctx = n_ctx;
            self.cache.clear();
        }
        // Split the borrow so the context and the token cache can be used
        // together (they are distinct fields).
        let Session { ctx, cache, .. } = self;
        let ctx = ctx.as_mut().expect("context created above");

        // Longest prefix already in the cache, leaving one token to decode so
        // the sampler always gets fresh logits for the final position.
        let mut common = reusable_prefix(cache, &tokens);
        if common < cache.len() && ctx.kv_cache_seq_rm(0, Some(common as u32), None).is_err() {
            // The backend cannot drop a partial sequence (sliding-window or
            // recurrent models): start the cache over.
            ctx.clear_kv_cache();
            cache.clear();
            common = 0;
        }
        cache.truncate(common);

        // If any decode fails, the cache no longer matches the context, so the
        // next call must start over rather than reuse a stale prefix.
        let result = decode_and_generate(ctx, cache, model, tx, stop, temp, n_ctx, &tokens, common);
        if result.is_err() {
            ctx.clear_kv_cache();
            cache.clear();
        }
        result
    }
}

/// Decode the prompt tail past `common` and stream the answer, keeping `cache`
/// (the tokens evaluated in `ctx`) up to date as generation proceeds.
/// Decide where a model's layers go, and say so in one line for the UI.
///
/// Without a GPU (or in a build with no GPU backend) this is llama.cpp's
/// default and nothing changes. With a discrete card we count the layers
/// ourselves instead of leaving `n_gpu_layers` at its default of -1, "all":
/// a model that does not fit aborts the load with an out-of-memory error
/// rather than falling back, and "all" is a bet we cannot win on an 8 GB
/// card with a 16k context. Unified memory (Apple, integrated GPUs) keeps
/// the default — there is one pool, and llama.cpp splits it better than a
/// VRAM budget we do not have.
fn load_params(path: &std::path::Path, n_ctx: u32) -> (LlamaModelParams, String) {
    // Match the floor `generate` applies, so the KV cache we budget for is
    // the one that actually gets allocated.
    let n_ctx = n_ctx.max(2048);
    let params = LlamaModelParams::default();
    let Some(gpu) = crate::hardware::gpu() else {
        return (params, String::new());
    };
    if gpu.unified {
        return (params, format!("{} ({})", gpu.name, gpu.backend));
    }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    // No readable header means no layer count, and guessing with a card's
    // worth of VRAM at stake is not worth it — keep the weights in RAM.
    let Some(dims) = crate::models::gguf_dims(path) else {
        return (
            params.with_n_gpu_layers(0),
            format!("CPU only — could not read the layer count for {}", gpu.name),
        );
    };
    let n = crate::models::gpu_layers(size, &dims, n_ctx, gpu.budget());
    let note = if n == 0 {
        format!(
            "CPU only — {} has no room at a {n_ctx} token context",
            gpu.name
        )
    } else if u64::from(n) > dims.layers {
        format!("all {} layers on {}", dims.layers, gpu.name)
    } else {
        format!("{n} of {} layers on {}", dims.layers, gpu.name)
    };
    (params.with_n_gpu_layers(n), note)
}

#[allow(clippy::too_many_arguments)]
fn decode_and_generate(
    ctx: &mut LlamaContext<'_>,
    cache: &mut Vec<LlamaToken>,
    model: &LlamaModel,
    tx: &Sender<LlmEvent>,
    stop: &AtomicBool,
    temp: f32,
    n_ctx: u32,
    tokens: &[LlamaToken],
    common: usize,
) -> Result<(), String> {
    // Decode only the tokens past the shared prefix, in chunks so batch memory
    // stays bounded no matter how large the window is.
    let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
    let prompt_start = std::time::Instant::now();
    let last_idx = tokens.len() - 1;
    let mut pos = common;
    for chunk in tokens[common..].chunks(N_BATCH as usize) {
        batch.clear();
        for token in chunk {
            batch
                .add(*token, pos as i32, &[0], pos == last_idx)
                .map_err(|e| e.to_string())?;
            pos += 1;
        }
        ctx.decode(&mut batch).map_err(|e| e.to_string())?;
    }
    let prompt_secs = prompt_start.elapsed().as_secs_f32();
    cache.extend_from_slice(&tokens[common..]);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(42);
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temp.max(0.05)),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::dist(seed),
    ]);

    let mut n_cur = tokens.len() as i32;
    let mut pending: Vec<u8> = Vec::new();
    let gen_start = std::time::Instant::now();
    let mut gen_tokens = 0usize;
    while (n_cur as u32) < n_ctx && !stop.load(Ordering::Relaxed) {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        gen_tokens += 1;
        if let Ok(bytes) = model.token_to_piece_bytes(token, 256, true, None) {
            pending.extend_from_slice(&bytes);
            let text = drain_valid_utf8(&mut pending);
            if !text.is_empty() {
                let _ = tx.send(LlmEvent::Token(text));
            }
        }
        cache.push(token);
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| e.to_string())?;
        n_cur += 1;
        ctx.decode(&mut batch).map_err(|e| e.to_string())?;
    }
    let _ = tx.send(LlmEvent::Stats {
        prompt_tokens: tokens.len(),
        prompt_secs,
        gen_tokens,
        gen_secs: gen_start.elapsed().as_secs_f32(),
    });
    Ok(())
}

/// Leading tokens shared by the cached sequence and the new prompt, capped so
/// at least one token is decoded (the sampler needs fresh logits).
fn reusable_prefix(cache: &[LlamaToken], tokens: &[LlamaToken]) -> usize {
    cache
        .iter()
        .zip(tokens)
        .take_while(|(a, b)| a == b)
        .count()
        .min(tokens.len().saturating_sub(1))
}

/// Extract the valid UTF-8 prefix from `pending`, leaving incomplete trailing
/// bytes (e.g. half of a multi-byte emoji split across tokens) for later.
fn drain_valid_utf8(pending: &mut Vec<u8>) -> String {
    match std::str::from_utf8(pending) {
        Ok(s) => {
            let s = s.to_string();
            pending.clear();
            s
        }
        Err(e) => {
            let valid = e.valid_up_to();
            let mut s = String::from_utf8_lossy(&pending[..valid]).into_owned();
            let mut consumed = valid;
            if let Some(bad) = e.error_len() {
                s.push('\u{FFFD}');
                consumed += bad;
            }
            pending.drain(..consumed);
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    #[test]
    fn prefix_reuse_keeps_only_the_shared_head() {
        // Same opening, different tail: only the shared head is reusable.
        assert_eq!(reusable_prefix(&toks(&[1, 2, 3, 4]), &toks(&[1, 2, 9])), 2);
    }

    #[test]
    fn prefix_reuse_always_leaves_one_token_to_decode() {
        // Fully cached prompt still decodes its last token for fresh logits.
        assert_eq!(reusable_prefix(&toks(&[1, 2, 3]), &toks(&[1, 2, 3])), 2);
        // A single-token prompt can never be fully reused.
        assert_eq!(reusable_prefix(&toks(&[7]), &toks(&[7])), 0);
    }

    #[test]
    fn prefix_reuse_handles_empty_cache_and_divergence() {
        assert_eq!(reusable_prefix(&[], &toks(&[1, 2, 3])), 0);
        assert_eq!(reusable_prefix(&toks(&[1, 2, 3]), &toks(&[4, 5])), 0);
        // Cache longer than the new prompt: capping keeps a token to decode.
        assert_eq!(reusable_prefix(&toks(&[1, 2, 3, 4]), &toks(&[1, 2])), 1);
    }
}
