use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

const N_CTX: u32 = 8192;

#[derive(Clone, PartialEq)]
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

#[derive(Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum LlmCmd {
    Load(PathBuf),
    Unload,
    /// Generation events (Token/GenDone/Error) are sent to `reply`, so the
    /// chat UI and API server requests can share one worker.
    Generate {
        messages: Vec<ChatMessage>,
        reply: Sender<LlmEvent>,
        /// Sampling temperature: ~0.7 for chat, lower (~0.25) for agent/tool
        /// use where malformed JSON and sloppy code are costly.
        temp: f32,
    },
}

pub enum LlmEvent {
    Loaded(String),
    Unloaded,
    Token(String),
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
}

pub fn spawn_worker(n_threads: usize) -> LlmHandle {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
    let (event_tx, event_rx) = std::sync::mpsc::channel::<LlmEvent>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = stop.clone();
    let event_tx_worker = event_tx.clone();
    std::thread::spawn(move || worker(cmd_rx, event_tx_worker, stop_worker, n_threads));
    LlmHandle {
        cmd_tx,
        event_tx,
        event_rx,
        stop,
    }
}

fn worker(cmd_rx: Receiver<LlmCmd>, tx: Sender<LlmEvent>, stop: Arc<AtomicBool>, n_threads: usize) {
    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            let _ = tx.send(LlmEvent::Error(format!("llama backend init failed: {e}")));
            return;
        }
    };
    let mut model: Option<LlamaModel> = None;

    for cmd in cmd_rx {
        match cmd {
            LlmCmd::Load(path) => {
                model = None; // free the old model before loading the new one
                let params = LlamaModelParams::default();
                match LlamaModel::load_from_file(&backend, &path, &params) {
                    Ok(m) => {
                        model = Some(m);
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let _ = tx.send(LlmEvent::Loaded(name));
                    }
                    Err(e) => {
                        let _ = tx.send(LlmEvent::Error(format!("failed to load model: {e}")));
                    }
                }
            }
            LlmCmd::Unload => {
                model = None;
                let _ = tx.send(LlmEvent::Unloaded);
            }
            LlmCmd::Generate {
                messages,
                reply,
                temp,
            } => {
                let Some(model) = &model else {
                    let _ = reply.send(LlmEvent::Error("no model loaded".into()));
                    continue;
                };
                if let Err(e) = generate(&backend, model, &messages, &reply, &stop, n_threads, temp)
                {
                    let _ = reply.send(LlmEvent::Error(e));
                }
                let _ = reply.send(LlmEvent::GenDone);
            }
        }
    }
}

fn generate(
    backend: &LlamaBackend,
    model: &LlamaModel,
    messages: &[ChatMessage],
    tx: &Sender<LlmEvent>,
    stop: &AtomicBool,
    n_threads: usize,
    temp: f32,
) -> Result<(), String> {
    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.as_str().to_string(), m.content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    // Old GGUF files (pre-2024) carry no embedded chat template. Fall back to
    // ChatML — llama.cpp resolves the name to its built-in template. Not the
    // format those models were trained on, but a workable degradation.
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
    if tokens.len() as u32 >= N_CTX - 256 {
        // Callers match on this prefix to offer their own remedy.
        return Err(format!(
            "context window full ({} tokens, limit {})",
            tokens.len(),
            N_CTX
        ));
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX))
        .with_n_batch(N_CTX)
        .with_n_threads(n_threads as i32)
        .with_n_threads_batch(n_threads as i32);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| e.to_string())?;

    let mut batch = LlamaBatch::new(N_CTX as usize, 1);
    let last_idx = tokens.len() - 1;
    for (i, token) in tokens.iter().enumerate() {
        batch
            .add(*token, i as i32, &[0], i == last_idx)
            .map_err(|e| e.to_string())?;
    }
    let prompt_start = std::time::Instant::now();
    ctx.decode(&mut batch).map_err(|e| e.to_string())?;
    let prompt_secs = prompt_start.elapsed().as_secs_f32();

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
    while (n_cur as u32) < N_CTX && !stop.load(Ordering::Relaxed) {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
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
