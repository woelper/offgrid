use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tiny_http::{Header, Response, StatusCode};

use crate::llm::{ChatMessage, LlmCmd, LlmEvent, Role};
use crate::models;

pub const DEFAULT_PORT: u16 = 11633;

pub struct ApiServer {
    stop: Arc<AtomicBool>,
}

impl ApiServer {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for ApiServer {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn start(
    port: u16,
    cmd_tx: Sender<LlmCmd>,
    models_dir: PathBuf,
    loaded_model: Arc<Mutex<Option<String>>>,
) -> Result<ApiServer, String> {
    let server =
        tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| format!("server: {e}"))?;
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    std::thread::spawn(move || {
        while !stop_thread.load(Ordering::Relaxed) {
            match server.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(Some(request)) => {
                    let cmd_tx = cmd_tx.clone();
                    let models_dir = models_dir.clone();
                    let loaded_model = loaded_model.clone();
                    std::thread::spawn(move || {
                        handle(request, cmd_tx, models_dir, loaded_model)
                    });
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    Ok(ApiServer { stop })
}

fn json_response(status: u16, body: serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let data = body.to_string().into_bytes();
    Response::new(
        StatusCode(status),
        vec![Header::from_bytes("Content-Type", "application/json").unwrap()],
        std::io::Cursor::new(data),
        None,
        None,
    )
}

fn handle(
    mut request: tiny_http::Request,
    cmd_tx: Sender<LlmCmd>,
    models_dir: PathBuf,
    loaded_model: Arc<Mutex<Option<String>>>,
) {
    let url = request.url().to_string();
    let method = request.method().as_str().to_string();

    match (method.as_str(), url.as_str()) {
        ("GET", "/v1/models") => {
            let loaded = loaded_model.lock().unwrap().clone();
            let mut names: Vec<String> = models::scan_local(&models_dir)
                .into_iter()
                .map(|m| m.name)
                .collect();
            if let Some(l) = &loaded {
                names.retain(|n| n != l);
                names.insert(0, l.clone());
            }
            let data: Vec<_> = names
                .iter()
                .map(|n| json!({"id": n, "object": "model", "owned_by": "offgrid"}))
                .collect();
            let _ = request.respond(json_response(200, json!({"object": "list", "data": data})));
        }
        ("POST", "/v1/chat/completions") => {
            let mut body = String::new();
            if request.as_reader().read_to_string(&mut body).is_err() {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "unreadable body"}}),
                ));
                return;
            }
            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&body) else {
                let _ = request
                    .respond(json_response(400, json!({"error": {"message": "invalid JSON"}})));
                return;
            };
            let model_name = loaded_model
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "offgrid".to_string());
            let messages: Vec<ChatMessage> = payload
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            let role = match m.get("role")?.as_str()? {
                                "system" => Role::System,
                                "assistant" => Role::Assistant,
                                _ => Role::User,
                            };
                            Some(ChatMessage {
                                role,
                                content: m.get("content")?.as_str()?.to_string(),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            if messages.is_empty() {
                let _ = request
                    .respond(json_response(400, json!({"error": {"message": "no messages"}})));
                return;
            }
            let stream = payload
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false);

            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            if cmd_tx
                .send(LlmCmd::Generate {
                    messages,
                    reply: reply_tx,
                })
                .is_err()
            {
                let _ = request.respond(json_response(
                    500,
                    json!({"error": {"message": "LLM worker unavailable"}}),
                ));
                return;
            }

            if stream {
                let sse = SseStream::new(reply_rx, model_name);
                let response = Response::new(
                    StatusCode(200),
                    vec![
                        Header::from_bytes("Content-Type", "text/event-stream").unwrap(),
                        Header::from_bytes("Cache-Control", "no-cache").unwrap(),
                    ],
                    sse,
                    None,
                    None,
                );
                let _ = request.respond(response);
            } else {
                let mut text = String::new();
                let mut error = None;
                for event in reply_rx {
                    match event {
                        LlmEvent::Token(t) => text.push_str(&t),
                        LlmEvent::Error(e) => error = Some(e),
                        LlmEvent::GenDone => break,
                        _ => {}
                    }
                }
                let response = match error {
                    Some(e) => json_response(500, json!({"error": {"message": e}})),
                    None => json_response(
                        200,
                        json!({
                            "id": "chatcmpl-offgrid",
                            "object": "chat.completion",
                            "created": unix_time(),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        }),
                    ),
                };
                let _ = request.respond(response);
            }
        }
        _ => {
            let _ = request
                .respond(json_response(404, json!({"error": {"message": "not found"}})));
        }
    }
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Adapts the LLM worker's event stream into an OpenAI SSE byte stream that
/// tiny_http can send incrementally (chunked transfer encoding).
struct SseStream {
    rx: Receiver<LlmEvent>,
    model: String,
    buf: Vec<u8>,
    pos: usize,
    sent_role: bool,
    done: bool,
}

impl SseStream {
    fn new(rx: Receiver<LlmEvent>, model: String) -> Self {
        Self {
            rx,
            model,
            buf: Vec::new(),
            pos: 0,
            sent_role: false,
            done: false,
        }
    }

    fn chunk(&self, delta: serde_json::Value, finish: Option<&str>) -> String {
        let body = json!({
            "id": "chatcmpl-offgrid",
            "object": "chat.completion.chunk",
            "created": unix_time(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish
            }]
        });
        format!("data: {body}\n\n")
    }

    fn refill(&mut self) {
        self.buf.clear();
        self.pos = 0;
        match self.rx.recv() {
            Ok(LlmEvent::Token(t)) => {
                let delta = if self.sent_role {
                    json!({"content": t})
                } else {
                    self.sent_role = true;
                    json!({"role": "assistant", "content": t})
                };
                self.buf = self.chunk(delta, None).into_bytes();
            }
            Ok(LlmEvent::GenDone) | Ok(LlmEvent::Error(_)) | Err(_) => {
                let mut out = self.chunk(json!({}), Some("stop"));
                out.push_str("data: [DONE]\n\n");
                self.buf = out.into_bytes();
                self.done = true;
            }
            Ok(_) => {}
        }
    }
}

impl Read for SseStream {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.pos < self.buf.len() {
                let n = (self.buf.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            self.refill();
        }
    }
}
