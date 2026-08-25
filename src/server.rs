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

/// Live info about the current/last remote agent run, fed by the event
/// drain thread.
#[derive(Default, Clone)]
struct RunInfo {
    /// Session log file name (known after the run's first event).
    log: Option<String>,
    /// Completed model turns.
    iterations: usize,
}

/// Shared request-handler context. `agent_busy`/`agent_stop` serialize remote
/// agent runs: one at a time, stoppable via POST /agent/stop.
struct Ctx {
    cmd_tx: Sender<LlmCmd>,
    models_dir: PathBuf,
    loaded_model: Arc<Mutex<Option<String>>>,
    n_ctx: u32,
    workspace: Option<PathBuf>,
    /// Shared with the UI and the Telegram bridge: one run at a time,
    /// whoever started it.
    active: crate::agent::ActiveRun,
    agent_info: Arc<Mutex<RunInfo>>,
}

#[allow(clippy::too_many_arguments)]
pub fn start(
    port: u16,
    lan: bool,
    cmd_tx: Sender<LlmCmd>,
    models_dir: PathBuf,
    loaded_model: Arc<Mutex<Option<String>>>,
    n_ctx: u32,
    workspace: Option<PathBuf>,
    active: crate::agent::ActiveRun,
) -> Result<ApiServer, String> {
    // 0.0.0.0 exposes the model, the session logs, and remote agent runs
    // (shell access!) to the local network — strictly opt-in.
    let addr = if lan { "0.0.0.0" } else { "127.0.0.1" };
    let server = tiny_http::Server::http((addr, port)).map_err(|e| format!("server: {e}"))?;
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let ctx = Arc::new(Ctx {
        cmd_tx,
        models_dir,
        loaded_model,
        n_ctx,
        workspace,
        active,
        agent_info: Arc::new(Mutex::new(RunInfo::default())),
    });
    std::thread::spawn(move || {
        while !stop_thread.load(Ordering::Relaxed) {
            match server.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(Some(request)) => {
                    let ctx = ctx.clone();
                    std::thread::spawn(move || handle(request, &ctx));
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    Ok(ApiServer { stop })
}

/// This machine's LAN IP: a connected UDP socket picks the outbound
/// interface without sending a packet. Needs a default route, not internet —
/// on a routerless offline box it returns None and callers fall back to
/// showing the bind address.
pub fn lan_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    Some(socket.local_addr().ok()?.ip().to_string())
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

fn text_response(status: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::new(
        StatusCode(status),
        vec![Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap()],
        std::io::Cursor::new(body.into_bytes()),
        None,
        None,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Session logs in the data dir, newest first.
fn list_logs() -> Vec<(String, u64, std::time::SystemTime)> {
    let mut logs: Vec<_> = std::fs::read_dir(crate::config::logs_dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !(name.starts_with("agent-") && name.ends_with(".log")) {
                return None;
            }
            let meta = e.metadata().ok()?;
            Some((name, meta.len(), meta.modified().ok()?))
        })
        .collect();
    logs.sort_by_key(|l| std::cmp::Reverse(l.2)); // newest first
    logs
}

fn handle(mut request: tiny_http::Request, ctx: &Ctx) {
    let url = request.url().to_string();
    let method = request.method().as_str().to_string();
    let cmd_tx = ctx.cmd_tx.clone();
    let models_dir = ctx.models_dir.clone();
    let loaded_model = ctx.loaded_model.clone();
    let n_ctx = ctx.n_ctx;

    match (method.as_str(), url.as_str()) {
        // Human-readable live view: status plus the tail of the current
        // session log, auto-refreshing. Point a browser at the server root.
        ("GET", "/") => {
            let running = ctx.active.lock().unwrap().is_some();
            let info = ctx.agent_info.lock().unwrap().clone();
            let log_name = info
                .log
                .clone()
                .or_else(|| list_logs().into_iter().next().map(|(n, ..)| n));
            let status = if running {
                format!(
                    "&#9679; RUNNING — turn {} — {}",
                    info.iterations,
                    log_name.as_deref().unwrap_or("(log not started yet)")
                )
            } else {
                format!(
                    "idle — last log: {}",
                    log_name.as_deref().unwrap_or("(none)")
                )
            };
            let tail = log_name
                .and_then(|n| std::fs::read_to_string(crate::config::logs_dir().join(n)).ok())
                .map(|text| {
                    let tail_start = text
                        .char_indices()
                        .map(|(i, _)| i)
                        .find(|&i| i >= text.len().saturating_sub(8 * 1024))
                        .unwrap_or(0);
                    html_escape(&text[tail_start..])
                })
                .unwrap_or_else(|| "no session logs yet — start an agent run".into());
            let page = format!(
                "<!doctype html><html><head><meta charset=\"utf-8\">\
                 <meta http-equiv=\"refresh\" content=\"5\">\
                 <title>offgrid agent</title>\
                 <style>body{{font-family:monospace;background:#1e1e1e;color:#ddd;\
                 margin:1.5em}}pre{{white-space:pre-wrap;word-break:break-word;\
                 border-top:1px solid #444;padding-top:1em}}\
                 a{{color:#7ab}}</style></head><body>\
                 <b>offgrid agent</b> — {status} — <a href=\"/logs\">all logs</a>\
                 <pre>{tail}</pre>\
                 <script>window.scrollTo(0, document.body.scrollHeight);</script>\
                 </body></html>"
            );
            let response = Response::new(
                StatusCode(200),
                vec![Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap()],
                std::io::Cursor::new(page.into_bytes()),
                None,
                None,
            );
            let _ = request.respond(response);
        }
        ("GET", "/logs") => {
            let data: Vec<_> = list_logs()
                .into_iter()
                .map(|(name, size, modified)| {
                    let epoch = modified
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    json!({"name": name, "size": size, "modified": epoch})
                })
                .collect();
            let _ = request.respond(json_response(200, json!({"logs": data})));
        }
        ("GET", path) if path.starts_with("/logs/") => {
            let name = &path["/logs/".len()..];
            // "latest" resolves to the newest log; anything else must be a
            // plain agent log filename (no path tricks).
            let name = if name == "latest" {
                match list_logs().into_iter().next() {
                    Some((n, ..)) => n,
                    None => {
                        let _ = request.respond(text_response(404, "no logs yet".into()));
                        return;
                    }
                }
            } else if name.starts_with("agent-")
                && name.ends_with(".log")
                && !name.contains(['/', '\\'])
            {
                name.to_string()
            } else {
                let _ = request.respond(text_response(404, "no such log".into()));
                return;
            };
            match std::fs::read_to_string(crate::config::logs_dir().join(&name)) {
                Ok(text) => {
                    let _ = request.respond(text_response(200, text));
                }
                Err(_) => {
                    let _ = request.respond(text_response(404, "no such log".into()));
                }
            }
        }
        ("GET", "/agent") => {
            let info = ctx.agent_info.lock().unwrap().clone();
            let state = ctx.active.lock().unwrap().clone();
            let _ = request.respond(json_response(
                200,
                json!({
                    "running": state.is_some(),
                    "source": state.as_ref().map(|s| s.source.label()),
                    "task": state.as_ref().map(|s| s.task.clone()),
                    "turns": state.as_ref().map(|s| s.turns),
                    "activity": state.as_ref().map(|s| s.activity.clone()),
                    "text": state.as_ref().map(|s| s.text.clone()),
                    "iterations": info.iterations,
                    "log": info.log,
                }),
            ));
        }
        ("POST", "/agent/say") => {
            let mut body = String::new();
            let _ = request.as_reader().read_to_string(&mut body);
            let text = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["text"].as_str().map(str::to_string))
                .unwrap_or_default();
            if text.trim().is_empty() {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "missing 'text'"}}),
                ));
                return;
            }
            let delivered = crate::agent::steer(&ctx.active, text.trim());
            let _ = request.respond(json_response(
                if delivered { 200 } else { 409 },
                json!({"delivered": delivered}),
            ));
        }
        ("POST", "/agent/stop") => {
            let stopped = if let Some(s) = ctx.active.lock().unwrap().as_ref() {
                s.stop.store(true, Ordering::Relaxed);
                true
            } else {
                false
            };
            let _ = request.respond(json_response(200, json!({"stopped": stopped})));
        }
        ("GET", "/agent/saved") => {
            let saved = ctx
                .workspace
                .as_ref()
                .and_then(|w| crate::agent::saved_run(w));
            let _ = request.respond(json_response(
                200,
                match saved {
                    Some(s) => json!({"saved": true, "task": s.task, "turns": s.turns}),
                    None => json!({"saved": false}),
                },
            ));
        }
        ("POST", "/agent") => {
            let mut body = String::new();
            if request.as_reader().read_to_string(&mut body).is_err() {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "unreadable body"}}),
                ));
                return;
            }
            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&body) else {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "invalid JSON"}}),
                ));
                return;
            };
            let resuming = payload
                .get("resume")
                .and_then(|r| r.as_bool())
                .unwrap_or(false);
            let task = payload
                .get("task")
                .and_then(|t| t.as_str())
                .filter(|t| !t.trim().is_empty());
            if task.is_none() && !resuming {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "missing 'task' (or pass resume: true)"}}),
                ));
                return;
            }
            let workspace = payload
                .get("workspace")
                .and_then(|w| w.as_str())
                .map(PathBuf::from)
                .or_else(|| ctx.workspace.clone());
            let Some(workspace) = workspace.filter(|w| w.is_dir()) else {
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "no valid workspace (pass 'workspace' or set one in the Code tab)"}}),
                ));
                return;
            };
            let web_tools = payload
                .get("web_tools")
                .and_then(|w| w.as_bool())
                .unwrap_or(false);
            if ctx.loaded_model.lock().unwrap().is_none() {
                let _ = request.respond(json_response(
                    409,
                    json!({"error": {"message": "no model loaded"}}),
                ));
                return;
            }
            if let Some(summary) = crate::agent::run_summary(&ctx.active) {
                let _ = request.respond(json_response(
                    409,
                    json!({"error": {"message": format!("busy — {summary}")}}),
                ));
                return;
            }
            // Remote runs always auto-approve commands: there is no one at
            // the approval prompt. That is why LAN mode is opt-in.
            let run = if resuming {
                match crate::agent::resume(
                    workspace.clone(),
                    ctx.cmd_tx.clone(),
                    true,
                    web_tools,
                    ctx.n_ctx,
                ) {
                    Some(run) => run,
                    None => {
                        let _ = request.respond(json_response(
                            409,
                            json!({"error": {"message": "no saved run to resume"}}),
                        ));
                        return;
                    }
                }
            } else {
                crate::agent::start(
                    workspace.clone(),
                    task.unwrap_or_default().to_string(),
                    ctx.cmd_tx.clone(),
                    true,
                    web_tools,
                    ctx.n_ctx,
                )
            };
            crate::agent::claim(
                &ctx.active,
                crate::agent::RunSource::Api,
                task.unwrap_or("(resumed)"),
                &run,
            );
            *ctx.agent_info.lock().unwrap() = RunInfo::default();
            let active = ctx.active.clone();
            let info = ctx.agent_info.clone();
            std::thread::spawn(move || {
                // Drain events until the run thread ends, keeping RunInfo
                // current; the session log on disk is the full record.
                let (mut tokens, mut turn_buf) = (0usize, String::new());
                for event in run.rx {
                    match event {
                        crate::agent::AgentEvent::Info(text) => {
                            if let Some(path) = text.strip_prefix("session log: ") {
                                info.lock().unwrap().log = std::path::Path::new(path)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string());
                            }
                        }
                        crate::agent::AgentEvent::Token(t) => {
                            tokens += 1;
                            turn_buf.push_str(&t);
                            if tokens.is_multiple_of(16) {
                                crate::agent::note_text(&active, &turn_buf);
                            }
                        }
                        crate::agent::AgentEvent::ToolCall { name, summary } => {
                            crate::agent::note_activity(&active, &name, &summary);
                        }
                        crate::agent::AgentEvent::TurnDone => {
                            info.lock().unwrap().iterations += 1;
                            crate::agent::note_turn(&active);
                            turn_buf.clear();
                        }
                        _ => {}
                    }
                }
                crate::agent::release(&active);
            });
            let _ = request.respond(json_response(
                200,
                json!({
                    "started": true,
                    "resumed": resuming,
                    "workspace": workspace.display().to_string(),
                    "web_tools": web_tools,
                    "auto_approve": true,
                    "watch": "GET /agent for status, GET /logs/latest for the transcript"
                }),
            ));
        }
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
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "invalid JSON"}}),
                ));
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
                let _ = request.respond(json_response(
                    400,
                    json!({"error": {"message": "no messages"}}),
                ));
                return;
            }
            let stream = payload
                .get("stream")
                .and_then(|s| s.as_bool())
                .unwrap_or(false);

            let temp = payload
                .get("temperature")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.7) as f32;
            // Opt-in web search, matching the desktop/TUI toggle. OpenAI has no
            // field for this, so it is a custom extension: pass `"web": true`
            // (alias `"web_search"`). The model searches before answering; the
            // tool round-trips stay server-side and only the answer streams back.
            // Clients that cannot add custom fields (opencode, aider, …) can
            // instead start offgrid with OFFGRID_WEB=1 to default it on for
            // every chat completion; such a client may still send `"web": false`
            // to override per request.
            let web = payload
                .get("web")
                .or_else(|| payload.get("web_search"))
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| std::env::var_os("OFFGRID_WEB").is_some());
            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            if web {
                crate::webchat::spawn(
                    messages,
                    cmd_tx.clone(),
                    reply_tx,
                    Arc::new(AtomicBool::new(false)),
                    temp,
                    n_ctx,
                );
            } else if cmd_tx
                .send(LlmCmd::Generate {
                    messages,
                    reply: reply_tx,
                    temp,
                    n_ctx,
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
            let _ = request.respond(json_response(
                404,
                json!({"error": {"message": "not found"}}),
            ));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Routing smoke for the remote-control endpoints: status and log
    /// listing answer without a model; starting an agent run without a
    /// loaded model is refused. Loopback only — CI-safe.
    #[test]
    fn remote_endpoints_respond() {
        let (cmd_tx, _keep_rx) = std::sync::mpsc::channel();
        let server = start(
            18654,
            false,
            cmd_tx,
            std::env::temp_dir(),
            Arc::new(Mutex::new(None)),
            16384,
            None,
            crate::agent::active_run(),
        )
        .expect("server start");

        let mut res = ureq::get("http://127.0.0.1:18654/agent").call().unwrap();
        let body = res.body_mut().read_to_string().unwrap();
        assert!(body.contains("\"running\":false"));
        assert!(body.contains("\"iterations\":0"));

        // Human live view at the root.
        let mut res = ureq::get("http://127.0.0.1:18654/").call().unwrap();
        let body = res.body_mut().read_to_string().unwrap();
        assert!(body.contains("offgrid agent") && body.contains("idle"));

        let mut res = ureq::get("http://127.0.0.1:18654/logs").call().unwrap();
        let body = res.body_mut().read_to_string().unwrap();
        assert!(body.contains("\"logs\""));

        // Path traversal must not resolve.
        let res = ureq::get("http://127.0.0.1:18654/logs/../config.json")
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        assert_eq!(res.status(), 404);

        // No model loaded -> agent run refused with 409.
        let body = format!(
            r#"{{"task": "do things", "workspace": {}}}"#,
            serde_json::json!(std::env::temp_dir().display().to_string())
        );
        let res = ureq::post("http://127.0.0.1:18654/agent")
            .config()
            .http_status_as_error(false)
            .build()
            .send(&body)
            .unwrap();
        assert_eq!(res.status(), 409);

        server.stop();
    }
}
