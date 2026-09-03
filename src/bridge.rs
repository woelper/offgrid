//! Telegram bridge: chat with the loaded model from a phone.
//!
//! Long polling only — the app reaches out over plain HTTPS, so there is no
//! inbound port, no public URL and no tunnel, and it keeps working on a
//! laptop that moves between networks. Deliberately the one part of offgrid
//! that talks to someone else's computer: off by default, and every sender
//! must be on an allowlist before the model answers them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::llm::{ChatMessage, LlmCmd, LlmEvent, Role};

/// Seconds the server holds a poll open. Telegram allows up to 50; 25 keeps
/// the connection fresh without a stall being felt as unresponsiveness.
const POLL_SECS: u64 = 25;
/// Telegram rejects messages longer than 4096 characters.
const MAX_MSG: usize = 4000;
/// Turns kept per chat before the oldest are dropped (the model's context is
/// the real limit; this just stops unbounded growth).
const MAX_HISTORY: usize = 20;

pub struct Bridge {
    stop: Arc<AtomicBool>,
    /// Chat ids seen from senders who are not on the allowlist, newest last —
    /// the UI offers a one-click "allow" for them.
    pub pending: Arc<Mutex<Vec<(i64, String)>>>,
    pub status: Arc<Mutex<String>>,
}

impl Bridge {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A dedicated agent: the global timeout must outlast a long poll, unlike
/// the agent's web tools which fail fast on purpose.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(10)))
        .timeout_global(Some(std::time::Duration::from_secs(POLL_SECS + 15)))
        .build()
        .into()
}

const TELEGRAM: &str = "https://api.telegram.org";

fn api(base: &str, token: &str, method: &str) -> String {
    format!("{base}/bot{token}/{method}")
}

/// One incoming text message.
struct Incoming {
    update_id: i64,
    chat_id: i64,
    from: String,
    text: String,
}

/// Pull the text messages out of a `getUpdates` response.
fn parse_updates(json: &str) -> Vec<Incoming> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    let Some(items) = v["result"].as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|u| {
            let update_id = u["update_id"].as_i64()?;
            let msg = u.get("message")?;
            let chat_id = msg["chat"]["id"].as_i64()?;
            let text = msg["text"].as_str()?.to_string();
            let from = msg["from"]["username"]
                .as_str()
                .map(|u| format!("@{u}"))
                .or_else(|| msg["from"]["first_name"].as_str().map(str::to_string))
                .unwrap_or_else(|| chat_id.to_string());
            Some(Incoming {
                update_id,
                chat_id,
                from,
                text,
            })
        })
        .collect()
}

/// Split a reply into Telegram-sized chunks, preferring paragraph then line
/// boundaries so code blocks and prose survive the cut.
fn split_message(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while rest.chars().count() > MAX_MSG {
        let cut_at = rest
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= MAX_MSG)
            .last()
            .unwrap_or(rest.len());
        let head = &rest[..cut_at];
        let split = head
            .rfind("\n\n")
            .or_else(|| head.rfind('\n'))
            .unwrap_or(cut_at);
        out.push(rest[..split].to_string());
        rest = rest[split..].trim_start_matches('\n');
    }
    if !rest.trim().is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// POST a JSON body, returning the parsed response. Hand-rolled rather than
/// ureq's `send_json` so the crate's `json` feature is not needed.
fn post_json(
    base: &str,
    token: &str,
    method: &str,
    body: serde_json::Value,
) -> Option<serde_json::Value> {
    let mut res = agent()
        .post(api(base, token, method))
        .header("Content-Type", "application/json")
        .send(body.to_string())
        .ok()?;
    let text = res.body_mut().read_to_string().ok()?;
    serde_json::from_str(&text).ok()
}

/// Send a message and return its id, so progress can be edited in place
/// instead of spamming one notification per agent tool call.
fn send_message_id(base: &str, token: &str, chat_id: i64, text: &str) -> Option<i64> {
    post_json(
        base,
        token,
        "sendMessage",
        json!({"chat_id": chat_id, "text": text}),
    )?["result"]["message_id"]
        .as_i64()
}

fn edit_message(base: &str, token: &str, chat_id: i64, message_id: i64, text: &str) {
    let text: String = text.chars().take(MAX_MSG).collect();
    post_json(
        base,
        token,
        "editMessageText",
        json!({"chat_id": chat_id, "message_id": message_id, "text": text}),
    );
}

fn send_message(base: &str, token: &str, chat_id: i64, text: &str) {
    for chunk in split_message(text) {
        post_json(
            base,
            token,
            "sendMessage",
            json!({"chat_id": chat_id, "text": chunk}),
        );
    }
}

fn send_typing(base: &str, token: &str, chat_id: i64) {
    post_json(
        base,
        token,
        "sendChatAction",
        json!({"chat_id": chat_id, "action": "typing"}),
    );
}

/// Ask the LLM worker for a reply, blocking until the turn is done.
fn generate(
    cmd_tx: &Sender<LlmCmd>,
    messages: Vec<ChatMessage>,
    n_ctx: u32,
) -> Result<String, String> {
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    cmd_tx
        .send(LlmCmd::Generate {
            messages,
            reply: reply_tx,
            temp: 0.7,
            n_ctx,
        })
        .map_err(|_| "LLM worker unavailable".to_string())?;
    let mut text = String::new();
    for event in reply_rx {
        match event {
            LlmEvent::Token(t) => text.push_str(&t),
            LlmEvent::Error(e) => return Err(e),
            LlmEvent::GenDone => break,
            _ => {}
        }
    }
    Ok(text)
}

/// Reasoning models emit `<think>` blocks; they are noise in a phone chat.
fn strip_think(s: &str) -> String {
    let mut out = s.to_string();
    while let Some(start) = out.find("<think>") {
        let end = out[start..]
            .find("</think>")
            .map(|e| start + e + "</think>".len())
            .unwrap_or(out.len());
        out.replace_range(start..end, "");
    }
    out.trim().to_string()
}

/// Progress lines kept in the live-edited run message.
const RUN_LINES: usize = 12;

/// Last couple of lines of what the model is writing right now.
fn live_tail(text: &str) -> String {
    let clean = strip_think(text);
    let skip = clean.chars().count().saturating_sub(220);
    let tail: String = clean.chars().skip(skip).collect();
    if tail.trim().is_empty() {
        "…working".to_string()
    } else {
        format!("…{}", tail.trim())
    }
}

/// Drive one agent run, reporting into a single message that is edited as
/// the run proceeds (one notification, not one per tool call). Blocks until
/// the run ends; callers put it on its own thread so polling continues.
#[allow(clippy::too_many_arguments)]
fn run_agent(
    base: &str,
    token: &str,
    chat_id: i64,
    task: &str,
    workspace: std::path::PathBuf,
    cmd_tx: Sender<LlmCmd>,
    web_tools: bool,
    n_ctx: u32,
    active: &crate::agent::ActiveRun,
    resuming: bool,
) {
    let head = format!(
        "{} {}\nin {}",
        if resuming { "⏵ resuming" } else { "▶" },
        task.lines().next().unwrap_or(task),
        workspace.display()
    );
    let msg_id = send_message_id(base, token, chat_id, &format!("{head}\n\nstarting…"));
    // Remote runs auto-approve: nobody is at the approval prompt.
    let run = if resuming {
        match crate::agent::resume(workspace, cmd_tx, true, web_tools, n_ctx) {
            Some(run) => run,
            None => {
                send_message(base, token, chat_id, "Nothing to resume.");
                return;
            }
        }
    } else {
        crate::agent::start(workspace, task.to_string(), cmd_tx, true, web_tools, n_ctx)
    };
    crate::agent::claim(active, crate::agent::RunSource::Telegram, task, &run);

    let mut lines: Vec<String> = Vec::new();
    let mut tokens = 0usize;
    let mut turn_buf = String::new();
    // Telegram throttles edits; once every few seconds reads as live.
    let mut last_edit = std::time::Instant::now();
    let mut last_turn = String::new();
    let mut error: Option<String> = None;
    let repaint = |lines: &[String], tail: &str| {
        if let Some(id) = msg_id {
            let body: Vec<&str> = lines.iter().map(String::as_str).collect();
            edit_message(
                base,
                token,
                chat_id,
                id,
                &format!("{head}\n\n{}\n{tail}", body.join("\n")),
            );
        }
    };

    for event in run.rx {
        match event {
            crate::agent::AgentEvent::Token(t) => {
                turn_buf.push_str(&t);
                tokens += 1;
                if tokens.is_multiple_of(16) {
                    crate::agent::note_text(active, &turn_buf);
                    if last_edit.elapsed() > std::time::Duration::from_secs(3) {
                        last_edit = std::time::Instant::now();
                        repaint(&lines, &live_tail(&turn_buf));
                    }
                }
            }
            crate::agent::AgentEvent::TurnDone => {
                crate::agent::note_turn(active);
                last_turn = strip_think(&turn_buf);
                turn_buf.clear();
            }
            crate::agent::AgentEvent::ToolCall { name, summary } => {
                crate::agent::note_activity(active, &name, &summary);
                let summary: String = summary.chars().take(80).collect();
                lines.push(format!("• {name}: {summary}"));
                if lines.len() > RUN_LINES {
                    lines.remove(0);
                }
                last_edit = std::time::Instant::now();
                repaint(&lines, "…working");
            }
            crate::agent::AgentEvent::ToolResult { ok: false, .. } => {
                if let Some(last) = lines.last_mut() {
                    last.push_str("  ✗");
                }
                repaint(&lines, "…working");
            }
            crate::agent::AgentEvent::Info(note) => {
                if let Some(instruction) = note.strip_prefix("new instruction: ") {
                    let instruction: String = instruction.chars().take(80).collect();
                    lines.push(format!("↪ you: {instruction}"));
                    if lines.len() > RUN_LINES {
                        lines.remove(0);
                    }
                    last_edit = std::time::Instant::now();
                    repaint(&lines, "…working");
                }
            }
            crate::agent::AgentEvent::Error(e) => error = Some(e),
            crate::agent::AgentEvent::Done { iterations } => {
                repaint(&lines, &format!("✓ finished in {iterations} turns"));
            }
            _ => {}
        }
    }
    crate::agent::release(active);

    match error {
        Some(e) => send_message(base, token, chat_id, &format!("Run failed: {e}")),
        None if !last_turn.is_empty() => send_message(base, token, chat_id, &last_turn),
        None => send_message(base, token, chat_id, "Run ended."),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn start(
    token: String,
    allowed: Vec<i64>,
    cmd_tx: Sender<LlmCmd>,
    loaded_model: Arc<Mutex<Option<String>>>,
    n_ctx: u32,
    workspace: Option<std::path::PathBuf>,
    web_tools: bool,
    code_enabled: bool,
    active: crate::agent::ActiveRun,
) -> Bridge {
    start_with_base(
        TELEGRAM.to_string(),
        token,
        allowed,
        cmd_tx,
        loaded_model,
        n_ctx,
        workspace,
        web_tools,
        code_enabled,
        active,
    )
}

/// The bridge against an arbitrary API base — tests point it at a local
/// server standing in for Telegram.
#[allow(clippy::too_many_arguments)]
fn start_with_base(
    base: String,
    token: String,
    allowed: Vec<i64>,
    cmd_tx: Sender<LlmCmd>,
    loaded_model: Arc<Mutex<Option<String>>>,
    n_ctx: u32,
    workspace: Option<std::path::PathBuf>,
    web_tools: bool,
    code_enabled: bool,
    active: crate::agent::ActiveRun,
) -> Bridge {
    let stop = Arc::new(AtomicBool::new(false));
    let pending: Arc<Mutex<Vec<(i64, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let status = Arc::new(Mutex::new("connecting…".to_string()));
    let (stop_t, pending_t, status_t) = (stop.clone(), pending.clone(), status.clone());

    std::thread::spawn(move || {
        // Histories are per chat so two people don't share a conversation.
        let mut histories: std::collections::HashMap<i64, Vec<ChatMessage>> =
            std::collections::HashMap::new();
        let mut offset: i64 = 0;
        let mut backoff = 1u64;

        while !stop_t.load(Ordering::Relaxed) {
            let url = format!(
                "{}?timeout={POLL_SECS}&offset={offset}",
                api(&base, &token, "getUpdates")
            );
            let body = match agent().get(&url).call() {
                Ok(mut res) => res.body_mut().read_to_string().unwrap_or_default(),
                Err(e) => {
                    // Offline or throttled: back off, keep the app unaffected.
                    *status_t.lock().unwrap() = format!("offline ({e})");
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(60);
                    continue;
                }
            };
            backoff = 1;
            *status_t.lock().unwrap() = "connected".to_string();

            for msg in parse_updates(&body) {
                offset = offset.max(msg.update_id + 1);
                if stop_t.load(Ordering::Relaxed) {
                    break;
                }
                // Allowlist first: an unknown sender never reaches the model.
                if !allowed.contains(&msg.chat_id) {
                    let mut pend = pending_t.lock().unwrap();
                    if !pend.iter().any(|(id, _)| *id == msg.chat_id) {
                        pend.push((msg.chat_id, msg.from.clone()));
                    }
                    drop(pend);
                    send_message(
                        &base,
                        &token,
                        msg.chat_id,
                        "This offgrid instance has not authorized this chat. Ask its \
                         owner to allow it in the Serve tab.",
                    );
                    continue;
                }
                let text = msg.text.trim();
                match text {
                    "/start" | "/help" => {
                        let code_line = if code_enabled {
                            "/code <task> runs the coding agent in the workspace, \
                             /stop aborts it, /resume continues an interrupted run. \
                             While it runs, anything you send is handed to it as a new \
                             instruction.\n"
                        } else {
                            ""
                        };
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            &format!(
                                "offgrid bridge. Send a message to chat with the loaded \
                                 model.\n{code_line}/new starts a fresh conversation, \
                                 /status shows what is going on."
                            ),
                        );
                        continue;
                    }
                    "/new" => {
                        histories.remove(&msg.chat_id);
                        send_message(&base, &token, msg.chat_id, "Started a new conversation.");
                        continue;
                    }
                    "/status" => {
                        let model = loaded_model.lock().unwrap().clone();
                        let turns = histories.get(&msg.chat_id).map_or(0, Vec::len);
                        let running = match crate::agent::run_summary(&active) {
                            Some(s) => format!("\n{s}\n/stop aborts it"),
                            None => String::new(),
                        };
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            &match model {
                                Some(m) => {
                                    format!("model: {m}\nturns in this chat: {turns}{running}")
                                }
                                None => "No model loaded.".to_string(),
                            },
                        );
                        continue;
                    }
                    "/stop" => {
                        // Aborts whatever is running, including a run
                        // started in the UI or over the API.
                        let state = active.lock().unwrap().clone();
                        match state {
                            Some(s) => {
                                s.stop.store(true, Ordering::Relaxed);
                                send_message(
                                    &base,
                                    &token,
                                    msg.chat_id,
                                    &format!("Stopping the run started from {}…", s.source.label()),
                                );
                            }
                            None => send_message(&base, &token, msg.chat_id, "No run is active."),
                        }
                        continue;
                    }
                    _ => {}
                }

                if text == "/resume" {
                    let saved = workspace.as_ref().and_then(|w| crate::agent::saved_run(w));
                    let reply = if !code_enabled {
                        Some("Code mode is disabled on this instance.".to_string())
                    } else if saved.is_none() {
                        Some("No interrupted run to resume.".to_string())
                    } else if loaded_model.lock().unwrap().is_none() {
                        Some("No model is loaded — load one in the Models tab first.".to_string())
                    } else if crate::agent::run_summary(&active).is_some() {
                        crate::agent::run_summary(&active)
                            .map(|s| format!("Busy — {s}. /stop aborts it."))
                    } else {
                        None
                    };
                    if let Some(text) = reply {
                        send_message(&base, &token, msg.chat_id, &text);
                        continue;
                    }
                    let (base_t, token_t) = (base.clone(), token.clone());
                    let task_t = saved.map(|s| s.task).unwrap_or_default();
                    let ws = workspace.clone().unwrap();
                    let (cmd_t, active_t) = (cmd_tx.clone(), active.clone());
                    let chat_id = msg.chat_id;
                    std::thread::spawn(move || {
                        run_agent(
                            &base_t, &token_t, chat_id, &task_t, ws, cmd_t, web_tools, n_ctx,
                            &active_t, true,
                        );
                    });
                    continue;
                }

                if let Some(task) = text.strip_prefix("/code") {
                    let task = task.trim();
                    let reply = if !code_enabled {
                        Some(
                            "Code mode is disabled on this instance — enable it in the \
                             Serve tab."
                                .to_string(),
                        )
                    } else if task.is_empty() {
                        Some("Usage: /code <task>".to_string())
                    } else if workspace.as_ref().is_none_or(|w| !w.is_dir()) {
                        Some("No workspace is set — pick one in the Code tab.".to_string())
                    } else if loaded_model.lock().unwrap().is_none() {
                        Some("No model is loaded — load one in the Models tab first.".to_string())
                    } else if crate::agent::run_summary(&active).is_some() {
                        crate::agent::run_summary(&active)
                            .map(|s| format!("Busy — {s}. /stop aborts it."))
                    } else {
                        None
                    };
                    if let Some(text) = reply {
                        send_message(&base, &token, msg.chat_id, &text);
                        continue;
                    }
                    // Run on its own thread so /stop and /status still answer.
                    let (base_t, token_t, task_t) = (base.clone(), token.clone(), task.to_string());
                    let ws = workspace.clone().unwrap();
                    let (cmd_t, active_t) = (cmd_tx.clone(), active.clone());
                    let chat_id = msg.chat_id;
                    std::thread::spawn(move || {
                        run_agent(
                            &base_t, &token_t, chat_id, &task_t, ws, cmd_t, web_tools, n_ctx,
                            &active_t, false,
                        );
                    });
                    continue;
                }

                // A message sent while the agent works is an instruction for
                // it, not a chat turn — that is the whole point of watching a
                // run from a phone.
                if crate::agent::steer(&active, text) {
                    send_message(
                        &base,
                        &token,
                        msg.chat_id,
                        "✔ passed to the agent — it will see this at the next turn. \
                         /status for progress, /stop to abort.",
                    );
                    continue;
                }
                if loaded_model.lock().unwrap().is_none() {
                    send_message(
                        &base,
                        &token,
                        msg.chat_id,
                        "No model is loaded — load one in the Models tab first.",
                    );
                    continue;
                }

                let history = histories.entry(msg.chat_id).or_default();
                history.push(ChatMessage {
                    role: Role::User,
                    content: text.to_string(),
                });
                if history.len() > MAX_HISTORY {
                    let drop_n = history.len() - MAX_HISTORY;
                    history.drain(..drop_n);
                }
                send_typing(&base, &token, msg.chat_id);
                match generate(&cmd_tx, history.clone(), n_ctx) {
                    Ok(reply) => {
                        let clean = strip_think(&reply);
                        history.push(ChatMessage {
                            role: Role::Assistant,
                            content: reply,
                        });
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            if clean.is_empty() {
                                "(no answer)"
                            } else {
                                &clean
                            },
                        );
                    }
                    Err(e) => {
                        // Drop the unanswered turn so the next try is clean.
                        history.pop();
                        send_message(&base, &token, msg.chat_id, &format!("Error: {e}"));
                    }
                }
            }
        }
        *status_t.lock().unwrap() = "stopped".to_string();
    });

    Bridge {
        stop,
        pending,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_updates_only() {
        let json = r#"{"ok":true,"result":[
            {"update_id":10,"message":{"chat":{"id":42},"from":{"username":"woelper"},
             "text":"hello"}},
            {"update_id":11,"message":{"chat":{"id":43},"from":{"first_name":"Jo"},
             "photo":[]}},
            {"update_id":12,"edited_message":{"chat":{"id":44},"text":"ignored"}}
        ]}"#;
        let msgs = parse_updates(json);
        assert_eq!(msgs.len(), 1); // photo and edited_message skipped
        assert_eq!(msgs[0].update_id, 10);
        assert_eq!(msgs[0].chat_id, 42);
        assert_eq!(msgs[0].from, "@woelper");
        assert_eq!(msgs[0].text, "hello");
        assert!(parse_updates("nonsense").is_empty());
    }

    #[test]
    fn splits_long_replies_on_boundaries() {
        let para = "x".repeat(3000);
        let text = format!("{para}\n\n{para}");
        let parts = split_message(&text);
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|p| p.chars().count() <= MAX_MSG));
        // Split at the paragraph break, not mid-run.
        assert_eq!(parts[0].trim(), para);
        // Short messages are untouched.
        assert_eq!(split_message("hi"), vec!["hi".to_string()]);
    }

    /// End to end against a local stand-in for Telegram: an allowed chat is
    /// answered by the model, a stranger is refused and queued for approval.
    #[test]
    fn answers_allowed_chats_and_refuses_strangers() {
        let server = tiny_http::Server::http(("127.0.0.1", 0)).unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let sent: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sent_srv = sent.clone();

        std::thread::spawn(move || {
            let mut polls = 0;
            for mut req in server.incoming_requests() {
                let url = req.url().to_string();
                if url.contains("getUpdates") {
                    polls += 1;
                    // One batch of updates, then nothing to report.
                    let body = if polls == 1 {
                        r#"{"ok":true,"result":[
                            {"update_id":1,"message":{"chat":{"id":7},
                             "from":{"username":"owner"},"text":"hi there"}},
                            {"update_id":2,"message":{"chat":{"id":99},
                             "from":{"username":"stranger"},"text":"let me in"}}
                        ]}"#
                    } else {
                        r#"{"ok":true,"result":[]}"#
                    };
                    let _ = req.respond(tiny_http::Response::from_string(body));
                } else {
                    let mut body = String::new();
                    let _ = req.as_reader().read_to_string(&mut body);
                    sent_srv.lock().unwrap().push(format!("{url} {body}"));
                    let _ = req.respond(tiny_http::Response::from_string(r#"{"ok":true}"#));
                }
            }
        });

        // Scripted model: echoes a fixed answer, with a think block to strip.
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let LlmCmd::Generate { reply, .. } = cmd {
                    let _ = reply.send(LlmEvent::Token(
                        "<think>pondering</think>Hello from the model.".into(),
                    ));
                    let _ = reply.send(LlmEvent::GenDone);
                }
            }
        });

        let bridge = start_with_base(
            format!("http://127.0.0.1:{port}"),
            "test-token".into(),
            vec![7], // only chat 7 is allowed
            cmd_tx,
            Arc::new(Mutex::new(Some("test-model".into()))),
            4096,
            None,
            false,
            false,
            crate::agent::active_run(),
        );

        // Wait for both replies to land (or give up after ~5s).
        for _ in 0..50 {
            if sent
                .lock()
                .unwrap()
                .iter()
                .any(|s| s.contains("Hello from"))
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        bridge.stop();

        let sent = sent.lock().unwrap().clone();
        let joined = sent.join("\n");
        // The allowed chat got the model's answer, without the think block.
        assert!(joined.contains("Hello from the model."), "sent: {joined}");
        assert!(
            !joined.contains("pondering"),
            "think block leaked: {joined}"
        );
        assert!(joined.contains("sendChatAction")); // typing indicator
        // The stranger was refused and is queued for approval in the UI.
        assert!(joined.contains("has not authorized this chat"));
        let pending = bridge.pending.lock().unwrap().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 99);
        assert_eq!(pending[0].1, "@stranger");
    }

    /// `/code` drives a real agent run and reports it back: progress edited
    /// into one message, the model's closing summary sent separately. With
    /// code mode off, the same command is refused.
    #[test]
    fn code_command_runs_the_agent_only_when_enabled() {
        for code_enabled in [false, true] {
            let ws = std::env::temp_dir().join(format!("offgrid-bridge-code-{code_enabled}"));
            let _ = std::fs::remove_dir_all(&ws);
            std::fs::create_dir_all(&ws).unwrap();

            let server = tiny_http::Server::http(("127.0.0.1", 0)).unwrap();
            let port = server.server_addr().to_ip().unwrap().port();
            let sent: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let sent_srv = sent.clone();
            std::thread::spawn(move || {
                let mut polls = 0;
                for mut req in server.incoming_requests() {
                    let url = req.url().to_string();
                    if url.contains("getUpdates") {
                        polls += 1;
                        let body = if polls == 1 {
                            r#"{"ok":true,"result":[{"update_id":1,"message":{
                                "chat":{"id":7},"from":{"username":"owner"},
                                "text":"/code write a haiku"}}]}"#
                        } else {
                            r#"{"ok":true,"result":[]}"#
                        };
                        let _ = req.respond(tiny_http::Response::from_string(body));
                    } else {
                        let mut body = String::new();
                        let _ = req.as_reader().read_to_string(&mut body);
                        sent_srv.lock().unwrap().push(format!("{url} {body}"));
                        let _ = req.respond(tiny_http::Response::from_string(
                            r#"{"ok":true,"result":{"message_id":5}}"#,
                        ));
                    }
                }
            });

            // Scripted model: one tool call, then a closing summary.
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
            std::thread::spawn(move || {
                let mut turn = 0;
                for cmd in cmd_rx {
                    if let LlmCmd::Generate { reply, .. } = cmd {
                        turn += 1;
                        let text = if turn == 1 {
                            "<tool_call>{\"name\": \"list_files\", \"arguments\": {}}</tool_call>"
                        } else {
                            "Wrote the haiku."
                        };
                        let _ = reply.send(LlmEvent::Token(text.into()));
                        let _ = reply.send(LlmEvent::GenDone);
                    }
                }
            });

            let bridge = start_with_base(
                format!("http://127.0.0.1:{port}"),
                "test-token".into(),
                vec![7],
                cmd_tx,
                Arc::new(Mutex::new(Some("test-model".into()))),
                4096,
                Some(ws.clone()),
                false,
                code_enabled,
                crate::agent::active_run(),
            );

            let want = if code_enabled {
                "Wrote the haiku."
            } else {
                "Code mode is disabled"
            };
            for _ in 0..50 {
                if sent.lock().unwrap().iter().any(|s| s.contains(want)) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            bridge.stop();

            let joined = sent.lock().unwrap().join("\n");
            assert!(
                joined.contains(want),
                "code_enabled={code_enabled}: {joined}"
            );
            if code_enabled {
                // Progress went into one edited message, not a flood.
                assert!(
                    joined.contains("editMessageText"),
                    "no live progress: {joined}"
                );
                assert!(
                    joined.contains("list_files"),
                    "tool call not reported: {joined}"
                );
            } else {
                assert!(!joined.contains("list_files"), "agent ran while disabled");
            }
            let _ = std::fs::remove_dir_all(&ws);
        }
    }

    #[test]
    fn strips_reasoning_blocks() {
        assert_eq!(strip_think("<think>hmm</think>Answer."), "Answer.");
        assert_eq!(strip_think("plain"), "plain");
    }
}
