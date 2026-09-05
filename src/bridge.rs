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

use crate::llm::{LlmCmd, LlmEvent};
use crate::session::{self, ChatBusy, Command, Conversation, Mode};

/// Seconds the server holds a poll open. Telegram allows up to 50; 25 keeps
/// the connection fresh without a stall being felt as unresponsiveness.
const POLL_SECS: u64 = 25;
/// Telegram rejects messages longer than 4096 characters.
const MAX_MSG: usize = 4000;
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

/// Messages waiting to be answered: the LLM worker takes one at a time.
type ChatQueue = Arc<Mutex<std::collections::VecDeque<i64>>>;

/// Buttons along the bottom of the chat, so the mode is always one tap away
/// (Telegram has no status bar to show which mode you are in).
fn keyboard() -> serde_json::Value {
    json!({
        "keyboard": [["/chat", "/code"], ["/last", "/status", "/stop"]],
        "resize_keyboard": true,
        "is_persistent": true
    })
}

fn send_with_keyboard(base: &str, token: &str, chat_id: i64, text: &str) {
    post_json(
        base,
        token,
        "sendMessage",
        json!({"chat_id": chat_id, "text": text, "reply_markup": keyboard()}),
    );
}

/// Answer one queued chat message. Tokens stream into the shared
/// conversation (so the desktop shows the same reply appearing) and into a
/// single Telegram message that is edited as they arrive — a slow local
/// model otherwise means a minute of silence.
fn answer_chat(
    base: &str,
    token: &str,
    chat_id: i64,
    conv: &Conversation,
    cmd_tx: &Sender<LlmCmd>,
    n_ctx: u32,
) {
    let messages = session::snapshot(conv);
    let msg_id = send_message_id(base, token, chat_id, "…");
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    if cmd_tx
        .send(LlmCmd::Generate {
            messages,
            reply: reply_tx,
            temp: 0.7,
            n_ctx,
        })
        .is_err()
    {
        send_message(base, token, chat_id, "Error: LLM worker unavailable");
        return;
    }
    session::push_assistant(conv);

    let mut text = String::new();
    let mut error = None;
    let mut last_edit = std::time::Instant::now();
    for event in reply_rx {
        match event {
            LlmEvent::Token(t) => {
                text.push_str(&t);
                session::append_assistant(conv, &t);
                if last_edit.elapsed() > std::time::Duration::from_secs(3) {
                    last_edit = std::time::Instant::now();
                    let partial = session::strip_think(&text);
                    if let (Some(id), false) = (msg_id, partial.is_empty()) {
                        let shown: String = partial.chars().take(MAX_MSG).collect();
                        edit_message(base, token, chat_id, id, &format!("{shown} ▍"));
                    }
                }
            }
            LlmEvent::Error(e) => error = Some(e),
            LlmEvent::GenDone => break,
            _ => {}
        }
    }

    match error {
        Some(e) => {
            session::pop_unanswered(conv);
            match msg_id {
                Some(id) => edit_message(base, token, chat_id, id, &format!("Error: {e}")),
                None => send_message(base, token, chat_id, &format!("Error: {e}")),
            }
        }
        None => {
            let clean = session::strip_think(&text);
            let clean = if clean.is_empty() {
                "(no answer)".to_string()
            } else {
                clean
            };
            // The first chunk lands in the streamed message; any overflow
            // follows as ordinary messages.
            let mut chunks = split_message(&clean).into_iter();
            match (msg_id, chunks.next()) {
                (Some(id), Some(first)) => edit_message(base, token, chat_id, id, &first),
                (None, Some(first)) => send_message(base, token, chat_id, &first),
                _ => {}
            }
            for rest in chunks {
                send_message(base, token, chat_id, &rest);
            }
        }
    }
}

/// Start (or resume) an agent run for a chat, refusing clearly when the
/// preconditions are not met. The run gets its own thread so polling —
/// and therefore /status and /stop — keeps working.
#[allow(clippy::too_many_arguments)]
fn start_run(
    base: &str,
    token: &str,
    chat_id: i64,
    task: String,
    resuming: bool,
    workspace: &Option<std::path::PathBuf>,
    loaded_model: &Arc<Mutex<Option<String>>>,
    active: &crate::agent::ActiveRun,
    cmd_tx: &Sender<LlmCmd>,
    web_tools: bool,
    n_ctx: u32,
) {
    let Some(ws) = workspace.clone().filter(|w| w.is_dir()) else {
        send_message(
            base,
            token,
            chat_id,
            "No workspace is set — pick one in the Code tab.",
        );
        return;
    };
    let task = if resuming {
        match crate::agent::saved_run(&ws) {
            Some(saved) => saved.task,
            None => {
                send_message(base, token, chat_id, "No interrupted run to resume.");
                return;
            }
        }
    } else {
        task
    };
    if loaded_model.lock().unwrap().is_none() {
        send_message(
            base,
            token,
            chat_id,
            "No model is loaded — load one in the Models tab first.",
        );
        return;
    }
    if let Some(summary) = crate::agent::run_summary(active) {
        send_message(
            base,
            token,
            chat_id,
            &format!("Busy — {summary}. /stop aborts it."),
        );
        return;
    }
    let (base_t, token_t) = (base.to_string(), token.to_string());
    let (cmd_t, active_t) = (cmd_tx.clone(), active.clone());
    std::thread::spawn(move || {
        run_agent(
            &base_t, &token_t, chat_id, &task, ws, cmd_t, web_tools, n_ctx, &active_t, resuming,
        );
    });
}

/// Progress lines kept in the live-edited run message.
const RUN_LINES: usize = 12;

/// Last couple of lines of what the model is writing right now.
fn live_tail(text: &str) -> String {
    let clean = session::strip_think(text);
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
                last_turn = session::strip_think(&turn_buf);
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
    llm_stop: Arc<AtomicBool>,
    conv: Conversation,
    busy: ChatBusy,
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
        llm_stop,
        conv,
        busy,
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
    llm_stop: Arc<AtomicBool>,
    conv: Conversation,
    busy: ChatBusy,
) -> Bridge {
    let stop = Arc::new(AtomicBool::new(false));
    let pending: Arc<Mutex<Vec<(i64, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let status = Arc::new(Mutex::new("connecting…".to_string()));
    let (stop_t, pending_t, status_t) = (stop.clone(), pending.clone(), status.clone());

    let queue: ChatQueue = Arc::new(Mutex::new(std::collections::VecDeque::new()));

    // Chat answers run here, not in the poll loop: a slow model would
    // otherwise block polling, so /status and /stop would go unanswered
    // exactly when they matter. One at a time, because the LLM worker is.
    {
        let (base_w, token_w) = (base.clone(), token.clone());
        let (conv_w, queue_w, cmd_w, stop_w) =
            (conv.clone(), queue.clone(), cmd_tx.clone(), stop.clone());
        let busy_w = busy.clone();
        let llm_stop_w = llm_stop.clone();
        std::thread::spawn(move || {
            while !stop_w.load(Ordering::Relaxed) {
                let next = queue_w.lock().unwrap().pop_front();
                match next {
                    Some(chat_id) => {
                        // The desktop may be answering the same conversation.
                        if !busy_w.claim() {
                            std::thread::sleep(std::time::Duration::from_millis(300));
                            queue_w.lock().unwrap().push_front(chat_id);
                            continue;
                        }
                        llm_stop_w.store(false, Ordering::Relaxed);
                        answer_chat(&base_w, &token_w, chat_id, &conv_w, &cmd_w, n_ctx);
                        busy_w.release();
                    }
                    None => std::thread::sleep(std::time::Duration::from_millis(200)),
                }
            }
        });
    }

    std::thread::spawn(move || {
        let mut modes: std::collections::HashMap<i64, Mode> = std::collections::HashMap::new();
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
                let mode = *modes.entry(msg.chat_id).or_default();
                let run_active = active.lock().unwrap().is_some();
                match session::parse(text, mode, run_active, code_enabled) {
                    Command::Empty => continue,
                    Command::Help => {
                        send_with_keyboard(
                            &base,
                            &token,
                            msg.chat_id,
                            &session::help(mode, code_enabled),
                        );
                    }
                    Command::SwitchMode(m) => {
                        modes.insert(msg.chat_id, m);
                        send_with_keyboard(
                            &base,
                            &token,
                            msg.chat_id,
                            match m {
                                Mode::Chat => "chat mode — what you type goes to the model.",
                                Mode::Code => {
                                    "code mode — what you type becomes an agent \
                                               task, or a new instruction while one is \
                                               running."
                                }
                            },
                        );
                    }
                    Command::CodeDisabled => {
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            "Code mode is disabled on this instance — enable it in the \
                             Serve tab.",
                        );
                    }
                    Command::New => {
                        session::clear(&conv);
                        send_message(&base, &token, msg.chat_id, "Started a new conversation.");
                    }
                    Command::Last => {
                        send_message(&base, &token, msg.chat_id, &session::recent(&conv, 6, 600));
                    }
                    Command::Status => {
                        let model = loaded_model.lock().unwrap().clone();
                        let running = match crate::agent::run_summary(&active) {
                            Some(s) => format!("\n{s}\n/stop aborts it"),
                            None => String::new(),
                        };
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            &match model {
                                Some(m) => format!(
                                    "{} mode\nmodel: {m}\nturns in this conversation: \
                                     {}{running}",
                                    mode.label(),
                                    session::turns(&conv)
                                ),
                                None => "No model loaded.".to_string(),
                            },
                        );
                    }
                    Command::Stop => {
                        // Aborts whatever is running: an agent run started
                        // anywhere, or a chat answer being generated now.
                        let state = active.lock().unwrap().clone();
                        let dropped = {
                            let mut q = queue.lock().unwrap();
                            let before = q.len();
                            q.retain(|id| *id != msg.chat_id);
                            before - q.len()
                        };
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
                            None => {
                                llm_stop.store(true, Ordering::Relaxed);
                                send_message(
                                    &base,
                                    &token,
                                    msg.chat_id,
                                    &format!("Stopped. {dropped} queued message(s) dropped."),
                                );
                            }
                        }
                    }
                    Command::Steer(text) => {
                        crate::agent::steer(&active, &text);
                        send_message(
                            &base,
                            &token,
                            msg.chat_id,
                            "✔ passed to the agent — it will see this at the next turn. \
                             /status for progress, /stop to abort.",
                        );
                    }
                    Command::Code(task) => {
                        start_run(
                            &base,
                            &token,
                            msg.chat_id,
                            task,
                            false,
                            &workspace,
                            &loaded_model,
                            &active,
                            &cmd_tx,
                            web_tools,
                            n_ctx,
                        );
                    }
                    Command::Resume => {
                        start_run(
                            &base,
                            &token,
                            msg.chat_id,
                            String::new(),
                            true,
                            &workspace,
                            &loaded_model,
                            &active,
                            &cmd_tx,
                            web_tools,
                            n_ctx,
                        );
                    }
                    Command::Chat(text) => {
                        if loaded_model.lock().unwrap().is_none() {
                            send_message(
                                &base,
                                &token,
                                msg.chat_id,
                                "No model is loaded — load one in the Models tab first.",
                            );
                            continue;
                        }
                        session::push_user(&conv, &text);
                        let queued = {
                            let mut q = queue.lock().unwrap();
                            q.push_back(msg.chat_id);
                            q.len()
                        };
                        send_typing(&base, &token, msg.chat_id);
                        if queued > 1 {
                            send_message(
                                &base,
                                &token,
                                msg.chat_id,
                                &format!("queued — {queued} messages waiting."),
                            );
                        }
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
            Arc::new(AtomicBool::new(false)),
            session::conversation(),
            ChatBusy::new(),
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
                Arc::new(AtomicBool::new(false)),
                session::conversation(),
                ChatBusy::new(),
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

    /// /code and /chat are sticky per chat: after switching, plain messages
    /// start agent runs instead of chat turns, and switching back restores
    /// chatting. Each chat keeps its own mode.
    /// A conversation started in the desktop UI continues on the phone:
    /// the model sees the earlier turns, and the phone's reply lands back
    /// in the same conversation the UI renders.
    #[test]
    fn telegram_continues_the_desktop_conversation() {
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
                            "text":"and what did I just ask?"}}]}"#
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

        // The model answers only if it can see the desktop's earlier turn.
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let LlmCmd::Generate {
                    messages, reply, ..
                } = cmd
                {
                    let text = if messages
                        .iter()
                        .any(|m| m.content.contains("capital of Peru"))
                    {
                        "You asked about Peru."
                    } else {
                        "I have no idea what you asked."
                    };
                    let _ = reply.send(LlmEvent::Token(text.into()));
                    let _ = reply.send(LlmEvent::GenDone);
                }
            }
        });

        // What the desktop UI would have put there.
        let conv = session::conversation();
        session::push_user(&conv, "what is the capital of Peru?");
        session::push_assistant(&conv);
        session::append_assistant(&conv, "Lima.");

        let bridge = start_with_base(
            format!("http://127.0.0.1:{port}"),
            "test-token".into(),
            vec![7],
            cmd_tx,
            Arc::new(Mutex::new(Some("test-model".into()))),
            4096,
            None,
            false,
            false,
            crate::agent::active_run(),
            Arc::new(AtomicBool::new(false)),
            conv.clone(),
            ChatBusy::new(),
        );

        for _ in 0..60 {
            if sent
                .lock()
                .unwrap()
                .iter()
                .any(|s| s.contains("You asked about Peru"))
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        bridge.stop();

        let joined = sent.lock().unwrap().join("\n");
        assert!(
            joined.contains("You asked about Peru"),
            "phone did not see the desktop conversation: {joined}"
        );
        // The phone's turns joined the same conversation the UI renders.
        let history = session::snapshot(&conv);
        assert!(history.iter().any(|m| m.content.contains("Lima")));
        assert!(
            history
                .iter()
                .any(|m| m.content.contains("what did I just ask"))
        );
        assert!(
            history
                .iter()
                .any(|m| m.content.contains("You asked about Peru"))
        );
    }

    #[test]
    fn mode_is_sticky_and_per_chat() {
        let ws = std::env::temp_dir().join("offgrid-mode-test");
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
                    // chat 7 switches to code mode then sends a bare task;
                    // chat 8 stays in chat mode and just talks.
                    let body = match polls {
                        1 => {
                            r#"{"ok":true,"result":[
                            {"update_id":1,"message":{"chat":{"id":7},
                             "from":{"username":"owner"},"text":"/code"}}]}"#
                        }
                        2 => {
                            r#"{"ok":true,"result":[
                            {"update_id":2,"message":{"chat":{"id":7},
                             "from":{"username":"owner"},"text":"list the files"}},
                            {"update_id":3,"message":{"chat":{"id":8},
                             "from":{"username":"other"},"text":"hello there"}}]}"#
                        }
                        _ => r#"{"ok":true,"result":[]}"#,
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

        // The model: one tool call for the agent, prose for chat.
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
        std::thread::spawn(move || {
            for cmd in cmd_rx {
                if let LlmCmd::Generate {
                    messages, reply, ..
                } = cmd
                {
                    let agentic = messages.iter().any(|m| m.content.contains("coding agent"));
                    let text = if agentic {
                        "Listed them.".to_string()
                    } else {
                        "Hi from chat.".to_string()
                    };
                    let _ = reply.send(LlmEvent::Token(text));
                    let _ = reply.send(LlmEvent::GenDone);
                }
            }
        });

        let bridge = start_with_base(
            format!("http://127.0.0.1:{port}"),
            "test-token".into(),
            vec![7, 8],
            cmd_tx,
            Arc::new(Mutex::new(Some("test-model".into()))),
            4096,
            Some(ws.clone()),
            false,
            true,
            crate::agent::active_run(),
            Arc::new(AtomicBool::new(false)),
            session::conversation(),
            ChatBusy::new(),
        );

        for _ in 0..60 {
            let joined = sent.lock().unwrap().join("\n");
            if joined.contains("Hi from chat.") && joined.contains("offgrid-mode-test") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        bridge.stop();

        let joined = sent.lock().unwrap().join("\n");
        // chat 7 switched mode and its bare message started a run…
        assert!(joined.contains("code mode"), "no mode switch: {joined}");
        // The run header names the task and the workspace (whose path
        // separator differs per platform — match on the directory name).
        assert!(
            joined.contains("list the files") && joined.contains("offgrid-mode-test"),
            "bare message did not start a run: {joined}"
        );
        // …while chat 8, untouched, was answered as a normal chat.
        assert!(
            joined.contains("Hi from chat."),
            "chat 8 not answered: {joined}"
        );
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn strips_reasoning_blocks() {
        assert_eq!(session::strip_think("<think>hmm</think>Answer."), "Answer.");
        assert_eq!(session::strip_think("plain"), "plain");
    }
}
