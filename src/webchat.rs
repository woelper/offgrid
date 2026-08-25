//! Web-augmented chat.
//!
//! Plain chat is a single streamed generation. This adds an opt-in pre-pass
//! that reuses the coding agent's web tools: before answering, the model may
//! call `web_search` / `fetch_url`, we run them, and then the *normal*
//! streaming path answers with the results folded in as context. The final
//! answer therefore looks and streams exactly like plain chat — only better
//! grounded — while the tool-call machinery stays out of the chat window.
//!
//! It is off by default: a search query leaves the machine, which is the one
//! thing this app otherwise never does.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use crate::agent::{fetch_url, parse_tool_call, web_search};
use crate::llm::{ChatMessage, LlmCmd, LlmEvent, Role};

/// Cap on tool calls before we force an answer. A search plus a page fetch or
/// two is plenty; more just burns a small model's context.
const MAX_TOOL_CALLS: usize = 3;

/// Low temperature for the routing turns — we want parseable tool-call JSON,
/// not creative prose. The final answer uses the caller's chat temperature.
const ROUTER_TEMP: f32 = 0.3;

const ROUTER_PROMPT: &str = "You can look things up on the web before answering. \
Two tools are available:\n\
- web_search: arguments {\"query\": \"...\"} — search the web\n\
- fetch_url: arguments {\"url\": \"https://...\"} — fetch a page as plain text\n\
To use one, reply with ONLY this and nothing else:\n\
<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"...\"}}</tool_call>\n\
If the question involves current events, version numbers, prices, news, or \
anything newer than your training data, use web_search FIRST. If you can \
answer confidently from your own knowledge, just answer normally with no tool \
call.";

/// Drive the web pre-pass on a background thread, then hand off to the normal
/// streaming chat path. `conversation` is the raw chat history (user/assistant
/// turns, no system message). Final-answer tokens, `Stats`, and `GenDone` are
/// sent to `event_tx` exactly like a plain `Generate`, so the existing UI pump
/// renders them with no changes; `Info` events report search progress.
pub fn spawn(
    conversation: Vec<ChatMessage>,
    cmd_tx: Sender<LlmCmd>,
    event_tx: Sender<LlmEvent>,
    stop: Arc<AtomicBool>,
    temp: f32,
    n_ctx: u32,
) {
    std::thread::spawn(move || {
        let context = gather(&conversation, &cmd_tx, &event_tx, &stop, n_ctx);

        if stop.load(Ordering::Relaxed) {
            // The user stopped during the pre-pass; close out the turn so the
            // UI leaves its "generating" state.
            let _ = event_tx.send(LlmEvent::GenDone);
            return;
        }

        // Final answer: the real conversation, optionally grounded, streamed
        // through the normal path.
        let mut messages = Vec::with_capacity(conversation.len() + 1);
        if let Some(ctx) = &context {
            messages.push(ChatMessage {
                role: Role::System,
                content: format!(
                    "Web results gathered for the user's latest question. Base your \
                     answer on them and cite the source URLs. If they do not actually \
                     answer it, say so rather than guessing.\n\n{ctx}"
                ),
            });
        }
        messages.extend(conversation);

        if cmd_tx
            .send(LlmCmd::Generate {
                messages,
                reply: event_tx.clone(),
                temp,
                n_ctx,
            })
            .is_err()
        {
            let _ = event_tx.send(LlmEvent::Error("LLM worker unavailable".into()));
            let _ = event_tx.send(LlmEvent::GenDone);
        }
    });
}

/// The pre-pass: ask the model whether to search, run the web tools it requests
/// (bounded), and return the gathered text — or `None` if it chose to answer
/// from memory, in which case the final turn is an ordinary chat answer.
fn gather(
    conversation: &[ChatMessage],
    cmd_tx: &Sender<LlmCmd>,
    event_tx: &Sender<LlmEvent>,
    stop: &AtomicBool,
    n_ctx: u32,
) -> Option<String> {
    let mut messages = Vec::with_capacity(conversation.len() + 1);
    messages.push(ChatMessage {
        role: Role::System,
        content: ROUTER_PROMPT.into(),
    });
    messages.extend(conversation.iter().cloned());

    let mut gathered: Vec<String> = Vec::new();
    for _ in 0..MAX_TOOL_CALLS {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let response = run_turn(&messages, cmd_tx, stop, n_ctx)?;
        let Some(call) = parse_tool_call(&response) else {
            break; // answered from memory — no grounding needed
        };
        let (label, result) = match call.name.as_str() {
            "web_search" => {
                let q = call.arg("query").unwrap_or_default().to_string();
                let _ = event_tx.send(LlmEvent::Info(format!("searching the web: {q}")));
                ("web_search", web_search(&q))
            }
            "fetch_url" => {
                let u = call.arg("url").unwrap_or_default().to_string();
                let _ = event_tx.send(LlmEvent::Info(format!("reading {u}")));
                ("fetch_url", fetch_url(&u))
            }
            // Chat exposes web tools only — never files or the shell.
            _ => break,
        };
        let text = result.unwrap_or_else(|e| format!("Error: {e}"));
        gathered.push(text.clone());
        // Feed the round-trip back so a follow-up (e.g. fetch after search)
        // can build on it.
        messages.push(ChatMessage {
            role: Role::Assistant,
            content: response,
        });
        messages.push(ChatMessage {
            role: Role::User,
            content: format!("Tool result ({label}):\n{text}"),
        });
    }

    if gathered.is_empty() {
        None
    } else {
        Some(gathered.join("\n\n---\n\n"))
    }
}

/// One non-streamed generation: collect the whole response on a private channel
/// so the router's tool-call JSON never reaches the chat window. Returns `None`
/// if the worker errored or went away.
fn run_turn(
    messages: &[ChatMessage],
    cmd_tx: &Sender<LlmCmd>,
    stop: &AtomicBool,
    n_ctx: u32,
) -> Option<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    cmd_tx
        .send(LlmCmd::Generate {
            messages: messages.to_vec(),
            reply: tx,
            temp: ROUTER_TEMP,
            n_ctx,
        })
        .ok()?;
    let mut out = String::new();
    for event in rx {
        match event {
            LlmEvent::Token(t) => {
                out.push_str(&t);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
            }
            LlmEvent::GenDone => break,
            LlmEvent::Error(_) => return None,
            _ => {}
        }
    }
    Some(out)
}
