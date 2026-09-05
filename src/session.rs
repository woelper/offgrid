//! Frontend-agnostic chat session: the conversation itself, the command
//! vocabulary, and the rules for what a message means.
//!
//! The desktop UI, the Telegram bridge and anything added later share one
//! conversation, so a chat started at the keyboard can be continued from a
//! phone with the model still knowing what was said. Only transport and
//! rendering stay in the frontends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::llm::{ChatMessage, Role};

/// Turns kept before the oldest are dropped. The model's context window is
/// the real limit; this only stops unbounded growth.
pub const MAX_HISTORY: usize = 40;

/// The one conversation, shared by every frontend.
pub type Conversation = Arc<Mutex<Vec<ChatMessage>>>;

pub fn conversation() -> Conversation {
    Arc::new(Mutex::new(Vec::new()))
}

/// Guards the single LLM worker: two frontends generating at once would
/// interleave their tokens into the same conversation.
#[derive(Clone, Default)]
pub struct ChatBusy(Arc<AtomicBool>);

impl ChatBusy {
    pub fn new() -> Self {
        Self::default()
    }

    /// True if this caller now owns generation.
    pub fn claim(&self) -> bool {
        !self.0.swap(true, Ordering::SeqCst)
    }

    pub fn release(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    #[allow(dead_code)] // used by frontends added later (REPL/TUI status)
    pub fn is_busy(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What a plain message means. Telegram and a terminal both have a single
/// input line, so the mode decides: talk to the model, or drive the agent.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub enum Mode {
    #[default]
    Chat,
    Code,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Chat => "chat",
            Mode::Code => "code",
        }
    }
}

/// A parsed instruction, independent of where it was typed.
#[derive(Clone, PartialEq, Debug)]
pub enum Command {
    /// Talk to the model.
    Chat(String),
    /// Start an agent run with this task.
    Code(String),
    /// Hand an instruction to the run already in flight.
    Steer(String),
    SwitchMode(Mode),
    Resume,
    Status,
    Stop,
    New,
    Help,
    /// Code mode asked for while `/code` is disabled.
    CodeDisabled,
    /// Whitespace only.
    Empty,
}

/// Turn typed text into a command. The same rules everywhere: a bare
/// message steers a live run, starts a task in code mode, or chats.
pub fn parse(text: &str, mode: Mode, run_active: bool, code_enabled: bool) -> Command {
    let text = text.trim();
    if text.is_empty() {
        return Command::Empty;
    }
    // Telegram sends "/cmd@botname" in groups.
    let (head, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    let head_lower = head.split('@').next().unwrap_or(head).to_ascii_lowercase();
    let rest = rest.trim();

    match head_lower.as_str() {
        "/start" | "/help" => Command::Help,
        "/status" => Command::Status,
        "/stop" => Command::Stop,
        "/new" | "/clear" => Command::New,
        "/resume" if code_enabled => Command::Resume,
        "/resume" => Command::CodeDisabled,
        "/chat" if rest.is_empty() => Command::SwitchMode(Mode::Chat),
        "/chat" => Command::Chat(rest.to_string()),
        "/code" if !code_enabled => Command::CodeDisabled,
        // `/code` alone switches mode; with a task it starts one right away.
        "/code" if rest.is_empty() => Command::SwitchMode(Mode::Code),
        "/code" => Command::Code(rest.to_string()),
        _ => {
            // Not a command: what it means depends on the situation.
            if run_active {
                Command::Steer(text.to_string())
            } else if mode == Mode::Code && code_enabled {
                Command::Code(text.to_string())
            } else {
                Command::Chat(text.to_string())
            }
        }
    }
}

/// Help text, shared so both frontends describe the same app.
pub fn help(mode: Mode, code_enabled: bool) -> String {
    let code_line = if code_enabled {
        "/code — switch to code mode: what you type becomes an agent task, \
         and anything sent while it runs is handed to it as a new \
         instruction. /stop aborts, /resume continues an interrupted run.\n"
    } else {
        ""
    };
    format!(
        "offgrid — now in {} mode.\n/chat — talk to the loaded model.\n\
         {code_line}/new starts a fresh conversation, /status shows what is \
         going on.",
        mode.label()
    )
}

/// Append a user turn and trim the backlog.
pub fn push_user(conv: &Conversation, text: &str) {
    let mut c = conv.lock().unwrap();
    c.push(ChatMessage {
        role: Role::User,
        content: text.to_string(),
    });
    trim(&mut c);
}

/// Append the empty assistant turn that streaming tokens flow into. Every
/// frontend renders the same conversation, so a reply typed on a phone
/// appears on the desktop as it is written.
pub fn push_assistant(conv: &Conversation) {
    conv.lock().unwrap().push(ChatMessage {
        role: Role::Assistant,
        content: String::new(),
    });
}

/// Add streamed text to the assistant turn in progress.
pub fn append_assistant(conv: &Conversation, text: &str) {
    let mut c = conv.lock().unwrap();
    if let Some(last) = c.last_mut()
        && last.role == Role::Assistant
    {
        last.content.push_str(text);
    }
}

/// Drop the turn in progress — used when generation fails, so the next try
/// starts from a clean transcript.
pub fn pop_unanswered(conv: &Conversation) {
    let mut c = conv.lock().unwrap();
    if let Some(last) = c.last()
        && last.role == Role::Assistant
        && last.content.is_empty()
    {
        c.pop();
    }
}

pub fn snapshot(conv: &Conversation) -> Vec<ChatMessage> {
    conv.lock().unwrap().clone()
}

pub fn clear(conv: &Conversation) {
    conv.lock().unwrap().clear();
}

pub fn turns(conv: &Conversation) -> usize {
    conv.lock().unwrap().len()
}

fn trim(c: &mut Vec<ChatMessage>) {
    if c.len() > MAX_HISTORY {
        let drop_n = c.len() - MAX_HISTORY;
        c.drain(..drop_n);
    }
}

/// Reasoning models emit `<think>` blocks; they are noise outside the
/// desktop UI, which renders them as a quote block.
pub fn strip_think(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_parse_the_same_everywhere() {
        let chat = Mode::Chat;
        let code = Mode::Code;
        // Bare text depends on mode…
        assert_eq!(
            parse("hello", chat, false, true),
            Command::Chat("hello".into())
        );
        assert_eq!(
            parse("fix the tests", code, false, true),
            Command::Code("fix the tests".into())
        );
        // …but a live run always takes precedence: you are steering it.
        assert_eq!(
            parse("also update the README", chat, true, true),
            Command::Steer("also update the README".into())
        );
        // Mode switches vs immediate tasks.
        assert_eq!(parse("/code", chat, false, true), Command::SwitchMode(code));
        assert_eq!(
            parse("/code do a thing", chat, false, true),
            Command::Code("do a thing".into())
        );
        assert_eq!(parse("/chat", code, false, true), Command::SwitchMode(chat));
        // Constants, case and group-suffix tolerant.
        assert_eq!(parse("/STATUS", chat, false, true), Command::Status);
        assert_eq!(parse("/stop@offgridbot", chat, true, true), Command::Stop);
        assert_eq!(parse("/clear", chat, false, true), Command::New);
        assert_eq!(parse("   ", chat, false, true), Command::Empty);
        // With code mode off, code commands are refused, not chatted.
        assert_eq!(parse("/code x", chat, false, false), Command::CodeDisabled);
        assert_eq!(parse("/resume", chat, false, false), Command::CodeDisabled);
        // …and a bare message in code mode falls back to chatting.
        assert_eq!(
            parse("hello", code, false, false),
            Command::Chat("hello".into())
        );
    }

    #[test]
    fn conversation_is_shared_and_trimmed() {
        let conv = conversation();
        push_user(&conv, "first");
        push_assistant(&conv);
        append_assistant(&conv, "hel");
        append_assistant(&conv, "lo");
        assert_eq!(snapshot(&conv)[1].content, "hello");

        // A failed turn leaves no empty assistant message behind.
        push_user(&conv, "second");
        push_assistant(&conv);
        pop_unanswered(&conv);
        assert_eq!(turns(&conv), 3);

        // Old turns fall off the front.
        for i in 0..MAX_HISTORY {
            push_user(&conv, &format!("msg {i}"));
        }
        assert_eq!(turns(&conv), MAX_HISTORY);
        assert!(!snapshot(&conv).iter().any(|m| m.content == "first"));

        clear(&conv);
        assert_eq!(turns(&conv), 0);
    }

    #[test]
    fn only_one_frontend_generates_at_a_time() {
        let busy = ChatBusy::new();
        assert!(busy.claim());
        assert!(!busy.claim()); // a second frontend must wait
        assert!(busy.is_busy());
        busy.release();
        assert!(busy.claim());
    }

    #[test]
    fn strips_reasoning_blocks() {
        assert_eq!(strip_think("<think>hmm</think>Answer."), "Answer.");
        assert_eq!(strip_think("plain"), "plain");
    }
}
