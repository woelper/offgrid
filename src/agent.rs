//! Simple coding agent: a per-run thread drives a tool-calling loop against
//! the LLM worker. Tool calls are prompt-based (`<tool_call>{json}</tool_call>`,
//! the format Qwen models are trained on) so any GGUF chat model can be used.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};

use crate::llm::{ChatMessage, LlmCmd, LlmEvent, Role};

const MAX_ITERATIONS: usize = 25;
const MAX_FILE_READ: u64 = 50 * 1024;
const MAX_LIST_ENTRIES: usize = 200;
const MAX_TOOL_OUTPUT: usize = 16 * 1024;
const COMMAND_TIMEOUT_SECS: u64 = 60;
const WEB_UA: &str = "Mozilla/5.0 (X11; Linux x86_64) offgrid/0.1";
const OFFLINE_HINT: &str =
    "Continue using local knowledge and mention in your summary that the web was unavailable.";

pub enum AgentEvent {
    /// Streamed model output for the current turn.
    Token(String),
    /// Status note for the transcript (e.g. context compaction).
    Info(String),
    /// Tokens currently occupied in the context window (after a turn).
    Ctx(usize),
    /// The model's turn finished; `content` is the full reply.
    TurnDone,
    ToolCall {
        name: String,
        summary: String,
    },
    ToolResult {
        output: String,
        /// Whether the tool completed successfully (derived from the output
        /// markers this crate itself produces — cross-platform by design).
        ok: bool,
    },
    NeedsApproval {
        command: String,
        reply: Sender<bool>,
    },
    Done {
        iterations: usize,
    },
    Error(String),
}

/// Verbatim session log for post-mortem debugging: raw model output, parse
/// results, full tool I/O. One file per run under the data dir.
struct SessionLog {
    file: Option<std::fs::File>,
    pub path: std::path::PathBuf,
}

impl SessionLog {
    fn new() -> Self {
        let dir = crate::config::logs_dir();
        let _ = std::fs::create_dir_all(&dir);
        // Seconds since epoch keeps names unique enough per machine.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = dir.join(format!("agent-{stamp}.log"));
        let file = std::fs::File::create(&path).ok();
        Self { file, path }
    }

    fn log(&mut self, tag: &str, text: &str) {
        if let Some(f) = &mut self.file {
            use std::io::Write as _;
            let _ = writeln!(f, "\n===== {tag} =====\n{text}");
            let _ = f.flush();
        }
    }
}

pub struct AgentRun {
    pub rx: Receiver<AgentEvent>,
    pub stop: Arc<AtomicBool>,
    /// Live-updatable: toggling auto-approve mid-run takes effect on the
    /// next command instead of only on the next run.
    pub auto_approve: Arc<AtomicBool>,
}

pub fn start(
    workspace: PathBuf,
    task: String,
    cmd_tx: Sender<LlmCmd>,
    auto_approve: bool,
    web_tools: bool,
    n_ctx: u32,
) -> AgentRun {
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let auto = Arc::new(AtomicBool::new(auto_approve));
    let auto_thread = auto.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_loop(
            &workspace,
            &task,
            &cmd_tx,
            &tx,
            &stop_thread,
            &auto_thread,
            web_tools,
            n_ctx,
        ) {
            let _ = tx.send(AgentEvent::Error(e));
        }
    });
    AgentRun {
        rx,
        stop,
        auto_approve: auto,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    workspace: &Path,
    task: &str,
    cmd_tx: &Sender<LlmCmd>,
    tx: &Sender<AgentEvent>,
    stop: &AtomicBool,
    auto_approve: &AtomicBool,
    web_tools: bool,
    n_ctx: u32,
) -> Result<(), String> {
    let mut log = SessionLog::new();
    let _ = tx.send(AgentEvent::Info(format!(
        "session log: {}",
        log.path.display()
    )));
    let mut messages = vec![
        ChatMessage {
            role: Role::System,
            content: system_prompt(workspace, web_tools),
        },
        ChatMessage {
            role: Role::User,
            content: task.to_string(),
        },
    ];
    log.log("SYSTEM PROMPT", &messages[0].content);
    log.log("TASK", task);
    if recent_history(workspace).is_some() {
        let _ = tx.send(AgentEvent::Info(
            "project history from earlier sessions injected into context".into(),
        ));
    }
    let mut files_touched: Vec<String> = Vec::new();

    let mut format_retries = 0usize;
    let mut compact_level = 0usize;
    for iteration in 1..=MAX_ITERATIONS {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(LlmCmd::Generate {
                messages: messages.clone(),
                reply: reply_tx,
                // Low temperature: agent runs need valid JSON and careful
                // code much more than they need creative variety.
                temp: 0.25,
                n_ctx,
            })
            .map_err(|_| "LLM worker unavailable".to_string())?;

        let mut response = String::new();
        let mut gen_error: Option<String> = None;
        for event in reply_rx {
            match event {
                LlmEvent::Token(t) => {
                    response.push_str(&t);
                    let _ = tx.send(AgentEvent::Token(t));
                }
                LlmEvent::GenDone => break,
                LlmEvent::Error(e) => gen_error = Some(e),
                LlmEvent::Stats {
                    prompt_tokens,
                    gen_tokens,
                    ..
                } => {
                    let _ = tx.send(AgentEvent::Ctx(prompt_tokens + gen_tokens));
                }
                _ => {}
            }
        }
        if let Some(e) = gen_error {
            // On context overflow, compact progressively harder and retry
            // instead of aborting the run.
            if e.starts_with("context window full") && compact_level < 2 {
                compact_level += 1;
                log.log("COMPACTION", &format!("level {}", compact_level));
                compact_transcript(&mut messages, compact_level);
                let _ = tx.send(AgentEvent::Info(format!(
                    "context window full — compacting transcript (level {compact_level}) and retrying"
                )));
                continue;
            }
            return Err(if e.starts_with("context window full") {
                format!(
                    "{e} — the task transcript is too long even after compaction; try a smaller task"
                )
            } else {
                e
            });
        }
        let _ = tx.send(AgentEvent::TurnDone);
        log.log(
            &format!("MODEL RESPONSE (turn {iteration}, raw)"),
            &response,
        );
        messages.push(ChatMessage {
            role: Role::Assistant,
            content: response.clone(),
        });

        let Some(call) = parse_tool_call(&response) else {
            // A reply that clearly tried to call a tool but could not be
            // parsed gets one corrective nudge instead of ending the run.
            let attempted = response.contains("<tool_call>")
                || (response.contains("\"name\"") && response.contains("\"arguments\""));
            if attempted && format_retries < 2 {
                format_retries += 1;
                log.log(
                    "PARSE FAILURE",
                    "response looked like a tool call but did not parse; nudging model",
                );
                let _ = tx.send(AgentEvent::Info(
                    "tool call could not be parsed — asking the model to retry".into(),
                ));
                messages.push(ChatMessage {
                    role: Role::User,
                    content: "Your tool call could not be parsed. Emit exactly one call as \
                              <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call> \
                              with valid JSON — or, if the task is finished, reply with a \
                              summary and no tool call."
                        .into(),
                });
                continue;
            }
            if !response.trim().is_empty() {
                append_history(workspace, task, &response, &files_touched);
            }
            let _ = tx.send(AgentEvent::Done {
                iterations: iteration,
            });
            return Ok(());
        };

        log.log("TOOL CALL", &format!("{}: {}", call.name, call.arguments));
        let _ = tx.send(AgentEvent::ToolCall {
            name: call.name.clone(),
            summary: call.summary(),
        });

        let output = if call.name == "run_command" && !auto_approve.load(Ordering::Relaxed) {
            let command = call.arg("command").unwrap_or_default().to_string();
            let (approve_tx, approve_rx) = std::sync::mpsc::channel();
            let _ = tx.send(AgentEvent::NeedsApproval {
                command,
                reply: approve_tx,
            });
            match approve_rx.recv() {
                Ok(true) => execute(&call, workspace, web_tools),
                Ok(false) => "Command denied by the user.".to_string(),
                Err(_) => break, // UI went away / run aborted
            }
        } else {
            execute(&call, workspace, web_tools)
        };
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let mut output = output;
        if output.len() > MAX_TOOL_OUTPUT {
            // Keep head AND tail: compilers put the errors at the end.
            let head_end = output
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= MAX_TOOL_OUTPUT / 4)
                .last()
                .unwrap_or(0);
            let tail_target = output.len() - (MAX_TOOL_OUTPUT - MAX_TOOL_OUTPUT / 4);
            let tail_start = output
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| i >= tail_target)
                .unwrap_or(output.len());
            output = format!(
                "{}\n[… middle of output omitted …]\n{}",
                &output[..head_end],
                &output[tail_start..]
            );
        }
        let ok = tool_output_ok(&output);
        if ok
            && call.name == "write_file"
            && let Some(path) = call.arg("path")
            && !files_touched.iter().any(|f| f == path)
        {
            files_touched.push(path.to_string());
        }
        // A successful write_file leaves a full copy of the file in the
        // assistant's turn — the biggest context hog. Replace it with plain
        // prose (NOT a syntactically valid tool call: the model imitates its
        // own turns, and a placeholder-shaped call once got written to disk
        // verbatim). The content is on disk and can be re-read if needed.
        if ok
            && call.name == "write_file"
            && let Some(last) = messages.last_mut()
            && last.role == Role::Assistant
            && let Some(pos) = last.content.find("<tool_call>")
        {
            let note = format!(
                "(wrote {} bytes to {} with the write_file tool)",
                call.arg("content").map(str::len).unwrap_or(0),
                call.arg("path").unwrap_or("")
            );
            last.content.truncate(pos);
            last.content.push_str(&note);
        }
        if output.is_empty() {
            // Commands like cp/rm succeed silently — say so explicitly, for
            // both the transcript and the model.
            output = "(no output — completed successfully)".to_string();
        }
        log.log(&format!("TOOL RESULT (ok={ok})"), &output);
        let _ = tx.send(AgentEvent::ToolResult {
            output: output.clone(),
            ok,
        });
        messages.push(ChatMessage {
            role: Role::User,
            content: format!("<tool_response>\n{output}\n</tool_response>"),
        });
    }

    let _ = tx.send(AgentEvent::Done {
        iterations: MAX_ITERATIONS,
    });
    Ok(())
}

/// Did a tool call succeed? All failure paths in this crate mark the output:
/// file/web tools prefix "Error:", run_command appends a non-zero
/// "[exit code: …]", denials and offline results carry fixed phrases.
fn tool_output_ok(output: &str) -> bool {
    !(output.starts_with("Error:")
        || output.contains("\n[exit code:")
        || output.starts_with("Command denied")
        || output.starts_with("Web access is unavailable"))
}

/// Shrink the transcript to free context. Level 1 trims old tool responses;
/// level 2 trims all tool responses and long assistant turns, keeping the
/// system prompt and the task intact.
fn compact_transcript(messages: &mut [ChatMessage], level: usize) {
    let len = messages.len();
    let keep_from = if level >= 2 {
        len.saturating_sub(1)
    } else {
        len.saturating_sub(4)
    };
    // Never touch the system prompt (0) and the task (1).
    for m in messages[..keep_from].iter_mut().skip(2) {
        if m.role == Role::User && m.content.starts_with("<tool_response>") && m.content.len() > 600
        {
            let head: String = m.content.chars().take(300).collect();
            m.content = format!("{head}\n[older tool output trimmed]\n</tool_response>");
        }
        if level >= 2 && m.role == Role::Assistant && m.content.len() > 1500 {
            let head: String = m.content.chars().take(700).collect();
            m.content = format!("{head}\n[rest of this turn trimmed]");
        }
    }
}

const HISTORY_RECENT: usize = 5;
const HISTORY_MAX_CHARS: usize = 2000;

fn history_path(workspace: &Path) -> PathBuf {
    workspace.join(".offgrid").join("history.md")
}

fn strip_think_blocks(s: &str) -> String {
    let mut out = s.to_string();
    while let Some(start) = out.find("<think>") {
        let end = out[start..]
            .find("</think>")
            .map(|e| start + e + "</think>".len())
            .unwrap_or(out.len());
        out.replace_range(start..end, "");
    }
    out
}

/// Append a finished task to the workspace's rolling history. The file is a
/// plain, user-editable markdown file in `.offgrid/history.md`.
fn append_history(workspace: &Path, task: &str, summary: &str, files: &[String]) {
    let path = history_path(workspace);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let task_line: String = task
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(120)
        .collect();
    let mut summary = strip_think_blocks(summary).trim().to_string();
    if summary.chars().count() > 600 {
        summary = summary.chars().take(600).collect::<String>() + "…";
    }
    let files_line = if files.is_empty() {
        String::new()
    } else {
        format!("\nFiles touched: {}", files.join(", "))
    };
    let entry = format!("\n## Task: {task_line}\n{summary}{files_line}\n");
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(entry.as_bytes());
    }
}

/// The most recent history entries, capped so they can't crowd the prompt.
fn recent_history(workspace: &Path) -> Option<String> {
    let text = std::fs::read_to_string(history_path(workspace)).ok()?;
    let mut entries: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if line.starts_with("## ") && !cur.trim().is_empty() {
            entries.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        entries.push(cur);
    }
    let start = entries.len().saturating_sub(HISTORY_RECENT);
    let recent = entries[start..].join("\n");
    let recent: String = if recent.chars().count() > HISTORY_MAX_CHARS {
        // Keep the tail: newest entries win.
        let skip = recent.chars().count() - HISTORY_MAX_CHARS;
        recent.chars().skip(skip).collect()
    } else {
        recent
    };
    (!recent.trim().is_empty()).then_some(recent)
}

fn system_prompt(workspace: &Path, web_tools: bool) -> String {
    let listing = list_files_impl(workspace, workspace, 1).unwrap_or_default();
    let agents_md = std::fs::read_to_string(workspace.join("AGENTS.md"))
        .map(|s| format!("\nProject instructions (AGENTS.md):\n{s}\n"))
        .unwrap_or_default();
    let history = recent_history(workspace)
        .map(|h| {
            format!(
                "\nWhat happened in this workspace in earlier sessions (newest last):\n{h}\n\
                 Trust this history: do not redo or rewrite work it describes as done.\n"
            )
        })
        .unwrap_or_default();
    let web = if web_tools {
        "- web_search: arguments {\"query\": \"...\"} — search the web\n\
         - fetch_url: arguments {\"url\": \"https://...\"} — fetch a web page as plain text\n\
         If the task involves \"latest\", \"current\", version numbers, URLs, or anything \
         possibly newer than your training data, call web_search FIRST instead of \
         answering from memory. Web tools may be offline: if one reports that web access \
         is unavailable, do NOT retry more than once — continue with local knowledge.\n"
    } else {
        ""
    };
    format!(
        "You are a coding agent working in the workspace directory {ws}. All paths are \
relative to it.\n\
\n\
You have these tools:\n\
- list_files: arguments {{\"path\": \"optional/subdir\"}} — list files recursively\n\
- read_file: arguments {{\"path\": \"file\"}} — read a file\n\
- write_file: arguments {{\"path\": \"file\", \"content\": \"...\"}} — create or overwrite a file\n\
- run_command: arguments {{\"command\": \"shell command\"}} — run a shell command in the workspace\n\
{web}\
\n\
To use a tool, end your reply with exactly one call in this format:\n\
<tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>\n\
The result will come back in a <tool_response> block. Use one tool at a time.\n\
\n\
Example turn:\n\
  user: What is in this project?\n\
  you: I'll look at the files.\n\
  <tool_call>{{\"name\": \"list_files\", \"arguments\": {{}}}}</tool_call>\n\
  (a <tool_response> block arrives, then you continue with the next step)\n\
\n\
After writing or changing code, ALWAYS run it or the project's tests with \
run_command and fix any errors until it succeeds — never declare code done \
without running it.\n\
The workspace may already contain a working project from earlier tasks. Read \
files before changing them, make the smallest change that fulfils the task, \
and NEVER rewrite an existing file from scratch unless the task explicitly \
asks for a rewrite.\n\
When the task is complete, reply with a short summary and NO tool call.\n\
{agents}\
{history}\
\n\
Top-level files in the workspace:\n{listing}",
        ws = workspace.display(),
        agents = agents_md,
        history = history,
        web = web,
    )
}

pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn arg(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    pub fn summary(&self) -> String {
        match self.name.as_str() {
            "run_command" => self.arg("command").unwrap_or("?").to_string(),
            "write_file" => format!(
                "{} ({} bytes)",
                self.arg("path").unwrap_or("?"),
                self.arg("content").map(str::len).unwrap_or(0)
            ),
            "web_search" => self.arg("query").unwrap_or("?").to_string(),
            "fetch_url" => self.arg("url").unwrap_or("?").to_string(),
            _ => self.arg("path").unwrap_or("").to_string(),
        }
    }
}

const KNOWN_TOOLS: [&str; 6] = [
    "list_files",
    "read_file",
    "write_file",
    "run_command",
    "web_search",
    "fetch_url",
];

pub fn parse_tool_call(response: &str) -> Option<ToolCall> {
    // Ignore anything inside <think> blocks by only looking after the last one.
    let searchable = match response.rfind("</think>") {
        Some(i) => &response[i..],
        None => response,
    };

    // 1) The requested format: <tool_call>{json}</tool_call>
    if let Some(start) = searchable.find("<tool_call>") {
        let rest = &searchable[start + "<tool_call>".len()..];
        let end = rest.find("</tool_call>").unwrap_or(rest.len());
        if let Some(call) = call_from_json(rest[..end].trim(), false) {
            return Some(call);
        }
    }

    // 2) Lenient: a ```json fenced block holding the call object.
    let mut rest = searchable;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        let end = body.find("```").unwrap_or(body.len());
        if let Some(call) = call_from_json(body[..end].trim(), true) {
            return Some(call);
        }
        rest = &after[body_start + end..];
    }

    // 3) Lenient: a bare, balanced JSON object mentioning a known tool.
    let mut from = 0;
    for _ in 0..20 {
        let Some(rel) = searchable[from..].find('{') else {
            break;
        };
        let start = from + rel;
        if let Some(end) = balanced_json_end(&searchable[start..])
            && let Some(call) = call_from_json(&searchable[start..start + end], true)
        {
            return Some(call);
        }
        from = start + 1;
    }
    None
}

/// Models often emit literal newlines/tabs inside JSON strings (invalid
/// JSON, especially in write_file content). Escape them so parsing succeeds.
pub fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        if in_str {
            if esc {
                out.push(c);
                esc = false;
                continue;
            }
            match c {
                '\\' => {
                    out.push(c);
                    esc = true;
                }
                '"' => {
                    out.push(c);
                    in_str = false;
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c => out.push(c),
            }
        } else {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
        }
    }
    out
}

/// Parse a candidate JSON string into a tool call. `strict_names` restricts to
/// known tools — used for the lenient fallbacks so ordinary JSON in a summary
/// is not mistaken for a call.
fn call_from_json(s: &str, strict_names: bool) -> Option<ToolCall> {
    let json: serde_json::Value = serde_json::from_str(s)
        .or_else(|_| serde_json::from_str(&escape_control_chars_in_strings(s)))
        .ok()?;
    let name = json.get("name")?.as_str()?.to_string();
    if strict_names && !KNOWN_TOOLS.contains(&name.as_str()) {
        return None;
    }
    let arguments = json
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    Some(ToolCall { name, arguments })
}

/// Byte offset one past the end of the balanced JSON object starting at `s[0]`
/// (which must be '{'), respecting strings and escapes.
fn balanced_json_end(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in s.bytes().enumerate() {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn execute(call: &ToolCall, workspace: &Path, web_tools: bool) -> String {
    let result = match call.name.as_str() {
        "web_search" if web_tools => web_search(call.arg("query").unwrap_or("")),
        "fetch_url" if web_tools => fetch_url(call.arg("url").unwrap_or("")),
        "web_search" | "fetch_url" => {
            Err("web tools are disabled — solve the task with local tools only".into())
        }
        "list_files" => resolve(workspace, call.arg("path").unwrap_or(""))
            .and_then(|dir| list_files_impl(&dir, workspace, 8)),
        "read_file" => resolve(workspace, call.arg("path").unwrap_or("")).and_then(|path| {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if size > MAX_FILE_READ {
                return Err(format!(
                    "file too large ({size} bytes, limit {MAX_FILE_READ})"
                ));
            }
            std::fs::read_to_string(&path).map_err(|e| e.to_string())
        }),
        "write_file" => resolve(workspace, call.arg("path").unwrap_or("")).and_then(|path| {
            let content = call.arg("content").unwrap_or("");
            if looks_like_placeholder(content) {
                return Err(format!(
                    "content looks like a placeholder ({content:?}), not real file \
                     content — send the actual file content"
                ));
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&path, content)
                .map(|_| format!("wrote {} bytes to {}", content.len(), path.display()))
                .map_err(|e| e.to_string())
        }),
        "run_command" => run_command(call.arg("command").unwrap_or(""), workspace),
        other => Err(format!("unknown tool '{other}'")),
    };
    match result {
        Ok(out) => out,
        Err(e) => format!("Error: {e}"),
    }
}

/// Catch the model writing a placeholder instead of real file content
/// (e.g. "[3519 bytes written to disk]", "...", "(content omitted)").
fn looks_like_placeholder(content: &str) -> bool {
    let t = content.trim();
    if t.len() > 120 {
        return false;
    }
    let bracketed = (t.starts_with('[') && t.ends_with(']'))
        || (t.starts_with('(') && t.ends_with(')'))
        || (t.starts_with('<') && t.ends_with('>'));
    bracketed
        && (t.contains("bytes")
            || t.contains("omitted")
            || t.contains("written")
            || t.contains("content")
            || t.contains("unchanged")
            || t.contains("same as"))
}

/// Resolve a path relative to the workspace and reject anything escaping it.
pub fn resolve(workspace: &Path, path: &str) -> Result<PathBuf, String> {
    let joined = if path.is_empty() {
        workspace.to_path_buf()
    } else {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            workspace.join(p)
        }
    };
    // Canonicalize the deepest existing ancestor so `..` cannot escape via
    // not-yet-existing paths.
    let mut existing = joined.clone();
    let mut suffix = PathBuf::new();
    while !existing.exists() {
        let Some(name) = existing.file_name().map(|n| n.to_owned()) else {
            return Err("invalid path".into());
        };
        suffix = if suffix.as_os_str().is_empty() {
            PathBuf::from(&name)
        } else {
            Path::new(&name).join(&suffix)
        };
        existing = existing
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "invalid path".to_string())?;
    }
    let canon = existing.canonicalize().map_err(|e| e.to_string())?;
    let ws = workspace
        .canonicalize()
        .map_err(|e| format!("workspace: {e}"))?;
    let full = if suffix.as_os_str().is_empty() {
        canon
    } else {
        canon.join(suffix)
    };
    if full.starts_with(&ws) {
        Ok(full)
    } else {
        Err(format!("path '{path}' is outside the workspace"))
    }
}

fn list_files_impl(dir: &Path, workspace: &Path, max_depth: usize) -> Result<String, String> {
    let mut lines = Vec::new();
    let mut queue = vec![(dir.to_path_buf(), 0usize)];
    while let Some((d, depth)) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if lines.len() >= MAX_LIST_ENTRIES {
                lines.push("[listing truncated]".to_string());
                return Ok(lines.join("\n"));
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if matches!(
                name.as_str(),
                ".git" | "target" | "node_modules" | ".venv" | ".offgrid"
            ) {
                continue;
            }
            let path = entry.path();
            let rel = path
                .strip_prefix(workspace)
                .unwrap_or(&path)
                .display()
                .to_string();
            if path.is_dir() {
                lines.push(format!("{rel}/"));
                if depth + 1 < max_depth {
                    queue.push((path, depth + 1));
                }
            } else {
                lines.push(rel);
            }
        }
    }
    if lines.is_empty() {
        lines.push("(empty)".to_string());
    }
    Ok(lines.join("\n"))
}

fn run_command(command: &str, workspace: &Path) -> Result<String, String> {
    run_command_with_timeout(command, workspace, COMMAND_TIMEOUT_SECS)
}

/// Run a shell command with a timeout enforced in-process. The GNU `timeout`
/// binary we used before does not exist on macOS or Windows (os error 2).
fn run_command_with_timeout(
    command: &str,
    workspace: &Path,
    timeout_secs: u64,
) -> Result<String, String> {
    use std::io::Read as _;
    use wait_timeout::ChildExt;

    if command.is_empty() {
        return Err("empty command".into());
    }
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new("bash");
        c.arg("-c").arg(command);
        c
    };
    let mut child = cmd
        .current_dir(workspace)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start shell: {e}"))?;

    // Drain the pipes on threads so a chatty child can't fill the pipe
    // buffer, block, and turn into a false timeout.
    let mut stdout_pipe = child.stdout.take().unwrap();
    let mut stderr_pipe = child.stderr.take().unwrap();
    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    let status = match child
        .wait_timeout(std::time::Duration::from_secs(timeout_secs))
        .map_err(|e| e.to_string())?
    {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("command timed out after {timeout_secs}s"));
        }
    };
    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();

    let mut result = String::new();
    result.push_str(&String::from_utf8_lossy(&stdout));
    let stderr = String::from_utf8_lossy(&stderr);
    if !stderr.trim().is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&stderr);
    }
    if !status.success() {
        result.push_str(&format!("\n[exit code: {}]", status.code().unwrap_or(-1)));
    }
    Ok(result)
}

fn web_agent() -> ureq::Agent {
    // Short timeouts so an offline machine fails fast instead of stalling the run.
    ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(4)))
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .build()
        .into()
}

fn web_get(url: &str) -> Result<String, String> {
    let mut res = web_agent()
        .get(url)
        .header("User-Agent", WEB_UA)
        .call()
        .map_err(|e| e.to_string())?;
    res.body_mut().read_to_string().map_err(|e| e.to_string())
}

fn offline(err: String) -> String {
    format!("Web access is unavailable (offline?): {err}. {OFFLINE_HINT}")
}

fn web_search(query: &str) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("empty query".into());
    }
    let q = urlencode(query.trim());
    // Primary: DuckDuckGo Lite. It sometimes serves a bot challenge; treat a
    // parse miss the same as being offline and fall back to Wikipedia.
    if let Ok(html) = web_get(&format!("https://lite.duckduckgo.com/lite/?q={q}")) {
        let results = parse_ddg_lite(&html);
        if !results.is_empty() {
            return Ok(results.join("\n\n"));
        }
    }
    match web_get(&format!(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=5"
    )) {
        Ok(json) => {
            let parsed: serde_json::Value =
                serde_json::from_str(&json).map_err(|e| e.to_string())?;
            let mut out = vec!["(search provider unavailable — Wikipedia results)".to_string()];
            if let Some(hits) = parsed["query"]["search"].as_array() {
                for hit in hits {
                    let title = hit["title"].as_str().unwrap_or_default();
                    let snippet = html_to_text(hit["snippet"].as_str().unwrap_or_default());
                    let slug = urlencode(&title.replace(' ', "_"));
                    out.push(format!(
                        "{title} — https://en.wikipedia.org/wiki/{slug}\n{snippet}"
                    ));
                }
            }
            if out.len() > 1 {
                Ok(out.join("\n\n"))
            } else {
                Ok(format!("No results found. {OFFLINE_HINT}"))
            }
        }
        Err(e) => Ok(offline(e)),
    }
}

fn fetch_url(url: &str) -> Result<String, String> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("only http(s) URLs are supported".into());
    }
    match web_get(url) {
        Ok(html) => Ok(html_to_text(&html)),
        Err(e) => Ok(offline(e)),
    }
}

/// Parse DuckDuckGo Lite results: `<a rel="nofollow" href="//duckduckgo.com/l/?uddg=<url>&amp;rut=...">title</a>`
/// followed by a `result-snippet` cell.
fn parse_ddg_lite(html: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("uddg=") {
        rest = &rest[pos + 5..];
        let end = rest.find(['&', '"']).unwrap_or(rest.len());
        let url = urldecode(&rest[..end]);
        let title = rest
            .find('>')
            .map(|gt| {
                let after = &rest[gt + 1..];
                html_to_text(&after[..after.find("</a>").unwrap_or(0)])
            })
            .unwrap_or_default();
        let snippet = rest
            .find("result-snippet")
            .and_then(|s| {
                let after = &rest[s..];
                let gt = after.find('>')?;
                let cell = &after[gt + 1..];
                Some(html_to_text(&cell[..cell.find("</td>").unwrap_or(0)]))
            })
            .unwrap_or_default();
        if !url.is_empty() && !title.is_empty() {
            out.push(format!("{title} — {url}\n{snippet}"));
        }
        if out.len() >= 5 {
            break;
        }
    }
    out
}

/// Strip scripts, styles and tags; decode common entities; collapse whitespace.
fn html_to_text(html: &str) -> String {
    let mut s = html.to_string();
    for tag in ["script", "style"] {
        loop {
            // ascii_lowercase keeps byte offsets identical to the original.
            let lower = s.to_ascii_lowercase();
            let Some(start) = lower.find(&format!("<{tag}")) else {
                break;
            };
            let end = lower[start..]
                .find(&format!("</{tag}>"))
                .map(|e| start + e + tag.len() + 3)
                .unwrap_or(s.len());
            s.replace_range(start..end, " ");
        }
    }
    let mut text = String::with_capacity(s.len() / 2);
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                text.push(' ');
            }
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(&line);
        out.push('\n');
    }
    out.trim().to_string()
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub const AGENTS_MD_TEMPLATE: &str = "\
# Project instructions

Describe this project for the coding agent: what it is, how to build and test
it, and any conventions to follow.
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_call() {
        let r = "I'll list the files.\n<tool_call>{\"name\": \"list_files\", \"arguments\": {\"path\": \"src\"}}</tool_call>";
        let call = parse_tool_call(r).unwrap();
        assert_eq!(call.name, "list_files");
        assert_eq!(call.arg("path"), Some("src"));
    }

    #[test]
    fn no_tool_call_in_plain_reply() {
        assert!(parse_tool_call("All done, the tests pass.").is_none());
    }

    #[test]
    fn ignores_tool_call_inside_think() {
        let r = "<think>maybe <tool_call>{\"name\": \"x\"}</tool_call></think>Done.";
        assert!(parse_tool_call(r).is_none());
    }

    #[test]
    fn history_roundtrip_keeps_newest_entries() {
        let ws = std::env::temp_dir().join("offgrid-history-test");
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();
        for i in 0..7 {
            append_history(
                &ws,
                &format!("task number {i}"),
                "<think>internal</think>Built the thing successfully.",
                &[format!("src/file{i}.rs")],
            );
        }
        let recent = recent_history(&ws).unwrap();
        // capped at the newest 5 entries
        assert!(!recent.contains("task number 0"));
        assert!(!recent.contains("task number 1"));
        assert!(recent.contains("task number 2"));
        assert!(recent.contains("task number 6"));
        // think blocks are stripped, summary and files are present
        assert!(!recent.contains("<think>"));
        assert!(recent.contains("Built the thing successfully."));
        assert!(recent.contains("src/file6.rs"));
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn write_file_rejects_placeholder_content() {
        assert!(looks_like_placeholder("[3519 bytes written to disk]"));
        assert!(looks_like_placeholder("(content unchanged)"));
        assert!(looks_like_placeholder("<file content omitted>"));
        assert!(!looks_like_placeholder("[package]\nname = \"demo\"")); // real TOML
        assert!(!looks_like_placeholder("fn main() {}"));
        let call = ToolCall {
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "x.txt", "content": "[10 bytes written to disk]"}),
        };
        let out = execute(&call, &std::env::temp_dir(), false);
        assert!(out.contains("placeholder"));
    }

    #[test]
    fn repairs_raw_newlines_inside_json_strings() {
        // Models often emit real newlines inside the content string.
        let r = "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"main.rs\", \"content\": \"fn main() {\n    println!(\\\"hi\\\");\n}\"}}</tool_call>";
        let call = parse_tool_call(r).expect("repaired parse");
        assert_eq!(call.name, "write_file");
        assert!(call.arg("content").unwrap().contains("println!"));
    }

    #[test]
    fn lenient_parses_fenced_json() {
        let r = "I'll check the files:\n```json\n{\"name\": \"list_files\", \"arguments\": {\"path\": \"src\"}}\n```";
        let call = parse_tool_call(r).unwrap();
        assert_eq!(call.name, "list_files");
        assert_eq!(call.arg("path"), Some("src"));
    }

    #[test]
    fn lenient_parses_bare_json_for_known_tools_only() {
        let r = "Running it now: {\"name\": \"run_command\", \"arguments\": {\"command\": \"ls\"}}";
        assert_eq!(parse_tool_call(r).unwrap().name, "run_command");
        // Unknown names in bare JSON are ordinary prose, not calls.
        let prose = "The config is {\"name\": \"my-app\", \"arguments\": {\"debug\": true}}";
        assert!(parse_tool_call(prose).is_none());
    }

    #[test]
    fn tolerates_missing_close_tag() {
        let r = "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}";
        assert_eq!(parse_tool_call(r).unwrap().name, "read_file");
    }

    #[test]
    fn parses_ddg_lite_results() {
        let html = r#"<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Fegui&amp;rut=abc123" class="result-link">egui - <b>Rust</b> docs</a></td></tr>
<tr><td class="result-snippet">egui is an immediate mode GUI library.</td></tr>
<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Femilk%2Fegui&amp;rut=def" class="result-link">GitHub - emilk/egui</a></td></tr>"#;
        let results = parse_ddg_lite(html);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("https://docs.rs/egui"));
        assert!(results[0].contains("egui - Rust docs"));
        assert!(results[0].contains("immediate mode GUI"));
        assert!(results[1].contains("https://github.com/emilk/egui"));
    }

    #[test]
    fn html_to_text_strips_scripts_and_tags() {
        let html = "<html><ScRipt>var x=1;</sCrIpt><style>.a{}</style><p>Hello&nbsp;<b>world</b> &amp; more</p></html>";
        let text = html_to_text(html);
        assert!(!text.contains("var x"));
        assert!(!text.contains(".a{}"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world & more"));
    }

    #[test]
    fn urldecode_roundtrip() {
        assert_eq!(
            urldecode("https%3A%2F%2Fdocs.rs%2Fegui"),
            "https://docs.rs/egui"
        );
        assert_eq!(urldecode(&urlencode("a b/c?d=e")), "a b/c?d=e");
    }

    #[test]
    fn compacts_old_tool_responses_only() {
        let long = format!("<tool_response>\n{}\n</tool_response>", "x".repeat(2000));
        let mut messages = vec![
            ChatMessage {
                role: Role::System,
                content: "sys".into(),
            },
            ChatMessage {
                role: Role::User,
                content: long.clone(),
            }, // old → trimmed
            ChatMessage {
                role: Role::Assistant,
                content: "a".into(),
            },
            ChatMessage {
                role: Role::User,
                content: long.clone(),
            }, // recent → kept
            ChatMessage {
                role: Role::Assistant,
                content: "b".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "task".into(),
            },
        ];
        compact_transcript(&mut messages, 1);
        // index 0/1 (system + task) are never touched; index 2+ within range is
        assert!(messages[2].content == "a" || messages[2].content.len() <= 400);
        assert_eq!(messages[3].content, long); // within the keep window at level 1
        compact_transcript(&mut messages, 2);
        assert!(messages[3].content.contains("[older tool output trimmed]"));
        assert!(messages[3].content.len() < 500);
        assert_eq!(messages[0].content, "sys"); // system prompt untouched
        assert_eq!(messages[5].content, "task"); // last message untouched
    }

    /// End-to-end proof that the agent loop uses web tools: a scripted fake
    /// LLM calls fetch_url against a local web server, and the page content
    /// must flow back into the conversation. Loopback only — CI-safe.
    #[test]
    fn agent_loop_uses_web_tools() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        // Tiny local server standing in for the internet.
        let server = tiny_http::Server::http(("127.0.0.1", 0)).unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let served = Arc::new(AtomicBool::new(false));
        let served_flag = served.clone();
        std::thread::spawn(move || {
            if let Ok(req) = server.recv() {
                served_flag.store(true, Ordering::SeqCst);
                let _ = req.respond(tiny_http::Response::from_string(
                    "<html><body>hello from the fake web</body></html>",
                ));
            }
        });

        // Scripted "model": turn 1 calls fetch_url, turn 2 finishes — and
        // records whether the tool result made it into its transcript.
        let result_reached_model = Arc::new(AtomicBool::new(false));
        let reached = result_reached_model.clone();
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<LlmCmd>();
        std::thread::spawn(move || {
            let mut turn = 0;
            for cmd in cmd_rx {
                if let LlmCmd::Generate {
                    messages, reply, ..
                } = cmd
                {
                    turn += 1;
                    let text = if turn == 1 {
                        format!(
                            "<tool_call>{{\"name\": \"fetch_url\", \"arguments\": \
                             {{\"url\": \"http://127.0.0.1:{port}/\"}}}}</tool_call>"
                        )
                    } else {
                        if messages
                            .iter()
                            .any(|m| m.content.contains("hello from the fake web"))
                        {
                            reached.store(true, Ordering::SeqCst);
                        }
                        "Done: the page says hello.".to_string()
                    };
                    let _ = reply.send(LlmEvent::Token(text));
                    let _ = reply.send(LlmEvent::GenDone);
                }
            }
        });

        let run = start(
            std::env::temp_dir(),
            "check the fake web".into(),
            cmd_tx,
            true,
            true, // web tools enabled
            8192,
        );
        let mut saw_call = false;
        let mut saw_result = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let event = run.rx.recv_timeout(remaining).expect("agent run timed out");
            match event {
                AgentEvent::ToolCall { name, .. } if name == "fetch_url" => saw_call = true,
                AgentEvent::ToolResult { output, .. } => {
                    saw_result = output.contains("hello from the fake web");
                }
                AgentEvent::Done { .. } => break,
                AgentEvent::Error(e) => panic!("agent error: {e}"),
                _ => {}
            }
        }
        assert!(saw_call, "the agent never called fetch_url");
        assert!(
            served.load(Ordering::SeqCst),
            "the web server was never hit"
        );
        assert!(
            saw_result,
            "the page content did not appear in the tool result"
        );
        assert!(
            result_reached_model.load(Ordering::SeqCst),
            "the tool result was not fed back to the model"
        );
    }

    #[test]
    fn fetch_url_degrades_gracefully_offline() {
        // .invalid never resolves — DNS fails fast, like being offline.
        let out = fetch_url("https://no-such-host.invalid/").unwrap();
        assert!(out.contains("unavailable"));
        assert!(out.contains("local knowledge"));
    }

    #[test]
    #[ignore = "hits the live network"]
    fn web_search_live() {
        let out = web_search("rust programming language").unwrap();
        println!("--- live search output ---\n{out}");
        assert!(!out.is_empty());
    }

    #[test]
    fn web_tools_disabled_is_rejected() {
        let call = ToolCall {
            name: "web_search".into(),
            arguments: serde_json::json!({"query": "x"}),
        };
        let out = execute(&call, &std::env::temp_dir(), false);
        assert!(out.contains("disabled"));
    }

    #[test]
    fn run_command_captures_output() {
        let ws = std::env::temp_dir();
        let out = run_command_with_timeout("echo hello", &ws, 10).unwrap();
        assert!(out.contains("hello"));
    }

    #[cfg(unix)]
    #[test]
    fn run_command_captures_stderr_and_exit_code() {
        let ws = std::env::temp_dir();
        let out = run_command_with_timeout("echo oops >&2; exit 3", &ws, 10).unwrap();
        assert!(out.contains("[stderr]"));
        assert!(out.contains("oops"));
        assert!(out.contains("[exit code: 3]"));
    }

    #[cfg(windows)]
    #[test]
    fn run_command_captures_exit_code() {
        let ws = std::env::temp_dir();
        let out = run_command_with_timeout("exit /b 3", &ws, 10).unwrap();
        assert!(out.contains("[exit code: 3]"));
    }

    #[test]
    fn run_command_times_out() {
        let ws = std::env::temp_dir();
        #[cfg(unix)]
        let cmd = "sleep 30";
        #[cfg(windows)]
        let cmd = "ping -n 31 127.0.0.1 > nul";
        let err = run_command_with_timeout(cmd, &ws, 1).unwrap_err();
        assert!(err.contains("timed out"));
    }

    #[test]
    fn sandbox_rejects_escape() {
        let ws = std::env::temp_dir();
        assert!(resolve(&ws, "../../etc/passwd").is_err());
        assert!(resolve(&ws, "/etc/passwd").is_err());
        assert!(resolve(&ws, "sub/dir/file.txt").is_ok());
    }

    #[test]
    fn resolve_overwrites_existing_file() {
        let ws = std::env::temp_dir().join("offgrid-resolve-test");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("existing.txt"), "old").unwrap();
        let path = resolve(&ws, "existing.txt").unwrap();
        assert!(!path.to_string_lossy().ends_with("/"));
        std::fs::write(&path, "new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        std::fs::remove_dir_all(&ws).unwrap();
    }

    #[test]
    fn resolve_has_no_trailing_slash() {
        let ws = std::env::temp_dir();
        let path = resolve(&ws, "does-not-exist-yet.txt").unwrap();
        assert!(path.to_string_lossy().ends_with("does-not-exist-yet.txt"));
        assert!(!path.to_string_lossy().ends_with("/"));
        std::fs::write(&path, "x").unwrap();
        std::fs::remove_file(&path).unwrap();
    }
}
