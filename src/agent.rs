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

pub enum AgentEvent {
    /// Streamed model output for the current turn.
    Token(String),
    /// The model's turn finished; `content` is the full reply.
    TurnDone,
    ToolCall { name: String, summary: String },
    ToolResult { output: String },
    NeedsApproval { command: String, reply: Sender<bool> },
    Done { iterations: usize },
    Error(String),
}

pub struct AgentRun {
    pub rx: Receiver<AgentEvent>,
    pub stop: Arc<AtomicBool>,
}

pub fn start(
    workspace: PathBuf,
    task: String,
    cmd_tx: Sender<LlmCmd>,
    auto_approve: bool,
) -> AgentRun {
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_loop(&workspace, &task, &cmd_tx, &tx, &stop_thread, auto_approve) {
            let _ = tx.send(AgentEvent::Error(e));
        }
    });
    AgentRun { rx, stop }
}

fn run_loop(
    workspace: &Path,
    task: &str,
    cmd_tx: &Sender<LlmCmd>,
    tx: &Sender<AgentEvent>,
    stop: &AtomicBool,
    auto_approve: bool,
) -> Result<(), String> {
    let mut messages = vec![
        ChatMessage {
            role: Role::System,
            content: system_prompt(workspace),
        },
        ChatMessage {
            role: Role::User,
            content: task.to_string(),
        },
    ];

    for iteration in 1..=MAX_ITERATIONS {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(LlmCmd::Generate {
                messages: messages.clone(),
                reply: reply_tx,
            })
            .map_err(|_| "LLM worker unavailable".to_string())?;

        let mut response = String::new();
        for event in reply_rx {
            match event {
                LlmEvent::Token(t) => {
                    response.push_str(&t);
                    let _ = tx.send(AgentEvent::Token(t));
                }
                LlmEvent::GenDone => break,
                LlmEvent::Error(e) => return Err(e),
                _ => {}
            }
        }
        let _ = tx.send(AgentEvent::TurnDone);
        messages.push(ChatMessage {
            role: Role::Assistant,
            content: response.clone(),
        });

        let Some(call) = parse_tool_call(&response) else {
            let _ = tx.send(AgentEvent::Done { iterations: iteration });
            return Ok(());
        };

        let _ = tx.send(AgentEvent::ToolCall {
            name: call.name.clone(),
            summary: call.summary(),
        });

        let output = if call.name == "run_command" && !auto_approve {
            let command = call.arg("command").unwrap_or_default().to_string();
            let (approve_tx, approve_rx) = std::sync::mpsc::channel();
            let _ = tx.send(AgentEvent::NeedsApproval {
                command,
                reply: approve_tx,
            });
            match approve_rx.recv() {
                Ok(true) => execute(&call, workspace),
                Ok(false) => "Command denied by the user.".to_string(),
                Err(_) => break, // UI went away / run aborted
            }
        } else {
            execute(&call, workspace)
        };
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let mut output = output;
        if output.len() > MAX_TOOL_OUTPUT {
            output.truncate(MAX_TOOL_OUTPUT);
            output.push_str("\n[output truncated]");
        }
        if output.is_empty() {
            output = "(no output)".to_string();
        }
        let _ = tx.send(AgentEvent::ToolResult {
            output: output.clone(),
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

fn system_prompt(workspace: &Path) -> String {
    let listing = list_files_impl(workspace, workspace, 1).unwrap_or_default();
    let agents_md = std::fs::read_to_string(workspace.join("AGENTS.md"))
        .map(|s| format!("\nProject instructions (AGENTS.md):\n{s}\n"))
        .unwrap_or_default();
    format!(
        "You are a coding agent working in the workspace directory {ws}. All paths are \
relative to it.\n\
\n\
You have these tools:\n\
- list_files: arguments {{\"path\": \"optional/subdir\"}} — list files recursively\n\
- read_file: arguments {{\"path\": \"file\"}} — read a file\n\
- write_file: arguments {{\"path\": \"file\", \"content\": \"...\"}} — create or overwrite a file\n\
- run_command: arguments {{\"command\": \"shell command\"}} — run a shell command in the workspace\n\
\n\
To use a tool, end your reply with exactly one call in this format:\n\
<tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>\n\
The result will come back in a <tool_response> block. Use one tool at a time.\n\
When the task is complete (verify your work when possible, e.g. by running code you \
wrote), reply with a short summary and NO tool call.\n\
{agents}\
\n\
Top-level files in the workspace:\n{listing}",
        ws = workspace.display(),
        agents = agents_md,
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
            _ => self.arg("path").unwrap_or("").to_string(),
        }
    }
}

pub fn parse_tool_call(response: &str) -> Option<ToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    // Ignore anything inside <think> blocks by only looking after the last one.
    let searchable = match response.rfind("</think>") {
        Some(i) => &response[i..],
        None => response,
    };
    let start = searchable.find(OPEN)?;
    let rest = &searchable[start + OPEN.len()..];
    let end = rest.find(CLOSE).unwrap_or(rest.len());
    let json: serde_json::Value = serde_json::from_str(rest[..end].trim()).ok()?;
    let name = json.get("name")?.as_str()?.to_string();
    let arguments = json.get("arguments").cloned().unwrap_or(serde_json::json!({}));
    Some(ToolCall { name, arguments })
}

fn execute(call: &ToolCall, workspace: &Path) -> String {
    let result = match call.name.as_str() {
        "list_files" => {
            resolve(workspace, call.arg("path").unwrap_or(""))
                .and_then(|dir| list_files_impl(&dir, workspace, 8))
        }
        "read_file" => resolve(workspace, call.arg("path").unwrap_or("")).and_then(|path| {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if size > MAX_FILE_READ {
                return Err(format!("file too large ({size} bytes, limit {MAX_FILE_READ})"));
            }
            std::fs::read_to_string(&path).map_err(|e| e.to_string())
        }),
        "write_file" => resolve(workspace, call.arg("path").unwrap_or("")).and_then(|path| {
            let content = call.arg("content").unwrap_or("");
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
    let full = canon.join(suffix);
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
            if matches!(name.as_str(), ".git" | "target" | "node_modules" | ".venv") {
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
    if command.is_empty() {
        return Err("empty command".into());
    }
    // `timeout` keeps a runaway command from hanging the agent thread forever.
    let output = Command::new("timeout")
        .arg(COMMAND_TIMEOUT_SECS.to_string())
        .arg("bash")
        .arg("-c")
        .arg(command)
        .current_dir(workspace)
        .output()
        .map_err(|e| e.to_string())?;
    let mut result = String::new();
    result.push_str(&String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&stderr);
    }
    if !output.status.success() {
        result.push_str(&format!("\n[exit code: {}]", output.status.code().unwrap_or(-1)));
    }
    Ok(result)
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
    fn tolerates_missing_close_tag() {
        let r = "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}";
        assert_eq!(parse_tool_call(r).unwrap().name, "read_file");
    }

    #[test]
    fn sandbox_rejects_escape() {
        let ws = std::env::temp_dir();
        assert!(resolve(&ws, "../../etc/passwd").is_err());
        assert!(resolve(&ws, "/etc/passwd").is_err());
        assert!(resolve(&ws, "sub/dir/file.txt").is_ok());
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
