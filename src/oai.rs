//! OpenAI tool calling on top of a plain chat template.
//!
//! llama.cpp's built-in template engine knows roles, not tools, so the API
//! does what a model's own Jinja template would: the tool list goes into the
//! system prompt, earlier calls and results are written back in the format the
//! model was trained on, and the reply is parsed for `<tool_call>` blocks.
//! The format is the Qwen3-Coder / Qwen3.6 one —
//! `<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>` —
//! and the parser also accepts the older JSON body (`{"name": …, "arguments": …}`),
//! so Hermes-style models work too.

use serde_json::{Value, json};

use crate::llm::{ChatMessage, Role};

/// Convert OpenAI chat messages (plus an optional `tools` list) into the
/// role/content pairs the worker understands.
pub fn convert_messages(messages: &[Value], tools: &[Value]) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = text_of(m.get("content"));
        match role {
            "system" | "developer" => push(&mut out, Role::System, content),
            "assistant" => {
                let mut text = content;
                for call in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let f = call.get("function").unwrap_or(call);
                    let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
                    let args = match f.get("arguments") {
                        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    if !text.trim().is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(&format_call(name, &args));
                }
                out.push(ChatMessage {
                    role: Role::Assistant,
                    content: text,
                });
            }
            // Tool results go back as a user turn, one <tool_response> each;
            // consecutive results share the turn like in the model's template.
            "tool" => {
                let block = format!("<tool_response>\n{content}\n</tool_response>");
                match out.last_mut() {
                    Some(last)
                        if last.role == Role::User
                            && last.content.ends_with("</tool_response>") =>
                    {
                        last.content.push('\n');
                        last.content.push_str(&block);
                    }
                    _ => push(&mut out, Role::User, block),
                }
            }
            _ => push(&mut out, Role::User, content),
        }
    }
    if !tools.is_empty() {
        let prompt = tools_prompt(tools);
        match out.first_mut() {
            Some(first) if first.role == Role::System => {
                first.content = format!("{prompt}\n\n{}", first.content);
            }
            _ => out.insert(
                0,
                ChatMessage {
                    role: Role::System,
                    content: prompt,
                },
            ),
        }
    }
    out
}

fn push(out: &mut Vec<ChatMessage>, role: Role, content: String) {
    out.push(ChatMessage { role, content });
}

/// `content` may be a string, null, or a list of parts; only text is kept.
fn text_of(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The system prompt section Qwen3.6's template writes for a tool list.
fn tools_prompt(tools: &[Value]) -> String {
    let mut s = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        s.push('\n');
        s.push_str(&tool.get("function").unwrap_or(tool).to_string());
    }
    s.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with \
         NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
         value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second \
         parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
         <IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner \
         <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
         - Required parameters MUST be specified\n- You may provide optional reasoning for your \
         function call in natural language BEFORE the function call, but NOT after\n- If there is \
         no function call available, answer the question like normal with your current knowledge \
         and do not tell the user about function calls\n</IMPORTANT>",
    );
    s
}

/// One earlier call, written the way the model would have produced it.
fn format_call(name: &str, args: &Value) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    if let Some(map) = args.as_object() {
        for (key, value) in map {
            let value = match value {
                Value::String(v) => v.clone(),
                other => other.to_string(),
            };
            s.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// A finished reply split into its parts.
#[derive(Debug, Default, PartialEq)]
pub struct Reply {
    pub reasoning: String,
    pub content: String,
    /// (function name, arguments as a JSON object string)
    pub calls: Vec<(String, String)>,
}

pub fn parse_reply(text: &str) -> Reply {
    let (reasoning, rest) = split_think(text);
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = rest;
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let body = &rest[start + "<tool_call>".len()..];
        let (inner, after) = match body.find("</tool_call>") {
            Some(end) => (&body[..end], &body[end + "</tool_call>".len()..]),
            None => (body, ""),
        };
        if let Some(call) = parse_call(inner) {
            calls.push(call);
        }
        rest = after;
    }
    // Text after the last call is a stray suffix the template forbids; keep
    // it only when there were no calls at all.
    if calls.is_empty() {
        content.push_str(rest);
    }
    Reply {
        reasoning: reasoning.trim().to_string(),
        content: content.trim().to_string(),
        calls,
    }
}

/// Reasoning is everything up to `</think>`; a reply that never closes a
/// think block it opened is all reasoning.
fn split_think(text: &str) -> (&str, &str) {
    let trimmed = text.trim_start();
    match trimmed.find("</think>") {
        Some(end) => (
            trimmed[..end].trim_start_matches("<think>"),
            &trimmed[end + "</think>".len()..],
        ),
        None if trimmed.starts_with("<think>") => (&trimmed["<think>".len()..], ""),
        None => ("", text),
    }
}

fn parse_call(inner: &str) -> Option<(String, String)> {
    let inner = inner.trim();
    if let Some(start) = inner.find("<function=") {
        let after = &inner[start + "<function=".len()..];
        let name_end = after.find('>')?;
        let name = after[..name_end].trim().to_string();
        let mut body = &after[name_end + 1..];
        if let Some(end) = body.find("</function>") {
            body = &body[..end];
        }
        let mut args = serde_json::Map::new();
        while let Some(p) = body.find("<parameter=") {
            let after = &body[p + "<parameter=".len()..];
            let Some(key_end) = after.find('>') else {
                break;
            };
            let key = after[..key_end].trim().to_string();
            let value_and_rest = &after[key_end + 1..];
            let (value, next) = match value_and_rest.find("</parameter>") {
                Some(e) => (
                    &value_and_rest[..e],
                    &value_and_rest[e + "</parameter>".len()..],
                ),
                None => (value_and_rest, ""),
            };
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            args.insert(key, typed(value));
            body = next;
        }
        return (!name.is_empty()).then(|| (name, Value::Object(args).to_string()));
    }
    // JSON body: {"name": ..., "arguments": {...}}
    let fixed = crate::agent::escape_control_chars_in_strings(inner);
    let v: Value = serde_json::from_str(&fixed).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(Value::String(s)) => s.clone(),
        Some(a) => a.to_string(),
        None => "{}".into(),
    };
    Some((name, args))
}

/// Parameter values arrive as text; numbers, booleans, arrays and objects are
/// passed on typed so the client's schema validation accepts them.
fn typed(value: &str) -> Value {
    let t = value.trim();
    if (matches!(t.chars().next(), Some('{' | '['))
        || matches!(t, "true" | "false" | "null")
        || t.parse::<f64>().is_ok())
        && let Ok(v) = serde_json::from_str(t)
    {
        return v;
    }
    Value::String(value.to_string())
}

/// OpenAI `tool_calls` entries for a parsed reply.
pub fn tool_calls_json(calls: &[(String, String)]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            json!({
                "index": i,
                "id": format!("call_{i}_{}", unix_nanos()),
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        })
        .collect()
}

fn unix_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Incremental split of a streamed reply into reasoning and content deltas.
/// Text from the first `<tool_call>` on is held back for `parse_reply`.
#[derive(Default)]
pub struct StreamSplitter {
    text: String,
    /// Bytes of `text` already sent (as reasoning or content).
    sent: usize,
    in_think: Option<bool>,
    /// Content sent so far, to know what is left at the end.
    streamed: String,
}

pub enum Delta {
    Reasoning(String),
    Content(String),
}

impl StreamSplitter {
    pub fn push(&mut self, token: &str) -> Vec<Delta> {
        self.text.push_str(token);
        let mut out = Vec::new();
        if self.in_think.is_none() {
            let t = self.text.trim_start();
            if t.is_empty() || ("<think>".starts_with(t) && t.len() < "<think>".len()) {
                return out; // cannot tell yet
            }
            self.in_think = Some(t.starts_with("<think>"));
            if self.in_think == Some(true) {
                self.sent = self.text.find("<think>").unwrap() + "<think>".len();
            }
        }
        if self.in_think == Some(true) {
            if let Some(end) = self.text[self.sent..].find("</think>") {
                let end = self.sent + end;
                if end > self.sent {
                    out.push(Delta::Reasoning(self.text[self.sent..end].to_string()));
                }
                self.sent = end + "</think>".len();
                self.in_think = Some(false);
            } else {
                let safe = safe_end(&self.text, self.sent, "</think>");
                if safe > self.sent {
                    out.push(Delta::Reasoning(self.text[self.sent..safe].to_string()));
                    self.sent = safe;
                }
                return out;
            }
        }
        if self.text[self.sent..].contains("<tool_call>") || self.held_back() {
            return out;
        }
        let safe = safe_end(&self.text, self.sent, "<tool_call>");
        if safe > self.sent {
            let piece = &self.text[self.sent..safe];
            // Skip the blank line between </think> and the answer.
            let piece = if self.streamed.is_empty() {
                piece.trim_start()
            } else {
                piece
            };
            if !piece.is_empty() {
                self.streamed.push_str(piece);
                out.push(Delta::Content(piece.to_string()));
            }
            self.sent = safe;
        }
        out
    }

    fn held_back(&self) -> bool {
        self.text[..self.sent].contains("<tool_call>")
    }

    /// The whole reply parsed, plus the content not streamed yet.
    pub fn finish(&self) -> (Reply, String) {
        let reply = parse_reply(&self.text);
        let streamed = self.streamed.trim();
        let rest = reply
            .content
            .strip_prefix(streamed)
            .unwrap_or(if streamed.is_empty() {
                &reply.content
            } else {
                ""
            })
            .to_string();
        (reply, rest)
    }
}

/// End of the prefix that cannot be the start of `tag`.
fn safe_end(text: &str, from: usize, tag: &str) -> usize {
    let tail = &text[from..];
    for keep in (1..tag.len()).rev() {
        if tail.len() >= keep
            && tag.starts_with(&tail[tail.len() - keep..])
            && tail.is_char_boundary(tail.len() - keep)
        {
            return text.len() - keep;
        }
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_xml_calls_and_reasoning() {
        let text = "<think>\nlook first\n</think>\n\nLet me read it.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";
        let r = parse_reply(text);
        assert_eq!(r.reasoning, "look first");
        assert_eq!(r.content, "Let me read it.");
        assert_eq!(
            r.calls,
            vec![("read".into(), r#"{"limit":20,"path":"src/main.rs"}"#.into())]
        );
    }

    #[test]
    fn parses_json_calls_and_multiline_values() {
        let r = parse_reply(
            "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>\n<tool_call>\n<function=write>\n<parameter=content>\na\nb\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(r.calls[0], ("bash".into(), r#"{"command":"ls"}"#.into()));
        assert_eq!(r.calls[1], ("write".into(), r#"{"content":"a\nb"}"#.into()));
    }

    #[test]
    fn plain_answer_has_no_calls() {
        let r = parse_reply("Just text.");
        assert_eq!(
            r,
            Reply {
                reasoning: String::new(),
                content: "Just text.".into(),
                calls: vec![]
            }
        );
    }

    #[test]
    fn history_round_trips_calls_and_results() {
        let msgs = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "file a"}),
            json!({"role": "tool", "tool_call_id": "2", "content": "file b"}),
        ];
        let tools = vec![json!({"type": "function", "function": {"name": "read"}})];
        let out = convert_messages(&msgs, &tools);
        assert!(out[0].role == Role::System);
        assert!(out[0].content.contains("<tools>\n{\"name\":\"read\"}"));
        assert_eq!(out[1].content, "hi");
        assert_eq!(
            out[2].content,
            "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            out[3].content,
            "<tool_response>\nfile a\n</tool_response>\n<tool_response>\nfile b\n</tool_response>"
        );
    }

    #[test]
    fn stream_holds_back_tool_calls() {
        let mut s = StreamSplitter::default();
        let mut reasoning = String::new();
        let mut content = String::new();
        for tok in [
            "<th",
            "ink>\nplan",
            "</thi",
            "nk>\n\nOk",
            " now.\n\n<tool",
            "_call>\n<function=x>\n</function>\n</tool_call>",
        ] {
            for d in s.push(tok) {
                match d {
                    Delta::Reasoning(t) => reasoning.push_str(&t),
                    Delta::Content(t) => content.push_str(&t),
                }
            }
        }
        let (reply, rest) = s.finish();
        content.push_str(&rest);
        assert_eq!(reasoning.trim(), "plan");
        assert_eq!(content.trim(), "Ok now.");
        assert_eq!(reply.calls.len(), 1);
    }
}
