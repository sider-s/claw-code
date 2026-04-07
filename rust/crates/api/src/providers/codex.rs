//! Codex Responses API provider.
//!
//! Translates between the internal `MessageRequest`/`MessageResponse` types and
//! the OpenAI Codex Responses API format used at
//! `https://chatgpt.com/backend-api/codex/responses`.
//!
//! Required headers: `Authorization`, `chatgpt-account-id`, `OpenAI-Beta`.

use std::collections::{BTreeMap, VecDeque};

use serde_json::{json, Value};

use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    ToolDefinition, Usage,
};

use runtime::{load_codex_oauth_credentials, CodexOAuthTokenSet};

use super::preflight_message_request;

pub const CODEX_API_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

/// Returns `true` if saved Codex OAuth credentials exist.
#[must_use]
pub fn has_codex_auth() -> bool {
    load_codex_oauth_credentials()
        .ok()
        .and_then(std::convert::identity)
        .is_some()
}

/// Load saved Codex OAuth credentials. Returns `None` if not present.
pub fn load_codex_auth() -> Option<CodexOAuthTokenSet> {
    load_codex_oauth_credentials().ok().flatten()
}

#[derive(Debug, Clone)]
pub struct CodexClient {
    http: reqwest::Client,
    access_token: String,
    account_id: String,
    base_url: String,
}

impl CodexClient {
    #[must_use]
    pub fn new(access_token: impl Into<String>, account_id: impl Into<String>) -> Self {
        Self {
            http: build_http_client_or_default(),
            access_token: access_token.into(),
            account_id: account_id.into(),
            base_url: CODEX_API_URL.to_string(),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        // For non-streaming, we stream internally and collect.
        let mut stream = self.stream_message(request).await?;
        let mut text_content = String::new();
        let mut tool_uses: Vec<(String, String, String)> = Vec::new(); // (id, name, json_args)
        let mut usage = Usage::default();

        while let Some(event) = stream.next_event().await? {
            match event {
                StreamEvent::ContentBlockDelta(delta_event) => match delta_event.delta {
                    ContentBlockDelta::TextDelta { text } => text_content.push_str(&text),
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        if let Some((_, _, args)) = tool_uses.last_mut() {
                            args.push_str(&partial_json);
                        }
                    }
                    _ => {}
                },
                StreamEvent::MessageDelta(delta_event) => {
                    usage = delta_event.usage;
                }
                StreamEvent::ContentBlockStart(start_event) => {
                    if let OutputContentBlock::ToolUse { id, name, .. } = start_event.content_block
                    {
                        tool_uses.push((id, name, String::new()));
                    }
                }
                _ => {}
            }
        }

        // Build finalized tool use blocks
        let tool_blocks: Vec<OutputContentBlock> = tool_uses
            .into_iter()
            .map(|(id, name, args)| {
                let input =
                    serde_json::from_str(&args).unwrap_or(Value::Object(serde_json::Map::new()));
                OutputContentBlock::ToolUse { id, name, input }
            })
            .collect();

        let mut content: Vec<OutputContentBlock> = Vec::new();
        if !text_content.is_empty() {
            content.push(OutputContentBlock::Text { text: text_content });
        }
        content.extend(tool_blocks);

        let stop_reason = if content
            .iter()
            .any(|b| matches!(b, OutputContentBlock::ToolUse { .. }))
        {
            Some("tool_use".to_string())
        } else {
            Some("end_turn".to_string())
        };

        Ok(MessageResponse {
            id: String::new(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model: request.model.clone(),
            stop_reason,
            stop_sequence: None,
            usage,
            request_id: None,
        })
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        preflight_message_request(request)?;
        let body = build_codex_request(request);
        let response = self
            .http
            .post(&self.base_url)
            .header("content-type", "application/json")
            .header("openai-beta", "responses=experimental")
            .header("chatgpt-account-id", &self.account_id)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Api {
                status,
                error_type: Some("codex_error".to_string()),
                message: Some(format!("Codex API error: {body}")),
                request_id: None,
                body,
                retryable: status.is_server_error(),
            });
        }

        Ok(MessageStream {
            response,
            buffer: String::new(),
            pending: VecDeque::new(),
            done: false,
            state: StreamState::new(request.model.clone()),
        })
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

fn build_codex_request(request: &MessageRequest) -> Value {
    let (instructions, input) = convert_messages(&request.messages, request.system.as_deref());

    let mut body = json!({
        "model": request.model,
        "instructions": if instructions.is_empty() { "You are a helpful assistant.".to_string() } else { instructions },
        "input": input,
        "store": false,
        "stream": true,
    });

    if let Some(tools) = &request.tools {
        body["tools"] = convert_tools(tools);
    }

    body
}

fn convert_messages(messages: &[InputMessage], system: Option<&str>) -> (String, Vec<Value>) {
    let instructions = system.unwrap_or("").to_string();
    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "user" => {
                for block in &msg.content {
                    match block {
                        InputContentBlock::Text { text } => {
                            input.push(json!({
                                "type": "message",
                                "role": "user",
                                "content": [{ "type": "input_text", "text": text }]
                            }));
                        }
                        InputContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            let output_text: String = content
                                .iter()
                                .map(|c| match c {
                                    crate::types::ToolResultContentBlock::Text { text } => {
                                        text.clone()
                                    }
                                    crate::types::ToolResultContentBlock::Json { value } => {
                                        value.to_string()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": output_text
                            }));
                        }
                        _ => {}
                    }
                }
            }
            "assistant" => {
                // Text content goes in a message; function_call is a
                // separate top-level input item (not nested in content).
                let mut text_parts: Vec<Value> = Vec::new();
                for block in &msg.content {
                    match block {
                        InputContentBlock::Text { text } => {
                            text_parts.push(json!({ "type": "output_text", "text": text }));
                        }
                        InputContentBlock::ToolUse {
                            id,
                            name,
                            input: tool_input,
                            ..
                        } => {
                            // Emit assistant text collected so far as a message
                            // before the function_call item.
                            if !text_parts.is_empty() {
                                input.push(json!({
                                    "type": "message",
                                    "role": "assistant",
                                    "content": std::mem::take(&mut text_parts)
                                }));
                            }
                            input.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": if tool_input.is_string() {
                                    tool_input.as_str().unwrap_or("{}").to_string()
                                } else {
                                    tool_input.to_string()
                                }
                            }));
                        }
                        _ => {}
                    }
                }
                if !text_parts.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text_parts
                    }));
                }
            }
            _ => {}
        }
    }

    (instructions, input)
}

fn convert_tools(tools: &[ToolDefinition]) -> Value {
    let converted: Vec<Value> = tools
        .iter()
        .map(|t| {
            let parameters = sanitize_tool_schema(&t.input_schema);
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description.as_deref().unwrap_or(""),
                "parameters": parameters
            })
        })
        .collect();
    Value::Array(converted)
}

/// Codex API requires `"type": "object"` schemas to have a `"properties"`
/// field. Patch any object schema that is missing one by inserting an empty
/// `"properties": {}`. This is applied recursively to nested schemas.
fn sanitize_tool_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut patched = map.clone();
            // If this is an object schema without `properties`, add an empty one.
            if patched.get("type").and_then(Value::as_str) == Some("object")
                && !patched.contains_key("properties")
            {
                patched.insert(
                    "properties".to_string(),
                    Value::Object(serde_json::Map::new()),
                );
            }
            // Recurse into nested schemas.
            for value in patched.values_mut() {
                *value = sanitize_tool_schema(value);
            }
            Value::Object(patched)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sanitize_tool_schema).collect()),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// SSE stream parsing
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MessageStream {
    response: reqwest::Response,
    buffer: String,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: StreamState,
}

#[derive(Debug)]
struct StreamState {
    model: String,
    content_index: u32,
    fn_call_map: BTreeMap<String, FnCallState>,
    emitted_start: bool,
    emitted_text_start: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FnCallState {
    call_id: String,
    name: String,
    arguments: String,
    content_index: u32,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            model,
            content_index: 0,
            fn_call_map: BTreeMap::new(),
            emitted_start: false,
            emitted_text_start: false,
        }
    }
}

impl MessageStream {
    pub fn request_id(&self) -> Option<&str> {
        None
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }

            let chunk = self.response.chunk().await.map_err(ApiError::from)?;

            let Some(chunk) = chunk else {
                // Stream ended — emit MessageStop
                if self.state.emitted_start {
                    self.pending
                        .push_back(StreamEvent::MessageStop(MessageStopEvent {}));
                }
                self.done = true;
                continue;
            };

            let text = String::from_utf8_lossy(&chunk);
            self.buffer.push_str(&text);

            // Process complete SSE events (separated by double newlines)
            while let Some(sep) = self.buffer.find("\n\n") {
                let event_block = self.buffer[..sep].to_string();
                self.buffer = self.buffer[sep + 2..].to_string();

                let mut event_type = String::new();
                let mut event_data = String::new();
                for line in event_block.lines() {
                    if let Some(rest) = line.strip_prefix("event:") {
                        event_type = rest.trim().to_string();
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        event_data = rest.trim().to_string();
                    }
                }

                if event_data.is_empty() || event_data == "[DONE]" {
                    continue;
                }

                let json: Value = match serde_json::from_str(&event_data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                self.process_sse_event(&event_type, &json);
            }
        }
    }

    fn process_sse_event(&mut self, event_type: &str, json: &Value) {
        // Emit MessageStart on first content event
        if !self.state.emitted_start {
            self.state.emitted_start = true;
            self.pending
                .push_back(StreamEvent::MessageStart(MessageStartEvent {
                    message: MessageResponse {
                        id: String::new(),
                        kind: "message".to_string(),
                        role: "assistant".to_string(),
                        content: vec![],
                        model: self.state.model.clone(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: Usage::default(),
                        request_id: None,
                    },
                }));
        }

        match event_type {
            "response.output_text.delta" => {
                let text = json
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !text.is_empty() {
                    // Emit ContentBlockStart for the text block on the first
                    // text delta so that the TUI renderer initialises properly.
                    if !self.state.emitted_text_start {
                        self.state.emitted_text_start = true;
                        self.pending.push_back(StreamEvent::ContentBlockStart(
                            ContentBlockStartEvent {
                                index: 0,
                                content_block: OutputContentBlock::Text {
                                    text: String::new(),
                                },
                            },
                        ));
                    }
                    self.pending.push_back(StreamEvent::ContentBlockDelta(
                        ContentBlockDeltaEvent {
                            index: 0,
                            delta: ContentBlockDelta::TextDelta { text },
                        },
                    ));
                }
            }
            "response.output_item.added" => {
                let item = json.get("item").cloned().unwrap_or_default();
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or(&item_id)
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();

                    self.state.content_index += 1;
                    let idx = self.state.content_index;

                    self.state.fn_call_map.insert(
                        item_id.clone(),
                        FnCallState {
                            call_id: call_id.clone(),
                            name: name.clone(),
                            arguments: String::new(),
                            content_index: idx,
                        },
                    );

                    self.pending.push_back(StreamEvent::ContentBlockStart(
                        ContentBlockStartEvent {
                            index: idx,
                            content_block: OutputContentBlock::ToolUse {
                                id: call_id,
                                name,
                                input: Value::Object(serde_json::Map::new()),
                            },
                        },
                    ));
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = json
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let delta = json
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();

                if let Some(fc) = self.state.fn_call_map.get_mut(&item_id) {
                    fc.arguments.push_str(&delta);
                    self.pending.push_back(StreamEvent::ContentBlockDelta(
                        ContentBlockDeltaEvent {
                            index: fc.content_index,
                            delta: ContentBlockDelta::InputJsonDelta {
                                partial_json: delta,
                            },
                        },
                    ));
                }
            }
            "response.output_item.done" => {
                let item = json.get("item").cloned().unwrap_or_default();
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if let Some(fc) = self.state.fn_call_map.get(&item_id) {
                        self.pending.push_back(StreamEvent::ContentBlockStop(
                            ContentBlockStopEvent {
                                index: fc.content_index,
                            },
                        ));
                    }
                }
            }
            "response.output_text.done" => {
                self.state.emitted_text_start = false;
                self.pending
                    .push_back(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                        index: 0,
                    }));
            }
            "response.completed" => {
                // Ensure text block is properly closed if it was started
                // but not closed via response.output_text.done.
                if self.state.emitted_text_start {
                    self.state.emitted_text_start = false;
                    self.pending
                        .push_back(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                            index: 0,
                        }));
                }
                // Extract usage if available
                let response_obj = json.get("response").cloned().unwrap_or_default();
                let usage_obj = response_obj.get("usage").cloned().unwrap_or_default();
                let input_tokens = usage_obj
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32;
                let output_tokens = usage_obj
                    .get("output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32;

                let stop_reason = if self.state.fn_call_map.is_empty() {
                    "end_turn"
                } else {
                    "tool_use"
                };

                self.pending
                    .push_back(StreamEvent::MessageDelta(MessageDeltaEvent {
                        delta: MessageDelta {
                            stop_reason: Some(stop_reason.to_string()),
                            stop_sequence: None,
                        },
                        usage: Usage {
                            input_tokens,
                            output_tokens,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                        },
                    }));
            }
            _ => {
                // Ignore other event types (response.created, response.in_progress, etc.)
            }
        }
    }
}
