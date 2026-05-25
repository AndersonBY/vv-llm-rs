use crate::{Message, MessageContent};

pub fn normalize_text_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut normalized: Vec<Message> = Vec::new();

    for message in messages {
        if let Some(previous) = normalized.last_mut() {
            if previous.role == message.role
                && previous.name == message.name
                && previous.tool_call_id == message.tool_call_id
                && is_text_only(previous)
                && is_text_only(&message)
            {
                let merged = [previous.text_content(), message.text_content()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n");
                previous.content = vec![MessageContent::Text { text: merged }];
                continue;
            }
        }
        normalized.push(message);
    }

    normalized
}

fn is_text_only(message: &Message) -> bool {
    !message.content.is_empty()
        && message
            .content
            .iter()
            .all(|content| matches!(content, MessageContent::Text { .. }))
}
