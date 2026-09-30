"""Small compatibility boundary for MiniMax's fenced structured replies."""

from __future__ import annotations

import json

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


def unfence_json(value: str) -> str:
    """Remove a full Markdown JSON fence only when it encloses valid JSON."""

    stripped = value.strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        candidate = stripped[8:-3].strip()
    elif stripped.startswith("```\n") and stripped.endswith("```"):
        candidate = stripped[4:-3].strip()
    else:
        candidate = value
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict) and payload.get("route") in {
        "therapeutic",
        "memory_control",
        "grounded_lookup",
        "guided_exercise",
    }:
        for key in ("clarification_question", "intent_summary", "reasoning"):
            if payload.get(key) is None:
                payload[key] = ""
        if payload.get("clarification_kind") is None:
            payload["clarification_kind"] = "none"
        if payload.get("confidence") is None:
            payload["confidence"] = "low"
        payload["reasoning"] = str(payload.get("reasoning", ""))[:240]
        return json.dumps(payload, ensure_ascii=False)
    return candidate


class MiniMaxChatCompletionsModel(OpenAIChatCompletionsModel):
    """Normalize MiniMax JSON envelopes before the Agents SDK validates them."""

    async def get_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = await super().get_response(*args, **kwargs)
        output_schema = kwargs.get("output_schema")
        if output_schema is None and len(args) >= 5:
            output_schema = args[4]
        if output_schema is None:
            return response
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    part.text = unfence_json(part.text)
        return response
