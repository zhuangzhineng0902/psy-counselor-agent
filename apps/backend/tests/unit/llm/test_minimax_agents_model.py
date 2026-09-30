"""MiniMax structured-output compatibility tests."""

from __future__ import annotations

import json

from llm.minimax_agents_model import unfence_json


def test_unfence_json_repairs_known_triage_shape() -> None:
    raw = "```json\n" + json.dumps(
        {
            "route": "therapeutic",
            "clarification_question": None,
            "clarification_kind": None,
            "reasoning": "x" * 300,
        }
    ) + "\n```"

    payload = json.loads(unfence_json(raw))
    assert payload["clarification_question"] == ""
    assert payload["clarification_kind"] == "none"
    assert payload["confidence"] == "low"
    assert len(payload["reasoning"]) == 240


def test_unfence_json_leaves_prose_untouched() -> None:
    assert unfence_json("```json\nnot json\n```") == "```json\nnot json\n```"
