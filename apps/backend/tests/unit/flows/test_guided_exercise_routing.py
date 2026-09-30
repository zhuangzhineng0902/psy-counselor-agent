"""Tests for guided-exercise routing helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.flows.guided_exercise import routing as guided_exercise_routing
from agent.skills.guided_exercises.catalog.registry import (
    available_exercise_definitions,
)
from agent.skills.guided_exercises.catalog.types import ExerciseDefinition, ExerciseStep
from agent.state import AgentState


def _definition(
    exercise_id: str,
    *,
    required_capability: str | None = None,
) -> ExerciseDefinition:
    return ExerciseDefinition(
        id=exercise_id,
        display_name=exercise_id.replace("_", " ").title(),
        selection_use_case=f"{exercise_id} support",
        steps=(
            ExerciseStep(
                instruction="Try one small step.",
                id="step",
                completion_mode="confirmation",
            ),
        ),
        selection_aliases=(exercise_id.replace("_", " "),),
        required_capability=required_capability,
    )


def test_available_aliases_respect_installed_capability_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basic = _definition("basic")
    gated = _definition(
        "gated",
        required_capability="advanced_exercises",
    )
    definitions = (basic, gated)

    def fake_available_exercise_definitions(
        **kwargs: object,
    ) -> tuple[ExerciseDefinition, ...]:
        return available_exercise_definitions(
            definitions=definitions,
            installed_skills=kwargs.get("installed_skills", ()),
            channel=kwargs.get("channel", "text"),
            therapeutic_approach=kwargs.get("therapeutic_approach"),
        )

    monkeypatch.setattr(
        guided_exercise_routing,
        "available_exercise_definitions",
        fake_available_exercise_definitions,
    )

    without_capability = guided_exercise_routing.available_exercise_aliases_for_state(
        {"installed_skills": [], "channel": "text"}
    )
    assert "basic" in without_capability
    assert "gated" not in without_capability

    with_capability = guided_exercise_routing.available_exercise_aliases_for_state(
        {"installed_skills": ["advanced_exercises"], "channel": "text"}
    )
    assert "basic" in with_capability
    assert "gated" in with_capability


def test_chinese_exercise_request_requires_clear_intent() -> None:
    assert guided_exercise_routing.message_explicitly_requests_guided_exercise(
        {}, "请带我做一个想法记录"
    )
    assert not guided_exercise_routing.message_explicitly_requests_guided_exercise(
        {}, "我只是想聊聊考试压力"
    )


@pytest.mark.asyncio
async def test_chinese_refusal_clears_active_exercise() -> None:
    state = cast(AgentState, {
        "message": "别分析了，听我说就好",
        "exercise_state": {
            "exercise_type": "thought_work_simple_record",
            "exercise_step": 2,
        },
    })

    async def no_memory(current: AgentState, context: Any) -> AgentState:
        return current

    result, selected = await guided_exercise_routing.prepare_guided_exercise_route(
        state,
        cast(Any, object()),
        load_turn_memory=no_memory,
    )
    assert not selected
    assert result["route"] == "therapeutic"
    assert result["exercise_state"]["exercise_type"] is None
