"""Regression checks for evidence review and pending experiment outcomes."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.skills.guided_exercises.catalog.definitions.thought_work import (
    BEHAVIORAL_EXPERIMENT_DEFINITION,
    EXERCISE_BEHAVIORAL_EXPERIMENT,
    THOUGHT_RECORD_DEFINITION,
)
from agent.skills.guided_exercises.lifecycle.step_classifier import classify_step_state
from agent.state import AgentState


class _AlwaysComplete:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, **kwargs: Any) -> Any:
        self.calls += 1
        return kwargs["response_schema"](step_state="complete", reasoning="test", confidence="high")


def test_thought_record_reviews_both_sides_of_evidence() -> None:
    ids = [step.id for step in THOUGHT_RECORD_DEFINITION.steps]
    assert ids == ["situation", "thought", "evidence_for", "evidence_against", "balanced_thought"]


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["实验我还没做", "还没有尝试", "计划尚未执行", "I haven't tried it yet", "我尚未开始"])
async def test_untried_experiment_cannot_complete(message: str) -> None:
    llm = _AlwaysComplete()
    step = BEHAVIORAL_EXPERIMENT_DEFINITION.steps[-1]
    result = await classify_step_state(
        state=cast(AgentState, {"message": message}),
        classifier_llm=llm,
        exercise_type=EXERCISE_BEHAVIORAL_EXPERIMENT,
        step_index=3,
        current_step=step,
    )
    assert result == "hold"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_negative_observed_outcome_can_be_recorded() -> None:
    llm = _AlwaysComplete()
    result = await classify_step_state(
        state=cast(AgentState, {"message": "我试了，在会上提问后真的被批评了。"}),
        classifier_llm=llm,
        exercise_type=EXERCISE_BEHAVIORAL_EXPERIMENT,
        step_index=3,
        current_step=BEHAVIORAL_EXPERIMENT_DEFINITION.steps[-1],
    )
    assert result == "complete"
    assert llm.calls == 1
