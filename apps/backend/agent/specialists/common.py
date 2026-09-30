"""Shared builders for OpenAI text agents."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence
from urllib.parse import urlparse

from agents import Agent, ModelSettings
from openai import AsyncOpenAI

from llm.openai_client import DEFAULT_OPENAI_MODEL
from llm.minimax_agents_model import MiniMaxChatCompletionsModel
from agent.runtime.context import OpenAITextRunContext


@dataclass(frozen=True)
class AgentDefinition:
    """Static metadata for one OpenAI text agent."""

    name: str
    handoff_description: str
    instructions: str


def build_agent(
    definition: AgentDefinition,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
    output_type: type[Any] | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build an OpenAI Agents SDK agent from OpenCouch metadata."""

    minimax = (urlparse(os.getenv("OPENAI_BASE_URL", "")).hostname or "") in {
        "api.minimax.cn",
        "api.minimax.io",
    }
    return Agent[OpenAITextRunContext](
        name=definition.name,
        handoff_description=definition.handoff_description,
        instructions=definition.instructions,
        model=(
            MiniMaxChatCompletionsModel(model=model, openai_client=AsyncOpenAI())
            if minimax
            else model
        ),
        model_settings=(
            ModelSettings(extra_body={"thinking": {"type": "disabled"}})
            if minimax
            else ModelSettings()
        ),
        tools=list(tools or ()),
        output_type=output_type,
    )


def definition_with_instructions(
    definition: AgentDefinition,
    instructions: str | None,
) -> AgentDefinition:
    """Override instructions while preserving identity and handoff metadata."""

    if instructions is None:
        return definition
    return AgentDefinition(
        name=definition.name,
        handoff_description=definition.handoff_description,
        instructions=instructions,
    )
