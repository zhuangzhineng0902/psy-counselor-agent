"""OpenAI provider adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, cast
from urllib.parse import urlparse

from openai import AsyncOpenAI

from llm.base import BaseLLMClient, StructuredResponseT

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAILLMClient(BaseLLMClient):
    """OpenAI implementation of `BaseLLMClient`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        """Initialize an OpenAI-backed model client.

        Args:
            api_key: Optional explicit API key override.
            model: Model identifier to use for requests.

        Returns:
            None.

        Raises:
            ValueError: If no OpenAI API key can be resolved.
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")

        self.model = model
        self.client = AsyncOpenAI(api_key=resolved_key)

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        """Generate a plain-text response with OpenAI.

        When ``use_search=True``, the OpenAI hosted web-search tool is
        attached to the Responses API call so the model can ground its
        reply against live web results.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            use_search: Whether to attach OpenAI's hosted web-search
                tool. When False, no tool is attached.

        Returns:
            The generated text response.

        Raises:
            ValueError: If OpenAI returns an empty text payload.
        """
        input_items: list[dict[str, str]] = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if use_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]

        response = await self.client.responses.create(**kwargs)
        text = response.output_text
        if not text:
            raise ValueError("OpenAI text generation returned an empty response.")
        return text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a plain-text response from OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.

        Yields:
            String chunks of the generated text as they arrive.
        """

        input_items: list[Any] = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        async with self.client.responses.stream(
            model=self.model,
            input=input_items,
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        """Generate a structured response with OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.
            use_search: Whether to attach OpenAI's hosted web-search tool.

        Returns:
            A parsed object matching `response_schema`.

        Raises:
            ValueError: If OpenAI does not return parsed structured output.
        """
        input_items: list[Any] = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "text_format": response_schema,
        }
        if use_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]

        # MiniMax exposes a Responses-compatible endpoint but sometimes wraps
        # JSON-schema output in a Markdown fence. Parse that one compatibility
        # shape locally, then keep the same Pydantic validation boundary.
        hostname = urlparse(os.getenv("OPENAI_BASE_URL", "")).hostname or ""
        if hostname in {"api.minimax.cn", "api.minimax.io"}:
            format_spec = {
                "type": "json_schema",
                "name": response_schema.__name__,
                "schema": response_schema.model_json_schema(),
                "strict": True,
            }
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    *input_items,
                    {
                        "role": "system",
                        "content": (
                            "Return only one JSON object with exactly these fields: "
                            + ", ".join(response_schema.model_fields)
                            + ". Follow the JSON schema; do not add explanations "
                            "or Markdown fences."
                        ),
                    },
                ],
                text={"format": format_spec},
                **({"tools": kwargs["tools"]} if use_search else {}),
            )
            raw_text = (response.output_text or "").strip()
            if raw_text.startswith("```json") and raw_text.endswith("```"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```") and raw_text.endswith("```"):
                raw_text = raw_text[3:-3].strip()
            return response_schema.model_validate_json(raw_text)

        response = await self.client.responses.parse(**kwargs)

        parsed = response.output_parsed
        if not isinstance(parsed, response_schema):
            raise ValueError(
                "OpenAI structured generation did not return parsed output."
            )

        return cast(StructuredResponseT, parsed)
