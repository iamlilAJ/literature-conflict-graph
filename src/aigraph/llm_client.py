"""OpenAI-compatible LLM client helpers.

The project prefers the Responses API when available, while keeping chat
completions as a compatibility fallback for older proxies and tests.
"""

from __future__ import annotations

import os
from typing import Any


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_TOKENS = 4000


def configured_model(model: str | None = None) -> str:
    return model or os.environ.get("AIGRAPH_MODEL") or DEFAULT_MODEL


def configured_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def configured_base_url(base_url: str | None = None) -> str | None:
    return base_url or os.environ.get("AIGRAPH_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def build_openai_client(api_key: str | None = None, base_url: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - exercised in real runs
        raise RuntimeError("openai package is required for LLM calls. Install with `pip install -e '.[real]'`.") from e
    kwargs: dict[str, Any] = {}
    key = configured_api_key(api_key)
    url = configured_base_url(base_url)
    if key:
        kwargs["api_key"] = key
    if url:
        kwargs["base_url"] = url
    kwargs["timeout"] = float(os.environ.get("AIGRAPH_LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    return OpenAI(**kwargs)


def call_llm_text(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    endpoint = os.environ.get("AIGRAPH_LLM_ENDPOINT", "responses").strip().lower()
    if endpoint == "responses":
        try:
            return _call_responses(
                client,
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AttributeError:
            pass
    return _call_chat(
        client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _call_responses(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_output_tokens": max_tokens,
    }
    # Some GPT-5 compatible providers reject temperature on reasoning models.
    if os.environ.get("AIGRAPH_INCLUDE_TEMPERATURE", "0") == "1":
        kwargs["temperature"] = temperature
    effort = os.environ.get("AIGRAPH_REASONING_EFFORT")
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    try:
        resp = client.responses.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning", None)
        kwargs.pop("temperature", None)
        resp = client.responses.create(**kwargs)
    return _extract_responses_text(resp)


def _call_chat(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _extract_responses_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return str(output_text)
    if isinstance(resp, dict):
        return _extract_responses_text_from_dict(resp)
    output = getattr(resp, "output", None)
    if output is None:
        return ""
    return _extract_responses_text_from_dict({"output": output})


def _extract_responses_text_from_dict(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            item = _model_dump_like(item)
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                part = _model_dump_like(part)
            text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _model_dump_like(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}
