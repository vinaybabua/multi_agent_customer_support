"""Thin wrapper around LiteLLM — the *only* place in the codebase that calls
`litellm.completion(...)`.

Concept: every model call in this POC (both the Info-Gathering node and the
Context Compaction node) goes through `complete_json`, and the model name
comes from config.py, not a string literal buried in a node. If you ever
swap providers (OpenAI -> Anthropic -> a local model via Ollama), this is
the one function whose behavior you're relying on, and config.py is the one
line you'd change.

LiteLLM retries transient completion failures twice. There is no fallback
model or streaming — this is a POC, and those additions would obscure the
teaching point rather than help it.
"""
import json
import re

import litellm

from config import LLM_MODEL

LLM_NUM_RETRIES = 2


def _load_json_payload(raw: str) -> dict:
    """Parse a model response that may be wrapped in markdown code fences."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM returned empty content")

    # Some providers emit fenced JSON like ```json ... ``` even when instructed
    # to return raw JSON. Strip the outer fence before parsing.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {text[:200]!r}") from exc


def complete_json(system_prompt: str, user_content: str, temperature: float = 0.0) -> dict:
    """Call the LLM and parse its reply as a JSON object.

    Both the info-gathering and compaction nodes ask the model to respond
    with *only* a JSON object (no prose, no markdown fences) — this helper
    is what turns that raw string back into a Python dict.
    """
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        num_retries=LLM_NUM_RETRIES,
    )
    raw = response.choices[0].message.content
    return _load_json_payload(raw)
