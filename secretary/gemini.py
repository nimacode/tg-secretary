"""Official Gemini generateContent API and paid-tier usage accounting."""
import base64
import logging
import re
from asyncio import sleep
from datetime import date, datetime, timezone
from typing import Any

import httpx

from . import db
from .config import settings
from .gemini_http import API_URL, GeminiAPIError

log = logging.getLogger(__name__)

# Standard paid-tier USD per 1M tokens: text input, audio input, output incl. thinking.
# https://ai.google.dev/gemini-api/docs/pricing
# Flash introductory rates double on 2027-01-01; implicit cache reads cost 10% of input.
PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.8-flash": (0.75, 0.75, 3.75),
    "gemini-3.6-flash": (0.75, 0.75, 3.75),
    "gemini-3.1-flash-lite": (0.25, 0.50, 1.50),
    "gemini-3.5-flash-lite": (0.30, 0.30, 2.50),
}
_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def cost_usd(model: str, usage: dict[str, Any], *, today: date | None = None) -> float:
    """Published paid-tier estimate, not a Google balance lookup. Unknown prices fail closed."""
    text_rate, audio_rate, output_rate = PRICES[model]
    today = today or datetime.now(timezone.utc).date()
    multiplier = 2 if model in {"gemini-3.8-flash", "gemini-3.6-flash"} and today >= date(2027, 1, 1) else 1
    prompt = usage.get("promptTokenCount", 0)
    cached = usage.get("cachedContentTokenCount", 0)
    audio = sum(d["tokenCount"] for d in usage.get("promptTokensDetails", []) if d.get("modality") == "AUDIO")
    cached_audio = sum(d["tokenCount"] for d in usage.get("cacheTokensDetails", []) if d.get("modality") == "AUDIO")
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return multiplier * (
        (prompt - cached - audio + cached_audio) * text_rate
        + (audio - cached_audio) * audio_rate
        + (cached - cached_audio) * text_rate * 0.1
        + cached_audio * audio_rate * 0.1
        + output * output_rate
    ) / 1_000_000


def _contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = {"user": "user", "assistant": "model"}[message["role"]]
        text = message["content"]
        if not text:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    if not contents or contents[-1]["role"] != "user":
        raise ValueError("Gemini requires a final non-empty user turn")
    return contents


async def complete(*, system: str, messages: list[dict[str, Any]], model: str | None = None,
                   max_tokens: int = 4096, json_output: bool = False, audio: bytes | None = None) -> str:
    """One native turn. Keep secrets in headers and never return thoughts or partial/refused text."""
    model = (model or settings.gemini_model).removeprefix("models/")
    if not re.fullmatch(r"gemini-[A-Za-z0-9._-]+", model):
        raise ValueError("Use a native Gemini model id, without the OpenRouter 'google/' prefix")
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required")
    if model not in PRICES and (settings.hosted or settings.credit_limit_usd > 0):
        raise ValueError(f"No Gemini price configured for metered model {model}")
    contents = _contents(messages)
    if audio is not None:
        contents[-1]["parts"].append({"inlineData": {
            "mimeType": "audio/ogg", "data": base64.b64encode(audio).decode("ascii"),
        }})
    config: dict[str, Any] = {"maxOutputTokens": max_tokens}
    if model.startswith("gemini-3"):
        config["thinkingConfig"] = {
            "thinkingLevel": "minimal" if "flash-lite" in model else "low", "includeThoughts": False,
        }
    elif model.startswith("gemini-2.5"):
        config["thinkingConfig"] = {"thinkingBudget": 128 if "pro" in model else 0, "includeThoughts": False}
    if json_output:
        # Unlike responseMimeType, responseFormat.text.mimeType is a protobuf enum.
        config["responseFormat"] = {"text": {"mimeType": "APPLICATION_JSON"}}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": contents, "generationConfig": config}
    for attempt in range(3):
        if (settings.hosted or settings.credit_limit_usd > 0) and await db.usage_total() >= settings.credit_limit_usd:
            raise RuntimeError("Gemini credit limit reached")
        resp = await client().post(
            f"{API_URL}/models/{model}:generateContent",
            headers={"x-goog-api-key": settings.gemini_api_key}, json=body,
        )
        if resp.status_code != 503 or attempt == 2:
            break
        delay = 2 ** (attempt + 1)
        log.warning("Gemini HTTP 503 (model=%s); retry %d/2 in %ds", model, attempt + 1, delay)
        await sleep(delay)
    if not resp.is_success:
        raise GeminiAPIError(resp)
    data = resp.json()
    usage = data.get("usageMetadata")
    if model in PRICES and usage is not None:
        try:
            await db.add_usage(
                model=model,
                input_tokens=usage.get("promptTokenCount", 0) - usage.get("cachedContentTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
                cache_read_tokens=usage.get("cachedContentTokenCount", 0), cache_write_tokens=0,
                cost_usd=cost_usd(model, usage),
            )
        except Exception:  # noqa: BLE001 — accounting must never eat a reply
            log.exception("Gemini usage row not recorded")
    else:
        log.warning("Gemini cost accounting unavailable (model=%s, usage_present=%s)", model, usage is not None)
    candidates = data.get("candidates") or []
    if data.get("promptFeedback", {}).get("blockReason") or not candidates:
        log.warning("Gemini returned no reply (blocked=%s)", bool(data.get("promptFeedback", {}).get("blockReason")))
        return ""
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        log.warning("Gemini reply discarded (finish=%s)", candidate.get("finishReason"))
        return ""
    return "".join(p["text"] for p in candidate.get("content", {}).get("parts", [])
                   if p.get("text") and not p.get("thought")).strip()
