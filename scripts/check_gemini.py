"""Exercise the configured official Gemini API with synthetic text/JSON, optionally OGG.

    uv run python scripts/check_gemini.py

Uses .env and the normal credit/usage accounting. Makes real, potentially billable
requests; never sends stored chats. Optional voice input is sent to Google.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secretary.gemini_http import GeminiAPIError  # noqa: E402
from secretary.setup import SetupError  # noqa: E402


async def check(args: argparse.Namespace) -> None:
    from secretary import db, gemini, llm, memory, setup, voice
    from secretary.config import settings

    if settings.provider != "gemini":
        raise ValueError("Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env first; no other provider will be used.")
    audio = None
    if args.voice:
        if args.voice.stat().st_size > voice.MAX_BYTES:
            raise ValueError("Voice sample exceeds the 10 MB limit.")
        audio = args.voice.read_bytes()
        if not audio:
            raise ValueError("Voice sample is empty.")
    try:
        ids = await asyncio.to_thread(setup.gemini_model_ids, settings.gemini_api_key)
        for model in (settings.gemini_model, settings.gemini_extractor_model):
            if model.removeprefix("models/") not in ids:
                raise ValueError(f"Configured model is not available to this key: {model}")
        print("PASS: key and both native models are listed. Checking generation separately...")
        await db.init_db()
        reply = await llm.generate_reply(
            system_prompt="For this connection check, return exactly GEMINI_OK with no punctuation or other text.",
            history=[], user_message="Check the connection.",
        )
        if reply != "GEMINI_OK":
            raise RuntimeError("Reply generation did not return the requested check value (empty, blocked or unexpected output).")
        print(f"PASS: reply generation ({settings.gemini_model}).")
        result = json.loads(await memory._chat(
            'Return only a JSON object with exactly these values: {"connection": "ready", "answer": 42}.',
            "Run the JSON connection check.", max_tokens=2000, json_output=True,
        ))
        if result != {"connection": "ready", "answer": 42}:
            raise RuntimeError("The extractor did not return the requested structured JSON values.")
        print(f"PASS: structured JSON used by memory and style ({settings.gemini_extractor_model}).")
        if audio is not None:
            transcript = await voice.transcribe(audio)
            if not transcript or args.expect_transcript.casefold() not in transcript.casefold():
                raise RuntimeError("Voice transcription failed or did not contain the expected phrase; transcript not printed.")
            print("PASS: OGG voice transcription contains the expected phrase.")
        print("Gemini connection check passed. No Telegram messages sent and no stored chats uploaded.")
    finally:
        await gemini.close()
        await db.close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, help="Optional Telegram OGG/Opus sample to send to Google.")
    parser.add_argument("--expect-transcript", help="Phrase that must appear in the voice transcript (not printed).")
    args = parser.parse_args()
    if (args.voice is not None and not (args.expect_transcript or "").strip()
            or args.voice is None and args.expect_transcript is not None):
        parser.error("--voice and a non-empty --expect-transcript must be supplied together")
    try:
        asyncio.run(check(args))
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors(include_input=False))
        print(f"FAIL: invalid configuration fields: {fields}", file=sys.stderr)
        return 1
    except (GeminiAPIError, SetupError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (httpx.TransportError, OSError) as exc:
        print(f"FAIL: {type(exc).__name__}; check network/proxy access and the voice file path.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
