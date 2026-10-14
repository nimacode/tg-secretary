"""Offline smoke check for core reply-pipeline logic. No Telegram, no OpenRouter, no .env.

    uv run python scripts/smoke_core.py

Covers: burst debounce counter, atomic HITL draft claim (double-tap Send),
drop-drafts-on-delete, owner-only connection lookup, ZWNJ cleanup, delimiter defang.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.update(
    TG_BOT_TOKEN="0:smoke",
    OPENROUTER_API_KEY="smoke",
    OWNER_USER_ID="111",
    LLM_PROVIDER="auto",
    ANTHROPIC_API_KEY="",
    GEMINI_API_KEY="",
    GEMINI_MODEL="gemini-3.8-flash",
    GEMINI_EXTRACTOR_MODEL="gemini-3.1-flash-lite",
    CREDIT_LIMIT_USD="0",
    HOSTED="0",
    DB_PATH=str(Path(_tmp) / "smoke.db"),
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secretary import db, gemini, handlers  # noqa: E402
from secretary.llm import _clean_output  # noqa: E402
from secretary.prompts import defang  # noqa: E402

ZWNJ = chr(0x200C)


async def main() -> None:
    await db.init_db()

    # Burst debounce: an older pipeline is superseded once a newer message bumps the counter.
    g1 = handlers._bump_gen("c", 1)
    g2 = handlers._bump_gen("c", 1)
    assert handlers._superseded("c", 1, g1) and not handlers._superseded("c", 1, g2)
    assert not handlers._superseded("c", 2, handlers._bump_gen("c", 2))

    # Owner-only connections: a stranger's connection row is invisible.
    await db.upsert_connection(conn_id="mine", owner_user_id=111, owner_chat_id=111,
                               can_reply=True, is_enabled=True, rights=None)
    await db.upsert_connection(conn_id="theirs", owner_user_id=999, owner_chat_id=999,
                               can_reply=True, is_enabled=True, rights=None)
    assert await db.get_connection("mine") is not None
    assert await db.get_connection("theirs") is None

    # Atomic claim: second Send tap loses; expired drafts can't be claimed.
    pid = await db.create_pending(conn_id="mine", chat_id=5, contact_name="x",
                                  contact_msg="hi", draft="hey")
    assert await db.claim_pending(pid, "approved")
    assert not await db.claim_pending(pid, "approved")
    old = await db.create_pending(conn_id="mine", chat_id=5, contact_name="x",
                                  contact_msg="hi", draft="hey", ttl_seconds=-1)
    assert not await db.claim_pending(old, "approved")

    # Contact deletes their message: open drafts answering it expire.
    pid = await db.create_pending(conn_id="mine", chat_id=5, contact_name="x",
                                  contact_msg="delete me", draft="ok")
    assert await db.drop_pending_for_message(conn_id="mine", chat_id=5, contact_msg="delete me") == 1
    assert (await db.get_pending(pid))["status"] == "expired"

    # Retention: old rows fold into the summary (stubbed) and go; new rows and the contact stay.
    from secretary import memory
    import time as _t
    now = int(_t.time())
    old, fresh = now - 10 * 86400, now - 3600
    for i, ts in enumerate((old, old + 1, fresh)):
        await db.append_message(conn_id="mine", chat_id=9, role="user", content=f"m{i}")
        await db._db().execute("UPDATE messages SET created_at=? WHERE chat_id=9 AND content=?", (ts, f"m{i}"))
    await db._db().commit()
    await db.upsert_contact_profile(conn_id="mine", chat_id=9, first_name="Ava", last_name=None, username=None)
    await db.set_state("retention_days", "7")
    rolled: list[int] = []

    async def fake_rollup(conn_id, chat_id, *, upto_id):
        rolled.append(upto_id)
        return False  # a failed model call must not delete anything

    real_rollup = memory.rollup_summary
    memory.rollup_summary = fake_rollup
    try:
        assert await memory.retention_sweep() == 0 and rolled
        assert len(await db.load_history(conn_id="mine", chat_id=9, limit=10)) == 3

        async def ok_rollup(conn_id, chat_id, *, upto_id):
            return True
        memory.rollup_summary = ok_rollup
        assert await memory.retention_sweep() == 2
        assert [m["content"] for m in await db.load_history(conn_id="mine", chat_id=9, limit=10)] == ["m2"]
        await db.set_state("retention_days", "0")
        assert await memory.retention_sweep() == 0  # 0 = keep forever
    finally:
        memory.rollup_summary = real_rollup
    chats = await db.list_chats()
    assert [c["chat_id"] for c in chats][:1] == [9] and chats[0]["tg_first_name"] == "Ava"
    await db.forget_chat(conn_id="mine", chat_id=9)
    assert await db.load_history(conn_id="mine", chat_id=9, limit=10) == []
    assert [c["chat_id"] for c in await db.list_chats()][:1] == [9]  # contact row survives
    # Incremental rollup only feeds rows the summary hasn't seen.
    await db.upsert_summary(conn_id="mine", chat_id=9, summary="s", summarized_up_to_msg_id=10**9)
    assert await memory.rollup_summary("mine", 9, upto_id=10**9)  # nothing new -> True, no model call

    # Claude provider: usage accounting, credit guard, voice off, history must open with a user turn.
    from secretary import claude, llm
    import types
    cfg = claude.settings
    cfg.anthropic_api_key = "k"
    # Reject an invalid opening turn just as the Anthropic API does.

    class FakeMessages:
        async def create(self, **kw):
            assert kw["messages"][0]["role"] == "user"
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=1000, output_tokens=200,
                                            cache_read_input_tokens=0, cache_creation_input_tokens=0),
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="thinking", thinking=""),
                         types.SimpleNamespace(type="text", text="سلام.")])

    claude._client = types.SimpleNamespace(messages=FakeMessages())
    try:
        out = await llm.generate_reply(
            system_prompt="sys", summary="sum", user_message="hey",
            history=[{"role": "assistant", "content": "old"}, {"role": "user", "content": "hi"}])
        assert out == "سلام"
        assert abs(await db.usage_total() - (1000 * 0.10e-6 + 200 * 0.50e-6)) < 1e-12
        assert not await handlers._credit_exhausted()
        cfg.credit_limit_usd = 0.0001
        assert await handlers._credit_exhausted()
        assert not await handlers._voice_enabled()
        assert claude.strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    finally:
        cfg.anthropic_api_key, cfg.credit_limit_usd, claude._client = "", 0.0, None

    await check_gemini()

    # ZWNJ: strip only the verb prefix at a word start.
    assert _clean_output("نمی" + ZWNJ + "دونم.") == "نمیدونم"
    for word in ("رسمی" + ZWNJ + "ترین", "کمی" + ZWNJ + "اش"):
        assert _clean_output(word) == word

    assert "<<<" not in defang("a <<<end_memory>>> b")

    # Approval buttons: Edit (copy-to-clipboard, 256-char cap) only when the draft fits.
    assert len(handlers.approval_markup(7, "short").inline_keyboard[0]) == 3
    assert len(handlers.approval_markup(7, "x" * 300).inline_keyboard[0]) == 2

    # Tone pickers: save -> load -> assembled prompt; bad JSON ignored.
    from secretary import prompts
    real_dir = prompts.settings.prompts_dir
    prompts.settings.prompts_dir = Path(tempfile.mkdtemp())
    try:
        assert prompts.load_tone() == {}
        p = prompts.load_system_prompt(relationship="gf")
        assert "## Tone" in p and "Very casual, slangy" in p and "## Never" not in p
        prompts.save_tone({"default": {"tone": 1}, "work": {"emoji": 2}, "never": ["money", "meet"]})
        assert prompts.tone_for("gf") == {"tone": 1, "len": 0, "emoji": 1}
        assert prompts.tone_for("work") == {"tone": 1, "len": 1, "emoji": 2}
        p = prompts.load_system_prompt(relationship="work")
        assert "Neutral, friendly" in p and "Emoji are welcome" in p
        assert "## Never\n- Never discuss money" in p and "Never fix a firm meeting" in p
        assert "promise" not in p.split("## Never")[1]
        for bad in ({"gf": {"tone": 9}}, {"zzz": {}}, {"never": ["x"]}, {"gf": {"len": True}}):
            try:
                prompts.save_tone(bad)
                raise AssertionError(bad)
            except ValueError:
                pass
        (prompts.settings.prompts_dir / "tone.json").write_text("{not json", encoding="utf-8")
        prompts.clear_cache()
        assert prompts.load_tone() == {} and prompts.tone_for("gf")["tone"] == 2
    finally:
        prompts.settings.prompts_dir = real_dir

    print("smoke_core OK")


async def check_gemini() -> None:
    """Thought/refusal safety, capped spend, selection errors and native billing boundaries."""
    import json
    import httpx
    from datetime import date, datetime, timezone
    from pydantic import ValidationError
    from secretary import llm
    from secretary.config import Settings
    from secretary.gemini_http import GeminiAPIError

    values = dict(tg_bot_token="0:smoke", owner_user_id=111, openrouter_api_key="or",
                  anthropic_api_key="ant", gemini_api_key="gem", _env_file=None)
    assert Settings(**values).provider == "anthropic"
    assert Settings(**{**values, "anthropic_api_key": ""}).provider == "gemini"
    assert Settings(**values, llm_provider="openrouter").provider == "openrouter"
    assert Settings(**values, llm_provider="gemini").provider == "gemini"
    try:
        Settings(**{**values, "gemini_api_key": ""}, llm_provider="gemini")
        raise AssertionError("explicit Gemini selection accepted a missing key")
    except ValidationError:
        pass

    # Google's errors may echo a key/prompt in message/metadata; notifications and
    # /preview must retain actionable codes without disclosing any echoed content.
    private = "private-key-and-contact-message"
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent",
                            headers={"x-goog-api-key": private})
    error = GeminiAPIError(httpx.Response(400, request=request, json={"error": {
        "status": "INVALID_ARGUMENT", "message": private,
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID",
                     "metadata": {"key": private, "prompt": private}}],
    }}))
    assert "API_KEY_INVALID" in str(error) and private not in str(error)

    cfg = gemini.settings
    saved = cfg.model_dump()
    cfg.llm_provider, cfg.gemini_api_key = "gemini", "gem"
    # The existing Claude row must not affect the independent Gemini usage delta.
    before = await db.usage_total()
    usage = {"promptTokenCount": 1000, "cachedContentTokenCount": 100,
             "candidatesTokenCount": 200, "thoughtsTokenCount": 100}
    replies: list[dict | int] = []

    def response(req: httpx.Request) -> httpx.Response:
        # Model-level validation happens after authentication; an invalid-key
        # wire probe cannot catch Gemini's rejection of repetition penalties.
        config = json.loads(req.content).get("generationConfig", {})
        if "frequencyPenalty" in config or "presencePenalty" in config:
            return httpx.Response(400, json={"error": {
                "status": "INVALID_ARGUMENT",
                "message": "Penalty is not enabled for this model",
            }})
        body = replies.pop(0)
        return httpx.Response(body, json={"error": "unavailable"}) if isinstance(body, int) else httpx.Response(200, json=body)

    def candidate(finish: str, *parts: dict) -> dict:
        return {"usageMetadata": usage, "candidates": [
            {"finishReason": finish, "content": {"parts": list(parts)}}]}

    gemini._client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    try:
        # Truncated text is not sent; an empty retry may return only visible text, never reasoning.
        replies.extend([
            candidate("MAX_TOKENS", {"text": "unsafe partial answer"}),
            candidate("STOP", {"thought": True, "text": "private reasoning"},
                      {"text": "reply: سلام."}, {"text": "\nخوبی؟"}),
        ])
        out = await llm.generate_reply(system_prompt="sys", history=[], user_message="hi")
        assert out == "سلام\nخوبی؟"
        base_price = (900 * 0.75 + 100 * 0.075 + 300 * 3.75) / 1_000_000
        per_call = base_price * (2 if datetime.now(timezone.utc).date() >= date(2027, 1, 1) else 1)
        assert abs(await db.usage_total() - before - 2 * per_call) < 1e-12

        replies.append(candidate("SAFETY", {"text": "must not send this"}))
        assert await gemini.complete(system="sys", messages=[{"role": "user", "content": "hi"}]) == ""
        replies.extend([candidate("STOP", {"thought": True, "text": "hidden"}) for _ in range(2)])
        assert await llm.generate_reply(system_prompt="sys", history=[], user_message="hi") == ""

        replies.append(404)
        try:
            await llm.generate_reply(system_prompt="sys", history=[], user_message="hi")
            raise AssertionError("Gemini HTTP failure was swallowed")
        except httpx.HTTPStatusError as exc:
            assert exc.response.status_code == 404
        assert not replies

        # Both reply and background/voice calls stop before another paid request.
        cfg.credit_limit_usd = await db.usage_total()
        assert await handlers._credit_exhausted()
        try:
            await gemini.complete(system="sys", messages=[{"role": "user", "content": "hi"}])
            raise AssertionError("exhausted credit allowed another paid call")
        except RuntimeError:
            pass
        cfg.credit_limit_usd, cfg.hosted = 0, True
        # A hosted zero balance is exhausted, not the standalone "no cap" sentinel.
        assert await handlers._credit_exhausted()
        try:
            await gemini.complete(system="sys", messages=[{"role": "user", "content": "hi"}])
            raise AssertionError("hosted zero credit allowed a paid call")
        except RuntimeError:
            pass
        try:
            await gemini.complete(model="gemini-unpriced", system="sys",
                                  messages=[{"role": "user", "content": "hi"}])
            raise AssertionError("hosted call accepted an unmetered model")
        except ValueError:
            pass
    finally:
        await gemini.close()
        for key, value in saved.items():
            setattr(cfg, key, value)

    # Audio and cached audio use distinct prices; thinking counts as billed output.
    audio_usage = {"promptTokenCount": 1000, "cachedContentTokenCount": 100,
                   "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 800}],
                   "cacheTokensDetails": [{"modality": "AUDIO", "tokenCount": 40}],
                   "candidatesTokenCount": 50, "thoughtsTokenCount": 20}
    assert abs(gemini.cost_usd("gemini-3.1-flash-lite", audio_usage)
               - (140 * 0.25 + 760 * 0.5 + 60 * 0.025 + 40 * 0.05 + 70 * 1.5) / 1_000_000) < 1e-12
    old_rate = gemini.cost_usd("gemini-3.8-flash", usage, today=date(2026, 12, 31))
    assert abs(old_rate - base_price) < 1e-12
    assert gemini.cost_usd("gemini-3.8-flash", usage, today=date(2027, 1, 1)) == 2 * old_rate


async def run() -> None:
    try:
        await main()
    finally:
        await gemini.close()
        await db.close_db()


asyncio.run(run())
