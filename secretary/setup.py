"""First-run setup wizard.

    uv run python -m secretary.setup

Validates the Telegram bot token (getMe) and the selected provider key, and
proves who owns the bot with a one-time /start code, then writes .env. Safe to
re-run: current values are offered as defaults and the old .env is backed up.

Must not import secretary.config at module level: Settings() fails at import
when .env does not exist yet. The dashboard reuses the helpers below.
"""
from __future__ import annotations

import getpass
import os
import re
import secrets
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values, set_key

from .gemini_http import API_URL, GeminiAPIError

ENV_PATH = Path(".env")
EXAMPLE_PATH = Path(".env.example")
TG_API = "https://api.telegram.org"
OR_API = "https://openrouter.ai/api/v1"
PROVIDERS = ("auto", "openrouter", "gemini", "anthropic")
PROVIDER_KEY = {"openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DEFAULT_MODEL = "google/gemini-3.8-flash"
TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L
OWNER_WAIT_SECONDS = 600
_PLACEHOLDERS = ("replace-with",)
_PLACEHOLDER_VALUES = {"123456789", "Alex"}  # OWNER_USER_ID / OWNER_FIRST_NAME in .env.example


class SetupError(Exception):
    """A check failed. The message is safe to show: it never contains a secret."""


class PollConflict(SetupError):
    """Telegram answered 409: another process is already polling this bot."""


# ----- pure helpers -----


def mask(secret: str) -> str:
    """Show only the first and last 4 characters of a secret."""
    return "****" if len(secret) < 12 else f"{secret[:4]}…{secret[-4:]}"


def real_value(value: str | None) -> str:
    """Treat .env.example placeholders as unset."""
    v = (value or "").strip()
    return "" if v.startswith(_PLACEHOLDERS) or v in _PLACEHOLDER_VALUES else v


def selected_provider(selection: str, values: Mapping[str, str]) -> str:
    """Resolve a proposed config without importing Settings or exposing secrets."""
    if selection not in PROVIDERS:
        raise SetupError("Unknown AI provider.")
    provider = selection
    if provider == "auto":
        provider = next((p for p in ("anthropic", "gemini", "openrouter")
                         if real_value(values.get(PROVIDER_KEY[p]))), "openrouter")
    if not real_value(values.get(PROVIDER_KEY[provider])):
        raise SetupError(f"{PROVIDER_KEY[provider]} is required for the selected provider.")
    return provider


def new_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))


def find_owner(updates: list[dict[str, Any]], code: str) -> dict[str, Any] | None:
    """Return the `from` user of the first private-chat `/start <code>` message."""
    for update in updates:
        msg = update.get("message") or {}
        if (msg.get("chat") or {}).get("type") != "private":
            continue
        if (msg.get("text") or "").strip() == f"/start {code}":
            return msg.get("from")
    return None


def ssh_hint(port: int, env: Mapping[str, str]) -> str | None:
    """`ssh -L` command for reaching the dashboard, or None when not on SSH."""
    parts = env.get("SSH_CONNECTION", "").split()  # client_ip client_port server_ip server_port
    if len(parts) != 4:
        return None
    ssh_port = "" if parts[3] == "22" else f" -p {parts[3]}"
    return f"ssh -L {port}:127.0.0.1:{port}{ssh_port} {env.get('USER') or 'you'}@{parts[2]}"


def merge_env(env_path: Path, example_path: Path, values: Mapping[str, str]) -> Path | None:
    """Write `values` into env_path and keep every other line as-is.

    Backs up an existing file to .env.<timestamp>.bak first and returns the
    backup path. A fresh install (no backup, returns None) starts from the
    example so every documented key is present.
    """
    backup = None
    if env_path.exists():
        backup = env_path.with_name(f"{env_path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        shutil.copy2(env_path, backup)
    elif example_path.exists():
        shutil.copy2(example_path, env_path)
    else:
        env_path.touch()
    for key, value in values.items():
        set_key(env_path, key, value)
    if os.name == "posix":
        env_path.chmod(0o600)
    return backup


# ----- network checks (sync; the dashboard runs them via asyncio.to_thread) -----


def _json(r: httpx.Response, service: str) -> dict[str, Any]:
    try:
        return r.json()
    except ValueError:
        raise SetupError(
            f"{service} answered HTTP {r.status_code} without JSON (a proxy blocking it?)"
        ) from None


def tg_get_me(token: str) -> dict[str, Any]:
    if not TOKEN_RE.match(token):
        raise SetupError("that doesn't look like a bot token (123456:ABC...)")
    r = httpx.get(f"{TG_API}/bot{token}/getMe", timeout=15)
    if r.status_code in (401, 404):
        raise SetupError("Telegram rejected this token")
    body = _json(r, "Telegram")
    if not body.get("ok"):
        raise SetupError(f"Telegram error: {body.get('description', r.status_code)}")
    return body["result"]


def or_key_info(key: str, timeout: float = 15) -> dict[str, Any]:
    r = httpx.get(f"{OR_API}/key", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    if r.status_code == 401:
        raise SetupError("OpenRouter rejected this key")
    body = _json(r, "OpenRouter")
    if r.status_code != 200 or "data" not in body:
        # A 403 "Access denied by security policy" is a blocked network, not a bad key.
        raise SetupError(
            f"OpenRouter answered HTTP {r.status_code}: {str(body.get('error', ''))[:200]} "
            "(403 'Access denied' usually means set HTTPS_PROXY)"
        )
    return body["data"]


def or_model_ids() -> set[str] | None:
    """All model ids on OpenRouter, or None when the list can't be fetched."""
    try:
        r = httpx.get(f"{OR_API}/models", timeout=20)
        r.raise_for_status()
        return {m["id"] for m in r.json()["data"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


def gemini_model_ids(key: str, timeout: float = 20) -> set[str]:
    """Validate a Gemini key and collect every native generateContent model ID."""
    ids: set[str] = set()
    page = ""
    seen: set[str] = set()
    while True:
        r = httpx.get(
            f"{API_URL}/models", headers={"x-goog-api-key": key},
            params={"pageToken": page} if page else {}, timeout=timeout,
        )
        if r.status_code != 200:
            raise SetupError(str(GeminiAPIError(r))) from None
        body = _json(r, "Gemini")
        for model in body.get("models", []):
            if "generateContent" in model.get("supportedGenerationMethods", []):
                name = model.get("name", "")
                if name.startswith("models/"):
                    ids.add(name.removeprefix("models/"))
        page = body.get("nextPageToken", "")
        if not page:
            return ids
        if page in seen:
            raise SetupError("Gemini returned a repeated catalogue page.")
        seen.add(page)


def anthropic_key_info(key: str) -> None:
    """Check an Anthropic key without generating a paid response."""
    r = httpx.get("https://api.anthropic.com/v1/models",
                  headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                  params={"limit": 1}, timeout=15)
    if r.status_code != 200:
        raise SetupError(f"Anthropic rejected this key (HTTP {r.status_code}).")


def tg_get_updates(token: str, offset: int, timeout: int) -> list[dict[str, Any]]:
    r = httpx.get(
        f"{TG_API}/bot{token}/getUpdates",
        params={"offset": offset, "timeout": timeout, "allowed_updates": '["message"]'},
        timeout=timeout + 15,
    )
    if r.status_code == 409:
        raise PollConflict("the bot is already running — stop it first: pm2 stop tg-secretary")
    body = _json(r, "Telegram")
    if not body.get("ok"):
        raise SetupError(f"Telegram error: {body.get('description', r.status_code)}")
    return body["result"]


def tg_send(token: str, chat_id: int, text: str) -> None:
    """Best-effort DM; a failure here must not stop setup."""
    try:
        httpx.post(f"{TG_API}/bot{token}/sendMessage",
                   json={"chat_id": chat_id, "text": text}, timeout=15)
    except httpx.HTTPError:
        pass


def wait_for_owner(token: str, code: str, seconds: int = OWNER_WAIT_SECONDS) -> dict[str, Any] | None:
    """Long-poll until someone sends `/start <code>`; None on timeout."""
    # ponytail: confirming the offset also drops updates the bot hasn't seen yet.
    # Harmless on first setup (no business chats connected yet); re-runs default to
    # keeping the current owner and skip this poll. Track seen ids if that changes.
    offset = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        updates = tg_get_updates(token, offset, 25)
        if not updates:
            continue
        offset = updates[-1]["update_id"] + 1
        owner = find_owner(updates, code)
        if owner is not None:
            tg_get_updates(token, offset, 0)  # confirm, so the bot doesn't replay /start
            return owner
    return None


# ----- interactive wizard -----

SECRET_KEYS = {"TG_BOT_TOKEN", *PROVIDER_KEY.values()}
BUSINESS_HELP = """  ! Business Mode is off for this bot. Turn it on:
    @BotFather → /mybots → <your bot> → Bot Settings → Business Mode → Turn on"""
NEXT_STEPS = """
Next:
  1. Start the bot:  pm2 start ecosystem.config.cjs   (without pm2: uv run python -m secretary)
  2. In Telegram: Settings → Business → Chatbots → add {bot},
     and allow "reply to messages" and "read messages".
  3. Send /help to {bot} for the owner commands."""


def ask(label: str, current: str = "", secret: bool = False) -> str:
    """Prompt until non-empty; Enter keeps `current` when there is one."""
    shown = mask(current) if secret else current
    hint = f" [{shown}]" if current else ""
    read = getpass.getpass if secret else input
    while True:
        value = read(f"  {label}{hint}: ").strip()
        if value or current:
            return value or current
        print("  ! required")


def ask_yes(label: str, default: bool = True) -> bool:
    value = input(f"  {label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not value else value.startswith("y")


def ask_owner_id() -> int:
    while True:
        value = ask("Your numeric Telegram user id (ask @userinfobot)")
        if value.isdigit():
            return int(value)
        print("  ! must be a number")


def _net(host: str, fn: Callable[..., Any], *args: Any) -> Any:
    """Run a network check. On a connection failure offer retry, or skip (returns None)."""
    while True:
        try:
            return fn(*args)
        except httpx.TransportError as e:
            print(f"  ! can't reach {host} ({type(e).__name__}) — check HTTPS_PROXY / network")
            if input("  [r]etry or [s]kip this check? [r]: ").strip().lower().startswith("s"):
                return None


def step_token(current: str) -> tuple[str, dict[str, Any] | None]:
    """Return (token, getMe result or None when the check was skipped)."""
    while True:
        token = ask("Bot token", current, secret=True)
        try:
            me = _net("api.telegram.org", tg_get_me, token)
        except SetupError as e:
            print(f"  ! {e}")
            current = ""
            continue
        if me is None:
            return token, None
        print(f"  ✓ @{me['username']}")
        while not me.get("can_connect_to_business"):
            print(BUSINESS_HELP)
            input("  press Enter to check again... ")
            again = _net("api.telegram.org", tg_get_me, token)
            if again is None:
                break  # check skipped
            me = again
        return token, me


def step_key(current: str, provider: str = "openrouter") -> str:
    check = {"openrouter": or_key_info, "gemini": gemini_model_ids, "anthropic": anthropic_key_info}[provider]
    host = {"openrouter": "openrouter.ai", "gemini": "generativelanguage.googleapis.com",
            "anthropic": "api.anthropic.com"}[provider]
    while True:
        key = ask(f"{provider} API key", current, secret=True)
        try:
            info = _net(host, check, key)
        except SetupError as e:
            print(f"  ! {e}")
            current = ""
            continue
        if provider == "openrouter" and info is not None:
            left = info.get("limit_remaining")
            print(f"  ✓ key works ({'no credit limit' if left is None else f'${left:.2f} left'})")
        return key


def step_owner(token: str, me: dict[str, Any] | None, current: str) -> tuple[int, str]:
    """Return (owner user id, Telegram first name or "")."""
    if current.isdigit() and ask_yes(f"Keep owner {current}?"):
        return int(current), ""
    if me is None:
        return ask_owner_id(), ""
    code = new_code()
    print("  Open this link on your own Telegram account and tap Start:")
    print(f"    https://t.me/{me['username']}?start={code}")
    print("  waiting up to 10 minutes... (Ctrl+C to type your user id instead)")
    try:
        while True:
            try:
                owner = _net("api.telegram.org", wait_for_owner, token, code)
            except PollConflict as e:
                print(f"  ! {e}")
                input("  press Enter to retry... ")
                continue
            except SetupError as e:
                print(f"  ! {e}")
                return ask_owner_id(), ""
            if owner is not None:
                break
            if not ask_yes("No /start arrived. Keep waiting?"):
                return ask_owner_id(), ""
    except KeyboardInterrupt:
        print()
        return ask_owner_id(), ""
    tg_send(token, owner["id"], "Verified — you are the owner of this secretary.")
    print(f"  ✓ {owner.get('first_name', '')} (id {owner['id']})")
    return int(owner["id"]), owner.get("first_name", "")


def step_model(current: str, provider: str = "openrouter", key: str = "", label: str = "Reply model") -> str:
    if provider == "anthropic":
        return current  # preserve the existing fixed Claude model
    ids = (_net("generativelanguage.googleapis.com", gemini_model_ids, key)
           if provider == "gemini" else or_model_ids())
    if ids is None:
        print("  ! couldn't fetch the model list — not checking the id")
    while True:
        model = ask(f"{label} (native Gemini id)" if provider == "gemini"
                    else f"{label} (any id from https://openrouter.ai/models)", current)
        native_id = provider != "gemini" or (model.startswith("gemini-") and "/" not in model)
        if native_id and (ids is None or model in ids):
            return model
        print("  ! unknown model id")
        current = ""


def first_login_link() -> tuple[str, int] | None:
    """Mint a one-time dashboard login in the bot DB, now that .env exists.
    Returns (url, port), or None when the dashboard is disabled."""
    import asyncio

    from . import db
    from .config import settings
    from .dashboard.auth import create_login_token

    if not settings.dashboard_enabled:
        return None

    async def mint() -> str:
        await db.init_db()
        try:
            return await create_login_token()
        finally:
            await db.close_db()

    port = settings.dashboard_port
    base = settings.dashboard_public_url.rstrip("/") or f"http://127.0.0.1:{port}"
    return f"{base}/login#t={asyncio.run(mint())}", port


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    cur = {k: real_value(v) for k, v in dotenv_values(ENV_PATH).items()} if ENV_PATH.exists() else {}
    print("tg-secretary setup" + (" — Enter keeps the value in [brackets]" if cur else ""))
    try:
        print("\n[1/4] Telegram bot token — @BotFather → /newbot")
        token, me = step_token(cur.get("TG_BOT_TOKEN", ""))
        print("\n[2/4] AI provider and API key")
        selection = cur.get("LLM_PROVIDER") or "auto"
        while True:
            selection = ask("Provider (auto / openrouter / gemini / anthropic)", selection)
            if selection in PROVIDERS:
                break
            print("  ! unknown provider")
            selection = ""
        provider = selection
        if provider == "auto":
            provider = next((p for p in ("anthropic", "gemini", "openrouter")
                             if cur.get(PROVIDER_KEY[p])), "openrouter")
        key = step_key(cur.get(PROVIDER_KEY[provider], ""), provider)
        print("\n[3/4] Owner — prove this Telegram account owns the bot")
        owner_id, tg_name = step_owner(token, me, cur.get("OWNER_USER_ID", ""))
        print("\n[4/4] Name and model")
        first = ask("Your first name (used in the persona)", cur.get("OWNER_FIRST_NAME") or tg_name)
        model_key = {"openrouter": "OPENROUTER_MODEL", "gemini": "GEMINI_MODEL",
                     "anthropic": "ANTHROPIC_MODEL"}[provider]
        default_model = {"openrouter": DEFAULT_MODEL, "gemini": "gemini-3.8-flash",
                         "anthropic": "claude-haiku-5-5"}[provider]
        model = step_model(cur.get(model_key) or default_model, provider, key)
        values = {
            "TG_BOT_TOKEN": token,
            "LLM_PROVIDER": selection,
            PROVIDER_KEY[provider]: key,
            "OWNER_USER_ID": str(owner_id),
            "OWNER_FIRST_NAME": first,
            model_key: model,
        }
        selected_provider(selection, {**cur, **values})
        if provider == "gemini":
            values["GEMINI_EXTRACTOR_MODEL"] = step_model(
                cur.get("GEMINI_EXTRACTOR_MODEL") or "gemini-3.1-flash-lite",
                provider, key, "Extractor and voice model",
            )
        # Saved so /dashboard can print the real tunnel command (the bot can't see SSH_CONNECTION).
        hint = ssh_hint(int(cur.get("DASHBOARD_PORT") or 8780), os.environ)
        if hint:
            values["DASHBOARD_SSH_HINT"] = hint
        print()
        for k, v in values.items():
            print(f"  {k:<20} {mask(v) if k in SECRET_KEYS else v}")
        if not ask_yes(f"Write these to {ENV_PATH}?"):
            print("aborted — nothing written")
            return 1
    except SetupError as e:
        print(f"\n! {e}\naborted — nothing written")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\naborted — nothing written")
        return 1
    backup = merge_env(ENV_PATH, EXAMPLE_PATH, values)
    print(f"\n✓ {ENV_PATH} written" + (f" (old one saved as {backup.name})" if backup else ""))
    print(NEXT_STEPS.format(bot=f"@{me['username']}" if me else "your bot"))
    try:
        link = first_login_link()
    except Exception as e:  # noqa: BLE001 — .env is written; the link is a convenience
        print(f"\n! couldn't create a dashboard link ({type(e).__name__}); send /dashboard to the bot later")
        link = None
    if link:
        url, port = link
        print("\nDashboard (once the bot is running):")
        hint = ssh_hint(port, os.environ)
        if hint:
            print(f"  1. On your computer, open a tunnel:  {hint}")
            print(f"  2. Open (works once, within 1 hour):  {url}")
        else:
            print(f"  Open (works once, within 1 hour):  {url}")
        print("  Lost it? Send /dashboard to the bot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
