"""Offline checks for the setup wizard helpers. No network, no .env needed.

    uv run python scripts/smoke_setup.py
"""
import sys
import tempfile
from pathlib import Path
import contextlib
import io
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import dotenv_values  # noqa: E402

from secretary.setup import (  # noqa: E402
    TOKEN_RE, find_owner, mask, merge_env, new_code, real_value, ssh_hint,
)
from secretary import setup  # noqa: E402

# Masking never reveals the middle of a secret.
assert mask("123456:ABCDEFGHIJKLMNOP") == "1234…MNOP"
assert mask("short") == "****"

# .env.example placeholders count as unset on a re-run.
assert real_value("replace-with-your-telegram-bot-token") == ""
assert real_value("123456789") == ""
assert real_value("Alex") == ""  # OWNER_FIRST_NAME placeholder in .env.example
assert real_value(None) == ""
assert real_value(" 42 ") == "42"

# Codes: 6 chars from an unambiguous, deep-link-safe alphabet.
code = new_code()
assert len(code) == 6 and code.isalnum() and not set(code) & set("0O1IL")

# Owner match: only a private-chat "/start <code>" counts.
updates = [
    {"update_id": 1, "message": {"chat": {"type": "group"}, "from": {"id": 9}, "text": f"/start {code}"}},
    {"update_id": 2, "message": {"chat": {"type": "private"}, "from": {"id": 8}, "text": "/start WRONG1"}},
    {"update_id": 3, "message": {"chat": {"type": "private"}, "from": {"id": 7, "first_name": "Sam"},
                                 "text": f"/start {code}"}},
    {"update_id": 4, "business_message": {"text": "hi"}},
]
assert find_owner(updates, code) == {"id": 7, "first_name": "Sam"}
assert find_owner(updates[:2], code) is None

# ssh hint only over SSH, using the server-side address.
assert ssh_hint(8780, {}) is None
assert ssh_hint(8780, {"SSH_CONNECTION": "1.2.3.4 5555 10.0.0.9 22", "USER": "deploy"}) == \
    "ssh -L 8780:127.0.0.1:8780 deploy@10.0.0.9"
# A non-standard sshd port must be in the command, or the tunnel can't connect.
assert ssh_hint(8780, {"SSH_CONNECTION": "1.2.3.4 5555 89.36.137.77 7744", "USER": "ai_user"}) == \
    "ssh -L 8780:127.0.0.1:8780 -p 7744 ai_user@89.36.137.77"

# Token shape gate keeps junk (and path tricks) out of the getMe URL.
assert TOKEN_RE.match("123456:" + "A" * 35)
assert not TOKEN_RE.match("123/../x:" + "A" * 35)

# .env merge: fresh install starts from the example; a re-run backs up and keeps unknown keys.
d = Path(tempfile.mkdtemp())
example, env = d / ".env.example", d / ".env"
example.write_text("# comment\nTG_BOT_TOKEN=replace-with-x\nHISTORY_TURNS=12\n", encoding="utf-8")
assert merge_env(env, example, {"TG_BOT_TOKEN": "tok"}) is None
vals = dotenv_values(env)
assert vals["TG_BOT_TOKEN"] == "tok" and vals["HISTORY_TURNS"] == "12"
env.write_text(env.read_text(encoding="utf-8") + "CUSTOM=keep\n", encoding="utf-8")
backup = merge_env(env, example, {"OWNER_FIRST_NAME": "O'Brien"})
assert backup is not None and "CUSTOM=keep" in backup.read_text(encoding="utf-8")
vals = dotenv_values(env)
assert vals["CUSTOM"] == "keep" and vals["OWNER_FIRST_NAME"] == "O'Brien" and vals["TG_BOT_TOKEN"] == "tok"
assert "# comment" in env.read_text(encoding="utf-8")

# Resolution preserves explicit selection and the existing Anthropic-first automatic order.
keys = {"OPENROUTER_API_KEY": "or-key", "GEMINI_API_KEY": "gemini-key", "ANTHROPIC_API_KEY": "claude-key"}
assert setup.selected_provider("auto", keys) == "anthropic"
assert setup.selected_provider("gemini", keys) == "gemini"
assert setup.selected_provider("openrouter", keys) == "openrouter"
for selection in ("invalid", "gemini"):
    try:
        setup.selected_provider(selection, {"OPENROUTER_API_KEY": "or-key"})
    except setup.SetupError:
        pass
    else:
        raise AssertionError("invalid/missing-key provider was accepted")

# The official catalogue is paginated, authenticated in a header, and native-only.
requests = []
gemini_key = "AIza-smoke-key-middle-never-show-7788"
anthropic_key = "sk-ant-smoke-key-middle-never-show-9922"


def catalogue(request):
    requests.append(request)
    assert request.url.host == "generativelanguage.googleapis.com", request.url
    assert request.headers["x-goog-api-key"] == gemini_key
    assert "key" not in request.url.params
    if request.url.params.get("pageToken") == "second":
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        ]})
    return httpx.Response(200, json={"nextPageToken": "second", "models": [
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embed-only", "supportedGenerationMethods": ["embedContent"]},
    ]})


with httpx.Client(transport=httpx.MockTransport(catalogue)) as network, patch.object(setup.httpx, "get", network.get):
    assert setup.gemini_model_ids(gemini_key) == {"gemini-3.8-flash", "gemini-3.1-flash-lite"}
    with patch("builtins.input", side_effect=["google/gemini-3.8-flash", "gemini-3.8-flash"]):
        assert setup.step_model("", "gemini", gemini_key) == "gemini-3.8-flash"
    # A complete Gemini setup never requires an OpenRouter key and masks its summary.
    wizard_env = d / "wizard.env"
    wizard_env.write_text(
        f"LLM_PROVIDER=gemini\nGEMINI_API_KEY={gemini_key}\nOWNER_FIRST_NAME=Sam\n",
        encoding="utf-8",
    )
    out = io.StringIO()
    with patch.object(setup, "ENV_PATH", wizard_env), patch.object(setup, "EXAMPLE_PATH", d / "missing"), \
         patch.object(setup, "step_token", return_value=("tok", None)), \
         patch.object(setup, "step_owner", return_value=(42, "Sam")), \
         patch.object(setup, "first_login_link", return_value=None), \
         patch.object(setup.getpass, "getpass", return_value=""), \
         patch("builtins.input", side_effect=["", "", "", "", "y"]), contextlib.redirect_stdout(out):
        assert setup.main() == 0
    saved = dotenv_values(wizard_env)
    assert saved["LLM_PROVIDER"] == "gemini"
    assert not saved.get("OPENROUTER_API_KEY")
    assert gemini_key not in out.getvalue() and mask(gemini_key) in out.getvalue()

with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403))) as network, \
     patch.object(setup.httpx, "get", network.get):
    try:
        setup.gemini_model_ids(gemini_key)
    except setup.SetupError as e:
        assert gemini_key not in str(e)
    else:
        raise AssertionError("a rejected Gemini key was accepted")

# Re-running an Anthropic-only config does not fetch any OpenRouter catalogue/key.
wizard_env.write_text(f"ANTHROPIC_API_KEY={anthropic_key}\nOWNER_FIRST_NAME=Sam\n", encoding="utf-8")
out = io.StringIO()
with patch.object(setup, "ENV_PATH", wizard_env), patch.object(setup, "EXAMPLE_PATH", d / "missing"), \
     patch.object(setup, "step_token", return_value=("tok", None)), \
     patch.object(setup, "step_owner", return_value=(42, "Sam")), \
     patch.object(setup, "first_login_link", return_value=None), \
     patch.object(setup, "anthropic_key_info", return_value=None), \
     patch.object(setup, "or_key_info", side_effect=AssertionError("unexpected OpenRouter key call")), \
     patch.object(setup, "or_model_ids", side_effect=AssertionError("unexpected OpenRouter catalogue")), \
     patch.object(setup.getpass, "getpass", return_value=""), \
     patch("builtins.input", side_effect=["", "", "y"]), contextlib.redirect_stdout(out):
    assert setup.main() == 0
assert not dotenv_values(wizard_env).get("OPENROUTER_API_KEY")
assert anthropic_key not in out.getvalue() and mask(anthropic_key) in out.getvalue()

print("smoke_setup OK")
