"""Gemini REST constants and errors, independent of tenant settings for setup checks."""
import re

import httpx

API_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAPIError(httpx.HTTPStatusError):
    """Expose actionable status codes, never Google's echoed prompts, keys or metadata."""

    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
        status = self._code(error.get("status"))
        details = error.get("details", [])
        reason = ""
        if isinstance(details, list):
            reason = next((self._code(d.get("reason")) for d in details
                           if isinstance(d, dict) and d.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"), "")
        codes = " / ".join(code for code in (status, reason) if code)
        hint = {
            400: "Check the native model ID and request parameters.",
            401: "Check the API key in Google AI Studio.",
            402: "The API project's billing credits are depleted; add credits in AI Studio.",
            403: "Check key restrictions, Generative Language API access and the server's supported region.",
            404: "The model is unavailable on this API; select a native model listed for this key.",
            429: "The API project's quota or rate limit is exhausted; check limits and billing in AI Studio.",
        }.get(response.status_code, "Check the Gemini API service status and network/proxy configuration.")
        if reason in {"API_KEY_INVALID", "API_KEY_EXPIRED"}:
            hint = "Create a valid API key in Google AI Studio and update GEMINI_API_KEY."
        elif status == "FAILED_PRECONDITION":
            hint = "Check project billing and Gemini API availability in the server's region."
        elif reason == "SERVICE_DISABLED":
            hint = "Enable the Generative Language API for the key's Google Cloud project."
        elif reason.startswith("API_KEY_") and response.status_code == 403:
            hint = "Check the key's API/IP restrictions; unrestricted standard keys are no longer accepted."
        message = f"Gemini HTTP {response.status_code}" + (f" ({codes})" if codes else "") + f": {hint}"
        super().__init__(message, request=response.request, response=response)

    @staticmethod
    def _code(value: object) -> str:
        return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", value) else ""
