"""Helpers for the secrets-detection benchmark."""

from __future__ import annotations

from typing import Any


def is_secret_detection_blocked(response: Any) -> bool:
    """Return True when the response indicates the secrets plugin blocked the request."""
    if getattr(response, "status_code", None) == 403:
        return True

    try:
        payload = response.json()
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    data = error.get("data")
    if isinstance(data, dict) and data.get("plugin_error_code") == "SECRETS_DETECTED":
        return True

    return False
