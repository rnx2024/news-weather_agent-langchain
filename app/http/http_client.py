# app/http_client.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import httpx

log = logging.getLogger(__name__)


def get_json_with_retry(
    url: str,
    params: Dict[str, Any],
    retries: int = 3,
    timeout: float = 10.0,
) -> Tuple[Dict[str, Any], str]:
    last_err = ""

    for attempt in range(1, retries + 1):
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except httpx.TimeoutException:
            last_err = "timeout"
            log.warning("Timeout from %s [attempt %s/%s]", url, attempt, retries)
            continue
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            last_err = str(status)
            log.error("HTTP %s from %s [attempt %s/%s]", status, url, attempt, retries)
            continue
        except httpx.RequestError as exc:
            last_err = str(exc)
            log.error("Request error from %s [attempt %s/%s]: %s", url, attempt, retries, exc)
            continue

        try:
            return response.json(), ""
        except ValueError:
            last_err = "invalid_json"
            log.error("Invalid JSON from %s [attempt %s/%s]", url, attempt, retries)
            continue

    return {}, last_err
