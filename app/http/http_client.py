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
            r = httpx.get(url, params=params, timeout=timeout)

            if r.status_code != 200:
                log.error("HTTP %s from %s [attempt %s]", r.status_code, url, attempt)
                last_err = str(r.status_code)
                continue

            return r.json(), ""

        except Exception as e:
            last_err = str(e)
            log.error("Exception during request to %s", url)

    return {}, last_err
