# app/http_utils.py
from __future__ import annotations

import logging
import httpx
from typing import Dict, Any, Tuple

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
                log.error(
                    "HTTP %s from %s [attempt %s]\nParams: %s\nBody: %s",
                    r.status_code,
                    url,
                    attempt,
                    params,
                    r.text,
                )
                last_err = f"{r.status_code}: {r.text}"
                continue

            return r.json(), ""

        except Exception as e:
            last_err = str(e)
            log.error("Exception during request to %s: %s", url, e)

    return {}, last_err
