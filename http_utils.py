from __future__ import annotations
import time
from typing import Dict, Tuple
import requests
from requests.exceptions import RequestException

def get_json_with_retry(url: str, params: Dict, tries: int = 2, timeout: int = 15) -> Tuple[dict, str]:
    last_err = ""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return (r.json() or {}, "")
            last_err = f"HTTP {r.status_code}"
        except RequestException as e:
            last_err = str(e)
        time.sleep(0.5 * (i + 1))
    return ({}, last_err)
