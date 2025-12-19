# app/admin_clean.py
from __future__ import annotations
from app.redis_client import redis

PATTERNS = ["cache:*", "sess:*", "cache:tool:*", "cache:weather:*", "cache:news:*"]

async def purge_by_patterns(patterns = PATTERNS) -> int:
    if redis is None:
        return 0
    deleted = 0
    for pat in patterns:
        cursor = "0"
        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pat, count=500)
            if keys:
                try:
                    deleted += await redis.unlink(*keys)  # non-blocking
                except Exception:
                    deleted += await redis.delete(*keys)  # fallback
            if cursor == "0":
                break
    return deleted

async def should_purge(threshold_mb: int = 28) -> bool:
    if redis is None:
        return False
    info = await redis.info(section="memory")
    used = int(info.get("used_memory", 0))
    return used > threshold_mb * 1024 * 1024

def _humanize_bytes(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

async def memory_usage(threshold_mb: int = 28) -> dict:
    """
    Returns: {
      "used_bytes": int,
      "used_human": "X.YZ MB",
      "threshold_mb": int,
      "percent_of_threshold": float,  # 0..100+
      "should_purge": bool
    }
    """
    if redis is None:
        return {
            "used_bytes": 0,
            "used_human": "0 B",
            "threshold_mb": threshold_mb,
            "percent_of_threshold": 0.0,
            "should_purge": False,
        }
    info = await redis.info(section="memory")
    used = int(info.get("used_memory", 0))
    percent = (used / (threshold_mb * 1024 * 1024)) * 100.0
    return {
        "used_bytes": used,
        "used_human": _humanize_bytes(used),
        "threshold_mb": threshold_mb,
        "percent_of_threshold": percent,
        "should_purge": used > threshold_mb * 1024 * 1024,
    }
