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
