import asyncio


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.values: dict[str, str] = {}

    async def _noop(self) -> None:
        await asyncio.sleep(0)

    async def hgetall(self, key: str) -> dict[str, str]:
        await self._noop()
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        await self._noop()
        bucket = self.hashes.setdefault(key, {})
        bucket.update({str(k): str(v) for k, v in mapping.items()})

    async def hget(self, key: str, field: str) -> str | None:
        await self._noop()
        return self.hashes.get(key, {}).get(field)

    async def hmget(self, key: str, *fields: str) -> list[str | None]:
        await self._noop()
        bucket = self.hashes.get(key, {})
        return [bucket.get(field) for field in fields]

    async def hdel(self, key: str, *fields: str) -> None:
        await self._noop()
        bucket = self.hashes.get(key, {})
        for field in fields:
            bucket.pop(field, None)

    async def expire(self, _key: str, _ttl: int) -> None:
        await self._noop()
        return None

    async def get(self, key: str) -> str | None:
        await self._noop()
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        await self._noop()
        _ = ex
        self.values[key] = str(value)
