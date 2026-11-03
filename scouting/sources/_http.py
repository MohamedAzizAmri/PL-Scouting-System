"""Small JSON-over-HTTP client with an on-disk cache, a rate limiter and retries.

football-data.org's free plan allows 10 requests/minute, so being a good citizen
(and not re-fetching the same thing on every run) is built in rather than bolted on.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str, url: str = ""):
        super().__init__(f"HTTP {status} for {url}: {message}")
        self.status = status
        self.url = url


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` calls per `period` seconds."""

    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_calls, self.period = max_calls, period
        self._clock, self._sleep = clock, sleep
        self._calls: deque[float] = deque()

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()

    def wait(self) -> None:
        now = self._clock()
        self._evict(now)
        if len(self._calls) >= self.max_calls:
            delay = self.period - (now - self._calls[0])
            if delay > 0:
                log.info("rate limit reached, sleeping %.1fs", delay)
                self._sleep(delay)
            now = self._clock()
            self._evict(now)
        self._calls.append(now)


class JsonClient:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        cache_dir: Path | None = None,
        max_age_hours: float = 6.0,
        timeout: float = 30.0,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        if headers:
            self.session.headers.update(headers)
        self.limiter, self.cache_dir = limiter, cache_dir
        self.max_age_hours, self.timeout, self.retries = max_age_hours, timeout, retries
        self._sleep = sleep

    # -- cache ------------------------------------------------------------- #
    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.removeprefix(self.base_url)).strip("_") or "root"
        return self.cache_dir / f"{slug}_{hashlib.sha1(key.encode()).hexdigest()[:8]}.json"

    def _read_cache(self, path: Path | None) -> Any | None:
        if path is None or not path.exists():
            return None
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        age_h = (time.time() - blob.get("fetched_at", 0)) / 3600
        return blob["data"] if age_h <= self.max_age_hours else None

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def is_cached(self, path: str, params: dict | None = None) -> bool:
        """True if a fresh cached response exists (so a get() would cost no request)."""
        return self._read_cache(self._cache_path(self._url(path), params)) is not None

    def invalidate(self, path: str, params: dict | None = None) -> None:
        cache_path = self._cache_path(self._url(path), params)
        if cache_path is not None and cache_path.exists():
            cache_path.unlink()

    # -- request ----------------------------------------------------------- #
    def get(self, path: str, params: dict | None = None, *, use_cache: bool = True) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        cache_path = self._cache_path(url, params)
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                log.debug("cache hit: %s", url)
                return cached

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.limiter:
                self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network trouble: back off and retry
                last_error = exc
                self._sleep(2 ** attempt)
                continue
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 60))
                log.warning("429 from %s, waiting %.0fs", url, wait)
                self._sleep(wait)
                last_error = HttpError(429, "rate limited", url)
                continue
            if resp.status_code >= 500:
                last_error = HttpError(resp.status_code, "server error", url)
                self._sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                try:
                    message = resp.json().get("message", resp.text[:200])
                except ValueError:
                    message = resp.text[:200]
                raise HttpError(resp.status_code, message, url)
            data = resp.json()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"fetched_at": time.time(), "data": data}))
            return data
        raise HttpError(getattr(last_error, "status", 0), f"gave up after retries: {last_error}", url)
