"""In-process pub/sub for pipeline progress.

Feeds both the WebSocket endpoint and the SSE fallback. For a multi-process
deployment, replace `Broker.publish` with a Redis pub/sub publish and have
each web worker subscribe — the API surface is intentionally identical.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any

MAX_REPLAY = 100


class Broker:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_REPLAY))
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self, topic: str, replay: bool = True) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subs[topic].add(q)
            if replay:
                for event in list(self._history[topic]):
                    q.put_nowait(event)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subs[topic].discard(q)
            if not self._subs[topic]:
                self._subs.pop(topic, None)

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Thread-safe: workers run in a thread pool but publish to the loop."""
        payload = {"topic": topic, "ts": time.time(), **event}
        self._history[topic].append(payload)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if asyncio.get_event_loop_policy() and _in_loop(loop):
            self._fanout(topic, payload)
        else:
            loop.call_soon_threadsafe(self._fanout, topic, payload)

    def _fanout(self, topic: str, payload: dict[str, Any]) -> None:
        for q in list(self._subs.get(topic, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:  # slow consumer: drop the oldest
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    pass

    def history(self, topic: str) -> list[dict[str, Any]]:
        return list(self._history[topic])


def _in_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


broker = Broker()


def job_topic(job_id: str) -> str:
    return f"job:{job_id}"


def project_topic(project_id: str) -> str:
    return f"project:{project_id}"
