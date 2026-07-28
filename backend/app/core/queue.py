"""Async job queue.

Deep-learning / DSP work must never run inside a request handler, so the API
only ever enqueues. Jobs are persisted in SQLite (so status survives a
restart) and executed by a pool of workers on a thread executor.

This is the local stand-in for Celery/Temporal: same contract (durable job
row, step-level progress, cancellation, retry), no broker to install. To move
to Celery, keep `JobContext` and swap `_run` for a task signature.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import db
from ..config import settings
from .events import broker, job_topic, project_topic


class JobCancelled(Exception):
    """Raised inside a job when the user cancels it."""


@dataclass
class Step:
    key: str
    label: str
    weight: float = 1.0


@dataclass
class JobContext:
    job_id: str
    project_id: str
    params: dict[str, Any]
    steps: list[Step]
    _step_i: int = field(default=0, init=False)
    _base: float = field(default=0.0, init=False)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def _total_weight(self) -> float:
        return sum(s.weight for s in self.steps) or 1.0

    def check_cancelled(self) -> None:
        if manager.is_cancelled(self.job_id):
            raise JobCancelled()

    def begin(self, key: str, message: str = "") -> None:
        self.check_cancelled()
        for i, s in enumerate(self.steps):
            if s.key == key:
                self._step_i = i
                self._base = sum(x.weight for x in self.steps[:i]) / self._total_weight
                break
        step = self.steps[self._step_i]
        self._emit(step, self._base, message or step.label)

    def progress(self, fraction: float, message: str = "") -> None:
        self.check_cancelled()
        step = self.steps[self._step_i]
        span = step.weight / self._total_weight
        overall = self._base + span * max(0.0, min(1.0, fraction))
        self._emit(step, overall, message or step.label)

    def log(self, message: str) -> None:
        broker.publish(job_topic(self.job_id), {"type": "log", "job_id": self.job_id, "message": message})

    def _emit(self, step: Step, overall: float, message: str) -> None:
        pct = round(max(0.0, min(1.0, overall)) * 100, 2)
        with db.tx() as conn:
            conn.execute(
                "UPDATE jobs SET step=?, step_index=?, step_total=?, progress=?, message=? WHERE id=?",
                (step.key, self._step_i + 1, len(self.steps), pct, message, self.job_id),
            )
        payload = {
            "type": "progress",
            "job_id": self.job_id,
            "project_id": self.project_id,
            "step": step.key,
            "step_label": step.label,
            "step_index": self._step_i + 1,
            "step_total": len(self.steps),
            "progress": pct,
            "message": message,
        }
        broker.publish(job_topic(self.job_id), payload)
        broker.publish(project_topic(self.project_id), payload)


JobHandler = Callable[[JobContext], dict[str, Any]]


class JobManager:
    def __init__(self) -> None:
        self._handlers: dict[str, tuple[JobHandler, list[Step]]] = {}
        self._queue: asyncio.Queue[str] | None = None
        self._workers: list[asyncio.Task] = []
        self._executor: ThreadPoolExecutor | None = None
        self._cancelled: set[str] = set()

    # registration ---------------------------------------------------------
    def register(self, kind: str, steps: list[Step]) -> Callable[[JobHandler], JobHandler]:
        def deco(fn: JobHandler) -> JobHandler:
            self._handlers[kind] = (fn, steps)
            return fn
        return deco

    def steps_for(self, kind: str) -> list[Step]:
        return self._handlers[kind][1] if kind in self._handlers else []

    # lifecycle ------------------------------------------------------------
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        broker.bind_loop(loop)
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.workers),
                                            thread_name_prefix="dub-worker")
        for i in range(max(1, settings.workers)):
            self._workers.append(asyncio.create_task(self._worker(i), name=f"job-worker-{i}"))
        # requeue anything left running by an unclean shutdown
        rows = db.get_conn().execute(
            "SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY created_at").fetchall()
        for row in rows:
            with db.tx() as conn:
                conn.execute("UPDATE jobs SET status='queued', progress=0 WHERE id=?", (row["id"],))
            self._queue.put_nowait(row["id"])

    async def stop(self) -> None:
        for t in self._workers:
            t.cancel()
        self._workers.clear()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # enqueue --------------------------------------------------------------
    def enqueue(self, kind: str, project_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if kind not in self._handlers:
            raise KeyError(f"unknown job kind: {kind}")
        job_id = db.new_id("job")
        steps = self._handlers[kind][1]
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO jobs (id, project_id, kind, status, step_total, params, created_at) "
                "VALUES (?,?,?,'queued',?,?,?)",
                (job_id, project_id, kind, len(steps), json.dumps(params or {}), db.now()),
            )
        if self._queue is not None:
            self._queue.put_nowait(job_id)
        broker.publish(project_topic(project_id),
                       {"type": "job_queued", "job_id": job_id, "kind": kind, "project_id": project_id})
        return {"id": job_id, "kind": kind, "status": "queued",
                "steps": [{"key": s.key, "label": s.label} for s in steps]}

    def cancel(self, job_id: str) -> bool:
        row = db.get_conn().execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] in ("done", "failed", "cancelled"):
            return False
        self._cancelled.add(job_id)
        with db.tx() as conn:
            conn.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                         (db.now(), job_id))
        return True

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled

    # execution ------------------------------------------------------------
    async def _worker(self, index: int) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                await self._run(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - worker must never die
                traceback.print_exc()
            finally:
                self._queue.task_done()

    async def _run(self, job_id: str) -> None:
        row = db.get_conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] == "cancelled":
            return
        kind = row["kind"]
        handler, steps = self._handlers[kind]
        ctx = JobContext(job_id=job_id, project_id=row["project_id"],
                         params=json.loads(row["params"] or "{}"), steps=list(steps))

        with db.tx() as conn:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (db.now(), job_id))
        broker.publish(job_topic(job_id), {"type": "started", "job_id": job_id, "kind": kind})
        broker.publish(project_topic(ctx.project_id),
                       {"type": "job_started", "job_id": job_id, "kind": kind})

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(self._executor, handler, ctx)
            with db.tx() as conn:
                conn.execute(
                    "UPDATE jobs SET status='done', progress=100, finished_at=?, result=?, message=? WHERE id=?",
                    (db.now(), json.dumps(result or {}), "Completed", job_id),
                )
            event = {"type": "completed", "job_id": job_id, "kind": kind,
                     "project_id": ctx.project_id, "result": result or {}}
        except JobCancelled:
            with db.tx() as conn:
                conn.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?", (db.now(), job_id))
            event = {"type": "cancelled", "job_id": job_id, "kind": kind, "project_id": ctx.project_id}
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            with db.tx() as conn:
                conn.execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                             (detail, db.now(), job_id))
            event = {"type": "failed", "job_id": job_id, "kind": kind,
                     "project_id": ctx.project_id, "error": detail}
        finally:
            self._cancelled.discard(job_id)

        broker.publish(job_topic(job_id), event)
        broker.publish(project_topic(ctx.project_id), event)


manager = JobManager()
