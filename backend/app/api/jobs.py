"""Job status, cancellation, and the realtime progress channels."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .. import db
from ..core.events import broker, job_topic, project_topic
from ..core.queue import manager

router = APIRouter(tags=["jobs"])


def _job_or_404(job_id: str) -> dict[str, Any]:
    row = db.get_conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "job not found")
    out = db.row_to_dict(row, ("params", "result"))
    assert out is not None
    steps = manager.steps_for(out["kind"])
    out["steps"] = [{"key": s.key, "label": s.label} for s in steps]
    return out


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id)


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    _job_or_404(job_id)
    return {"cancelled": manager.cancel(job_id)}


@router.get("/api/jobs")
def list_jobs(project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT * FROM jobs"
    args: list[Any] = []
    if project_id:
        sql += " WHERE project_id=?"
        args.append(project_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.get_conn().execute(sql, args).fetchall()
    return [d for d in (db.row_to_dict(r, ("params", "result")) for r in rows) if d]


# --------------------------------------------------------------------------
# WebSocket — primary progress channel
# --------------------------------------------------------------------------
async def _pump(websocket: WebSocket, topic: str) -> None:
    await websocket.accept()
    queue = await broker.subscribe(topic)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
                continue
            await websocket.send_text(json.dumps(event))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await broker.unsubscribe(topic, queue)


@router.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await _pump(websocket, job_topic(job_id))


@router.websocket("/ws/projects/{project_id}")
async def ws_project(websocket: WebSocket, project_id: str) -> None:
    await _pump(websocket, project_topic(project_id))


# --------------------------------------------------------------------------
# Server-Sent Events — fallback for environments without WebSockets
# --------------------------------------------------------------------------
async def _sse(request: Request, topic: str):
    queue = await broker.subscribe(topic)
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        await broker.unsubscribe(topic, queue)


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    _job_or_404(job_id)
    return StreamingResponse(_sse(request, job_topic(job_id)), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/projects/{project_id}/events")
async def project_events(project_id: str, request: Request) -> StreamingResponse:
    return StreamingResponse(_sse(request, project_topic(project_id)),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
