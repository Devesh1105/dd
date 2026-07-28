"""Project, segment and export endpoints."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import db
from ..config import settings
from ..core.queue import manager
from ..media import ffmpeg
from ..pipeline import orchestrator
from ..schemas import DubRequest, ProjectUpdate, ScriptUpload, SegmentBulkUpdate, SegmentUpdate

router = APIRouter(prefix="/api/projects", tags=["projects"])

ALLOWED_EXT = ffmpeg.AUDIO_EXT | ffmpeg.VIDEO_EXT


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:120]
    return cleaned or "upload"


def _project_or_404(project_id: str) -> dict[str, Any]:
    row = db.get_conn().execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "project not found")
    out = db.row_to_dict(row, ("settings",))
    assert out is not None
    return out


def _segments(project_id: str) -> list[dict[str, Any]]:
    rows = db.get_conn().execute(
        "SELECT * FROM segments WHERE project_id=? ORDER BY idx", (project_id,)).fetchall()
    return [dict(r) for r in rows]


@router.post("")
async def create_project(
    file: UploadFile = File(...),
    name: str = Form("Untitled project"),
    source_language: str = Form("auto"),
    target_language: str = Form("es"),
    auto_start: bool = Form(True),
    settings_json: str = Form("{}"),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type {suffix}")
    if suffix in ffmpeg.VIDEO_EXT and not ffmpeg.have_ffmpeg():
        raise HTTPException(400, "video input requires ffmpeg to be installed")
    if suffix in (ffmpeg.AUDIO_EXT - {".wav"}) and not ffmpeg.have_ffmpeg():
        raise HTTPException(400, f"{suffix} input requires ffmpeg; upload a .wav instead")

    project_id = db.new_id("prj")
    dest_dir = settings.project_dir(project_id)
    dest = dest_dir / f"source{suffix or '.wav'}"

    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                fh.close()
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB limit")
            fh.write(chunk)
    if size == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, "uploaded file is empty")

    try:
        extra = json.loads(settings_json or "{}")
    except json.JSONDecodeError:
        extra = {}

    with db.tx() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, source_path, source_kind, source_language, "
            "target_language, sample_rate, status, settings, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (project_id, name.strip() or "Untitled project", str(dest),
             "video" if suffix in ffmpeg.VIDEO_EXT else "audio", source_language, target_language,
             settings.sample_rate, "created", json.dumps(extra), db.now(), db.now()),
        )

    job = manager.enqueue("dub", project_id) if auto_start else None
    project = _project_or_404(project_id)
    project["job"] = job
    return project


@router.get("")
def list_projects(limit: int = 100) -> list[dict[str, Any]]:
    rows = db.get_conn().execute(
        "SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for row in rows:
        p = db.row_to_dict(row, ("settings",))
        assert p is not None
        p["segment_count"] = db.get_conn().execute(
            "SELECT COUNT(*) c FROM segments WHERE project_id=?", (p["id"],)).fetchone()["c"]
        out.append(p)
    return out


@router.get("/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = _project_or_404(project_id)
    conn = db.get_conn()
    project["segments"] = _segments(project_id)
    project["speakers"] = [
        {"speaker": r["speaker"], "voice_id": r["voice_id"], "total_seconds": r["total_seconds"]}
        for r in conn.execute("SELECT * FROM speakers WHERE project_id=? ORDER BY speaker",
                              (project_id,)).fetchall()
    ]
    project["assets"] = {
        r["role"]: {"role": r["role"], "meta": json.loads(r["meta"] or "{}"),
                    "url": f"/api/projects/{project_id}/media/{r['role']}"}
        for r in conn.execute("SELECT * FROM assets WHERE project_id=?", (project_id,)).fetchall()
    }
    project["jobs"] = [
        db.row_to_dict(r, ("params", "result"))
        for r in conn.execute(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 10",
            (project_id,)).fetchall()
    ]
    return project


@router.patch("/{project_id}")
def update_project(project_id: str, body: ProjectUpdate) -> dict[str, Any]:
    project = _project_or_404(project_id)
    merged = {**(project["settings"] or {}), **(body.settings or {})}
    with db.tx() as conn:
        conn.execute(
            "UPDATE projects SET name=?, source_language=?, target_language=?, settings=?, "
            "updated_at=? WHERE id=?",
            (body.name or project["name"], body.source_language or project["source_language"],
             body.target_language or project["target_language"], json.dumps(merged),
             db.now(), project_id),
        )
    return _project_or_404(project_id)


@router.delete("/{project_id}")
def delete_project(project_id: str) -> dict[str, bool]:
    _project_or_404(project_id)
    with db.tx() as conn:
        conn.execute("DELETE FROM segments WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM speakers WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM assets WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM jobs WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM voices WHERE owner=?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    shutil.rmtree(settings.project_dir(project_id), ignore_errors=True)
    return {"deleted": True}


# --------------------------------------------------------------------------
# pipeline control
# --------------------------------------------------------------------------
@router.post("/{project_id}/dub")
def start_dub(project_id: str, body: DubRequest | None = None) -> dict[str, Any]:
    project = _project_or_404(project_id)
    if body:
        merged = {**(project["settings"] or {}), **(body.settings or {})}
        with db.tx() as conn:
            conn.execute("UPDATE projects SET target_language=?, settings=?, updated_at=? WHERE id=?",
                         (body.target_language or project["target_language"], json.dumps(merged),
                          db.now(), project_id))
    return manager.enqueue("dub", project_id)


@router.post("/{project_id}/render")
def start_render(project_id: str) -> dict[str, Any]:
    _project_or_404(project_id)
    if not _segments(project_id):
        raise HTTPException(400, "nothing to render — run the dub pipeline first")
    return manager.enqueue("render", project_id)


@router.post("/{project_id}/script")
def set_script(project_id: str, body: ScriptUpload) -> dict[str, Any]:
    """Attach a known transcript; the pipeline aligns it instead of running ASR."""
    project = _project_or_404(project_id)
    merged = {**(project["settings"] or {}), "script": body.script}
    with db.tx() as conn:
        conn.execute("UPDATE projects SET settings=?, source_language=?, updated_at=? WHERE id=?",
                     (json.dumps(merged),
                      body.language if body.language != "auto" else project["source_language"],
                      db.now(), project_id))
    return _project_or_404(project_id)


# --------------------------------------------------------------------------
# segments (the transcript & translation matrix)
# --------------------------------------------------------------------------
@router.get("/{project_id}/segments")
def get_segments(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return _segments(project_id)


@router.patch("/{project_id}/segments/{segment_id}")
def update_segment(project_id: str, segment_id: str, body: SegmentUpdate) -> dict[str, Any]:
    row = db.get_conn().execute("SELECT * FROM segments WHERE id=? AND project_id=?",
                                (segment_id, project_id)).fetchone()
    if row is None:
        raise HTTPException(404, "segment not found")
    current = dict(row)
    fields = body.model_dump(exclude_none=True)
    if "locked" in fields:
        fields["locked"] = int(bool(fields["locked"]))
    if not fields:
        return current
    start = float(fields.get("start", current["start"]))
    end = float(fields.get("end", current["end"]))
    if end <= start:
        raise HTTPException(400, "segment end must be after start")
    fields["start"], fields["end"] = start, end
    assignments = ", ".join(f"{k}=?" for k in fields)
    with db.tx() as conn:
        conn.execute(f"UPDATE segments SET {assignments} WHERE id=?",
                     (*fields.values(), segment_id))
    return dict(db.get_conn().execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone())


@router.put("/{project_id}/segments")
def bulk_update_segments(project_id: str, body: SegmentBulkUpdate) -> dict[str, Any]:
    _project_or_404(project_id)
    allowed = {"start", "end", "speaker", "source_text", "target_text", "emotion",
               "voice_id", "locked"}
    updated = 0
    with db.tx() as conn:
        for item in body.segments:
            seg_id = item.get("id")
            if not seg_id:
                continue
            fields = {k: v for k, v in item.items() if k in allowed and v is not None}
            if "locked" in fields:
                fields["locked"] = int(bool(fields["locked"]))
            if not fields:
                continue
            assignments = ", ".join(f"{k}=?" for k in fields)
            cur = conn.execute(f"UPDATE segments SET {assignments} WHERE id=? AND project_id=?",
                               (*fields.values(), seg_id, project_id))
            updated += cur.rowcount
    return {"updated": updated}


@router.post("/{project_id}/speakers/{speaker}/voice/{voice_id}")
def assign_speaker_voice(project_id: str, speaker: str, voice_id: str) -> dict[str, Any]:
    """Re-cast a speaker: updates the mapping and every unlocked line."""
    project = _project_or_404(project_id)
    with db.tx() as conn:
        conn.execute("UPDATE speakers SET voice_id=? WHERE project_id=? AND speaker=?",
                     (voice_id, project_id, speaker))
        conn.execute("UPDATE segments SET voice_id=? WHERE project_id=? AND speaker=? AND locked=0",
                     (voice_id, project_id, speaker))
        merged = {**(project["settings"] or {})}
        merged.setdefault("voice_map", {})[speaker] = voice_id
        conn.execute("UPDATE projects SET settings=? WHERE id=?", (json.dumps(merged), project_id))
    return {"speaker": speaker, "voice_id": voice_id}


# --------------------------------------------------------------------------
# media + exports
# --------------------------------------------------------------------------
@router.get("/{project_id}/media/{role}")
def get_media(project_id: str, role: str):
    asset = orchestrator.get_asset(project_id, role)
    if asset is None or not Path(asset["path"]).exists():
        raise HTTPException(404, f"no {role} asset for this project")
    path = Path(asset["path"])
    media_type = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "video/mp4",
        ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=f"{project_id}_{role}{path.suffix}")


@router.get("/{project_id}/segments/{segment_id}/audio")
def get_segment_audio(project_id: str, segment_id: str):
    row = db.get_conn().execute("SELECT audio_path FROM segments WHERE id=? AND project_id=?",
                                (segment_id, project_id)).fetchone()
    if row is None or not row["audio_path"] or not Path(row["audio_path"]).exists():
        raise HTTPException(404, "segment audio not rendered yet")
    return FileResponse(row["audio_path"], media_type="audio/wav")


@router.get("/{project_id}/waveform")
def get_waveform(project_id: str, role: str = "original", buckets: int = 1600) -> dict[str, Any]:
    asset = orchestrator.get_asset(project_id, role)
    if asset is None:
        raise HTTPException(404, f"no {role} asset")
    meta = asset["meta"] or {}
    peaks = meta.get("peaks")
    if not peaks:
        from ..audio.wavio import read_wav, to_mono

        data, _ = read_wav(asset["path"])
        peaks = ffmpeg.waveform_peaks(to_mono(data), buckets)
    return {"role": role, "peaks": peaks, "duration": meta.get("duration")}


@router.get("/{project_id}/export.srt", response_class=PlainTextResponse)
def export_srt(project_id: str, field: str = "target_text") -> str:
    _project_or_404(project_id)
    if field not in ("target_text", "source_text"):
        raise HTTPException(400, "field must be target_text or source_text")

    def ts(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(_segments(project_id), start=1):
        text = (seg[field] or "").strip()
        if not text:
            continue
        lines.append(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{text}\n")
    return "\n".join(lines)


@router.get("/{project_id}/export.json")
def export_json(project_id: str) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return {
        "project": {k: project[k] for k in
                    ("id", "name", "source_language", "target_language", "duration", "status")},
        "segments": [
            {k: seg[k] for k in
             ("idx", "start", "end", "speaker", "source_text", "target_text", "emotion",
              "voice_id", "fit_rate")}
            for seg in _segments(project_id)
        ],
    }
