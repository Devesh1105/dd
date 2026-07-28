/* AI Dubbing Studio — single-file frontend (no build step, no dependencies). */
'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  system: null,
  projects: [],
  voices: [],
  project: null,
  peaks: {},
  ws: null,
  activeJob: null,
  playhead: 0,
  selected: null,
  saveTimers: {},
};

const SPEAKER_COLORS = ['#6c8cff', '#38d6b0', '#ffb547', '#ff6b9d', '#a78bfa', '#4dd0e1',
                        '#f97316', '#84cc16'];

/* ------------------------------------------------------------------ api */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  const type = res.headers.get('content-type') || '';
  if (type.includes('application/json')) return res.json();
  if (type.startsWith('audio/')) return res.blob();
  return res.text();
}

const getJSON = (p) => api(p);
const postJSON = (p, body) => api(p, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
});
const patchJSON = (p, body) => api(p, {
  method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

function toast(message, isError) {
  const el = $('toast');
  el.textContent = message;
  el.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = 'toast'; }, 4200);
}

const fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

/* ------------------------------------------------------------------ boot */
async function init() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  });

  try {
    state.system = await getJSON('/api/system');
  } catch (err) {
    $('engineStatus').textContent = 'backend unreachable';
    toast('Cannot reach the backend: ' + err.message, true);
    return;
  }
  renderEngineStatus();
  populateLanguages();
  renderSystem();

  wireUpload();
  wireEditor();
  wireVoiceStudio();

  await Promise.all([loadProjects(), loadVoices(), loadArchetypes()]);
}

function showView(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  if (name === 'editor' && state.project) drawTimeline();
}

function renderEngineStatus() {
  const p = state.system.active_providers;
  $('engineStatus').textContent =
    `asr:${p.asr} · mt:${p.mt} · tts:${p.tts} · ffmpeg:${state.system.ffmpeg ? 'yes' : 'no'}`;
}

function populateLanguages() {
  const langs = state.system.languages;
  const fill = (el, includeAuto, initial) => {
    el.innerHTML = (includeAuto ? '<option value="auto">Auto-detect</option>' : '') +
      langs.map((l) => `<option value="${l.code}">${l.name}</option>`).join('');
    if (initial) el.value = initial;
  };
  fill($('srcLang'), true, 'auto');
  fill($('tgtLang'), false, 'es');
  fill($('edTarget'), false, 'es');
  fill($('designLang'), false, 'en');
}

/* ------------------------------------------------------------------ upload */
function wireUpload() {
  const zone = $('dropzone');
  const input = $('fileInput');
  zone.addEventListener('click', () => input.click());
  ['dragenter', 'dragover'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((e) =>
    zone.addEventListener(e, () => zone.classList.remove('drag')));
  zone.addEventListener('drop', (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; showPicked(); }
  });
  input.addEventListener('change', showPicked);

  function showPicked() {
    const f = input.files[0];
    if (!f) return;
    $('dzTitle').textContent = f.name;
    $('dzHint').textContent = `${(f.size / 1048576).toFixed(1)} MB`;
    if (!$('projName').value) $('projName').value = f.name.replace(/\.[^.]+$/, '');
  }

  $('uploadForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const file = input.files[0];
    if (!file) return toast('Choose a file first', true);

    const body = new FormData();
    body.append('file', file);
    body.append('name', $('projName').value || file.name);
    body.append('source_language', $('srcLang').value);
    body.append('target_language', $('tgtLang').value);
    body.append('auto_start', 'false');
    body.append('settings_json', JSON.stringify({
      separate_background: $('optSeparate').checked,
      clone_speakers: $('optClone').checked,
      duck_background: $('optDuck').checked,
      max_speakers: Number($('optSpeakers').value) || 6,
    }));

    const btn = $('uploadBtn');
    btn.disabled = true;
    btn.textContent = 'Uploading…';
    $('uploadMsg').className = 'msg';
    $('uploadMsg').textContent = '';
    try {
      const project = await api('/api/projects', { method: 'POST', body });
      const script = $('optScript').value.trim();
      if (script) {
        await postJSON(`/api/projects/${project.id}/script`,
                       { script, language: $('srcLang').value });
      }
      const job = await postJSON(`/api/projects/${project.id}/dub`, {});
      await openProject(project.id);
      watchJob(job.id);
      showView('editor');
      toast('Dubbing started');
      $('uploadForm').reset();
      $('dzTitle').textContent = 'Drop audio or video here';
      loadProjects();
    } catch (err) {
      $('uploadMsg').className = 'msg error';
      $('uploadMsg').textContent = err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Create & start dubbing';
    }
  });
}

async function loadProjects() {
  state.projects = await getJSON('/api/projects');
  const list = $('projectList');
  if (!state.projects.length) {
    list.innerHTML = '<div class="hint">No projects yet — upload something to get started.</div>';
    return;
  }
  list.innerHTML = '';
  state.projects.forEach((p) => {
    const row = document.createElement('div');
    row.className = 'list-item';
    row.innerHTML = `
      <div class="meta">
        <div class="name"></div>
        <div class="info">${p.source_language} → ${p.target_language} ·
          ${p.duration ? fmtTime(p.duration) : '—'} · ${p.segment_count} lines</div>
      </div>
      <span class="pill ${p.status}">${p.status}</span>
      <button class="del" title="Delete">✕</button>`;
    row.querySelector('.name').textContent = p.name;
    row.addEventListener('click', () => { openProject(p.id); showView('editor'); });
    row.querySelector('.del').addEventListener('click', async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete "${p.name}" and all its media?`)) return;
      await api(`/api/projects/${p.id}`, { method: 'DELETE' });
      if (state.project && state.project.id === p.id) {
        state.project = null;
        $('editorBody').hidden = true;
        $('editorEmpty').hidden = false;
      }
      loadProjects();
      toast('Project deleted');
    });
    list.appendChild(row);
  });
}

/* ------------------------------------------------------------------ editor */
function wireEditor() {
  $('btnRerender').addEventListener('click', async () => {
    try {
      await saveMixSettings();
      watchJob((await postJSON(`/api/projects/${state.project.id}/render`)).id);
    } catch (err) { toast(err.message, true); }
  });

  $('btnRedub').addEventListener('click', async () => {
    if (!confirm('Re-run the whole pipeline? Transcript edits will be replaced.')) return;
    try {
      await saveMixSettings();
      const job = await postJSON(`/api/projects/${state.project.id}/dub`,
                                 { target_language: $('edTarget').value });
      watchJob(job.id);
    } catch (err) { toast(err.message, true); }
  });

  $('btnCancel').addEventListener('click', async () => {
    if (!state.activeJob) return;
    await postJSON(`/api/jobs/${state.activeJob}/cancel`);
    toast('Cancelling…');
  });

  $('edTarget').addEventListener('change', async () => {
    await patchJSON(`/api/projects/${state.project.id}`, { target_language: $('edTarget').value });
    toast('Target language updated — run a full re-dub to retranslate');
  });

  $('zoom').addEventListener('input', drawTimeline);
  window.addEventListener('resize', drawTimeline);

  const canvas = $('timeline');
  canvas.addEventListener('click', (ev) => {
    if (!state.project) return;
    const rect = canvas.getBoundingClientRect();
    const view = timelineView();
    const t = view.start + ((ev.clientX - rect.left) / rect.width) * view.span;
    seek(t);
    const seg = state.project.segments.find((s) => t >= s.start && t <= s.end);
    if (seg) selectSegment(seg.id);
  });

  const player = $('player');
  player.addEventListener('timeupdate', () => {
    state.playhead = player.currentTime;
    $('timeLabel').textContent =
      `${fmtTime(player.currentTime)} / ${fmtTime(player.duration || (state.project?.duration || 0))}`;
    drawTimeline();
  });
  player.addEventListener('ended', () => { $('btnPlay').textContent = '▶'; });

  $('btnPlay').addEventListener('click', () => {
    if (player.paused) { player.play(); $('btnPlay').textContent = '❚❚'; }
    else { player.pause(); $('btnPlay').textContent = '▶'; }
  });
  $('trackSelect').addEventListener('change', () => {
    const at = player.currentTime;
    loadTrack($('trackSelect').value);
    player.currentTime = at;
  });

  [['mixSpeech', (v) => v.toFixed(2)], ['mixBg', (v) => v.toFixed(2)],
   ['mixDuck', (v) => `${v} dB`], ['mixReverb', (v) => v.toFixed(2)],
   ['mixSpeed', (v) => `${v.toFixed(2)}×`]].forEach(([id, fmt]) => {
    const el = $(id);
    el.addEventListener('input', () => {
      el.nextElementSibling.textContent = fmt(Number(el.value));
    });
  });
}

async function openProject(id) {
  state.project = await getJSON(`/api/projects/${id}`);
  state.selected = null;
  $('editorEmpty').hidden = true;
  $('editorBody').hidden = false;

  const p = state.project;
  $('edTitle').textContent = p.name;
  $('edSub').textContent =
    `${p.source_language} → ${p.target_language} · ${fmtTime(p.duration)} · ` +
    `${p.segments.length} lines · ${p.speakers.length} speaker(s) · ${p.status}`;
  $('edTarget').value = p.target_language;

  const s = p.settings || {};
  $('mixSpeech').value = s.speech_gain ?? 1;
  $('mixBg').value = s.background_gain ?? 1;
  $('mixDuck').value = s.duck_depth_db ?? -11;
  $('mixReverb').value = s.reverb ?? 0.1;
  $('mixSpeed').value = s.speed ?? 1;
  $('mixOverflow').checked = !!s.allow_overflow;
  document.querySelectorAll('.mixer input[type=range]').forEach((el) =>
    el.dispatchEvent(new Event('input')));

  state.peaks = {};
  await Promise.all(['original', 'dubbed', 'mixed'].map(loadPeaks));
  renderMatrix();
  renderCast();
  renderExports();
  drawTimeline();
  loadTrack($('trackSelect').value);
  subscribeProject(p.id);

  const running = p.jobs.find((j) => j.status === 'running' || j.status === 'queued');
  if (running) watchJob(running.id);
}

async function loadPeaks(role) {
  if (!state.project.assets[role]) return;
  try {
    state.peaks[role] = (await getJSON(
      `/api/projects/${state.project.id}/waveform?role=${role}`)).peaks || [];
  } catch (_) { /* asset not rendered yet */ }
}

function loadTrack(role) {
  const player = $('player');
  if (!state.project || !state.project.assets[role]) { player.removeAttribute('src'); return; }
  player.src = `/api/projects/${state.project.id}/media/${role}?t=${Date.now()}`;
}

function seek(t) {
  const player = $('player');
  if (player.src) player.currentTime = Math.max(0, t);
  state.playhead = t;
  drawTimeline();
}

/* ---------------- timeline canvas ---------------- */
function timelineView() {
  const duration = state.project?.duration || 1;
  const zoom = Number($('zoom').value) || 1;
  const span = duration / zoom;
  let start = state.playhead - span / 2;
  start = Math.max(0, Math.min(start, duration - span));
  if (zoom <= 1) start = 0;
  return { start, span, duration };
}

function drawTimeline() {
  const canvas = $('timeline');
  if (!state.project || canvas.offsetWidth === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.offsetWidth;
  const height = 260;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const view = timelineView();
  const xOf = (t) => ((t - view.start) / view.span) * width;

  // time ruler
  ctx.font = '10px ui-monospace, monospace';
  const stepChoices = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  const step = stepChoices.find((s) => (s / view.span) * width > 62) || 600;
  ctx.strokeStyle = '#1e2434';
  ctx.fillStyle = '#8e97b0';
  for (let t = Math.ceil(view.start / step) * step; t < view.start + view.span; t += step) {
    const x = xOf(t);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    ctx.fillText(fmtTime(t), x + 4, 11);
  }

  drawWave(ctx, state.peaks.original, 18, 54, '#39415c', width, view, 'Original');
  drawWave(ctx, state.peaks.dubbed || state.peaks.mixed, 80, 54, '#6c8cff', width, view, 'Dubbed');

  // speaker lanes
  const speakers = [...new Set(state.project.segments.map((s) => s.speaker))].sort();
  const laneTop = 146;
  const laneH = Math.max(16, Math.min(26, (height - laneTop - 12) / Math.max(1, speakers.length)));
  speakers.forEach((speaker, i) => {
    const y = laneTop + i * laneH;
    const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
    ctx.fillStyle = '#8e97b0';
    ctx.font = '10px system-ui';
    ctx.fillText(speaker, 4, y + laneH / 2 + 3);

    state.project.segments.filter((s) => s.speaker === speaker).forEach((seg) => {
      const x1 = xOf(seg.start);
      const x2 = xOf(seg.end);
      if (x2 < 0 || x1 > width) return;
      const w = Math.max(2, x2 - x1);
      const selected = state.selected === seg.id;
      ctx.fillStyle = selected ? color : color + '66';
      ctx.strokeStyle = color;
      ctx.lineWidth = selected ? 2 : 1;
      roundRect(ctx, Math.max(46, x1), y + 3, Math.min(w, width - x1), laneH - 7, 4);
      ctx.fill();
      if (selected) ctx.stroke();

      if (w > 44) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(Math.max(46, x1), y, Math.min(w, width - x1), laneH);
        ctx.clip();
        ctx.fillStyle = '#0b0e18';
        ctx.font = '10px system-ui';
        const label = (seg.target_text || seg.source_text || '—').slice(0, 60);
        ctx.fillText(label, Math.max(50, x1 + 4), y + laneH / 2 + 3);
        ctx.restore();
      }
    });
  });

  // playhead
  const px = xOf(state.playhead);
  if (px >= 0 && px <= width) {
    ctx.strokeStyle = '#38d6b0';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, height); ctx.stroke();
  }

  $('legend').innerHTML = speakers.map((s, i) =>
    `<span><span class="sw" style="background:${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}"></span>${s}</span>`
  ).join('') + '<span>Click the timeline to seek · drag the zoom slider to inspect</span>';
}

function drawWave(ctx, peaks, top, h, color, width, view, label) {
  ctx.fillStyle = '#8e97b0';
  ctx.font = '10px system-ui';
  ctx.fillText(label, 4, top - 3);
  ctx.fillStyle = '#0f1219';
  ctx.fillRect(44, top, width - 44, h);
  if (!peaks || !peaks.length) {
    ctx.fillStyle = '#39415c';
    ctx.fillText('not rendered yet', 52, top + h / 2 + 3);
    return;
  }
  const mid = top + h / 2;
  ctx.fillStyle = color;
  const n = peaks.length;
  for (let x = 44; x < width; x++) {
    const t = view.start + ((x - 44) / (width - 44)) * view.span;
    const i = Math.floor((t / view.duration) * n);
    if (i < 0 || i >= n) continue;
    const amp = Math.min(1, peaks[i]) * (h / 2 - 1);
    ctx.fillRect(x, mid - amp, 1, Math.max(1, amp * 2));
  }
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ---------------- transcript matrix ---------------- */
function renderMatrix() {
  const box = $('matrix');
  const segments = state.project.segments;
  if (!segments.length) {
    box.innerHTML = '<div class="hint">No segments yet. Run the dub pipeline to populate the transcript.</div>';
    return;
  }
  const voiceOptions = state.voices.map((v) =>
    `<option value="${v.id}">${escapeHtml(v.name)}</option>`).join('');
  const emotions = (state.system.emotions || ['neutral', 'happy', 'excited', 'sad', 'angry',
    'shouting', 'whisper', 'sarcastic', 'laughing', 'crying', 'fearful', 'calm', 'menacing']);

  box.innerHTML = '';
  segments.forEach((seg) => {
    const row = document.createElement('div');
    row.className = 'matrix-row' + (state.selected === seg.id ? ' active' : '');
    row.dataset.id = seg.id;
    row.innerHTML = `
      <div class="seg-time">
        <div>${fmtTime(seg.start)}</div>
        <div>${(seg.end - seg.start).toFixed(1)}s</div>
        <button class="play-seg" title="Play this line">▶ line</button>
        <div class="fit"></div>
      </div>
      <div>
        <select class="seg-speaker"></select>
        <select class="seg-voice">${voiceOptions}</select>
        <select class="seg-emotion">${emotions.map((e) =>
          `<option value="${e}">${e}</option>`).join('')}</select>
      </div>
      <div><textarea class="seg-source" spellcheck="false"></textarea></div>
      <div><textarea class="seg-target" spellcheck="false"></textarea></div>`;

    const speakers = [...new Set(segments.map((s) => s.speaker))].sort();
    const spSel = row.querySelector('.seg-speaker');
    spSel.innerHTML = speakers.map((s) => `<option value="${s}">${s}</option>`).join('');
    spSel.value = seg.speaker;
    row.querySelector('.seg-voice').value = seg.voice_id || '';
    row.querySelector('.seg-emotion').value = seg.emotion || 'neutral';
    row.querySelector('.seg-source').value = seg.source_text || '';
    row.querySelector('.seg-target').value = seg.target_text || '';
    updateFit(row, seg);

    row.addEventListener('click', (ev) => {
      if (ev.target.closest('button, select, textarea')) return;
      selectSegment(seg.id);
    });
    row.querySelector('.play-seg').addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const audio = new Audio(
        `/api/projects/${state.project.id}/segments/${seg.id}/audio?t=${Date.now()}`);
      audio.play().catch(() => toast('This line has not been rendered yet', true));
    });

    const bind = (selector, field, transform) => {
      const el = row.querySelector(selector);
      const evt = el.tagName === 'SELECT' ? 'change' : 'input';
      el.addEventListener(evt, () => {
        const value = transform ? transform(el.value) : el.value;
        seg[field] = value;
        if (field === 'target_text') updateFit(row, seg);
        queueSave(seg.id, { [field]: value });
        if (field === 'speaker' || field === 'target_text') drawTimeline();
      });
    };
    bind('.seg-speaker', 'speaker');
    bind('.seg-voice', 'voice_id');
    bind('.seg-emotion', 'emotion');
    bind('.seg-source', 'source_text');
    bind('.seg-target', 'target_text');

    box.appendChild(row);
  });
}

function updateFit(row, seg) {
  const el = row.querySelector('.fit');
  const text = (seg.target_text || '').trim();
  const slot = seg.end - seg.start;
  if (!text) { el.textContent = ''; el.className = 'fit'; return; }
  const lang = state.project.target_language;
  const rate = (state.system.chars_per_second || {})[lang] || 15;
  const needed = text.length / rate;
  const ratio = needed / Math.max(0.2, slot);
  el.textContent = `${Math.round(ratio * 100)}% of slot`;
  el.className = 'fit' + (ratio > 1.45 ? ' over' : ratio > 1.12 ? ' tight' : '');
  el.title = ratio > 1.45
    ? 'Too long — this line will be clipped or padded. Shorten it.'
    : ratio > 1.12 ? 'Will be sped up to fit the original timing.' : 'Fits comfortably.';
}

function selectSegment(id) {
  state.selected = id;
  document.querySelectorAll('.matrix-row').forEach((r) =>
    r.classList.toggle('active', r.dataset.id === id));
  const row = document.querySelector(`.matrix-row[data-id="${id}"]`);
  if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  drawTimeline();
}

function queueSave(segmentId, patch) {
  clearTimeout(state.saveTimers[segmentId]);
  state.saveTimers[segmentId] = setTimeout(async () => {
    try {
      await patchJSON(`/api/projects/${state.project.id}/segments/${segmentId}`, patch);
    } catch (err) { toast('Could not save: ' + err.message, true); }
  }, 500);
}

/* ---------------- cast + exports ---------------- */
function renderCast() {
  const box = $('cast');
  const speakers = state.project.speakers;
  if (!speakers.length) {
    box.innerHTML = '<div class="hint">Speakers appear after diarization runs.</div>';
    return;
  }
  box.innerHTML = '';
  speakers.forEach((sp, i) => {
    const row = document.createElement('div');
    row.className = 'cast-row';
    row.innerHTML = `
      <span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:3px;
            background:${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}"></span>
      <span class="who">${sp.speaker}</span>
      <select>${state.voices.map((v) =>
        `<option value="${v.id}">${escapeHtml(v.name)}</option>`).join('')}</select>
      <span class="secs">${(sp.total_seconds || 0).toFixed(1)}s</span>
      <button class="audition">▶</button>`;
    const sel = row.querySelector('select');
    sel.value = sp.voice_id || '';
    sel.addEventListener('change', async () => {
      await postJSON(
        `/api/projects/${state.project.id}/speakers/${sp.speaker}/voice/${sel.value}`);
      state.project.segments.forEach((seg) => {
        if (seg.speaker === sp.speaker && !seg.locked) seg.voice_id = sel.value;
      });
      renderMatrix();
      toast(`${sp.speaker} re-cast — re-render to hear it`);
    });
    row.querySelector('.audition').addEventListener('click', () =>
      auditionVoice(sel.value, 'This is how this character will sound in the dub.'));
    box.appendChild(row);
  });
}

function renderExports() {
  const id = state.project.id;
  const assets = state.project.assets;
  const links = [];
  if (assets.mixed) links.push(['Dubbed audio (WAV)', `/api/projects/${id}/media/mixed`]);
  if (assets.output_video) links.push(['Dubbed video', `/api/projects/${id}/media/output_video`]);
  if (assets.dubbed) links.push(['Speech only', `/api/projects/${id}/media/dubbed`]);
  if (assets.background) links.push(['Background stem', `/api/projects/${id}/media/background`]);
  links.push(['Subtitles (SRT)', `/api/projects/${id}/export.srt`]);
  links.push(['Project JSON', `/api/projects/${id}/export.json`]);
  $('exports').innerHTML = links.map(([label, href]) =>
    `<a href="${href}" download>${label}</a>`).join('');
}

async function saveMixSettings() {
  await patchJSON(`/api/projects/${state.project.id}`, {
    settings: {
      speech_gain: Number($('mixSpeech').value),
      background_gain: Number($('mixBg').value),
      duck_depth_db: Number($('mixDuck').value),
      reverb: Number($('mixReverb').value),
      speed: Number($('mixSpeed').value),
      allow_overflow: $('mixOverflow').checked,
    },
  });
}

/* ---------------- realtime progress ---------------- */
function subscribeProject(projectId) {
  // Already listening to this project? Re-subscribing would replay the
  // broker's event history and re-trigger every completion handler.
  if (state.ws && state.wsProject === projectId &&
      (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  if (state.ws) { state.ws.onclose = null; state.ws.close(); }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/projects/${projectId}`);
  state.ws = ws;
  state.wsProject = projectId;
  ws.onmessage = (ev) => handleEvent(JSON.parse(ev.data));
  ws.onclose = () => {
    if (state.project && state.project.id === projectId && state.wsProject === projectId) {
      state.ws = null;
      setTimeout(() => subscribeProject(projectId), 2500);
    }
  };
}

function watchJob(jobId) {
  state.activeJob = jobId;
  $('progressWrap').hidden = false;
  $('btnCancel').hidden = false;
  $('progressLog').textContent = '';
  $('progressBar').style.width = '0%';
  $('progressStep').textContent = 'Queued…';
  $('progressPct').textContent = '0%';
}

async function handleEvent(ev) {
  if (ev.type === 'ping') return;
  if (ev.type === 'job_queued' || ev.type === 'job_started') {
    // The broker replays history on subscribe, so an event alone is not proof
    // the job is still running — confirm before adopting it.
    if (state.activeJob || !ev.job_id) return;
    try {
      const job = await getJSON(`/api/jobs/${ev.job_id}`);
      if (job.status === 'queued' || job.status === 'running') watchJob(ev.job_id);
    } catch (_) { /* job vanished */ }
    return;
  }
  // Terminal and progress events only matter for the job we are watching.
  // Without this, replayed 'completed' events reopen the project forever.
  if (!ev.job_id || ev.job_id !== state.activeJob) return;

  if (ev.type === 'progress') {
    $('progressWrap').hidden = false;
    $('btnCancel').hidden = false;
    $('progressBar').style.width = `${ev.progress}%`;
    $('progressPct').textContent = `${Math.round(ev.progress)}%`;
    $('progressStep').textContent = `Step ${ev.step_index}/${ev.step_total}: ${ev.message}`;
  } else if (ev.type === 'log') {
    const log = $('progressLog');
    log.textContent += ev.message + '\n';
    log.scrollTop = log.scrollHeight;
  } else if (ev.type === 'completed') {
    $('progressBar').style.width = '100%';
    $('progressPct').textContent = '100%';
    $('progressStep').textContent = 'Completed';
    $('btnCancel').hidden = true;
    state.activeJob = null;
    toast('Dub complete');
    setTimeout(() => { $('progressWrap').hidden = true; }, 2500);
    openProject(state.project.id).then(loadProjects);
  } else if (ev.type === 'failed') {
    $('progressStep').textContent = 'Failed: ' + ev.error;
    $('btnCancel').hidden = true;
    state.activeJob = null;
    toast('Job failed: ' + ev.error, true);
    loadProjects();
  } else if (ev.type === 'cancelled') {
    $('progressStep').textContent = 'Cancelled';
    $('btnCancel').hidden = true;
    state.activeJob = null;
    $('progressWrap').hidden = true;
    loadProjects();
  }
}

/* ------------------------------------------------------------------ voices */
function wireVoiceStudio() {
  $('archetypeSelect').addEventListener('change', (ev) => {
    const preset = (state.archetypes || []).find((a) => a.id === ev.target.value);
    if (!preset) return;
    $('designPrompt').value = preset.prompt;
    if (!$('designName').value) $('designName').value = preset.name;
  });

  $('btnAudition').addEventListener('click', async () => {
    const prompt = $('designPrompt').value.trim();
    if (!prompt) return toast('Describe the voice first', true);
    const btn = $('btnAudition');
    btn.disabled = true;
    try {
      const line = encodeURIComponent($('designLine').value || 'Hello.');
      const blob = await api(`/api/voices/preview-prompt?text=${line}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: $('designName').value || 'preview', prompt }),
      });
      playBlob(blob);
    } catch (err) { toast(err.message, true); }
    finally { btn.disabled = false; }
  });

  $('btnSaveVoice').addEventListener('click', async () => {
    const prompt = $('designPrompt').value.trim();
    if (!prompt) return toast('Describe the voice first', true);
    try {
      const voice = await postJSON('/api/voices/design', {
        name: $('designName').value || 'Designed voice',
        prompt, language: $('designLang').value,
      });
      $('designMsg').className = 'msg ok';
      $('designMsg').textContent = `Saved "${voice.name}".`;
      await loadVoices();
    } catch (err) {
      $('designMsg').className = 'msg error';
      $('designMsg').textContent = err.message;
    }
  });

  setupFilePicker('cloneZone', 'cloneFile', 'cloneTitle');
  setupFilePicker('convZone', 'convFile', 'convTitle');

  $('btnClone').addEventListener('click', async () => {
    const file = $('cloneFile').files[0];
    if (!file) return toast('Choose a reference recording', true);
    const body = new FormData();
    body.append('file', file);
    body.append('name', $('cloneName').value || file.name.replace(/\.[^.]+$/, ''));
    body.append('kind', $('cloneKind').value);
    const btn = $('btnClone');
    btn.disabled = true; btn.textContent = 'Cloning…';
    try {
      const voice = await api('/api/voices/clone', { method: 'POST', body });
      const report = voice.report || {};
      $('cloneMsg').className = 'msg ok';
      $('cloneMsg').textContent =
        `Created "${voice.name}" — ${report.duration}s, ${report.snr_db} dB SNR, ` +
        `quality: ${report.quality}. ${(report.warnings || []).join(' ')}`;
      await loadVoices();
    } catch (err) {
      $('cloneMsg').className = 'msg error';
      $('cloneMsg').textContent = err.message;
    } finally { btn.disabled = false; btn.textContent = 'Create clone'; }
  });

  $('btnConvert').addEventListener('click', async () => {
    const file = $('convFile').files[0];
    if (!file) return toast('Choose audio to convert', true);
    const body = new FormData();
    body.append('file', file);
    body.append('strength', $('convStrength').value);
    const btn = $('btnConvert');
    btn.disabled = true; btn.textContent = 'Converting…';
    try {
      const blob = await api(`/api/voices/${$('convVoice').value}/convert`,
                             { method: 'POST', body });
      playBlob(blob);
      $('convMsg').className = 'msg ok';
      $('convMsg').textContent = 'Converted — playing now.';
    } catch (err) {
      $('convMsg').className = 'msg error';
      $('convMsg').textContent = err.message;
    } finally { btn.disabled = false; btn.textContent = 'Convert'; }
  });

  $('voiceSearch').addEventListener('input', () => renderVoiceGrid($('voiceSearch').value));
}

function setupFilePicker(zoneId, inputId, titleId) {
  const zone = $(zoneId);
  const input = $(inputId);
  zone.addEventListener('click', () => input.click());
  ['dragenter', 'dragover'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((e) => zone.addEventListener(e, () => zone.classList.remove('drag')));
  zone.addEventListener('drop', (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) {
      input.files = ev.dataTransfer.files;
      $(titleId).textContent = input.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    if (input.files[0]) $(titleId).textContent = input.files[0].name;
  });
}

async function loadArchetypes() {
  const data = await getJSON('/api/voices/archetypes');
  state.system.emotions = data.emotions;
  state.archetypes = Object.values(data.categories).flat();
  $('archetypeSelect').innerHTML = '<option value="">— start from a template —</option>' +
    Object.entries(data.categories).map(([cat, items]) =>
      `<optgroup label="${cat}">` +
      items.map((i) => `<option value="${i.id}">${escapeHtml(i.name)}</option>`).join('') +
      '</optgroup>').join('');
  $('promptHint').innerHTML = 'Structure your prompt as: ' +
    data.prompt_structure.map((s) => `<br>· ${escapeHtml(s)}`).join('');
}

async function loadVoices() {
  state.voices = await getJSON('/api/voices');
  renderVoiceGrid($('voiceSearch').value);
  $('convVoice').innerHTML = state.voices.map((v) =>
    `<option value="${v.id}">${escapeHtml(v.name)}</option>`).join('');
  if (state.project) { renderCast(); renderMatrix(); }
}

function renderVoiceGrid(filter) {
  const q = (filter || '').toLowerCase();
  const voices = state.voices.filter((v) => !q ||
    v.name.toLowerCase().includes(q) || (v.prompt || '').toLowerCase().includes(q) ||
    (v.tags || []).join(' ').toLowerCase().includes(q));
  const grid = $('voiceGrid');
  if (!voices.length) { grid.innerHTML = '<div class="hint">No matching voices.</div>'; return; }

  const emotions = state.system.emotions || ['neutral'];
  grid.innerHTML = '';
  voices.forEach((v) => {
    const card = document.createElement('div');
    card.className = 'voice-card';
    card.innerHTML = `
      <div class="vc-head">
        <span class="vc-name"></span>
        <span class="pill ${v.kind === 'preset' ? '' : 'ready'}">${v.kind}</span>
      </div>
      <div class="vc-prompt"></div>
      <div class="vc-stats">f0 ${Math.round(v.params.f0)}Hz · rasp ${v.params.rasp.toFixed(2)} ·
        pace ${v.params.speed.toFixed(2)}×${v.training_seconds ? ` · ${v.training_seconds.toFixed(0)}s ref` : ''}</div>
      <div class="vc-actions">
        <select class="vc-emotion">${emotions.map((e) =>
          `<option value="${e}">${e}</option>`).join('')}</select>
        <button class="vc-play">▶</button>
        ${v.owner === 'system' ? '' : '<button class="vc-del">✕</button>'}
      </div>`;
    card.querySelector('.vc-name').textContent = v.name;
    card.querySelector('.vc-prompt').textContent = v.prompt || '—';
    card.querySelector('.vc-play').addEventListener('click', () =>
      auditionVoice(v.id, null, card.querySelector('.vc-emotion').value));
    const del = card.querySelector('.vc-del');
    if (del) del.addEventListener('click', async () => {
      if (!confirm(`Delete voice "${v.name}"?`)) return;
      try {
        await api(`/api/voices/${v.id}`, { method: 'DELETE' });
        await loadVoices();
        toast('Voice deleted');
      } catch (err) { toast(err.message, true); }
    });
    grid.appendChild(card);
  });
}

async function auditionVoice(voiceId, text, emotion) {
  if (!voiceId) return toast('No voice selected', true);
  try {
    const blob = await postJSON(`/api/voices/${voiceId}/preview`, {
      text: text || "Is that all you've got? You'll have to do better than that.",
      emotion: emotion || 'neutral',
    });
    playBlob(blob);
  } catch (err) { toast(err.message, true); }
}

function playBlob(blob) {
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.addEventListener('ended', () => URL.revokeObjectURL(url));
  audio.play().catch((err) => toast('Playback blocked: ' + err.message, true));
}

/* ------------------------------------------------------------------ system */
function renderSystem() {
  const s = state.system;
  const caps = Object.entries(s.capabilities).map(([k, v]) =>
    `<div class="cap"><div class="k">${k.replace(/_/g, ' ')}</div>
     <div class="v" style="color:${v ? 'var(--accent-2)' : 'var(--muted)'}">
     ${v ? 'available' : 'not installed'}</div></div>`).join('');

  const tables = Object.entries(s.providers).map(([capability, entries]) => `
    <h2>${capability.toUpperCase()}</h2>
    <table><thead><tr><th>Provider</th><th>Status</th><th>Requires</th><th>Description</th></tr></thead>
    <tbody>${Object.entries(entries).map(([name, info]) => `
      <tr>
        <td>${name}${s.active_providers[capability] === name ? ' ←  active' : ''}</td>
        <td class="${info.available ? 'on' : 'off'}">${info.available ? 'ready' : 'unavailable'}</td>
        <td class="off">${escapeHtml(info.requires || '—')}</td>
        <td class="off">${escapeHtml(info.description || '')}</td>
      </tr>`).join('')}</tbody></table>`).join('');

  $('systemInfo').innerHTML =
    `<div class="cap-grid">${caps}</div>
     <div class="hint">Sample rate ${s.sample_rate} Hz · ${s.workers} pipeline worker(s) ·
      ffmpeg ${s.ffmpeg ? 'detected' : 'not found (WAV only)'}</div>` + tables;
}

function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

init();
