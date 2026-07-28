/* 3D companion renderer.
 *
 * Raw WebGL — no three.js, no CDN, nothing to install. The body, head and hair
 * are procedural toon-shaded meshes; the face is drawn every frame into a 2D
 * canvas and uploaded as a texture. That split is deliberate: crisp anime eyes
 * and mouths are far easier to draw in 2D than to model, and it makes visemes
 * and expressions a drawing problem rather than a rigging problem.
 */
'use strict';

const Waifu = (() => {

/* ------------------------------------------------------------------ math */
const M4 = {
  identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  multiply(a, b) {
    const o = new Float32Array(16);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + c] * b[r * 4 + k];
      o[r * 4 + c] = s;
    }
    return o;
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[11] = -1;
    o[10] = (far + near) / (near - far);
    o[14] = (2 * far * near) / (near - far);
    return o;
  },
  translation(x, y, z) { const o = M4.identity(); o[12] = x; o[13] = y; o[14] = z; return o; },
  scaling(x, y, z) { const o = M4.identity(); o[0] = x; o[5] = y; o[10] = z; return o; },
  rotationX(a) { const c = Math.cos(a), s = Math.sin(a); const o = M4.identity();
    o[5] = c; o[6] = s; o[9] = -s; o[10] = c; return o; },
  rotationY(a) { const c = Math.cos(a), s = Math.sin(a); const o = M4.identity();
    o[0] = c; o[2] = -s; o[8] = s; o[10] = c; return o; },
  rotationZ(a) { const c = Math.cos(a), s = Math.sin(a); const o = M4.identity();
    o[0] = c; o[1] = s; o[4] = -s; o[5] = c; return o; },
  lookAt(eye, target, up) {
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
      -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
      -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]), 1,
    ]);
  },
  normalMatrix(m) {   // inverse-transpose of the upper 3x3, as a mat3
    const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6],
          g = m[8], h = m[9], i = m[10];
    const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g) || 1e-6;
    const inv = [
      (e*i - f*h)/det, (c*h - b*i)/det, (b*f - c*e)/det,
      (f*g - d*i)/det, (a*i - c*g)/det, (c*d - a*f)/det,
      (d*h - e*g)/det, (b*g - a*h)/det, (a*e - b*d)/det,
    ];
    // transpose of the inverse
    return new Float32Array([inv[0], inv[3], inv[6], inv[1], inv[4], inv[7],
                             inv[2], inv[5], inv[8]]);
  },
};
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0]/l, v[1]/l, v[2]/l]; };
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '#ffffff');
  return m ? [parseInt(m[1], 16)/255, parseInt(m[2], 16)/255, parseInt(m[3], 16)/255] : [1, 1, 1];
}

/* ------------------------------------------------------------- geometry */
/** Sphere with the texture seam at the BACK, so u=0.5 is dead centre of the
 *  face. Getting this wrong puts the drawn face on the side of the head.
 *  `faceOpening` skips quads over the face, turning the same mesh into a
 *  hair shell that frames the features instead of burying them. */
function sphere(rows, cols, rx, ry, rz, opts = {}) {
  const positions = [], normals = [], uvs = [], indices = [];
  const flatten = opts.flattenBack || 0;
  const taper = opts.taper || 0;
  const inside = [];
  for (let i = 0; i <= rows; i++) {
    const v = i / rows, phi = v * Math.PI;
    for (let j = 0; j <= cols; j++) {
      const u = j / cols, theta = (u - 0.5) * Math.PI * 2;
      const x = Math.sin(phi) * Math.sin(theta);
      const y = Math.cos(phi);
      const z = Math.sin(phi) * Math.cos(theta);
      const back = clamp(-z, 0, 1);
      const s = 1 - taper * clamp(-y, 0, 1);
      positions.push(x * rx * s, y * ry, z * rz * (1 - flatten * back));
      normals.push(x, y, z);
      uvs.push(u, v);
      inside.push(opts.faceOpening ? (z > opts.faceOpening.z && y < opts.faceOpening.y) : false);
    }
  }
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const a = i * (cols + 1) + j, b = a + cols + 1;
      if (inside[a] || inside[b] || inside[a + 1] || inside[b + 1]) continue;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { positions, normals, uvs, indices };
}

/** Tapered capsule along Y — torso, limbs, twintails. */
function capsule(segments, rings, radiusTop, radiusBottom, height) {
  const positions = [], normals = [], uvs = [], indices = [];
  for (let i = 0; i <= rings; i++) {
    const v = i / rings;
    const r = lerp(radiusTop, radiusBottom, v);
    const y = height * (0.5 - v);
    for (let j = 0; j <= segments; j++) {
      const u = j / segments, theta = u * Math.PI * 2;
      const nx = Math.sin(theta), nz = Math.cos(theta);
      positions.push(nx * r, y, nz * r);
      normals.push(nx, (radiusBottom - radiusTop) / Math.max(height, 1e-3), nz);
      uvs.push(u, v);
    }
  }
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j, b = a + segments + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  // caps
  const capCentre = (y, r, dir) => {
    const base = positions.length / 3;
    positions.push(0, y, 0); normals.push(0, dir, 0); uvs.push(0.5, 0.5);
    for (let j = 0; j <= segments; j++) {
      const theta = (j / segments) * Math.PI * 2;
      positions.push(Math.sin(theta) * r, y, Math.cos(theta) * r);
      normals.push(0, dir, 0); uvs.push(0.5, 0.5);
    }
    for (let j = 0; j < segments; j++) {
      if (dir > 0) indices.push(base, base + j + 1, base + j + 2);
      else indices.push(base, base + j + 2, base + j + 1);
    }
  };
  capCentre(height / 2, radiusTop, 1);
  capCentre(-height / 2, radiusBottom, -1);
  return { positions, normals, uvs, indices };
}

/* --------------------------------------------------------------- shaders */
const VERT = `
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec2 aUV;
uniform mat4 uProj, uView, uModel;
uniform mat3 uNormalMat;
varying vec3 vNormal, vWorld;
varying vec2 vUV;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorld = world.xyz;
  vNormal = normalize(uNormalMat * aNormal);
  vUV = aUV;
  gl_Position = uProj * uView * world;
}`;

const FRAG = `
precision mediump float;
varying vec3 vNormal, vWorld;
varying vec2 vUV;
uniform vec3 uColor, uShadow, uRim, uCamera;
uniform sampler2D uFace;
uniform float uUseFace;
void main() {
  vec3 n = normalize(vNormal);
  vec3 lightDir = normalize(vec3(-0.45, 0.75, 0.85));
  float ndl = dot(n, lightDir);

  // toon banding: two hard steps with a soft edge, the anime-cel look
  float band = smoothstep(0.02, 0.10, ndl) * 0.55 + smoothstep(0.45, 0.55, ndl) * 0.45;
  vec3 base = uColor;

  if (uUseFace > 0.5) {
    vec4 face = texture2D(uFace, vUV);
    base = mix(base, face.rgb, face.a);
  }

  vec3 lit = mix(uShadow * base, base, band);

  vec3 viewDir = normalize(uCamera - vWorld);
  float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.6);
  lit += uRim * rim * 0.55;

  gl_FragColor = vec4(lit, 1.0);
}`;

/* ------------------------------------------------------------ face canvas */
/** Draws eyes, brows, mouth and blush into a 2D canvas used as a texture. */
class Face {
  constructor(size = 1024) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = size;
    this.ctx = this.canvas.getContext('2d');
    this.size = size;
  }

  draw(look) {
    const { ctx, size: S } = this;
    ctx.clearRect(0, 0, S, S);

    // The head sphere is built with its seam at the back, so u=0.5 is the
    // centre of the face and v runs top (0) to bottom (1).
    const cx = S * 0.5, cy = S * 0.535;
    const w = S * 0.066, h = S * 0.058;

    ctx.save();
    ctx.translate(cx, cy);

    this._blush(ctx, w, h, look);
    this._eye(ctx, -w * 1.0, 0, w, h, look, -1);
    this._eye(ctx, w * 1.0, 0, w, h, look, 1);
    this._brow(ctx, -w * 1.0, -h * 2.2, w, look, -1);
    this._brow(ctx, w * 1.0, -h * 2.2, w, look, 1);
    this._mouth(ctx, 0, h * 2.05, w, look);
    ctx.restore();
    return this.canvas;
  }

  _blush(ctx, w, h, look) {
    const amount = look.blush;
    if (amount <= 0.01) return;
    for (const sx of [-1, 1]) {
      const g = ctx.createRadialGradient(sx * w * 1.75, h * 1.35, 0, sx * w * 1.75, h * 1.35, w * 0.95);
      g.addColorStop(0, `rgba(255,120,140,${0.55 * amount})`);
      g.addColorStop(1, 'rgba(255,120,140,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(sx * w * 1.75, h * 1.35, w * 0.95, h * 0.58, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _eye(ctx, x, y, w, h, look, side) {
    const open = look.eyeOpen;
    const rx = w * 0.82 * look.eyeSize;
    const ry = h * 1.05 * look.eyeSize * Math.max(0.04, open);

    ctx.save();
    ctx.translate(x, y);

    // sclera
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();

    if (open > 0.12) {
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.clip();

      const px = look.gaze[0] * rx * 0.32;
      const py = look.gaze[1] * ry * 0.3;
      const irisR = rx * 0.74;

      // iris with a vertical gradient — the classic anime "deep eye"
      const g = ctx.createLinearGradient(px, py - irisR, px, py + irisR);
      g.addColorStop(0, look.eyeColorDark);
      g.addColorStop(0.55, look.eyeColor);
      g.addColorStop(1, look.eyeColorLight);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(px, py, irisR * 0.92, irisR, 0, 0, Math.PI * 2);
      ctx.fill();

      // pupil
      ctx.fillStyle = 'rgba(20,16,30,0.92)';
      ctx.beginPath();
      ctx.ellipse(px, py, irisR * 0.42, irisR * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();

      // highlights
      ctx.fillStyle = 'rgba(255,255,255,0.95)';
      ctx.beginPath();
      ctx.ellipse(px - irisR * 0.36, py - irisR * 0.42, irisR * 0.3, irisR * 0.34, -0.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.beginPath();
      ctx.ellipse(px + irisR * 0.3, py + irisR * 0.4, irisR * 0.16, irisR * 0.18, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // upper lash line — thick on top, thin below, anime style
    ctx.strokeStyle = '#241d2e';
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(2, rx * 0.20);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, Math.PI * 1.08, Math.PI * 1.92);
    ctx.stroke();
    ctx.lineWidth = Math.max(1, rx * 0.07);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * 0.96, ry * 0.98, 0, Math.PI * 0.2, Math.PI * 0.8);
    ctx.stroke();

    // outer lash flick
    ctx.lineWidth = Math.max(2, rx * 0.16);
    ctx.beginPath();
    ctx.moveTo(side * rx * 0.86, -ry * 0.52);
    ctx.lineTo(side * rx * 1.22, -ry * 0.95);
    ctx.stroke();
    ctx.restore();
  }

  _brow(ctx, x, y, w, look, side) {
    ctx.save();
    ctx.translate(x, y + look.browY * w * 0.6);
    ctx.rotate(side * look.browAngle);
    ctx.strokeStyle = look.browColor;
    ctx.lineCap = 'round';
    ctx.lineWidth = w * 0.19;
    ctx.beginPath();
    ctx.moveTo(-w * 0.62, w * 0.10);
    ctx.quadraticCurveTo(0, -w * 0.16, w * 0.66, w * 0.02);
    ctx.stroke();
    ctx.restore();
  }

  _mouth(ctx, x, y, w, look) {
    const open = look.mouthOpen;      // 0..1 vertical opening
    const wide = look.mouthWide;      // 0..1 horizontal spread
    const curve = look.mouthCurve;    // -1 frown .. +1 smile
    const mw = w * (0.42 + 0.62 * wide);
    const mh = w * 1.15 * open;

    ctx.save();
    ctx.translate(x, y);

    if (mh < w * 0.05) {
      // closed: a single expressive line
      ctx.strokeStyle = '#8d4a53';
      ctx.lineCap = 'round';
      ctx.lineWidth = w * 0.10;
      ctx.beginPath();
      ctx.moveTo(-mw * 0.55, 0);
      ctx.quadraticCurveTo(0, curve * w * 0.34, mw * 0.55, 0);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // open: mouth interior, tongue, and a bright lower lip edge
    ctx.fillStyle = '#7a2d3c';
    ctx.beginPath();
    ctx.moveTo(-mw, -curve * w * 0.1);
    ctx.quadraticCurveTo(0, -mh * 0.75 - curve * w * 0.3, mw, -curve * w * 0.1);
    ctx.quadraticCurveTo(0, mh * 1.25, -mw, -curve * w * 0.1);
    ctx.fill();

    ctx.fillStyle = 'rgba(232,110,130,0.85)';
    ctx.beginPath();
    ctx.ellipse(0, mh * 0.42, mw * 0.6, mh * 0.36, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255,190,190,0.5)';
    ctx.lineWidth = w * 0.05;
    ctx.beginPath();
    ctx.moveTo(-mw * 0.85, mh * 0.1);
    ctx.quadraticCurveTo(0, mh * 1.15, mw * 0.85, mh * 0.1);
    ctx.stroke();
    ctx.restore();
  }
}

/* ------------------------------------------------------- expression table */
// Each expression is a target for the face parameters the animator lerps toward.
const EXPRESSIONS = {
  neutral:   { browY: 0.0,  browAngle: 0.00, curve: 0.15, blushBoost: 0.0, eyeOpen: 1.0, sway: 1.0 },
  happy:     { browY: -0.1, browAngle: -0.05, curve: 0.85, blushBoost: 0.1, eyeOpen: 0.9, sway: 1.2 },
  excited:   { browY: -0.25, browAngle: -0.10, curve: 1.0, blushBoost: 0.2, eyeOpen: 1.15, sway: 1.9 },
  laughing:  { browY: -0.2, browAngle: -0.12, curve: 1.0, blushBoost: 0.25, eyeOpen: 0.28, sway: 2.1 },
  sad:       { browY: 0.28, browAngle: 0.22, curve: -0.7, blushBoost: 0.0, eyeOpen: 0.78, sway: 0.5 },
  crying:    { browY: 0.34, browAngle: 0.3,  curve: -0.9, blushBoost: 0.3, eyeOpen: 0.5, sway: 0.6 },
  angry:     { browY: 0.3,  browAngle: -0.34, curve: -0.5, blushBoost: 0.15, eyeOpen: 1.1, sway: 1.5 },
  shouting:  { browY: 0.34, browAngle: -0.4, curve: -0.2, blushBoost: 0.2, eyeOpen: 1.2, sway: 2.2 },
  flustered: { browY: 0.2,  browAngle: 0.14, curve: -0.15, blushBoost: 0.55, eyeOpen: 1.05, sway: 1.6 },
  whisper:   { browY: 0.06, browAngle: 0.05, curve: 0.1, blushBoost: 0.1, eyeOpen: 0.72, sway: 0.4 },
  sarcastic: { browY: -0.16, browAngle: -0.2, curve: 0.5, blushBoost: 0.0, eyeOpen: 0.8, sway: 0.8 },
  calm:      { browY: 0.04, browAngle: 0.02, curve: 0.3, blushBoost: 0.0, eyeOpen: 0.88, sway: 0.7 },
  menacing:  { browY: 0.22, browAngle: -0.3, curve: -0.35, blushBoost: 0.0, eyeOpen: 0.72, sway: 0.6 },
  fearful:   { browY: 0.3,  browAngle: 0.26, curve: -0.6, blushBoost: 0.1, eyeOpen: 1.2, sway: 2.4 },
};

// Viseme → (openness, width, lip rounding)
const VISEME_SHAPE = {
  rest: [0.00, 0.30, 0.0], A: [1.00, 0.62, 0.0], E: [0.55, 0.80, 0.0],
  I:    [0.32, 0.92, 0.0], O: [0.78, 0.22, 1.0], U: [0.42, 0.12, 1.0],
  MBP:  [0.00, 0.34, 0.0], FV: [0.18, 0.62, 0.0], TH: [0.30, 0.55, 0.0],
  SS:   [0.16, 0.78, 0.0], L:  [0.38, 0.55, 0.0], WQ: [0.34, 0.18, 1.0],
};

/* --------------------------------------------------------------- renderer */
class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl', { antialias: true, alpha: true });
    if (!gl) throw new Error('WebGL is not available in this browser');
    this.gl = gl;

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    this.program = this._program(VERT, FRAG);
    this.loc = {
      aPos: gl.getAttribLocation(this.program, 'aPos'),
      aNormal: gl.getAttribLocation(this.program, 'aNormal'),
      aUV: gl.getAttribLocation(this.program, 'aUV'),
      uProj: gl.getUniformLocation(this.program, 'uProj'),
      uView: gl.getUniformLocation(this.program, 'uView'),
      uModel: gl.getUniformLocation(this.program, 'uModel'),
      uNormalMat: gl.getUniformLocation(this.program, 'uNormalMat'),
      uColor: gl.getUniformLocation(this.program, 'uColor'),
      uShadow: gl.getUniformLocation(this.program, 'uShadow'),
      uRim: gl.getUniformLocation(this.program, 'uRim'),
      uCamera: gl.getUniformLocation(this.program, 'uCamera'),
      uFace: gl.getUniformLocation(this.program, 'uFace'),
      uUseFace: gl.getUniformLocation(this.program, 'uUseFace'),
    };

    this.meshes = {
      head: this._mesh(sphere(44, 56, 0.30, 0.34, 0.29, { flattenBack: 0.08 })),
      // same sphere, slightly larger, with the face region removed
      hairShell: this._mesh(sphere(44, 56, 0.325, 0.362, 0.315, {
        flattenBack: 0.05, faceOpening: { z: 0.30, y: 0.34 },
      })),
      torso: this._mesh(capsule(28, 12, 0.17, 0.21, 0.52)),
      neck: this._mesh(capsule(16, 4, 0.062, 0.075, 0.12)),
      limb: this._mesh(capsule(16, 6, 0.055, 0.042, 0.44)),
      tail: this._mesh(capsule(18, 12, 0.075, 0.028, 0.5)),
      bang: this._mesh(capsule(12, 6, 0.055, 0.075, 0.2)),
      ground: this._mesh(sphere(16, 24, 1.2, 0.015, 0.9)),
    };

    this.faceCanvas = new Face(1024);
    this.faceTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.faceTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  _program(vs, fs) {
    const gl = this.gl;
    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('shader: ' + gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(p));
    }
    return p;
  }

  _mesh(geo) {
    const gl = this.gl;
    const buf = (data, Type, target) => {
      const b = gl.createBuffer();
      gl.bindBuffer(target, b);
      gl.bufferData(target, new Type(data), gl.STATIC_DRAW);
      return b;
    };
    return {
      pos: buf(geo.positions, Float32Array, gl.ARRAY_BUFFER),
      nrm: buf(geo.normals, Float32Array, gl.ARRAY_BUFFER),
      uv: buf(geo.uvs, Float32Array, gl.ARRAY_BUFFER),
      idx: buf(geo.indices, Uint16Array, gl.ELEMENT_ARRAY_BUFFER),
      count: geo.indices.length,
    };
  }

  _bind(mesh) {
    const gl = this.gl, l = this.loc;
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.pos);
    gl.enableVertexAttribArray(l.aPos);
    gl.vertexAttribPointer(l.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.nrm);
    gl.enableVertexAttribArray(l.aNormal);
    gl.vertexAttribPointer(l.aNormal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.uv);
    gl.enableVertexAttribArray(l.aUV);
    gl.vertexAttribPointer(l.aUV, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idx);
  }

  drawMesh(mesh, model, color, opts = {}) {
    const gl = this.gl, l = this.loc;
    this._bind(mesh);
    // Open shells (the hair, which has the face cut out of it) need their
    // inner surface drawn too, otherwise you see straight through the rim.
    if (opts.doubleSided) gl.disable(gl.CULL_FACE);
    gl.uniformMatrix4fv(l.uModel, false, model);
    gl.uniformMatrix3fv(l.uNormalMat, false, M4.normalMatrix(model));
    gl.uniform3fv(l.uColor, color);
    const shadow = opts.shadow || [color[0] * 0.68, color[1] * 0.66, color[2] * 0.74];
    gl.uniform3fv(l.uShadow, shadow);
    gl.uniform3fv(l.uRim, opts.rim || [0.55, 0.6, 0.85]);
    gl.uniform1f(l.uUseFace, opts.face ? 1 : 0);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    if (opts.doubleSided) gl.enable(gl.CULL_FACE);
  }

  uploadFace(look) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.faceTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
                  this.faceCanvas.draw(look));
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, this.canvas.clientWidth), h = Math.max(1, this.canvas.clientHeight);
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    return w / h;
  }
}

/* --------------------------------------------------------------- companion */
class Companion {
  constructor(canvas) {
    this.renderer = new Renderer(canvas);
    this.canvas = canvas;
    this.appearance = {
      hair_style: 'twintails', outfit_style: 'dress', hair_color: '#f2739d', eye_color: '#4fc3f7',
      skin_color: '#ffe0d0', outfit_color: '#3d4b7a', accent_color: '#ffffff',
      eye_size: 1.0, blush: 0.35, height: 1.0,
    };
    this.expression = 'neutral';
    this.visemes = null;
    this.speakStart = 0;
    this.audio = null;
    this.pointer = { x: 0, y: 0 };
    this.blink = { next: 1.5, t: 0, closing: 0 };
    this.state = {
      mouthOpen: 0, mouthWide: 0.3, mouthCurve: 0.15, browY: 0, browAngle: 0,
      eyeOpen: 1, blush: 0.35, sway: 1, gaze: [0, 0],
    };
    this.time = 0;
    this.running = false;

    canvas.addEventListener('pointermove', (ev) => {
      const r = canvas.getBoundingClientRect();
      this.pointer.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
      this.pointer.y = ((ev.clientY - r.top) / r.height) * 2 - 1;
    });
    canvas.addEventListener('pointerleave', () => { this.pointer.x = 0; this.pointer.y = 0; });
  }

  setAppearance(appearance) { Object.assign(this.appearance, appearance || {}); }
  setExpression(name) { if (EXPRESSIONS[name]) this.expression = name; }

  /** Play a rendered line: audio + viseme track, kept in sync by audio clock. */
  speak({ audio, visemes, emotion }) {
    this.stop();
    if (emotion) this.setExpression(emotion);
    this.visemes = visemes || null;
    const el = new Audio(audio);
    this.audio = el;
    el.addEventListener('ended', () => {
      this.visemes = null;
      this.audio = null;
      if (this.onSpeechEnd) this.onSpeechEnd();
    });
    return el.play().then(() => { this.speakStart = performance.now() / 1000; })
      .catch((err) => { this.visemes = null; this.audio = null; throw err; });
  }

  stop() {
    if (this.audio) { this.audio.pause(); this.audio = null; }
    this.visemes = null;
  }

  get speaking() { return !!this.audio && !this.audio.paused; }

  _visemeAt(t) {
    const track = this.visemes;
    if (!track || !track.length) return null;
    let i = 0;
    while (i + 1 < track.length && track[i + 1].t <= t) i++;
    const cur = track[i];
    const next = track[i + 1] || cur;
    const span = Math.max(1e-3, next.t - cur.t);
    // ease between shapes so the mouth moves rather than snapping
    const blend = clamp((t - cur.t) / span, 0, 1);
    const a = VISEME_SHAPE[cur.v] || VISEME_SHAPE.rest;
    const b = VISEME_SHAPE[next.v] || VISEME_SHAPE.rest;
    const ease = blend < 0.5 ? 2 * blend * blend : 1 - Math.pow(-2 * blend + 2, 2) / 2;
    return {
      open: lerp(a[0], b[0], ease) * lerp(cur.w || 1, next.w || 1, ease),
      wide: lerp(a[1], b[1], ease),
    };
  }

  frame(dt) {
    const s = this.state;
    const target = EXPRESSIONS[this.expression] || EXPRESSIONS.neutral;
    this.time += dt;

    // blinking — random intervals, fast close, slower open
    this.blink.t += dt;
    if (this.blink.t > this.blink.next) {
      this.blink.closing = 1;
      this.blink.t = 0;
      this.blink.next = 1.6 + Math.random() * 4.0;
    }
    if (this.blink.closing > 0) {
      this.blink.closing = Math.max(0, this.blink.closing - dt * 7.5);
    }
    const blinkFactor = 1 - Math.sin(clamp(this.blink.closing, 0, 1) * Math.PI);

    const k = clamp(dt * 12, 0, 1);
    s.browY = lerp(s.browY, target.browY, k);
    s.browAngle = lerp(s.browAngle, target.browAngle, k);
    s.blush = lerp(s.blush, clamp(this.appearance.blush + target.blushBoost, 0, 1), k);
    s.sway = lerp(s.sway, target.sway, k * 0.5);
    s.eyeOpen = lerp(s.eyeOpen, target.eyeOpen, k) * blinkFactor;

    const shape = this.speaking && this.visemes
      ? this._visemeAt(performance.now() / 1000 - this.speakStart)
      : null;
    const mouthK = clamp(dt * 26, 0, 1);   // fast: lip-sync must not lag audio
    s.mouthOpen = lerp(s.mouthOpen, shape ? clamp(shape.open, 0, 1) : 0, mouthK);
    s.mouthWide = lerp(s.mouthWide, shape ? shape.wide : 0.3, mouthK);
    s.mouthCurve = lerp(s.mouthCurve, target.curve, k);

    // eyes follow the pointer, head follows a little less
    s.gaze[0] = lerp(s.gaze[0], clamp(this.pointer.x * 1.2, -1, 1), k * 0.6);
    s.gaze[1] = lerp(s.gaze[1], clamp(this.pointer.y * 0.9, -1, 1), k * 0.6);

    this.render();
  }

  render() {
    const r = this.renderer, gl = r.gl, s = this.state, app = this.appearance;
    const aspect = r.resize();

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(r.program);

    const eye = [0, 1.12, 2.75];
    const view = M4.lookAt(eye, [0, 0.92, 0], [0, 1, 0]);
    gl.uniformMatrix4fv(r.loc.uProj, false, M4.perspective(0.62, aspect, 0.1, 50));
    gl.uniformMatrix4fv(r.loc.uView, false, view);
    gl.uniform3fv(r.loc.uCamera, eye);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, r.faceTexture);
    gl.uniform1i(r.loc.uFace, 0);

    r.uploadFace({
      eyeOpen: clamp(s.eyeOpen, 0, 1.3),
      eyeSize: app.eye_size,
      eyeColor: app.eye_color,
      eyeColorDark: shade(app.eye_color, -0.45),
      eyeColorLight: shade(app.eye_color, 0.4),
      browColor: shade(app.hair_color, -0.35),
      browY: s.browY, browAngle: s.browAngle,
      mouthOpen: s.mouthOpen, mouthWide: s.mouthWide, mouthCurve: s.mouthCurve,
      blush: s.blush, gaze: s.gaze,
    });

    const skin = hexToRgb(app.skin_color);
    const hair = hexToRgb(app.hair_color);
    const outfit = hexToRgb(app.outfit_color);
    const accent = hexToRgb(app.accent_color);
    const scale = app.height;

    // idle motion: breathing, weight shift, gentle head sway
    const breathe = Math.sin(this.time * 1.5) * 0.012 * s.sway;
    const swayX = Math.sin(this.time * 0.7) * 0.035 * s.sway;
    const bob = Math.sin(this.time * 1.5 + 0.6) * 0.018 * s.sway;
    const headYaw = s.gaze[0] * 0.34 + swayX * 0.5;
    const headPitch = -s.gaze[1] * 0.2 + Math.sin(this.time * 1.1) * 0.02 * s.sway;
    const talkBounce = this.speaking ? Math.sin(this.time * 11) * 0.008 : 0;

    const root = M4.multiply(M4.translation(swayX * 0.4, bob + talkBounce, 0),
                             M4.scaling(scale, scale, scale));

    // ---- shadow -----------------------------------------------------------
    r.drawMesh(r.meshes.ground, M4.multiply(M4.translation(0, 0.012, 0), M4.scaling(0.3, 1, 0.28)),
               [0.05, 0.05, 0.1], { shadow: [0.04, 0.04, 0.08], rim: [0, 0, 0] });

    // Layout runs feet(0) → legs(0.44) → torso(0.44-1.02) → neck → head(1.24).
    // ---- legs -------------------------------------------------------------
    for (const side of [-1, 1]) {
      r.drawMesh(r.meshes.limb,
        M4.multiply(root, M4.translation(side * 0.075, 0.22, 0)), shade3(skin, -0.04));
    }

    // ---- body -------------------------------------------------------------
    const torso = M4.multiply(root,
      M4.multiply(M4.translation(0, 0.74, 0),
                  M4.multiply(M4.rotationZ(swayX * 0.25),
                              M4.scaling(1 + breathe, 1 + breathe * 0.6, 1 + breathe))));
    r.drawMesh(r.meshes.torso, torso, outfit);

    // hips: a flared skirt, or trousers that clothe the upper legs
    if (app.outfit_style === 'trousers') {
      for (const side of [-1, 1]) {
        r.drawMesh(r.meshes.limb,
          M4.multiply(root, M4.multiply(M4.translation(side * 0.075, 0.30, 0),
                                        M4.scaling(1.25, 0.7, 1.25))),
          shade3(outfit, -0.14));
      }
    } else {
      r.drawMesh(r.meshes.torso,
        M4.multiply(root, M4.multiply(M4.translation(0, 0.52, 0), M4.scaling(1.34, 0.40, 1.34))),
        shade3(outfit, -0.14));
    }
    // collar
    r.drawMesh(r.meshes.torso,
      M4.multiply(root, M4.multiply(M4.translation(0, 0.99, 0), M4.scaling(0.86, 0.12, 0.86))),
      accent, { shadow: shade3(accent, -0.3) });

    // ---- arms -------------------------------------------------------------
    const armLift = (EXPRESSIONS[this.expression] || EXPRESSIONS.neutral).sway - 1;
    for (const side of [-1, 1]) {
      const swing = Math.sin(this.time * 0.7 + (side > 0 ? 0 : Math.PI)) * 0.10 * s.sway;
      const arm = M4.multiply(root,
        M4.multiply(M4.translation(side * 0.245, 0.74, 0.01),
          M4.multiply(M4.rotationZ(side * (0.20 + armLift * 0.16) + swing),
                      M4.scaling(0.85, 0.92, 0.85))));
      r.drawMesh(r.meshes.limb, arm, shade3(skin, 0.02));
      // sleeve
      r.drawMesh(r.meshes.limb,
        M4.multiply(arm, M4.multiply(M4.translation(0, 0.13, 0), M4.scaling(1.22, 0.5, 1.22))),
        outfit);
    }

    // ---- head -------------------------------------------------------------
    const headBase = M4.multiply(root,
      M4.multiply(M4.translation(0, 1.30, 0),
        M4.multiply(M4.rotationY(headYaw), M4.rotationX(headPitch))));

    r.drawMesh(r.meshes.neck,
      M4.multiply(root, M4.translation(0, 1.05, 0)), shade3(skin, -0.10));
    r.drawMesh(r.meshes.head, headBase, skin, { face: true, rim: [0.6, 0.55, 0.8] });

    // ---- hair -------------------------------------------------------------
    this._drawHair(headBase, hair, accent, root);
  }

  _drawHair(headBase, hair, accent, root) {
    const r = this.renderer;
    const style = this.appearance.hair_style;
    const hairShadow = shade3(hair, -0.34);
    const opts = { shadow: hairShadow, rim: [accent[0] * 0.7, accent[1] * 0.7, accent[2] * 0.9] };

    // shell — the face region is already cut out of this mesh
    r.drawMesh(r.meshes.hairShell,
      M4.multiply(headBase, M4.translation(0, 0.012, -0.012)), hair,
      { ...opts, doubleSided: true });

    // bangs falling across the forehead
    for (let i = -2; i <= 2; i++) {
      const x = i * 0.072;
      r.drawMesh(r.meshes.bang,
        M4.multiply(headBase,
          M4.multiply(M4.translation(x, 0.20, 0.235 - Math.abs(i) * 0.022),
            M4.multiply(M4.rotationX(0.22),
              M4.multiply(M4.rotationZ(i * 0.13),
                          M4.scaling(1.0, 1.0 - Math.abs(i) * 0.08, 0.85))))),
        hair, opts);
    }

    // side locks framing the face
    for (const side of [-1, 1]) {
      r.drawMesh(r.meshes.bang,
        M4.multiply(headBase,
          M4.multiply(M4.translation(side * 0.255, 0.03, 0.075),
            M4.multiply(M4.rotationZ(side * 0.10), M4.scaling(0.75, 1.9, 0.75)))),
        hair, opts);
    }

    const tail = (tx, ty, tz, rotZ, rotX, sx, sy) =>
      r.drawMesh(r.meshes.tail,
        M4.multiply(headBase,
          M4.multiply(M4.translation(tx, ty, tz),
            M4.multiply(M4.rotationZ(rotZ),
              M4.multiply(M4.rotationX(rotX), M4.scaling(sx, sy, sx))))),
        hair, opts);

    const swing = Math.sin(this.time * 1.4) * 0.09 * this.state.sway;

    if (style === 'twintails') {
      for (const side of [-1, 1]) {
        r.drawMesh(r.meshes.bang,
          M4.multiply(headBase, M4.multiply(M4.translation(side * 0.29, 0.19, -0.02),
                                            M4.scaling(0.85, 0.6, 0.85))),
          accent, { shadow: shade3(accent, -0.3), rim: [0.4, 0.4, 0.6] });
        tail(side * 0.35, -0.06, -0.05, side * (0.40 + swing * side), 0.08, 0.95, 1.05);
      }
    } else if (style === 'ponytail') {
      r.drawMesh(r.meshes.bang,
        M4.multiply(headBase, M4.multiply(M4.translation(0, 0.22, -0.26), M4.scaling(0.9, 0.6, 0.9))),
        accent, { shadow: shade3(accent, -0.3), rim: [0.4, 0.4, 0.6] });
      tail(0, -0.10, -0.30, swing * 0.6, -0.36, 1.05, 1.3);
    } else if (style === 'long') {
      tail(0, -0.30, -0.11, swing * 0.3, 0.03, 1.7, 1.5);
      for (const side of [-1, 1]) tail(side * 0.20, -0.26, -0.06, side * 0.10, 0.02, 0.85, 1.15);
    } else if (style === 'bob') {
      tail(0, -0.13, -0.09, 0, 0.02, 1.75, 0.62);
    } else { // short
      tail(0, -0.03, -0.10, 0, 0.05, 1.5, 0.34);
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    let last = performance.now();
    const loop = (now) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (this.canvas.clientWidth > 0 && this.canvas.offsetParent !== null) this.frame(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  destroy() { this.running = false; this.stop(); }
}

function shade(hex, amount) {
  const [r, g, b] = hexToRgb(hex);
  const f = (v) => Math.round(clamp(amount >= 0 ? v + (1 - v) * amount : v * (1 + amount), 0, 1) * 255);
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
function shade3(rgb, amount) {
  return rgb.map((v) => clamp(amount >= 0 ? v + (1 - v) * amount : v * (1 + amount), 0, 1));
}

return { Companion, EXPRESSIONS, VISEME_SHAPE };
})();
