/* Diagrams of the Beckmann-map tutorial (projects/sbm/tutorial/). Built on the five-atom toy of btm_anim.js (window.BTMToy):
   the exact flow-matching field b_t, the exact autonomous current b = J / nu on the shipped grid (assets/btm_fields.js) and its
   basins. Diagrams, in page order:
     1 "pairs"    the interpolant I_t = (1 - t) x0 + t x1 of random pairs, slider t, re-pair button;
     2 "field"    the flow-matching field b_t(x) on a lattice, slider t, hover shows the posterior mean z_t(x);
     3 "auto"     the autonomous field: streamlines, particles flowing, hover follows one flow line, basins toggle;
     4 "map"      the basins are the map: hover / tap a point, see T(x) in one call and the flow line it stands for;
     5 "compare"  the same 300 noise draws: flow matching with N Euler steps (left) against the map (right).
   Everything is computed from the exact fields; nothing is trained in the browser. */
(function (global) {
  "use strict";
  const Toy = global.BTMToy;
  if (!Toy) return;
  const { ATOMS, bFM, zhat, FG, stepDir, CLK, BOX, BAS, arrivedAt, loadFields, sepsOf } = Toy;
  const J = ATOMS.length, TAU = 2 * Math.PI;
  const CREAM = "#FBF6E9", INK = "#2F2A24", MUTED = "#5A6670", LINE = "#B8AA95", CHARCOAL = "#4A4A4A", GREY = "#8F8E88";

  function hex2rgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  function mix(c, k, base) { const a = hex2rgb(c), b = hex2rgb(base); return a.map((v, i) => Math.round(k * v + (1 - k) * b[i])); }
  function rgba(c, a) { const [r, g, b] = typeof c === "string" ? hex2rgb(c) : c; return `rgba(${r},${g},${b},${a})`; }
  const TINTS = ATOMS.map(a => mix(a.c, 0.26, CREAM));

  // ---------- draws: the same 300 noise points everywhere, data draws by the atom weights ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const rng = mulberry32(205);
  function gauss() { const r = Math.sqrt(-2 * Math.log(1 - rng())), th = TAU * rng(); return [r * Math.cos(th), r * Math.sin(th)]; }
  function drawAtom() { let u = rng(), k = 0; while (k < J - 1 && u > ATOMS[k].p) { u -= ATOMS[k].p; k++; } return k; }
  const N = 300, X0 = [], K1 = [];
  for (let i = 0; i < N; i++) { X0.push(gauss()); K1.push(drawAtom()); }
  function rePair() { for (let i = 0; i < N; i++) K1[i] = drawAtom(); }

  function nearestAtom(x, y) { let b = 0, d = Infinity; for (let k = 0; k < J; k++) { const e = Math.hypot(x - ATOMS[k].z[0], y - ATOMS[k].z[1]); if (e < d) { d = e; b = k; } } return b; }
  function distToAtoms(x, y) { let d = Infinity; for (const a of ATOMS) d = Math.min(d, Math.hypot(x - a.z[0], y - a.z[1])); return d; }

  // ---------- flow matching with N Euler steps: positions after every step, from the exact b_t ----------
  function eulerPath(x0, n) {
    const pts = [[x0[0], x0[1]]]; let x = x0[0], y = x0[1];
    for (let k = 0; k < n; k++) { const b = bFM(x, y, k / n); x += b[0] / n; y += b[1] / n; pts.push([x, y]); }
    return pts;
  }
  // ---------- the map: the endpoint of the exact autonomous flow line through x ----------
  function mapDest(x, y) {   // atom index; needs the field grid
    const p = { x, y };
    for (let k = 0; k < 4000; k++) { stepDir(p, FG.field, 0.02, CLK.auto.vmin, CLK.auto.vmax); const a = arrivedAt(p.x, p.y); if (a >= 0) return a; }
    return nearestAtom(p.x, p.y);
  }
  function flowLine(x, y, forward) {   // the flow line through (x, y) in one direction, as a polyline (forward stops at an atom)
    const pts = [[x, y]], p = { x, y }, f = forward ? FG.field : (u, v) => { const b = FG.field(u, v); return [-b[0], -b[1]]; };
    for (let k = 0; k < 3000; k++) {
      stepDir(p, f, 0.015, CLK.auto.vmin, CLK.auto.vmax); pts.push([p.x, p.y]);
      if (forward && arrivedAt(p.x, p.y) >= 0) break;
      if (!forward && (Math.abs(p.x) > BOX + 0.3 || Math.abs(p.y) > BOX + 0.3)) break;
    }
    return pts;
  }
  function streamlines() {
    const seeds = [];
    for (let k = 0; k < 13; k++) { const u = -2.35 + 4.7 * k / 12; seeds.push([u, BOX - 0.02], [u, -BOX + 0.02]); }
    for (let k = 0; k < 9; k++) { const v = -2.0 + 4.0 * k / 8; seeds.push([-BOX + 0.02, v], [BOX - 0.02, v]); }
    return seeds.map(s => flowLine(s[0], s[1], true));
  }
  function meanDist(pts) { let s = 0; for (const p of pts) s += distToAtoms(p[0], p[1]); return s / pts.length; }
  function masses(dests) { const m = new Array(J).fill(0); for (const d of dests) m[d] += 1 / dests.length; return m; }

  global.SBMTutorial = { X0, K1, eulerPath, mapDest, meanDist, masses, nearestAtom, loadFields };
  if (typeof document === "undefined") return;

  // ---------- canvas stage: two layers (bg static, fg animated), >= 2 device pixels per CSS pixel ----------
  class Stage {
    constructor(el) {
      this.el = el; this.bg = el.querySelector("canvas.bg"); this.fg = el.querySelector("canvas.fg"); this.resize();
    }
    resize() {
      const dpr = Math.min(Math.max(global.devicePixelRatio || 1, 2), 3), w = this.el.clientWidth, h = this.el.clientHeight;
      if (!w || !h) return false;
      for (const c of [this.bg, this.fg]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      this.dpr = dpr; this.W = this.bg.width; this.H = this.bg.height; this.sx = this.W / (2 * BOX); this.sy = this.H / (2 * BOX);
      return true;
    }
    X(u) { return (u + BOX) * this.sx; }
    Y(v) { return (BOX - v) * this.sy; }
    toModel(ev) { const r = this.el.getBoundingClientRect(); return [-BOX + 2 * BOX * (ev.clientX - r.left) / r.width, BOX - 2 * BOX * (ev.clientY - r.top) / r.height]; }
    ctx(layer) { return (layer === "fg" ? this.fg : this.bg).getContext("2d"); }
    clear(layer, fill) { const c = this.ctx(layer); c.clearRect(0, 0, this.W, this.H); if (fill) { c.fillStyle = fill; c.fillRect(0, 0, this.W, this.H); } return c; }
    frame(c) { c.strokeStyle = LINE; c.lineWidth = this.dpr; c.strokeRect(0.5 * this.dpr, 0.5 * this.dpr, this.W - this.dpr, this.H - this.dpr); }
    path(c, pts) { for (let k = 0; k < pts.length; k++) { const q = pts[k]; if (k) c.lineTo(this.X(q[0]), this.Y(q[1])); else c.moveTo(this.X(q[0]), this.Y(q[1])); } }
    line(c, pts, w, col, dash) { c.save(); c.lineWidth = w * this.dpr; c.strokeStyle = col; c.lineCap = "round"; c.lineJoin = "round"; if (dash) c.setLineDash(dash.map(d => d * this.dpr)); c.beginPath(); this.path(c, pts); c.stroke(); c.restore(); }
    dot(c, x, y, r, col, stroke) { c.beginPath(); c.arc(this.X(x), this.Y(y), r * this.dpr, 0, TAU); c.fillStyle = col; c.fill(); if (stroke) { c.lineWidth = 1.2 * this.dpr; c.strokeStyle = stroke; c.stroke(); } }
    arrow(c, from, to, col, w, head) {   // straight arrow in model coordinates, head size in CSS px
      const dpr = this.dpr, x1 = this.X(from[0]), y1 = this.Y(from[1]), x2 = this.X(to[0]), y2 = this.Y(to[1]), ang = Math.atan2(y2 - y1, x2 - x1), H = (head || 7) * dpr;
      c.save(); c.strokeStyle = col; c.fillStyle = col; c.lineWidth = w * dpr; c.lineCap = "round";
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2 - 0.8 * H * Math.cos(ang), y2 - 0.8 * H * Math.sin(ang)); c.stroke();
      c.beginPath(); c.moveTo(x2, y2); c.lineTo(x2 - H * Math.cos(ang - 0.42), y2 - H * Math.sin(ang - 0.42)); c.lineTo(x2 - H * Math.cos(ang + 0.42), y2 - H * Math.sin(ang + 0.42)); c.closePath(); c.fill(); c.restore();
    }
    atoms(c, labels) {
      const dpr = this.dpr;
      for (const a of ATOMS) {
        const r = (3.5 + 9 * Math.sqrt(a.p)) * dpr;
        c.beginPath(); c.arc(this.X(a.z[0]), this.Y(a.z[1]), r, 0, TAU); c.fillStyle = a.c; c.fill(); c.lineWidth = 1.6 * dpr; c.strokeStyle = "#ffffff"; c.stroke();
        if (labels) { c.font = `italic ${12.5 * dpr}px "Latin Modern Roman", "Times New Roman", serif`; c.fillStyle = a.c; c.textBaseline = "bottom"; c.textAlign = "left"; c.fillText(a.p.toFixed(2), this.X(a.z[0]) + r + 3 * dpr, this.Y(a.z[1]) - r + 2 * dpr); }
      }
    }
    label(c, text, x, y, col) {   // text with a light halo, model coordinates, kept inside the stage
      const dpr = this.dpr; c.font = `italic ${13.5 * dpr}px "Latin Modern Roman", "Times New Roman", serif`; c.textBaseline = "bottom"; c.textAlign = "left";
      const w = c.measureText(text).width, tx = Math.min(Math.max(this.X(x) + 8 * dpr, 4 * dpr), this.W - w - 6 * dpr), ty = Math.max(this.Y(y) - 8 * dpr, 18 * dpr);
      c.lineWidth = 4 * dpr; c.strokeStyle = "rgba(251,246,233,0.9)"; c.strokeText(text, tx, ty); c.fillStyle = col || INK; c.fillText(text, tx, ty);
    }
    basins(c) {   // the basin tints at the label grid's resolution, downsampled by the canvas
      const nb = BAS.nb, img = document.createElement("canvas"); img.width = nb; img.height = nb;
      const ic = img.getContext("2d"), data = ic.createImageData(nb, nb), d = data.data, cr = hex2rgb(CREAM);
      for (let i = 0; i < nb; i++) for (let j = 0; j < nb; j++) { const k = BAS.idx[i * nb + j], col = k >= 0 ? TINTS[k] : cr, o = ((nb - 1 - j) * nb + i) * 4; d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255; }
      ic.putImageData(data, 0, 0); c.imageSmoothingEnabled = true; c.imageSmoothingQuality = "high"; c.drawImage(img, 0, 0, this.W, this.H);
    }
    seps(c) { const s = sepsOf(BAS); if (!s) return; for (const [w, col] of [[3.4, "rgba(255,255,255,0.95)"], [1.3, "#3B3833"]]) for (const pts of s) this.line(c, pts, w, col); }
  }
  const chips = (vals, fmt) => vals.map((v, k) => `<span class="chip" style="background:${ATOMS[k].c}"></span>${fmt(v)}`).join(" ");
  const reduced = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stages = [];
  function makeStage(root) { const s = new Stage(root.querySelector(".anim-stage")); stages.push(s); return s; }
  function read(root) { return root.querySelector(".anim-read"); }
  const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;

  // ================= 1. pairs and the interpolant =================
  function initPairs() {
    const root = document.getElementById("d-pairs"); if (!root) return;
    const st = makeStage(root), slider = root.querySelector("input[type=range]"), tval = root.querySelector(".tval"), play = root.querySelector(".play"), repair = root.querySelector(".repair"), rd = read(root);
    let t = parseFloat(slider.value), playing = false, dir = 1, last = null;
    function bg() {
      const c = st.clear("bg", CREAM);
      for (let i = 0; i < 70; i++) { const z = ATOMS[K1[i]].z; st.line(c, [X0[i], z], 0.8, rgba(ATOMS[K1[i]].c, 0.28)); }
      st.atoms(c, true); st.frame(c);
    }
    function fg() {
      const c = st.clear("fg");
      for (let i = 0; i < N; i++) { const z = ATOMS[K1[i]].z, x = (1 - t) * X0[i][0] + t * z[0], y = (1 - t) * X0[i][1] + t * z[1]; st.dot(c, x, y, 2.1, t < 0.02 ? GREY : ATOMS[K1[i]].c); }
      tval.textContent = t.toFixed(2);
      rd.innerHTML = t < 0.02 ? "t = 0: the noise cloud, every point still grey because its partner has not mattered yet" : t > 0.98 ? "t = 1: the data, a fraction p<sub>j</sub> of the points on each atom" : `t = ${t.toFixed(2)}: the population &rho;<sub>t</sub>, each point a fraction t of the way along its line`;
    }
    slider.addEventListener("input", () => { t = parseFloat(slider.value); playing = false; play.textContent = "Play"; fg(); });
    play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; last = null; if (playing) requestAnimationFrame(frame); });
    repair.addEventListener("click", () => { rePair(); bg(); fg(); });
    function frame(now) {
      if (!playing) return; if (last === null) last = now; const dt = Math.min((now - last) / 1000, 0.05); last = now;
      t += dir * dt / 3.2; if (t >= 1) { t = 1; dir = -1; } if (t <= 0) { t = 0; dir = 1; }
      slider.value = t.toFixed(3); fg(); requestAnimationFrame(frame);
    }
    bg(); fg(); return { redraw() { bg(); fg(); } };
  }

  // ================= 2. the flow-matching field b_t =================
  function initField() {
    const root = document.getElementById("d-field"); if (!root) return;
    const st = makeStage(root), slider = root.querySelector("input[type=range]"), tval = root.querySelector(".tval"), rd = read(root);
    let t = parseFloat(slider.value), hover = null;
    function bg() {
      const c = st.clear("bg", CREAM), dpr = st.dpr, n = 11, L = 0.3, s = Math.min(t, 0.97);
      c.lineWidth = 1.4 * dpr; c.lineCap = "round"; c.strokeStyle = "rgba(78,76,70,0.6)"; c.fillStyle = "rgba(78,76,70,0.75)";
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        let x = -2.25 + 4.5 * a / (n - 1), y = -2.25 + 4.5 * b / (n - 1); const pts = [[x, y]];
        for (let k = 0; k < 10; k++) { const v = bFM(x, y, s), m = Math.hypot(v[0], v[1]) || 1; x += (L / 10) * v[0] / m; y += (L / 10) * v[1] / m; pts.push([x, y]); }
        st.arrow(c, pts[0], pts[pts.length - 1], "rgba(78,76,70,0.65)", 1.4, 6);
      }
      st.atoms(c, false); st.frame(c); tval.textContent = t.toFixed(2);
      rd.innerHTML = t < 0.03 ? "t = 0: nothing is known yet, every arrow points at the data mean" : t < 0.6 ? `t = ${t.toFixed(2)}: the field has started to split between the atoms` : `t = ${t.toFixed(2)}: near the end each arrow has chosen its atom, and the speed grows like 1 / (1 &minus; t)`;
    }
    function fg() {
      const c = st.clear("fg"); if (!hover) return;
      const [x, y] = hover, z = zhat(x, y, Math.min(t, 0.97)), col = ATOMS[nearestAtom(z[0], z[1])].c;
      st.line(c, [[x, y], z], 1.2, rgba(CHARCOAL, 0.6), [3, 4]); st.dot(c, z[0], z[1], 4, "#ffffff", col); st.dot(c, x, y, 3.5, INK);
      st.label(c, "x", x, y); st.label(c, "ẑ_t(x) = E[x₁ | I_t = x]", z[0], z[1], col);
      const b = bFM(x, y, Math.min(t, 0.97)), m = Math.hypot(b[0], b[1]) || 1, L = Math.min(0.9, 0.25 * m);
      st.arrow(c, [x, y], [x + L * b[0] / m, y + L * b[1] / m], INK, 1.8, 7);
    }
    slider.addEventListener("input", () => { t = parseFloat(slider.value); bg(); fg(); });
    st.el.addEventListener("pointermove", ev => { hover = st.toModel(ev); fg(); });
    st.el.addEventListener("pointerleave", () => { hover = null; fg(); });
    bg(); return { redraw() { bg(); fg(); } };
  }

  // ================= 3. the autonomous field: streamlines, particles, basins =================
  function initAuto() {
    const root = document.getElementById("d-auto"); if (!root) return;
    const st = makeStage(root), play = root.querySelector(".play"), basinsBtn = root.querySelector(".basins"), rd = read(root);
    let showBasins = false, playing = !reduced, last = null, streams = null, hover = null, clock = 0;
    const parts = X0.map(d => ({ x: d[0], y: d[1], px: d[0], py: d[1], dest: -1, done: false }));
    function reset() { for (let i = 0; i < N; i++) { const p = parts[i]; p.x = p.px = X0[i][0]; p.y = p.py = X0[i][1]; p.done = false; p.dest = FG.ready ? mapDest(p.x, p.y) : -1; } clock = 0; st.clear("fg"); }
    function bg() {
      const c = st.clear("bg", CREAM);
      if (showBasins && BAS.done) st.basins(c);
      if (!streams && FG.ready) streams = streamlines();
      if (streams) for (const pts of streams) st.line(c, pts, 0.9, showBasins ? "rgba(255,255,255,0.7)" : "rgba(60,58,54,0.32)");
      if (showBasins && BAS.done) st.seps(c);
      st.atoms(c, false); st.frame(c);
      if (!FG.ready) rd.innerHTML = "computing the field &hellip;";
      else if (showBasins && BAS.masses) rd.innerHTML = `basin mass: ${chips(BAS.masses, v => v.toFixed(3))} &middot; data weights ${chips(ATOMS.map(a => a.p), v => v.toFixed(2))}`;
      else rd.innerHTML = "one field for all times; hover to follow a flow line";
    }
    function fg() {
      const c = st.ctx("fg"), dpr = st.dpr;
      c.globalCompositeOperation = "destination-out"; c.fillStyle = "rgba(0,0,0,0.14)"; c.fillRect(0, 0, st.W, st.H); c.globalCompositeOperation = "source-over";
      c.lineCap = "round"; c.lineWidth = 2.2 * dpr;
      for (const p of parts) {
        const col = p.dest >= 0 ? ATOMS[p.dest].c : GREY; c.strokeStyle = col; c.fillStyle = col;
        if (!p.done || p.px !== p.x) { c.beginPath(); c.moveTo(st.X(p.px), st.Y(p.py)); c.lineTo(st.X(p.x), st.Y(p.y)); c.stroke(); }
        c.beginPath(); c.arc(st.X(p.x), st.Y(p.y), 1.7 * dpr, 0, TAU); c.fill();
      }
      if (hover) {
        const [x, y] = hover, fwd = flowLine(x, y, true), back = flowLine(x, y, false), e = fwd[fwd.length - 1], col = ATOMS[nearestAtom(e[0], e[1])].c;
        for (const [pts, w, cc] of [[back, 5, "rgba(255,255,255,0.9)"], [fwd, 5, "rgba(255,255,255,0.9)"], [back, 2.4, col], [fwd, 2.4, col]]) st.line(c, pts, w, cc);
        st.dot(c, x, y, 4.5, "#ffffff", col); st.label(c, "the one flow line through x", x, y, INK);
      }
    }
    function frame(now) {
      if (!playing) return; if (last === null) last = now; const dt = Math.min((now - last) / 1000, 0.05); last = now;
      if (FG.ready) {
        let alive = 0;
        for (const p of parts) { if (p.done) continue; alive++; p.px = p.x; p.py = p.y; stepDir(p, FG.field, 0.55 * dt, CLK.auto.vmin, CLK.auto.vmax); const a = arrivedAt(p.x, p.y); if (a >= 0) { p.done = true; p.x = ATOMS[a].z[0]; p.y = ATOMS[a].z[1]; } }
        clock += dt; if (alive === 0 && clock > 6.5 || clock > 9) reset();
      }
      fg(); requestAnimationFrame(frame);
    }
    play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; last = null; if (playing) requestAnimationFrame(frame); });
    basinsBtn.addEventListener("click", () => { showBasins = !showBasins; basinsBtn.textContent = showBasins ? "Hide basins" : "Show basins"; bg(); });
    st.el.addEventListener("pointermove", ev => { hover = st.toModel(ev); if (!playing) fg(); });
    st.el.addEventListener("pointerleave", () => { hover = null; if (!playing) fg(); });
    play.textContent = playing ? "Pause" : "Play";
    bg(); reset(); fg(); if (playing) requestAnimationFrame(frame);
    return { redraw() { streams = null; bg(); fg(); }, ready() { streams = null; bg(); reset(); fg(); } };
  }

  // ================= 4. the map: hover or tap a point, T(x) in one call =================
  function initMap() {
    const root = document.getElementById("d-map"); if (!root) return;
    const st = makeStage(root), rd = read(root);
    let pt = [-0.35, 0.95];
    function bg() { const c = st.clear("bg", CREAM); if (BAS.done) { st.basins(c); st.seps(c); } st.atoms(c, true); st.frame(c); }
    function fg() {
      const c = st.clear("fg"); if (!FG.ready) { rd.innerHTML = "computing the field &hellip;"; return; }
      const [x, y] = pt, k = mapDest(x, y), z = ATOMS[k].z, col = ATOMS[k].c, line = flowLine(x, y, true);
      st.line(c, line, 4, "rgba(255,255,255,0.9)"); st.line(c, line, 1.6, rgba(CHARCOAL, 0.9), [1, 5]);
      st.arrow(c, [x, y], [z[0] - 0.08 * (z[0] - x) / Math.hypot(z[0] - x, z[1] - y), z[1] - 0.08 * (z[1] - y) / Math.hypot(z[0] - x, z[1] - y)], col, 2.6, 9);
      st.dot(c, x, y, 4.5, "#ffffff", INK); st.label(c, "x", x, y);
      st.label(c, "T(x): one call, no integration", (x + z[0]) / 2, (y + z[1]) / 2, col);
      rd.innerHTML = `the dotted line is the flow line the ODE would follow; the map jumps to its endpoint. basin mass: ${chips(BAS.masses || [], v => v.toFixed(3))}`;
    }
    const move = ev => { pt = st.toModel(ev); fg(); };
    st.el.addEventListener("pointermove", move); st.el.addEventListener("pointerdown", move);
    bg(); fg(); return { redraw() { bg(); fg(); }, ready() { bg(); fg(); } };
  }

  // ================= 5. the same noise: flow matching with N steps against the map =================
  function initCompare() {
    const root = document.getElementById("d-compare"); if (!root) return;
    const [left, right] = [...root.querySelectorAll(".anim-stage")].map(el => { const s = new Stage(el); stages.push(s); return s; });
    const reads = root.querySelectorAll(".anim-read"), nBtns = [...root.querySelectorAll(".seg button")], go = root.querySelector(".go");
    let n = 1, paths = null, dests = null, anim = null;
    function setN(v) { n = v; for (const b of nBtns) b.classList.toggle("on", parseInt(b.dataset.n, 10) === n); paths = X0.map(x => eulerPath(x, n)); }
    function bgL() { const c = left.clear("bg", CREAM); left.atoms(c, false); left.frame(c); }
    function bgR() { const c = right.clear("bg", CREAM); if (BAS.done) { right.basins(c); right.seps(c); } right.atoms(c, false); right.frame(c); }
    function drawL(prog) {   // prog in [0, n]: step index plus fraction
      const c = left.clear("fg"), k = Math.min(n - 1, Math.floor(prog)), u = Math.min(1, prog - k);
      for (let i = 0; i < N; i++) {
        const P = paths[i], a = P[k], b = P[k + 1], x = a[0] + (b[0] - a[0]) * u, y = a[1] + (b[1] - a[1]) * u, end = prog >= n;
        for (let q = 0; q < k; q++) left.line(c, [P[q], P[q + 1]], 0.7, rgba(CHARCOAL, 0.18));
        left.line(c, [a, [x, y]], 0.7, rgba(CHARCOAL, 0.18));
        left.dot(c, x, y, 2.1, end ? ATOMS[nearestAtom(x, y)].c : GREY);
      }
      const ends = paths.map(P => P[n]), md = meanDist(ends), stepTxt = prog >= n ? `${n} network call${n > 1 ? "s" : ""} per sample` : `step ${Math.min(n, Math.floor(prog) + 1)} of ${n}`;
      reads[0].innerHTML = prog >= n ? `${stepTxt} &middot; mean distance to the nearest atom <b>${md.toFixed(2)}</b>` : stepTxt;
    }
    function drawR(u) {
      const c = right.clear("fg");
      for (let i = 0; i < N; i++) {
        const z = ATOMS[dests[i]].z, x = X0[i][0] + (z[0] - X0[i][0]) * u, y = X0[i][1] + (z[1] - X0[i][1]) * u;
        if (u > 0) right.line(c, [X0[i], [x, y]], 0.7, rgba(ATOMS[dests[i]].c, 0.25));
        right.dot(c, x, y, 2.1, u > 0 ? ATOMS[dests[i]].c : GREY);
      }
      reads[1].innerHTML = u >= 1 ? `1 network call per sample &middot; mean distance <b>0.00</b> &middot; basin mass ${chips(masses(dests), v => v.toFixed(2))}` : u > 0 ? "jumping to the endpoint &hellip;" : "the same 300 noise draws";
    }
    function run() {
      if (!FG.ready) return;
      if (!dests) dests = X0.map(x => mapDest(x[0], x[1]));
      if (anim) cancelAnimationFrame(anim);
      const t0 = performance.now(), per = Math.max(140, Math.min(650, 2600 / n)), total = per * n, flyT = 700;
      if (reduced) { drawL(n); drawR(1); return; }
      function f(now) {
        const el = now - t0, pL = Math.min(n, el / per), pR = ease(Math.min(1, el / flyT));
        drawL(pL); drawR(pR);
        if (el < Math.max(total, flyT)) anim = requestAnimationFrame(f); else { drawL(n); drawR(1); }
      }
      anim = requestAnimationFrame(f);
    }
    for (const b of nBtns) b.addEventListener("click", () => { setN(parseInt(b.dataset.n, 10)); run(); });
    go.addEventListener("click", run);
    setN(n); bgL(); bgR(); drawL(0); drawR(0);
    return { redraw() { bgL(); bgR(); drawL(0); drawR(0); }, ready() { bgR(); run(); } };
  }

  function init() {
    const loaded = loadFields(global.BTM_FIELDS);
    const diagrams = [initPairs(), initField(), initAuto(), initMap(), initCompare()].filter(Boolean);
    if (!loaded) {   // no shipped grids: compute the field in chunks, then the basins
      let row = 0; const G = Toy.GRID;
      (function step() {
        if (row < G.n) { const r1 = Math.min(G.n, row + 6); Toy.computeAutoGrid([row, r1]); row = r1; setTimeout(step, 0); return; }
        for (const F of Toy.FIELDS) F.ready = true;
        (function basins() { Toy.computeBasinRows(8, 0); if (!BAS.done) { setTimeout(basins, 0); return; } for (const d of diagrams) if (d.ready) d.ready(); })();
      })();
    } else for (const d of diagrams) if (d.ready) d.ready();
    let rt = null;
    global.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { for (const s of stages) s.resize(); for (const d of diagrams) d.redraw(); }, 120); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { for (const d of diagrams) d.redraw(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof window !== "undefined" ? window : globalThis);
