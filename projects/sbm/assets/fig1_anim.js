/* Animated Figure 1: particles flowing along the exact BTM current and the guided SBM current of the two-atom toy.
   Needs window.SBM_FIELDS (assets/fig1_fields.js): the log kernel ratio on a grid. The normalised field is
   V(x) = zbar(x) - x with zbar = a tanh(0.5 (log R + log(p/(1-p)))); its v-component is -v.
   SBM adds the guidance correction 2 (1 - 1/omega) a e1 inside the band S_omega(v) < u < 0, where the conditional map
   says +e1 and the null map says -e1. S_omega is the stable manifold of the axial saddle of (f + beta_omega, -v) with
   beta_omega = 1.6 m (1 - 1/omega), m the maximal inward pull on the axis; at omega = 2 this is the paper's panel (c). */
(function (global) {
  "use strict";
  const D = global.SBM_FIELDS;
  if (!D) return;

  // ---------- fields ----------
  function decode(b64) {
    const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }
  const LOGR = decode(D.logR_b64);
  const NU = D.nu, NV = D.nv, DU = (D.umax - D.umin) / (NU - 1), DV = D.vmax / (NV - 1);
  const LW = Math.log(D.p / (1 - D.p)), A = D.a;

  function logRat(u, v) {
    v = Math.abs(v);
    let fu = (u - D.umin) / DU, fv = v / DV;
    fu = Math.min(Math.max(fu, 0), NU - 1 - 1e-9);
    fv = Math.min(Math.max(fv, 0), NV - 1 - 1e-9);
    const iu = Math.floor(fu), iv = Math.floor(fv), au = fu - iu, av = fv - iv, i = iu * NV + iv;
    return (1 - au) * ((1 - av) * LOGR[i] + av * LOGR[i + 1]) + au * ((1 - av) * LOGR[i + NV] + av * LOGR[i + NV + 1]);
  }
  function f(u, v) { return A * Math.tanh(0.5 * (logRat(u, v) + LW)) - u; }

  function bisect(g, lo, hi) {
    let glo = g(lo);
    for (let k = 0; k < 80; k++) {
      const mid = 0.5 * (lo + hi), gm = g(mid);
      if ((gm < 0) === (glo < 0)) { lo = mid; glo = gm; } else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // axial quantities of the plain conditional field
  const U_P = bisect(u => f(u, 0), -0.99, 0.99);          // BTM saddle (separatrix on the axis)
  let M = 0, U_M = -0.9;
  for (let k = 0; k <= 400; k++) {
    const u = -0.99 + (U_P + 0.99) * k / 400, val = -f(u, 0);
    if (val > M) { M = val; U_M = u; }
  }

  // stable manifold of the saddle of (f + beta, -v): integrate backward in time from (u0, eps)
  function separatrix(beta) {
    const u0 = beta > 0 ? bisect(u => f(u, 0) + beta, U_M, U_P) : U_P;
    const vs = [0], us = [u0];
    let u = u0, v = 1e-4;
    const h = 0.01;
    const rhs = (u, v) => [-(f(u, v) + beta), v];
    for (let k = 0; k < 20000 && v < 4.6; k++) {
      const k1 = rhs(u, v), k2 = rhs(u + 0.5 * h * k1[0], v + 0.5 * h * k1[1]);
      const k3 = rhs(u + 0.5 * h * k2[0], v + 0.5 * h * k2[1]), k4 = rhs(u + h * k3[0], v + h * k3[1]);
      u += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      vs.push(v); us.push(u);
    }
    const n = vs.length, slope = (us[n - 1] - us[n - 20]) / (vs[n - 1] - vs[n - 20]);
    const S = function (v) {
      v = Math.abs(v);
      if (v >= vs[n - 1]) return us[n - 1] + slope * (v - vs[n - 1]);
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (vs[mid] <= v) lo = mid; else hi = mid; }
      const t = (v - vs[lo]) / (vs[hi] - vs[lo]);
      return us[lo] + t * (us[hi] - us[lo]);
    };
    S.u0 = u0; S.beta = beta;
    return S;
  }

  // Gaussian source mass of {u > S(|v|)}
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  function massRight(S) {
    const n = 2800, lo = -7, hi = 7, h = (hi - lo) / n;
    let acc = 0;
    for (let k = 0; k <= n; k++) {
      const v = lo + k * h, w = (k === 0 || k === n) ? 0.5 : 1;
      acc += w * phi(v) * (1 - Phi(S(v)));
    }
    return acc * h;
  }

  const betaOf = omega => 1.6 * M * (1 - 1 / omega);
  const pushOf = omega => 2 * (1 - 1 / omega) * A;

  const Toy = { f, U_P, M, U_M, separatrix, massRight, betaOf, pushOf };
  global.SBMToy = Toy;
  if (typeof document === "undefined") return;

  // ---------- animation ----------
  const UMAX = 2.6, VMAX = 2.2;
  const C = {
    plus: "#FBF6E9", minus: "#DAECF8", flip: "#D3E5D2",
    sep_btm: "#4A4A4A", sep_sbm: "#1B8A6B", sep_null: "#8F8E88", stream: "rgba(120,118,112,0.35)",
    p_plus: "#1B8A6B", p_minus: "#1769E0", p_flip: "#BB5078", ink: "#2F2A24", atom: "#111111",
  };

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const rng = mulberry32(20261103);
  function gauss() {
    const r = Math.sqrt(-2 * Math.log(1 - rng())), th = 2 * Math.PI * rng();
    return [r * Math.cos(th), r * Math.sin(th)];
  }
  const N = 360, LIVES = 64;
  const DRAWS = [];
  for (let i = 0; i < N * LIVES; i++) DRAWS.push(gauss());

  const S_BTM = separatrix(0);
  const MASS_BTM = massRight(S_BTM);

  class Panel {
    constructor(root, guided) {
      this.root = root; this.guided = guided;
      this.bg = root.querySelector("canvas.bg"); this.fg = root.querySelector("canvas.fg");
      this.read = root.parentElement.querySelector(".anim-read");
      this.omega = 2; this.S = S_BTM; this.push = 0; this.mass = MASS_BTM;
      this.parts = [];
      for (let i = 0; i < N; i++) this.parts.push(this.spawn(i, 0));
      this.arrived = { plus: 0, minus: 0 };
      this.streams = [];
      this.resize();
    }
    setOmega(omega) {
      this.omega = omega;
      if (this.guided) {
        this.S = omega > 1 ? separatrix(betaOf(omega)) : S_BTM;
        this.push = pushOf(omega);
        this.mass = massRight(this.S);
      }
      for (const p of this.parts) p.dest = this.destOf(p.x, p.y);
      this.arrived = { plus: 0, minus: 0 };
      this.streams = this.streamlines();
      this.drawBackground();
      this.clearTrails();
    }
    inBand(x, y) { return this.guided && x < 0 && x > this.S(y); }
    field(x, y) {
      let fu = f(x, y);
      if (this.push > 0 && this.inBand(x, y)) fu += this.push;
      return [fu, -y];
    }
    destOf(x, y) {
      if (x > this.S(y)) return (this.guided && x < S_BTM(y)) ? "flip" : "plus";
      return "minus";
    }
    spawn(i, life) {
      const d = DRAWS[i + (life % LIVES) * N];
      return { x: d[0], y: d[1], px: d[0], py: d[1], life, age: 0, dest: this.destOf(d[0], d[1]), i };
    }
    step(dt) {
      for (let k = 0; k < this.parts.length; k++) {
        const p = this.parts[k];
        p.px = p.x; p.py = p.y;
        const k1 = this.field(p.x, p.y);
        const k2 = this.field(p.x + 0.5 * dt * k1[0], p.y + 0.5 * dt * k1[1]);
        const k3 = this.field(p.x + 0.5 * dt * k2[0], p.y + 0.5 * dt * k2[1]);
        const k4 = this.field(p.x + dt * k3[0], p.y + dt * k3[1]);
        p.x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        p.y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        p.age += dt;
        const dp = Math.hypot(p.x - 1, p.y), dm = Math.hypot(p.x + 1, p.y);
        if (dp < 0.045 || dm < 0.045 || p.age > 14) {
          if (dp < 0.045) this.arrived.plus++; else if (dm < 0.045) this.arrived.minus++;
          this.parts[k] = this.spawn(p.i, p.life + 1);
        }
      }
    }
    streamlines() {
      const seeds = [];
      for (let k = 0; k < 15; k++) { const u = -2.4 + 4.8 * k / 14; seeds.push([u, VMAX - 0.02]); seeds.push([u, -VMAX + 0.02]); }
      for (let k = 0; k < 7; k++) { const v = -1.8 + 3.6 * k / 6; seeds.push([-UMAX + 0.02, v]); seeds.push([UMAX - 0.02, v]); }
      const lines = [];
      for (const s of seeds) {
        let x = s[0], y = s[1]; const pts = [[x, y]]; const dt = 0.02;
        for (let k = 0; k < 2500; k++) {
          const k1 = this.field(x, y), k2 = this.field(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1]);
          x += dt * k2[0]; y += dt * k2[1]; pts.push([x, y]);
          if (Math.hypot(x - 1, y) < 0.04 || Math.hypot(x + 1, y) < 0.04) break;
        }
        lines.push(pts);
      }
      return lines;
    }
    resize() {
      const dpr = Math.min(Math.max(global.devicePixelRatio || 1, 2), 3);   // >= 2 device pixels per CSS pixel: smooth lines on every screen
      const w = this.root.clientWidth, h = this.root.clientHeight;
      if (!w || !h) return;
      for (const c of [this.bg, this.fg]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      this.sx = this.bg.width / (2 * UMAX); this.sy = this.bg.height / (2 * VMAX); this.dpr = dpr;
      this.drawBackground(); this.clearTrails();
    }
    X(u) { return (u + UMAX) * this.sx; }
    Y(v) { return (VMAX - v) * this.sy; }
    curve(ctx, S, vmin, vmax) {
      let first = true;
      for (let k = 0; k <= 160; k++) {
        const v = vmin + (vmax - vmin) * k / 160, u = Math.max(-UMAX - 0.1, Math.min(UMAX + 0.1, S(v)));
        if (first) { ctx.moveTo(this.X(u), this.Y(v)); first = false; } else ctx.lineTo(this.X(u), this.Y(v));
      }
    }
    drawBackground() {
      const ctx = this.bg.getContext("2d"), W = this.bg.width, H = this.bg.height, dpr = this.dpr || 1;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.plus; ctx.fillRect(0, 0, W, H);
      // region sent to -e1: left of the separatrix
      ctx.beginPath(); ctx.moveTo(0, 0); this.curve(ctx, this.S, VMAX, -VMAX); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = C.minus; ctx.fill();
      if (this.guided && this.omega > 1) {
        // reassigned band: between the SBM and the BTM separatrix
        ctx.beginPath(); this.curve(ctx, this.S, VMAX, -VMAX); this.curve(ctx, S_BTM, -VMAX, VMAX); ctx.closePath();
        ctx.fillStyle = C.flip; ctx.fill();
      }
      // streamlines
      ctx.lineWidth = 0.8 * dpr; ctx.strokeStyle = C.stream; ctx.lineCap = "round";
      for (const pts of this.streams) {
        ctx.beginPath();
        for (let k = 0; k < pts.length; k++) { const q = pts[k]; if (k === 0) ctx.moveTo(this.X(q[0]), this.Y(q[1])); else ctx.lineTo(this.X(q[0]), this.Y(q[1])); }
        ctx.stroke();
      }
      // separatrices
      ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.lineWidth = 1 * dpr; ctx.strokeStyle = C.sep_null;
      ctx.beginPath(); ctx.moveTo(this.X(0), 0); ctx.lineTo(this.X(0), H); ctx.stroke();
      if (this.guided && this.omega > 1) {
        ctx.setLineDash([1.5 * dpr, 3 * dpr]); ctx.lineWidth = 1.2 * dpr; ctx.strokeStyle = C.sep_btm;
        ctx.beginPath(); this.curve(ctx, S_BTM, -VMAX, VMAX); ctx.stroke();
        ctx.setLineDash([]); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = C.sep_sbm;
        ctx.beginPath(); this.curve(ctx, this.S, -VMAX, VMAX); ctx.stroke();
      } else {
        ctx.setLineDash([]); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = C.sep_btm;
        ctx.beginPath(); this.curve(ctx, S_BTM, -VMAX, VMAX); ctx.stroke();
      }
      // atoms, sized by the class weight
      ctx.fillStyle = C.atom;
      ctx.beginPath(); ctx.arc(this.X(1), this.Y(0), 7 * dpr, 0, 2 * Math.PI); ctx.fill();
      ctx.beginPath(); ctx.arc(this.X(-1), this.Y(0), 4 * dpr, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = C.ink; ctx.font = `italic ${15 * dpr}px "Latin Modern Roman", "Times New Roman", serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText("+e₁", this.X(1), this.Y(0) + 10 * dpr);
      ctx.fillText("−e₁", this.X(-1), this.Y(0) + 8 * dpr);
      // frame
      ctx.strokeStyle = "#B8AA95"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(0.5 * dpr, 0.5 * dpr, W - dpr, H - dpr);
    }
    clearTrails() { const ctx = this.fg.getContext("2d"); ctx.clearRect(0, 0, this.fg.width, this.fg.height); }
    drawParticles() {
      const ctx = this.fg.getContext("2d"), dpr = this.dpr || 1;
      // fade the previous frame so trails decay
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,0.16)"; ctx.fillRect(0, 0, this.fg.width, this.fg.height);
      ctx.globalCompositeOperation = "source-over";
      ctx.lineCap = "round"; ctx.lineWidth = 2.2 * dpr;
      for (const p of this.parts) {
        ctx.strokeStyle = p.dest === "flip" ? C.p_flip : (p.dest === "plus" ? C.p_plus : C.p_minus);
        if (p.age > 0) { ctx.beginPath(); ctx.moveTo(this.X(p.px), this.Y(p.py)); ctx.lineTo(this.X(p.x), this.Y(p.y)); ctx.stroke(); }
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath(); ctx.arc(this.X(p.x), this.Y(p.y), 1.6 * dpr, 0, 2 * Math.PI); ctx.fill();
      }
    }
    readout() {
      const tot = this.arrived.plus + this.arrived.minus;
      const live = tot ? ` · arrived so far: ${(100 * this.arrived.plus / tot).toFixed(0)} % at +e₁ (${tot})` : "";
      this.read.textContent = `source mass sent to +e₁: ${(100 * this.mass).toFixed(1)} %${live}`;
    }
  }

  function init() {
    const root = document.getElementById("fig1-anim");
    if (!root) return;
    const stages = root.querySelectorAll(".anim-stage:not(.static)");
    const btm = new Panel(stages[0], false), sbm = new Panel(stages[1], true);
    const slider = root.querySelector("#omega"), omegaLabel = root.querySelector("#omega-val");
    const title = root.querySelector("#title-sbm"), playBtn = root.querySelector("#play");
    btm.setOmega(1);
    function applyOmega() {
      const om = parseFloat(slider.value);
      sbm.setOmega(om);
      omegaLabel.textContent = om.toFixed(2);
      title.textContent = om > 1 ? `(c) SBM, ω = ${om.toFixed(2)}` : "(c) SBM, ω = 1 (BTM)";
      btm.readout(); sbm.readout();
    }
    slider.addEventListener("input", applyOmega);
    applyOmega();
    const reduced = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let playing = !reduced, last = null, acc = 0;
    playBtn.textContent = playing ? "Pause" : "Play";
    playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; last = null; if (playing) requestAnimationFrame(frame); });
    function frame(t) {
      if (!playing) return;
      if (last === null) last = t;
      let dt = Math.min((t - last) / 1000, 0.05); last = t;
      const h = 0.75 * dt;            // model time per real second
      btm.step(h); sbm.step(h);
      btm.drawParticles(); sbm.drawParticles();
      acc += dt;
      if (acc > 0.5) { btm.readout(); sbm.readout(); acc = 0; }
      requestAnimationFrame(frame);
    }
    // first frame even when paused; redraw the labels once the page fonts are in
    btm.drawParticles(); sbm.drawParticles();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { btm.drawBackground(); sbm.drawBackground(); });
    if (playing) requestAnimationFrame(frame);
    let rt = null;
    global.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { btm.resize(); sbm.resize(); }, 120); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof window !== "undefined" ? window : globalThis);
