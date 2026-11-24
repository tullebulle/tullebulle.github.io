/* The coloured-particle toy of the paper's flag/smiley figure, run live.

   State x = (y, c) with position y in R^2 and colour c in R^3 over {Blue, White, Red}.
   p0 = N(0, I_2) x N(0, I_3);  p1 = U([-1,1]^2) x U({e_B, e_W, e_R}).  Linear interpolant
   x_t = (1 - t) x0 + t x1, so both posteriors are closed form:

     y1 | y_t   is N(y_t / t, ((1-t)/t)^2) truncated to [-1,1], per coordinate;
     p(k | c_t) is proportional to exp( t c_t[k] / (1-t)^2 ).

   The base velocity is the exact b_t(x) = (E[x1 | x_t] - x_t) / (1 - t). Look-aheads are exact draws
   from p_{1|t} taken with common random numbers, as the outputs X_{0,1}(eps; t, x_t) of a Meta Flow
   Map would be, and steering is the paper's gradient-free estimator,
       (sigma_t^2 / 2) grad V_t(x_t)  ~  ( sum_i softmax_i(lambda r(x1_i)) x1_i - mean_i x1_i ) / (1 - t).
   The Norway reward steers colour alone, so the square survives; the smiley reward steers both. */
(function (global) {
  "use strict";

  const NP = 4000, NSTEP = 100, STORE = 2, NF = NSTEP / STORE + 1, NLOOK = 6;
  const COLS = [[38, 94, 178], [240, 240, 234], [196, 26, 50]];   // model colour order: Blue, White, Red
  // The colour noise is wider than the position noise on purpose. With a common scale the colour
  // posterior collapses by t = 0.85, before the position posterior says where the particle will land,
  // so the look-aheads all agree and the reward has nothing left to steer. A wider colour prior keeps
  // the category undecided until the position is known, which is when the reward becomes informative.
  let SC = 10.0;

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function invNorm(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) return -invNorm(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  function erf(x) {                       // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const Phi = z => 0.5 * (1 + erf(z * 0.7071067811865476));
  const phi = z => 0.3989422804014327 * Math.exp(-0.5 * z * z);
  const clamp = (v, a, b) => v < a ? a : (v > b ? b : v);

  // --- the truncated-normal posterior of one coordinate, written so it stays stable as t -> 0 and t -> 1 ---
  // the standardised window is [al, be] and s is the posterior sd; the endpoint lives in [-1, 1]
  function winOf(t, y) { const om = 1 - t; return [(-t - y) / om, (t - y) / om]; }
  function tnMean(al, be, s) {
    const w = be - al;
    if (!(w > 1e-7)) return 0;                                   // window is tiny: posterior is flat on [-1,1]
    const Z = Phi(be) - Phi(al);
    if (!(Z > 1e-12)) return clamp(-1 + s * -al, -1, 1);
    return clamp(-1 + s * ((phi(al) - phi(be)) / Z - al), -1, 1);
  }
  function tnDraw(al, be, s, u) {
    const w = be - al;
    if (!(w > 1e-7)) return -1 + 2 * u;
    const Pa = Phi(al), Z = Phi(be) - Pa;
    if (!(Z > 1e-12)) return clamp(-1 + s * -al, -1, 1);
    return clamp(-1 + s * (invNorm(clamp(Pa + u * Z, 1e-12, 1 - 1e-12)) - al), -1, 1);
  }
  // p(k | c_t), returned into out
  function colPost(t, c, o, out) {
    const k = t / ((1 - t) * (1 - t) * SC * SC);
    let mx = -Infinity;
    for (let j = 0; j < 3; j++) { const v = k * c[o + j]; out[j] = v; if (v > mx) mx = v; }
    let s = 0;
    for (let j = 0; j < 3; j++) { out[j] = Math.exp(out[j] - mx); s += out[j]; }
    for (let j = 0; j < 3; j++) out[j] /= s;
    return out;
  }

  // ---------- rewards ----------
  const sig = z => 1 / (1 + Math.exp(-z));
  const box = (z, a, b, k) => sig(k * (z - a)) * sig(k * (b - z));
  const cup = (a, b) => 1 - (1 - a) * (1 - b);                   // soft union
  const KS = 26;
  const NOR = { x0: -0.25, wW: 0.34, wB: 0.15 };   // white band widened a little: a thinner one is below the resolution the look-aheads reach before the colour sets
  // scores in model colour order (B, W, R)
  function sNorway(y1, y2, out) {
    const W = cup(box(y1, NOR.x0 - NOR.wW, NOR.x0 + NOR.wW, KS), box(y2, -NOR.wW, NOR.wW, KS));
    const B = cup(box(y1, NOR.x0 - NOR.wB, NOR.x0 + NOR.wB, KS), box(y2, -NOR.wB, NOR.wB, KS));
    out[0] = B; out[1] = W * (1 - B); out[2] = 1 - W;
    return out;
  }
  const SMI = { xe: 0.40, ye: 0.40, se: 0.17, y0: -0.60, kap: 0.78, sm: 0.085, mx: 0.63, my0: -0.95, my1: 0.05 };
  function sSmiley(y1, y2, out) {
    const S = box(y1, -1, 1, 14) * box(y2, -1, 1, 14);
    const dxR = y1 + SMI.xe, dxB = y1 - SMI.xe, dy = y2 - SMI.ye, q = 2 * SMI.se * SMI.se;
    const ER = Math.exp(-(dxR * dxR + dy * dy) / q), EB = Math.exp(-(dxB * dxB + dy * dy) / q);
    const d = y2 - SMI.y0 - SMI.kap * y1 * y1;
    const M = Math.exp(-d * d / (2 * SMI.sm * SMI.sm)) * box(y1, -SMI.mx, SMI.mx, KS) * box(y2, SMI.my0, SMI.my1, KS);
    out[0] = S * EB; out[1] = S * M; out[2] = S * ER;
    return out;
  }
  const REWARD = { norway: sNorway, smiley: sSmiley, none: null };
  // the smiley features cover little of the plane, so it needs more look-aheads to find them at all
  const NLOOK_OF = { norway: 14, smiley: 20, none: 1 };
  // Tuned guidance strengths, as in the paper: the ideal coefficient is replaced by a constant gamma.
  // The colour posterior collapses well before the position one becomes informative, so without a
  // gamma above one the reward arrives too late to change which colour a particle settles on.
  // The estimator carries a 1/(1-t) factor, so late in the flow one Euler step of raw guidance can
  // throw a particle clear across the square. The cap is therefore on the displacement a step may
  // make, not on the estimator itself: |h * g / (1-t)| <= step.
  const GAM = {
    norway: { col: 2, pos: 0, stepC: 0.25, stepP: 0.05 },
    smiley: { col: 2, pos: 8, stepC: 0.25, stepP: 0.05 },
    none: { col: 0, pos: 0, stepC: 0, stepP: 0 },
  };
  // the Norway reward is on colour only; the smiley reward moves position too
  const WINDOW = {
    norway: { col: [0.85, 0.995], pos: null },
    smiley: { col: [0.62, 0.99], pos: [0.15, 0.985] },
    none: { col: null, pos: null },
  };

  // ---------- one run of NP particles ----------
  // posF holds positions per stored frame, colF the rendered colour, scoreF the mean reward
  function makeRun() {
    return { posF: new Float32Array(NF * NP * 2), colF: new Uint8Array(NF * NP * 3), scoreF: new Float32Array(NF) };
  }
  const CHUNK = 400;                                 // particles handled per resumable slice
  const CT = 0.30;                                   // colour temperature used only for drawing
  function paint(c, o, out, j) {
    let mx = -Infinity;
    for (let k = 0; k < 3; k++) if (c[o + k] > mx) mx = c[o + k];
    let s = 0; const w = [0, 0, 0];
    for (let k = 0; k < 3; k++) { w[k] = Math.exp((c[o + k] - mx) / CT); s += w[k]; }
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < 3; k++) { const v = w[k] / s; r += v * COLS[k][0]; g += v * COLS[k][1]; b += v * COLS[k][2]; }
    out[j] = r; out[j + 1] = g; out[j + 2] = b;
  }

  class Run {
    constructor(seed, mode, lam, guided, opts) {
      opts = opts || {};
      this.mode = mode; this.lam = lam; this.guided = guided && mode !== "none";
      this.s = REWARD[mode]; this.win = opts.win || WINDOW[mode];
      this.nl = opts.nlook || NLOOK_OF[mode] || NLOOK;
      this.gam = opts.gam || GAM[mode];
      this.out = makeRun(); this.step = 0; this.frames = 0; this.pi = 0;
      const rng = mulberry32(seed);
      const g = () => { const r = Math.sqrt(-2 * Math.log(1 - rng())), th = 2 * Math.PI * rng(); return r * Math.cos(th); };
      this.y = new Float32Array(NP * 2); this.c = new Float32Array(NP * 3);
      for (let i = 0; i < NP; i++) { this.y[2 * i] = g(); this.y[2 * i + 1] = g(); for (let k = 0; k < 3; k++) this.c[3 * i + k] = SC * g(); }
      // Look-ahead noises: fixed per particle, so each one's look-aheads move continuously with its
      // state, but drawn independently across particles. Sharing one set across all particles pushes
      // them identically and collapses the cloud onto filaments.
      this.lu = new Float32Array(NP * this.nl * 3);
      for (let i = 0; i < NP; i++) {
        const ph = rng();                                  // per-particle phase keeps the u's stratified
        for (let l = 0; l < this.nl; l++) {
          const o = (i * this.nl + l) * 3;
          this.lu[o] = ((l + 0.5) / this.nl + ph) % 1;
          this.lu[o + 1] = rng(); this.lu[o + 2] = rng();
        }
      }
      this.p = [0, 0, 0]; this.sc = [0, 0, 0];
      this.mU = [0, 0, 0, 0, 0]; this.mW = [0, 0, 0, 0, 0];
      this.record();
    }
    record() {
      const f = this.frames, po = this.out.posF, co = this.out.colF;
      let tot = 0;
      for (let i = 0; i < NP; i++) {
        po[(f * NP + i) * 2] = this.y[2 * i]; po[(f * NP + i) * 2 + 1] = this.y[2 * i + 1];
        paint(this.c, 3 * i, co, (f * NP + i) * 3);
        if (this.s) {
          this.s(this.y[2 * i], this.y[2 * i + 1], this.sc);
          let b = 0; for (let k = 1; k < 3; k++) if (this.c[3 * i + k] > this.c[3 * i + b]) b = k;
          tot += this.sc[b];
        }
      }
      this.out.scoreF[f] = this.s ? tot / NP : 0;
      this.frames++;
    }
    // One Euler step of the (guided) probability flow, resumable: it processes at most `limit`
    // particles per call so a heavy step is spread over several frames instead of dropping one.
    advance(limit) {
      const t = this.step / NSTEP, h = 1 / NSTEP, om = 1 - t, tc = Math.min(t, 1 - 1e-4);
      const sPos = om / Math.max(tc, 1e-6);
      const onC = this.guided && this.win.col && t >= this.win.col[0] && t <= this.win.col[1];
      const onP = this.guided && this.win.pos && t >= this.win.pos[0] && t <= this.win.pos[1];
      const lam = this.lam, p = this.p, sc = this.sc, mU = this.mU, mW = this.mW;
      const end = Math.min(NP, this.pi + (limit || NP));
      for (let i = this.pi; i < end; i++) {
        const y0 = this.y[2 * i], y1 = this.y[2 * i + 1];
        const w0 = winOf(tc, y0), w1 = winOf(tc, y1);
        const m0 = tnMean(w0[0], w0[1], sPos), m1 = tnMean(w1[0], w1[1], sPos);
        colPost(tc, this.c, 3 * i, p);
        let gy0 = 0, gy1 = 0, gc0 = 0, gc1 = 0, gc2 = 0;
        if (onC || onP) {
          for (let j = 0; j < 5; j++) { mU[j] = 0; mW[j] = 0; }
          let Z = 0;
          const lb = i * this.nl * 3;
          for (let l = 0; l < this.nl; l++) {
            const o = lb + 3 * l;
            const a0 = tnDraw(w0[0], w0[1], sPos, this.lu[o + 1]);
            const a1 = tnDraw(w1[0], w1[1], sPos, this.lu[o + 2]);
            let k = 2, cum = 0; const u = this.lu[o];
            for (let j = 0; j < 3; j++) { cum += p[j]; if (u < cum) { k = j; break; } }
            this.s(a0, a1, sc);
            const wt = Math.exp(lam * sc[k]);
            Z += wt;
            mU[0] += a0; mU[1] += a1; mU[2 + k] += 1;
            mW[0] += wt * a0; mW[1] += wt * a1; mW[2 + k] += wt;
          }
          for (let j = 0; j < 5; j++) { mU[j] /= this.nl; mW[j] /= Z; }
          const gp = this.gam.pos, gc = this.gam.col;
          const capP = this.gam.stepP * om / h, capC = this.gam.stepC * om / h;
          if (onP) {
            gy0 = gp * (mW[0] - mU[0]); gy1 = gp * (mW[1] - mU[1]);
            const n = Math.hypot(gy0, gy1); if (n > capP) { const f = capP / n; gy0 *= f; gy1 *= f; }
          }
          if (onC) {
            gc0 = gc * (mW[2] - mU[2]); gc1 = gc * (mW[3] - mU[3]); gc2 = gc * (mW[4] - mU[4]);
            const n = Math.hypot(gc0, gc1, gc2); if (n > capC) { const f = capC / n; gc0 *= f; gc1 *= f; gc2 *= f; }
          }
        }
        this.y[2 * i] = y0 + h * (m0 - y0 + gy0) / om;
        this.y[2 * i + 1] = y1 + h * (m1 - y1 + gy1) / om;
        this.c[3 * i] += h * (p[0] - this.c[3 * i] + gc0) / om;
        this.c[3 * i + 1] += h * (p[1] - this.c[3 * i + 1] + gc1) / om;
        this.c[3 * i + 2] += h * (p[2] - this.c[3 * i + 2] + gc2) / om;
      }
      this.pi = end;
      if (this.pi < NP) return false;
      this.pi = 0; this.step++;
      if (this.step % STORE === 0) this.record();
      return true;
    }
    build(ms) {                                   // advance for at most ms milliseconds, resuming mid-step
      const t0 = (global.performance || Date).now();
      while (this.step < NSTEP) {
        this.advance(CHUNK);
        if ((global.performance || Date).now() - t0 > ms) break;
      }
      return this.step >= NSTEP;
    }
  }

  const Toy = { NP, NF, NSTEP, NLOOK, NLOOK_OF, COLS, Run, sNorway, sSmiley, tnMean, tnDraw, winOf, colPost, REWARD, WINDOW, GAM, SMI, NOR, setSC: v => { SC = v; }, getSC: () => SC };
  global.MultiMFMFlag = Toy;
  if (typeof document === "undefined") return;

  // ---------- drawing ----------
  const C = { paper: "#FBF6E9", frame: "#B8AA95", ink: "#1D2D35", panel: "#121A21", grid: "rgba(255,255,255,0.10)", label: "#6B6F6E" };
  const FONT = '"Latin Modern Roman", "CMU Serif", "Times New Roman", Times, serif';
  const D = { intro: 0.7, flow: 7.5, hold: 2.8, fade: 0.5 };
  const smooth = v => { v = v < 0 ? 0 : (v > 1 ? 1 : v); return v * v * (3 - 2 * v); };

  class Flag {
    constructor(root) {
      this.root = root;
      this.stage = root.querySelector(".anim-stage");
      this.bg = this.stage.querySelector("canvas.bg"); this.fg = this.stage.querySelector("canvas.fg");
      this.read = root.querySelector(".anim-read");
      this.mode = "norway"; this.lam = 6; this.seed = 71041;
      this.playing = true;
      this.resize();
      this.rebuild();
    }
    rebuild() {
      this.base = new Run(this.seed, this.mode, 0, false);
      this.guid = new Run(this.seed, this.mode, this.lam, true);
      this.go("intro"); this.t = 0; this.alpha = 0;
    }
    newNoise() { this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff; this.rebuild(); }
    go(ph) { this.phase = ph; this.tau = 0; }

    tick(dt) {
      // keep the integration ahead of the playhead, spending only a slice of each frame on it
      const need = Math.min(NF - 1, Math.ceil(this.t * (NF - 1)) + 6);
      if (this.base.frames <= need) this.base.build(2);
      if (this.guid.frames <= need) this.guid.build(8);
      if (this.playing) {
        this.tau += dt;
        if (this.phase === "intro") { if (this.tau > D.intro) this.go("flow"); }
        else if (this.phase === "flow") {
          const want = Math.min(1, this.t + dt / D.flow);
          const ready = (Math.min(this.base.frames, this.guid.frames) - 2) / (NF - 1);
          this.t = Math.min(want, Math.max(0, ready));
          if (this.t >= 1) { this.t = 1; this.go("hold"); }
        }
        else if (this.phase === "hold") { if (this.tau > D.hold) this.go("fade"); }
        else if (this.phase === "fade") { if (this.tau > D.fade) { this.newNoise(); return; } }
      }
      const ease = (v, to, tc) => v + (to - v) * (1 - Math.exp(-dt / tc));
      this.alpha = ease(this.alpha, this.phase === "fade" ? 0 : 1, this.phase === "fade" ? 0.12 : 0.18);
    }

    resize() {
      const dpr = Math.min(Math.max(global.devicePixelRatio || 1, 2), 3);
      const w = this.stage.clientWidth, h = this.stage.clientHeight;
      if (!w || !h) return;
      for (const c of [this.bg, this.fg]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      this.W = this.bg.width; this.H = this.bg.height;
      this.px = dpr * Math.max(0.74, Math.min(1, w / 940));
      this.fs = 17 * this.px;
      this.wide = this.W / this.H > 1.1;
      const pad = 1.6 * this.fs, gap = 1.5 * this.fs;
      let side;
      if (this.wide) {
        side = Math.min((this.W - 2 * pad - gap) / 2, this.H - 2 * pad);
        const x0 = (this.W - 2 * side - gap) / 2, y0 = (this.H - side) / 2;
        this.boxes = [[x0, y0, side], [x0 + side + gap, y0, side]];
      } else {
        side = Math.min(this.W - 2 * pad, (this.H - 2 * pad - gap) / 2);
        const x0 = (this.W - side) / 2, y0 = (this.H - 2 * side - gap) / 2;
        this.boxes = [[x0, y0, side], [x0, y0 + side + gap, side]];
      }
      this.drawStatic();
    }
    drawStatic() {
      const ctx = this.bg.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.W, this.H);
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, this.W, this.H);
      for (const b of this.boxes) { ctx.fillStyle = C.panel; ctx.fillRect(b[0], b[1], b[2], b[2]); }
      ctx.strokeStyle = C.frame; ctx.lineWidth = 1 * this.px;
      ctx.strokeRect(0.5 * this.px, 0.5 * this.px, this.W - this.px, this.H - this.px);
    }
    draw() {
      const ctx = this.fg.getContext("2d"), px = this.px;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.W, this.H);
      const f = this.t * (NF - 1), i0 = Math.min(NF - 2, Math.floor(f)), sfr = f - i0;
      const r = 1.65 * px, grow = this.phase === "intro" ? smooth(this.tau / 0.35) : 1;
      // the noise is three times wider than the data square, so the view closes in as the flow runs
      const view = (1 - this.t) * 2.75 + this.t * 1.04;
      ctx.save();
      ctx.globalAlpha = this.alpha * 0.30 * smooth((this.t - 0.25) / 0.35);
      ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1 * px;
      for (const b of this.boxes) {
        const sc = 0.5 * b[2] / view, cx = b[0] + 0.5 * b[2], cy = b[1] + 0.5 * b[2];
        ctx.save();
        ctx.beginPath(); ctx.rect(b[0], b[1], b[2], b[2]); ctx.clip();
        ctx.strokeRect(cx - sc, cy - sc, 2 * sc, 2 * sc);
        ctx.restore();
      }
      ctx.restore();
      ctx.globalAlpha = this.alpha * 0.92;
      // Batch by quantised colour: one fillStyle and one fill per bucket instead of per particle.
      // Setting fillStyle from a fresh string 8,000 times a frame is what makes this slow.
      const bk = this.bk || (this.bk = new Map());
      for (let p = 0; p < 2; p++) {
        const run = p === 0 ? this.base : this.guid, b = this.boxes[p];
        if (run.frames < 1) continue;
        for (const v of bk.values()) v.n = 0;
        // fall back to the newest finished frame when the next one is not built yet
        const f0 = Math.min(i0, run.frames - 1), f1 = Math.min(i0 + 1, run.frames - 1);
        const fr = f1 > f0 ? sfr : 0;
        const po = run.out.posF, co = run.out.colF, sc = 0.5 * b[2] / view, cx = b[0] + 0.5 * b[2], cy = b[1] + 0.5 * b[2];
        for (let i = 0; i < NP; i++) {
          const a = (f0 * NP + i) * 2, c = (f1 * NP + i) * 2;
          const x = po[a] + (po[c] - po[a]) * fr, y = po[a + 1] + (po[c + 1] - po[a + 1]) * fr;
          const ca = (f0 * NP + i) * 3, cc = (f1 * NP + i) * 3;
          const rr = co[ca] + (co[cc] - co[ca]) * fr, gg = co[ca + 1] + (co[cc + 1] - co[ca + 1]) * fr, bb = co[ca + 2] + (co[cc + 2] - co[ca + 2]) * fr;
          const qr = (rr * 0.0235294) | 0, qg = (gg * 0.0235294) | 0, qb = (bb * 0.0235294) | 0;   // 7 levels per channel
          const key = (qr * 7 + qg) * 7 + qb;
          let v = bk.get(key);
          if (!v) { v = { n: 0, xy: new Float32Array(4 * NP), col: "rgb(" + ((qr * 42.5) | 0) + "," + ((qg * 42.5) | 0) + "," + ((qb * 42.5) | 0) + ")" }; bk.set(key, v); }
          v.xy[2 * v.n] = cx + x * sc; v.xy[2 * v.n + 1] = cy - y * sc; v.n++;
        }
        // the noise reaches well outside the data square, so each panel keeps its own particles
        ctx.save();
        ctx.beginPath(); ctx.rect(b[0], b[1], b[2], b[2]); ctx.clip();
        const d = Math.max(1, Math.round(2 * r * grow)), o = d / 2;
        for (const v of bk.values()) {
          if (!v.n) continue;
          ctx.fillStyle = v.col;
          for (let k = 0; k < v.n; k++) ctx.fillRect(v.xy[2 * k] - o, v.xy[2 * k + 1] - o, d, d);
        }
        ctx.restore();
      }
      ctx.globalAlpha = 1;
    }
    readout() {
      const f = Math.min(NF - 1, Math.round(this.t * (NF - 1)));
      const nm = { norway: "Norway flag", smiley: "smiley", none: "no reward" }[this.mode];
      let s = "flow time t = " + this.t.toFixed(2) + " · " + nm;
      if (this.mode !== "none" && this.base.frames > f && this.guid.frames > f) {
        s += " · mean reward " + this.base.out.scoreF[f].toFixed(2) + " unguided, " + this.guid.out.scoreF[f].toFixed(2) + " steered";
      }
      if (s !== this.last) { this.read.textContent = s; this.last = s; }
    }
  }

  function init() {
    const root = document.getElementById("flag-anim");
    if (!root) return;
    const anim = new Flag(root);
    Toy.anim = anim;
    const playBtn = root.querySelector("#flag-play"), newBtn = root.querySelector("#flag-new");
    const lam = root.querySelector("#flag-lam"), lamVal = root.querySelector("#flag-lam-val");
    const segs = root.querySelectorAll("#flag-mode button");
    const reduced = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let visible = true, last = null, raf = 0, acc = 0;
    function frame(now) {
      raf = 0;
      const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.05); last = now;
      anim.tick(dt); anim.draw();
      acc += dt; if (acc > 0.12 || dt === 0) { anim.readout(); acc = 0; }
      const settled = !anim.playing && anim.base.step >= NSTEP && anim.guid.step >= NSTEP
        && Math.abs(anim.alpha - (anim.phase === "fade" ? 0 : 1)) < 0.004;
      if (visible && !document.hidden && !settled) raf = requestAnimationFrame(frame);
    }
    function wake() { if (!raf && visible && !document.hidden) { last = null; raf = requestAnimationFrame(frame); } }
    function setPlaying(p) { anim.playing = p; playBtn.textContent = p ? "Pause" : "Play"; wake(); }

    playBtn.addEventListener("click", () => setPlaying(!anim.playing));
    newBtn.addEventListener("click", () => { anim.newNoise(); setPlaying(true); });
    lam.addEventListener("input", () => { anim.lam = parseFloat(lam.value); lamVal.textContent = anim.lam.toFixed(0); anim.rebuild(); wake(); });
    segs.forEach(b => b.addEventListener("click", () => {
      anim.mode = b.dataset.m;
      segs.forEach(o => { const on = o === b; o.classList.toggle("on", on); o.setAttribute("aria-pressed", on ? "true" : "false"); });
      anim.rebuild(); wake();
    }));

    if (reduced) {
      anim.playing = false; playBtn.textContent = "Play";
      anim.base.build(1e9); anim.guid.build(1e9);
      anim.t = 1; anim.alpha = 1; anim.phase = "hold";
    }
    if ("IntersectionObserver" in global) new IntersectionObserver(es => { visible = es[0].isIntersecting; wake(); }, { rootMargin: "120px" }).observe(anim.stage);
    document.addEventListener("visibilitychange", wake);
    let rt = null;
    global.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { anim.resize(); wake(); }, 120); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { anim.drawStatic(); wake(); });
    wake();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof window !== "undefined" ? window : globalThis);
