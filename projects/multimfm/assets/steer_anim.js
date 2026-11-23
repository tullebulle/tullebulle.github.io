/* Animated Figure 2 of the multiMFM paper: inference-time steering with posterior look-aheads, on a toy whose
   posterior is known in closed form, drawn in the layout of the paper's figure.

   Data: four components k = 1..4. The continuous coordinate is y1 ~ N(MU[k], SD[k]^2) and the discrete one is the
   simplex vertex a1 = e_k, so position and category are coupled. The simplex of four categories is the tetrahedron
   in R^3, drawn in perspective as in the paper. Noise: y0 ~ N(0, 1) and a0 ~ N(A0, SA^2 I_3) below the tetrahedron.
   Interpolant x_t = (1 - t) x0 + t x1.

   The base velocity is exact, b_t(x) = (E[x1 | x_t = x] - x) / (1 - t). The look-aheads are NLOOK draws from the exact
   joint posterior p_{1|t}(y1, a1 | y_t, a_t), made from fixed noises so that they move continuously with (t, x_t), as
   the outputs X_{0,1}(eps; t, x_t) of a Meta Flow Map do. Steering is the paper's gradient-free estimator (Eq. 4),
       (sigma_t^2 / 2) grad V_t(x_t)  ~  ( sum_i softmax_i(r(x1_i)) x1_i  -  mean_i x1_i ) / (1 - t),
   with the reward r(x1) = lambda 1[a1 = e_target] and guidance switched on at t = T_ON. The gold arrows are that
   difference of means, the red path integrates base velocity plus it, and the dashed path is the base flow alone. */
(function (global) {
  "use strict";

  // ---------- the toy ----------
  const K = 4;
  const W_K = [0.27, 0.28, 0.24, 0.21];
  const MU = [-2.05, -0.72, 0.68, 1.98], SD = [0.30, 0.28, 0.26, 0.24];

  // the 3-simplex: a regular tetrahedron, circumradius 1, centred at the origin, apex up (+Z)
  const VTX = (function () {
    const rb = Math.sqrt(8) / 3, zb = -1 / 3, d = Math.PI / 180;
    const ring = a => [rb * Math.cos(a * d), rb * Math.sin(a * d), zb];
    return [ring(210), [0, 0, 1], ring(330), ring(90)];   // front-left, apex, front-right, back
  })();
  const A0 = [0.02, 0.05, -2.00], SA = 0.36;
  const T_ON = 0.28, NSTEP = 400, NLOOK = 32;

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  // inverse normal CDF (Acklam), |error| < 1.2e-9: turns stratified uniforms into evenly spread Gaussian draws
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

  // the look-ahead noises eps_i = (u_i, z_i): u picks the component, z the position inside it. Fixed for the page.
  const LOOK = (function () {
    const rng = mulberry32(51172);
    const perm = n => { const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const s = a[i]; a[i] = a[j]; a[j] = s; } return a; };
    const pu = perm(NLOOK), pz = perm(NLOOK), pj = perm(NLOOK);
    return pu.map((u, i) => ({
      u: (u + 0.5) / NLOOK, z: invNorm((pz[i] + 0.5) / NLOOK),
      bend: 0.26 + 0.40 * rng(), side: rng() - 0.5,                     // shape of the drawn curve
      jr: Math.cbrt((pj[i] + 0.5) / NLOOK),                             // where it lands around its vertex
      ja: pj[i] * 2.39996323, jb: Math.acos(1 - 2 * ((pj[i] * 0.6180339887) % 1)),
    }));
  })();

  // p(k | x_t) and the Gaussian law of y1 given k and y_t
  function posterior(t, y, a) {
    const om = 1 - t, va = Math.max(om * om * SA * SA, 1e-12);
    const lp = new Array(K), mean = new Array(K), sd = new Array(K);
    let mx = -Infinity;
    for (let k = 0; k < K; k++) {
      const vy = Math.max(om * om + t * t * SD[k] * SD[k], 1e-12), dy = y - t * MU[k];
      let q = 0;
      for (let j = 0; j < 3; j++) { const d = a[j] - om * A0[j] - t * VTX[k][j]; q += d * d; }
      lp[k] = Math.log(W_K[k]) - 0.5 * dy * dy / vy - 0.5 * Math.log(vy) - 0.5 * q / va;
      mean[k] = MU[k] + t * SD[k] * SD[k] * dy / vy;
      sd[k] = SD[k] * om / Math.sqrt(vy);
      if (lp[k] > mx) mx = lp[k];
    }
    let s = 0;
    const p = lp.map(v => { const e = Math.exp(v - mx); s += e; return e; });
    for (let k = 0; k < K; k++) p[k] /= s;
    return { p, mean, sd };
  }
  function postMean(post) {
    const m = [0, 0, 0, 0];
    for (let k = 0; k < K; k++) { m[0] += post.p[k] * post.mean[k]; for (let j = 0; j < 3; j++) m[j + 1] += post.p[k] * VTX[k][j]; }
    return m;
  }
  function lookaheads(post) {
    const out = new Array(NLOOK);
    for (let i = 0; i < NLOOK; i++) {
      const e = LOOK[i];
      let k = K - 1, c = 0;
      for (let j = 0; j < K; j++) { c += post.p[j]; if (e.u < c) { k = j; break; } }
      out[i] = { k, y: post.mean[k] + post.sd[k] * e.z };
    }
    return out;
  }
  // plain and reward-weighted means of the look-aheads, as [y, a_x, a_y, a_z]
  function aims(la, target, lam) {
    const mU = [0, 0, 0, 0], mW = [0, 0, 0, 0], eLam = Math.exp(lam);
    let Z = 0, hits = 0;
    for (const s of la) {
      const hit = s.k === target, w = hit ? eLam : 1, v = VTX[s.k];
      if (hit) hits++;
      Z += w;
      mU[0] += s.y; mW[0] += w * s.y;
      for (let j = 0; j < 3; j++) { mU[j + 1] += v[j]; mW[j + 1] += w * v[j]; }
    }
    for (let j = 0; j < 4; j++) { mU[j] /= NLOOK; mW[j] /= Z; }
    return { mU, mW, hits, share: hits * eLam / Z };
  }
  // the base path U and the steered path G from the same noise; they coincide up to T_ON
  function integrate(x0, target, lam) {
    const U = [x0.slice()], G = [x0.slice()], h = 1 / NSTEP;
    let u = x0.slice(), g = x0.slice();
    for (let i = 0; i < NSTEP; i++) {
      const t = i * h, om = 1 - t;
      const mu = postMean(posterior(t, u[0], u.slice(1)));
      u = u.map((v, j) => v + h * (mu[j] - v) / om);
      if (t < T_ON - 1e-9) g = u.slice();
      else {
        const post = posterior(t, g[0], g.slice(1)), mg = postMean(post);
        const a = lam > 0 ? aims(lookaheads(post), target, lam) : null;
        g = g.map((v, j) => v + h * (mg[j] - v + (a ? a.mW[j] - a.mU[j] : 0)) / om);
      }
      U.push(u); G.push(g);
    }
    return { U, G };
  }
  function nearestVertex(x) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < K; k++) {
      let d = 0; for (let j = 0; j < 3; j++) { const q = x[j + 1] - VTX[k][j]; d += q * q; }
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }
  const I_ON = Math.round(T_ON * NSTEP);
  function hitsAtOnset(G, target, lam) {
    const g = G[I_ON];
    return aims(lookaheads(posterior(T_ON, g[0], g.slice(1))), target, lam).hits;
  }

  const Toy = { K, W_K, MU, SD, VTX, A0, SA, T_ON, NSTEP, NLOOK, posterior, postMean, lookaheads, aims, integrate, nearestVertex, hitsAtOnset };
  global.MultiMFMToy = Toy;
  if (typeof document === "undefined") return;

  // ---------- drawing ----------
  const C = {
    paper: "#FBF6E9", halo: "rgba(251,246,233,0.86)", frame: "#B8AA95", ink: "#1D2D35", soft: "#6B6F6E",
    floor: "rgba(255,255,255,0.62)", floorEdge: "rgba(29,45,53,0.16)",
    p0Fill: "rgba(222,219,210,0.80)", p0Line: "#7C786F", p0Text: "#77736A",
    p1Fill: "#DCE9D4", p1Line: "#2E5C4E", p1Text: "#3B7A64",
    red: "#A01228", rose: "#C8617A", gold: "#DCA51F", goldText: "#A87806", blue: "#2389C9",
  };
  const GRAY = [70, 76, 82], ROSE = [200, 97, 122], RED = [160, 18, 40];
  const FONT = '"Latin Modern Roman", "CMU Serif", "Times New Roman", Times, serif';
  const YR = 3.0, JIT = 0.13, P0_H = 0.62;        // P0_H: p0(y0) is drawn at a lower height scale than p1(y1)
  const VR = W_K.map(w => 0.30 * Math.sqrt(w / 0.25));     // vertex balls, area proportional to the class weight

  // camera for the tetrahedron: yaw about the vertical axis, then an elevation tilt. depth grows away from the eye.
  const YAW = 0.40, ELE = 0.34;
  const CA = Math.cos(YAW), SAW = Math.sin(YAW), CE = Math.cos(ELE), SE = Math.sin(ELE);
  function cam(p) {
    const u = p[0] * CA + p[1] * SAW, v = -p[0] * SAW + p[1] * CA;
    return [u, p[2] * CE + v * SE, v * CE - p[2] * SE];      // [right, up, depth]
  }
  const FACES = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];   // face f omits vertex f
  const EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const faceDepth = f => (cam(VTX[FACES[f][0]])[2] + cam(VTX[FACES[f][1]])[2] + cam(VTX[FACES[f][2]])[2]) / 3;
  // the face opposite vertex f has outward normal along -V_f; it faces the eye when that normal's depth is negative
  const faceFront = f => cam([-VTX[f][0], -VTX[f][1], -VTX[f][2]])[2] < 0;

  const fontOf = (st, fs) => (st === "i" || st === "si" ? "italic " : "") + (st[0] === "s" ? 0.68 * fs : fs).toFixed(2) + "px " + FONT;
  // a line of maths: parts are [text, style] with style i (italic), r (upright), s (upright subscript), si (italic subscript)
  function mtext(ctx, parts, x, y, fs, color, align, halo) {
    let w = 0; const ws = [];
    for (const p of parts) { ctx.font = fontOf(p[1], fs); const m = ctx.measureText(p[0]).width; ws.push(m); w += m; }
    const x0 = align === "center" ? x - w / 2 : (align === "right" ? x - w : x);
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    for (let pass = halo ? 0 : 1; pass < 2; pass++) {
      let cx = x0;
      for (let i = 0; i < parts.length; i++) {
        const st = parts[i][1]; ctx.font = fontOf(st, fs);
        const yy = y + (st[0] === "s" ? 0.26 * fs : 0);
        if (pass === 0) { ctx.lineWidth = fs * 0.34; ctx.strokeStyle = halo; ctx.lineJoin = "round"; ctx.strokeText(parts[i][0], cx, yy); }
        else { ctx.fillStyle = color; ctx.fillText(parts[i][0], cx, yy); }
        cx += ws[i];
      }
    }
    return w;
  }
  const TX = {
    p0y: [["p", "i"], ["0", "s"], ["(", "r"], ["y", "i"], ["0", "s"], [")", "r"]],
    p1y: [["p", "i"], ["1", "s"], ["(", "r"], ["y", "i"], ["1", "s"], [")", "r"]],
    p0a: [["p", "i"], ["0", "s"], ["(", "r"], ["a", "i"], ["0", "s"], [")", "r"]],
    p1a: [["p", "i"], ["1", "s"], ["(", "r"], ["a", "i"], ["1", "s"], [")", "r"]],
    gy: [["∇", "r"], ["y", "si"], ["V", "i"], ["t", "si"], ["(", "r"], ["x", "i"], ["t", "si"], [")", "r"]],
    ga: [["∇", "r"], ["a", "si"], ["V", "i"], ["t", "si"], ["(", "r"], ["x", "i"], ["t", "si"], [")", "r"]],
    xt: [["x", "i"], ["t", "si"], [" = (", "r"], ["y", "i"], ["t", "si"], [", ", "r"], ["a", "i"], ["t", "si"], [")", "r"]],
    r: [["r", "i"], ["(", "r"], ["x", "i"], ["1", "s"], [")", "r"]],
    t0: [["t", "i"], [" = 0", "r"]], t1: [["t", "i"], [" = 1", "r"]],
    e: k => [["e", "i"], [String(k + 1), "s"]],
    ae: k => [["a", "i"], ["1", "s"], [" = ", "r"], ["e", "i"], [String(k + 1), "s"]],
  };
  const rgba = (c, a) => "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a.toFixed(3) + ")";
  const mix = (a, b, s) => [Math.round(a[0] + (b[0] - a[0]) * s), Math.round(a[1] + (b[1] - a[1]) * s), Math.round(a[2] + (b[2] - a[2]) * s)];
  const clamp01 = v => Math.max(0, Math.min(1, v));
  const smooth = v => { v = clamp01(v); return v * v * (3 - 2 * v); };
  const gaussPdf = (y, m, s) => Math.exp(-0.5 * (y - m) * (y - m) / (s * s)) / (s * 2.5066282746);
  const p1y = y => { let v = 0; for (let k = 0; k < K; k++) v += W_K[k] * gaussPdf(y, MU[k], SD[k]); return v; };

  // a compact cloud of noise draws below the tetrahedron, with depth so it reads as a solid blob
  const CLOUD = (function () {
    const rng = mulberry32(9041), pts = [];
    for (let i = 0; i < 230; i++) {
      const g = () => { const r = Math.sqrt(-2 * Math.log(1 - rng())), th = 2 * Math.PI * rng(); return [r * Math.cos(th), r * Math.sin(th)]; };
      const a = g(), b = g(), r = Math.hypot(a[0], a[1], b[0]);
      if (r > 2.1) { i--; continue; }
      pts.push([A0[0] + SA * a[0], A0[1] + SA * a[1], A0[2] + SA * b[0]]);
    }
    return pts;
  })();

  // seconds spent in each phase of one loop
  const D = { intro: 0.7, pre: 2.3, hold: 2.1, post: 5.4, end: 2.8, fade: 0.5 };

  class Steer {
    constructor(root) {
      this.root = root;
      this.stage = root.querySelector(".anim-stage");
      this.bg = this.stage.querySelector("canvas.bg"); this.fg = this.stage.querySelector("canvas.fg");
      this.read = root.querySelector(".anim-read");
      this.target = 2; this.lam = 5;
      this.rng = mulberry32(20261120);
      this.ends = LOOK.map(() => ({ y: 0, a: [0, 0, 0], rho: 0 }));
      this.playing = true; this.dirty = true;
      this.resize();
      this.newNoise();
    }

    // ----- state -----
    newNoise() {
      let x0, paths;
      for (let tries = 0; tries < 60; tries++) {
        const g = () => { const r = Math.sqrt(-2 * Math.log(1 - this.rng())), th = 2 * Math.PI * this.rng(); return [r * Math.cos(th), r * Math.sin(th)]; };
        const a = g(), b = g(), c = g();
        if (Math.abs(c[0]) > 1.5 || Math.hypot(a[0], a[1], b[0]) > 2.1) continue;
        x0 = [c[0], A0[0] + SA * a[0], A0[1] + SA * a[1], A0[2] + SA * b[0]];
        paths = integrate(x0, this.target, this.lam);
        // show draws that the base flow sends elsewhere and that at least one look-ahead connects to the reward
        if (nearestVertex(paths.U[NSTEP]) !== this.target && (this.lam <= 0 || hitsAtOnset(paths.G, this.target, this.lam) > 0)) break;
      }
      this.x0 = x0; this.paths = paths;
      this.snapshot();
      this.go("intro"); this.t = 0.05; this.u = 0;
      this.glow = 0; this.steer = 0; this.fan = 0.55; this.snap = 0; this.alpha = 0; this.pulse = -1; this.pulseClock = 0;
      this.snapEnds = true; this.dirty = true;
    }
    recompute() {            // same noise, new reward
      this.paths = integrate(this.x0, this.target, this.lam);
      this.snapshot(); this.dirty = true;
    }
    snapshot() {             // the look-aheads at the moment guidance switches on: redrawn at the end of the loop
      const g = this.paths.G[I_ON], la = lookaheads(posterior(T_ON, g[0], g.slice(1)));
      this.shot = { st: g, a: aims(la, this.target, this.lam), ends: la.map(s => ({ y: s.y, a: VTX[s.k].slice(), rho: s.k === this.target ? 1 : 0 })) };
    }
    go(phase) { this.phase = phase; this.tau = 0; this.pulsed = false; }
    at(path, t) {
      const f = clamp01(t) * NSTEP, i = Math.min(NSTEP - 1, Math.floor(f)), s = f - i, a = path[i], b = path[i + 1];
      return a.map((v, j) => v + s * (b[j] - v));
    }
    active() { return this.lam > 0 && this.shot.a.hits > 0; }

    tick(dt) {
      const live = this.active();
      if (this.playing) {
        this.tau += dt;
        if (this.phase === "intro") { if (this.tau > D.intro) this.go("pre"); }
        else if (this.phase === "pre") { this.t = Math.min(T_ON, this.t + dt * T_ON / D.pre); if (this.t >= T_ON) { this.t = T_ON; this.go("hold"); } }
        else if (this.phase === "hold") { if (this.tau > (live ? D.hold : 0.5)) { this.go("post"); this.u = 0; this.pulseClock = 0; } }
        else if (this.phase === "post") {
          this.u = Math.min(1, this.u + dt / D.post);
          this.t = T_ON + (1 - T_ON) * (0.5 * this.u + 0.5 * this.u * this.u);     // unhurried while the look-aheads change over
          if (this.u >= 1) { this.t = 1; this.go("end"); }
        }
        else if (this.phase === "end") { if (this.tau > D.end) this.go("fade"); }
        else if (this.phase === "fade") { if (this.tau > D.fade) { this.newNoise(); return; } }
        // the reward signal travelling back along the rewarded look-aheads
        if (live) {
          if (this.phase === "hold" && !this.pulsed && this.tau > 0.75) { this.pulse = 0; this.pulsed = true; }
          if (this.phase === "post" && this.t < 0.74) { this.pulseClock += dt; if (this.pulseClock > 1.5) { this.pulseClock = 0; this.pulse = 0; } }
        }
        if (this.pulse >= 0) { this.pulse += dt / 0.8; if (this.pulse > 1) this.pulse = -1; }
      }
      const past = this.phase === "post" || this.phase === "end" || this.phase === "fade";
      const ease = (v, target, tc) => v + (target - v) * (1 - Math.exp(-dt / tc));
      this.fan = ease(this.fan, 1, 0.25);
      this.glow = ease(this.glow, live && (past || (this.phase === "hold" && this.tau > 0.25)) ? 1 : 0, 0.16);
      this.steer = ease(this.steer, live && (past || (this.phase === "hold" && this.tau > 1.35)) ? 1 : 0, 0.16);
      this.snap = ease(this.snap, (this.phase === "end" && this.tau > 0.2) ? 1 : 0, this.phase === "end" ? 0.3 : 0.12);
      this.alpha = ease(this.alpha, this.phase === "fade" ? 0 : 1, this.phase === "fade" ? 0.12 : 0.18);

      // the look-aheads from the current steered state; the drawn endpoints ease toward them
      const g = this.at(this.paths.G, this.t);
      const la = lookaheads(posterior(Math.min(this.t, 1 - 1e-6), g[0], g.slice(1)));
      this.cur = { g, u: this.at(this.paths.U, this.t), a: aims(la, this.target, this.lam) };
      const k = this.snapEnds ? 1 : 1 - Math.exp(-dt / 0.11);
      for (let i = 0; i < NLOOK; i++) {
        const e = this.ends[i], s = la[i], v = VTX[s.k];
        e.y += (s.y - e.y) * k;
        for (let j = 0; j < 3; j++) e.a[j] += (v[j] - e.a[j]) * k;
        e.rho += ((s.k === this.target ? 1 : 0) - e.rho) * k;
      }
      this.snapEnds = false;
    }

    // ----- layout -----
    resize() {
      const dpr = Math.min(Math.max(global.devicePixelRatio || 1, 2), 3);
      const w = this.stage.clientWidth, h = this.stage.clientHeight;
      if (!w || !h) return;
      for (const c of [this.bg, this.fg]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      const W = this.bg.width, H = this.bg.height;
      this.W = W; this.H = H; this.wide = W / H > 1.25;
      this.px = dpr * Math.max(0.74, Math.min(1, w / 940));           // one design pixel
      this.fs = 17.5 * this.px;
      const L = this.wide ? { x: 0, y: 0, w: 0.588 * W, h: H } : { x: 0, y: 0, w: W, h: 0.70 * W };
      const R = this.wide ? { x: 0.592 * W, y: 0, w: 0.408 * W, h: H } : { x: 0, y: 0.70 * W, w: W, h: H - 0.70 * W };
      this.L = L; this.R = R;
      // left: oblique view of the (y, t) plane, densities standing on it
      const o = [L.x + 0.042 * L.w, L.y + 0.855 * L.h], ey = [0.665 * L.w / (2 * YR), 0.075 * L.h / (2 * YR)], et = [0.243 * L.w, -0.47 * L.h], hh = 0.40 * L.h;
      this.PL = (y, t, d) => [o[0] + (y + YR) * ey[0] + t * et[0], o[1] + (y + YR) * ey[1] + t * et[1] - (d || 0) * hh];
      this.eyDir = (function () { const n = Math.hypot(ey[0], ey[1]); return [ey[0] / n, ey[1] / n]; })();
      // right: perspective view of the tetrahedron, with the noise cloud below it
      let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
      const bound = (p, pad) => {
        const q = cam(p);
        lo[0] = Math.min(lo[0], q[0] - pad); hi[0] = Math.max(hi[0], q[0] + pad);
        lo[1] = Math.min(lo[1], q[1] - pad); hi[1] = Math.max(hi[1], q[1] + pad);
      };
      VTX.forEach((v, k) => bound(v, VR[k] + 0.40));                       // room for the ball and its label
      bound([A0[0], A0[1], A0[2] - 1.9 * SA], 0.10);
      bound([A0[0] + 1.9 * SA, A0[1], A0[2]], 0.10);
      bound([A0[0] - 1.9 * SA, A0[1], A0[2]], 0.10);
      if (!this.wide) hi[1] += 0.42;        // headroom under the stacked layout's caption
      const s = Math.min(R.w * 0.98 / (hi[0] - lo[0]), R.h * 0.98 / (hi[1] - lo[1]));
      const cx = R.x + R.w / 2 - s * (lo[0] + hi[0]) / 2, cy = R.y + R.h / 2 + s * (lo[1] + hi[1]) / 2;
      this.PR = a => { const q = cam(a); return [cx + s * q[0], cy - s * q[1]]; };
      this.sR = s;
      this.drawStatic(); this.dirty = true;
    }

    // ----- the tetrahedron -----
    // a ball at a simplex vertex, shaded so the set reads as a solid in space
    ball(ctx, k, on, tg) {
      const p = this.PR(VTX[k]), r = VR[k] * this.sR * (0.80 + 0.20 * (1 - clamp01((cam(VTX[k])[2] + 1.2) / 2.4)));
      const hot = on && k === tg;
      const g = ctx.createRadialGradient(p[0] - 0.36 * r, p[1] - 0.42 * r, 0.08 * r, p[0], p[1], r);
      if (hot) { g.addColorStop(0, "#F3D6DB"); g.addColorStop(0.55, "#DB9DA9"); g.addColorStop(1, "#A8314A"); }
      else { g.addColorStop(0, "#F2F7EE"); g.addColorStop(0.55, "#D4E4CB"); g.addColorStop(1, "#8FAE92"); }
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = g; ctx.fill();
      ctx.strokeStyle = hot ? C.red : "rgba(46,92,78,0.75)"; ctx.lineWidth = (hot ? 1.8 : 1.2) * this.px; ctx.stroke();
      return { p, r };
    }
    drawTetra(ctx, on, tg) {
      const px = this.px, PR = this.PR;
      const poly = pts => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); };
      const order = [0, 1, 2, 3].sort((a, b) => faceDepth(b) - faceDepth(a));      // far faces first
      const tgP = PR(VTX[tg]), lam = clamp01(this.lam / 4);
      for (const f of order) {
        const pts = FACES[f].map(i => PR(VTX[i])), front = faceFront(f);
        poly(pts);
        ctx.fillStyle = front ? "rgba(255,255,255,0.30)" : "rgba(238,236,226,0.52)";
        ctx.fill();
        if (on && FACES[f].indexOf(tg) >= 0) {        // the reward lights the faces that meet the rewarded vertex
          const g = ctx.createRadialGradient(tgP[0], tgP[1], 0, tgP[0], tgP[1], 1.35 * this.sR);
          g.addColorStop(0, rgba(RED, (front ? 0.46 : 0.26) * lam));
          g.addColorStop(0.5, rgba(RED, (front ? 0.15 : 0.09) * lam));
          g.addColorStop(1, rgba(RED, 0));
          ctx.save(); poly(pts); ctx.clip(); ctx.fillStyle = g; ctx.fillRect(this.R.x, this.R.y, this.R.w, this.R.h); ctx.restore();
        }
      }
      // edges: an edge hidden behind the solid (both its faces turned away) is drawn light and dashed
      for (const e of EDGES) {
        const hidden = [0, 1, 2, 3].every(f => FACES[f].indexOf(e[0]) < 0 || FACES[f].indexOf(e[1]) < 0 || !faceFront(f));
        const a = PR(VTX[e[0]]), b = PR(VTX[e[1]]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.setLineDash(hidden ? [4 * px, 4 * px] : []);
        ctx.strokeStyle = hidden ? "rgba(29,45,53,0.30)" : "rgba(29,45,53,0.85)";
        ctx.lineWidth = (hidden ? 1.1 : 1.8) * px; ctx.stroke();
      }
      ctx.setLineDash([]);
      const byDepth = [0, 1, 2, 3].sort((a, b) => cam(VTX[b])[2] - cam(VTX[a])[2]);
      return byDepth.map(k => ({ k, g: this.ball(ctx, k, on, tg) }));
    }

    // ----- static layer -----
    drawStatic() {
      const ctx = this.bg.getContext("2d"), W = this.W, H = this.H, px = this.px, fs = this.fs, PL = this.PL, PR = this.PR, tg = this.target;
      const on = this.lam > 0;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.paper; ctx.fillRect(0, 0, W, H);
      const poly = pts => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); };
      ctx.lineJoin = "round"; ctx.lineCap = "round";

      // the (y, t) plane, shaded so the far edge sits back
      poly([PL(-YR, 0), PL(YR, 0), PL(YR, 1), PL(-YR, 1)]); ctx.closePath();
      const near = PL(0, 0), far = PL(0, 1);
      const pg = ctx.createLinearGradient(near[0], near[1], far[0], far[1]);
      pg.addColorStop(0, "rgba(255,255,255,0.72)"); pg.addColorStop(1, "rgba(243,240,229,0.42)");
      ctx.fillStyle = pg; ctx.fill();
      ctx.strokeStyle = C.floorEdge; ctx.lineWidth = 1 * px;
      for (const q of [0.52, 0.76]) { poly([PL(-YR, q), PL(YR, q)]); ctx.stroke(); }     // kept clear of T_ON, where the moving time line pauses
      poly([PL(YR, 0), PL(YR, 1)]); ctx.stroke();
      // time arrow
      const a0 = PL(-YR, -0.04), a1 = PL(-YR, 1.2);
      this.arrow(ctx, a0[0], a0[1], a1[0], a1[1], C.ink, 1.7 * px, 9 * px);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.7 * px;
      poly([PL(-YR, 0), PL(YR, 0)]); ctx.stroke();
      poly([PL(-YR, 1), PL(YR, 1)]); ctx.stroke();
      let q = PL(-YR, 0); mtext(ctx, TX.t0, q[0] + 2 * px, q[1] + 1.15 * fs, 0.95 * fs, C.ink, "left");
      q = PL(-YR, 1); mtext(ctx, TX.t1, q[0] - 0.55 * fs, q[1] + 0.1 * fs, 0.95 * fs, C.ink, "right");

      // p1(y1) standing on t = 1, with the rewarded component tinted
      const dens = (f, t) => { const pts = [PL(-YR, t)]; for (let i = 0; i <= 240; i++) { const y = -YR + 2 * YR * i / 240; pts.push(PL(y, t, f(y))); } pts.push(PL(YR, t)); return pts; };
      poly(dens(p1y, 1)); ctx.closePath(); ctx.fillStyle = C.p1Fill; ctx.fill();
      poly(dens(p1y, 1).slice(1, -1)); ctx.strokeStyle = C.p1Line; ctx.lineWidth = 1.6 * px; ctx.stroke();
      // the reward, as in the paper's figure: a bell over the rewarded mode, narrower than the data mode and
      // standing a little above it. The tilt itself is on the category; this is drawn at a readable height.
      this.rbell = null;
      if (on) {
        const a = clamp01(this.lam / 4), rs = 0.66 * SD[tg], rh = 1.36 * p1y(MU[tg]);
        const bell = y => rh * Math.exp(-0.5 * (y - MU[tg]) * (y - MU[tg]) / (rs * rs));
        poly(dens(bell, 1)); ctx.closePath(); ctx.fillStyle = rgba(ROSE, 0.34 * a); ctx.fill();
        const lo = MU[tg] - 3.4 * rs, hi = MU[tg] + 3.4 * rs, pts = [];
        for (let i = 0; i <= 120; i++) { const y = lo + (hi - lo) * i / 120; pts.push(PL(y, 1, bell(y))); }
        poly(pts); ctx.strokeStyle = rgba(RED, 0.88 * a); ctx.lineWidth = 2 * px; ctx.stroke();
        this.rbell = { f: bell, rs, rh };
      }
      for (let k = 0; k < K; k++) {
        const top = on && k === tg ? Math.max(p1y(MU[k]), this.rbell.rh) : p1y(MU[k]);
        q = PL(MU[k], 1, top);
        mtext(ctx, TX.ae(k), q[0], q[1] - 0.5 * fs, 0.74 * fs, on && k === tg ? C.red : C.p1Text, "center", C.halo);
      }
      q = PL(MU[0] - 0.85, 1, 0.56); mtext(ctx, TX.p1y, q[0], q[1], 1.14 * fs, C.p1Text, "center");

      // the simplex: noise cloud below, then the solid
      const cl = CLOUD.map(p => ({ q: PR(p), d: cam(p)[2] })).sort((a, b) => b.d - a.d);
      for (const c of cl) {
        const near01 = clamp01((2.0 - c.d) / 2.4);
        ctx.beginPath(); ctx.arc(c.q[0], c.q[1], (1.2 + 1.7 * near01) * px, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(84,82,76," + (0.12 + 0.26 * near01).toFixed(3) + ")"; ctx.fill();
      }
      const balls = this.drawTetra(ctx, on, tg);
      this.ballAt = {}; for (const b of balls) this.ballAt[b.k] = b.g;

      // vertex labels, placed away from the centre of the solid
      const cen = [0, 1, 2, 3].reduce((s, k) => { const p = PR(VTX[k]); return [s[0] + p[0] / 4, s[1] + p[1] / 4]; }, [0, 0]);
      for (let k = 0; k < K; k++) {
        const b = this.ballAt[k], dx = b.p[0] - cen[0], dy = b.p[1] - cen[1];
        const col = on && k === tg ? C.red : C.p1Text;
        if (this.wide && dy < -1.1 * Math.abs(dx)) mtext(ctx, TX.e(k), b.p[0], b.p[1] - b.r - 0.45 * fs, 1.04 * fs, col, "center", C.halo);
        else {
          const n = Math.hypot(dx, dy) || 1, off = b.r + 0.42 * fs;
          mtext(ctx, TX.e(k), b.p[0] + dx / n * off, b.p[1] + dy / n * off + 0.32 * fs, 1.04 * fs, col, dx < 0 ? "right" : "left", C.halo);
        }
      }
      const apex = [0, 1, 2, 3].reduce((a, k) => this.ballAt[k].p[1] < this.ballAt[a].p[1] ? k : a, 0);
      const ab = this.ballAt[apex];
      mtext(ctx, TX.p1a, ab.p[0] + ab.r + 0.9 * fs, ab.p[1] + 0.1 * fs, 1.14 * fs, C.p1Text, "left", C.halo);
      q = PR([A0[0] + 2.0 * SA, A0[1], A0[2] - 0.6 * SA]); mtext(ctx, TX.p0a, q[0] + 0.5 * fs, q[1], 1.14 * fs, C.p0Text, "left", C.halo);

      // r(x1): one reward, read on both coordinates
      if (on) {
        const by = MU[tg] + 1.15 * this.rbell.rs;                       // on the bell's flank, below the a1 = e_k label
        const bump = PL(by, 1, this.rbell.f(by));
        const b = this.ballAt[tg];
        const corner = [b.p[0] + 0.30 * (cen[0] - b.p[0]), b.p[1] + 0.30 * (cen[1] - b.p[1])];
        if (this.wide) {
          // keep the label near its own bump, so the leader does not cross the other mode labels
          const lx = Math.max(this.L.x + 0.42 * this.L.w, Math.min(this.L.x + 0.97 * this.L.w, bump[0] + 0.20 * this.L.w)), ly = 0.085 * H;
          ctx.strokeStyle = rgba(RED, 0.85); ctx.lineWidth = 1 * px;
          poly([[lx - 0.9 * fs, ly + 0.35 * fs], bump]); ctx.stroke();
          poly([[lx + 0.9 * fs, ly + 0.35 * fs], corner]); ctx.stroke();
          mtext(ctx, TX.r, lx, ly, 1.3 * fs, C.red, "center", C.paper);
        } else {
          // the rewarded density is already drawn in red and labelled; name the reward once, by the rewarded vertex
          ctx.strokeStyle = rgba(RED, 0.8); ctx.lineWidth = 1 * px;
          const rx = b.p[0] + (b.p[0] < cen[0] ? -1 : 1) * (b.r + 0.35 * fs), ry = b.p[1] + b.r + 1.9 * fs;
          poly([[rx, ry - 0.35 * fs], [b.p[0] + (b.p[0] < cen[0] ? -0.5 : 0.5) * b.r, b.p[1] + 0.72 * b.r]]); ctx.stroke();
          mtext(ctx, TX.r, rx, ry, 1.2 * fs, C.red, b.p[0] < cen[0] ? "right" : "left", C.halo);
        }
      }
      ctx.strokeStyle = C.frame; ctx.lineWidth = 1 * px; ctx.strokeRect(0.5 * px, 0.5 * px, W - px, H - px);
    }

    arrow(ctx, x0, y0, x1, y1, color, lw, head) {
      const dx = x1 - x0, dy = y1 - y0, n = Math.hypot(dx, dy);
      if (n < 1e-3) return;
      const ux = dx / n, uy = dy / n, hl = Math.min(head, n * 0.7), bx = x1 - ux * hl, by = y1 - uy * hl;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(bx + ux * hl * 0.25, by + uy * hl * 0.25); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(bx - uy * hl * 0.42, by + ux * hl * 0.42); ctx.lineTo(bx + uy * hl * 0.42, by - ux * hl * 0.42); ctx.closePath(); ctx.fill();
    }

    // ----- look-aheads -----
    // control point and end of look-ahead i in each panel; the curves leave along the current direction of travel
    fanGeom(st, t, aim, ends, i) {
      const e = LOOK[i], d = ends[i], PL = this.PL, PR = this.PR;
      const lS = PL(st[0], t), lC = PL(st[0] + e.bend * (aim[0] - st[0]), t + e.bend * (1 - t)), lQ = PL(d.y, 1);
      const sa = Math.sin(e.jb), jit = [JIT * e.jr * sa * Math.cos(e.ja), JIT * e.jr * sa * Math.sin(e.ja), JIT * e.jr * Math.cos(e.jb)];
      const q = [0, 1, 2].map(j => d.a[j] + jit[j]);
      const dv = [0, 1, 2].map(j => q[j] - st[j + 1]), dn = Math.hypot(dv[0], dv[1], dv[2]);
      const av = [0, 1, 2].map(j => aim[j + 1] - st[j + 1]), an = Math.hypot(av[0], av[1], av[2]);
      const m = 0.72 * clamp01(an / 0.35);
      const c = [0, 1, 2].map(j => st[j + 1] + e.bend * ((1 - m) * dv[j] + m * dn * av[j] / (an || 1)));
      c[0] -= 0.14 * e.side * dv[1]; c[1] += 0.14 * e.side * dv[0];       // a little swirl so the sheaf has body
      // how near the eye this look-ahead ends, so the sheaf in the right view carries depth
      const dep = clamp01((1.25 - cam(q)[2]) / 2.2);
      return { lS, lC, lQ, rS: PR(st.slice(1)), rC: PR(c), rQ: PR(q), dep };
    }
    drawFan(ctx, st, t, aim, ends, o) {
      const px = this.px, lit = o.glow;
      ctx.lineCap = "round";
      const part = (S, Cp, Q, s) => {        // the first fraction s of a quadratic curve
        const c = [S[0] + (Cp[0] - S[0]) * s, S[1] + (Cp[1] - S[1]) * s], m = [Cp[0] + (Q[0] - Cp[0]) * s, Cp[1] + (Q[1] - Cp[1]) * s];
        ctx.beginPath(); ctx.moveTo(S[0], S[1]); ctx.quadraticCurveTo(c[0], c[1], c[0] + (m[0] - c[0]) * s, c[1] + (m[1] - c[1]) * s); ctx.stroke();
      };
      const geoms = [];
      for (let pass = 0; pass < 2; pass++) {          // rewarded look-aheads on top
        for (let i = 0; i < NLOOK; i++) {
          const h = lit * ends[i].rho;
          if ((h > 0.5) !== (pass === 1)) continue;
          const g = geoms[i] || (geoms[i] = this.fanGeom(st, t, aim, ends, i));
          // rewarded look-aheads are rose and translucent: the solid crimson line is the steered path alone
          const col = mix(GRAY, ROSE, h), a = o.alpha * (0.34 - 0.06 * lit + 0.20 * h);
          const w = (1.05 + 0.35 * h) * px, near = 0.58 + 0.42 * g.dep;
          ctx.strokeStyle = rgba(col, a); ctx.lineWidth = w;
          part(g.lS, g.lC, g.lQ, o.grow);
          ctx.strokeStyle = rgba(col, a * near); ctx.lineWidth = w * (0.82 + 0.26 * g.dep);
          part(g.rS, g.rC, g.rQ, o.grow);
          if (o.grow > 0.97) {
            ctx.fillStyle = rgba(col, o.alpha * (0.58 + 0.28 * h));
            ctx.beginPath(); ctx.arc(g.lQ[0], g.lQ[1], (1.8 + 0.5 * h) * px, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = rgba(col, o.alpha * (0.58 + 0.28 * h) * near);
            ctx.beginPath(); ctx.arc(g.rQ[0], g.rQ[1], (1.6 + 0.5 * h + 0.5 * g.dep) * px, 0, 2 * Math.PI); ctx.fill();
          }
        }
      }
      return geoms;
    }
    drawArrows(ctx, st, t, a, s, alpha, labels) {
      if (s < 0.02 || alpha < 0.02) return;
      const px = this.px, fs = this.fs, lS = this.PL(st[0], t), rS = this.PR(st.slice(1));
      const dy = a.mW[0] - a.mU[0];
      const dv = [1, 2, 3].map(j => a.mW[j] - a.mU[j]), dn = Math.hypot(dv[0], dv[1], dv[2]);
      ctx.save(); ctx.globalAlpha = alpha;
      const reach = 0.066 * (this.wide ? this.W : 1.5 * this.W);
      const ll = s * reach * Math.tanh(Math.abs(dy) / 1.1), sg = dy < 0 ? -1 : 1;
      if (ll > 3 * px) {
        const tip = [lS[0] + sg * this.eyDir[0] * ll, lS[1] + sg * this.eyDir[1] * ll];
        this.arrow(ctx, lS[0], lS[1], tip[0], tip[1], C.gold, 2.6 * px, 9.5 * px);
        if (labels && ll > 0.5 * reach) mtext(ctx, TX.gy, tip[0] + sg * 0.35 * fs, tip[1] + 1.2 * fs, 0.98 * fs, C.goldText, sg > 0 ? "left" : "right", C.halo);
      }
      if (dn > 1e-6) {
        const far = this.PR([1, 2, 3].map(j => st[j] + dv[j - 1] / dn * 0.9));   // direction in the projected view
        const ux = far[0] - rS[0], uy = far[1] - rS[1], un = Math.hypot(ux, uy) || 1;
        const rl = s * reach * Math.tanh(dn / 0.75);
        if (rl > 3 * px) {
          const tip = [rS[0] + ux / un * rl, rS[1] + uy / un * rl];
          this.arrow(ctx, rS[0], rS[1], tip[0], tip[1], C.gold, 2.6 * px, 9.5 * px);
          if (labels && rl > 0.5 * reach) mtext(ctx, TX.ga, tip[0] + (ux < 0 ? -0.3 : 0.3) * fs, tip[1] + (uy < 0 ? -0.55 : 1.15) * fs, 0.98 * fs, C.goldText, ux < 0 ? "right" : "left", C.halo);
        }
      }
      ctx.restore();
    }

    // ----- dynamic layer -----
    draw() {
      const ctx = this.fg.getContext("2d"), W = this.W, H = this.H, px = this.px, fs = this.fs, PL = this.PL, PR = this.PR;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
      if (!this.cur) return;
      const t = this.t, g = this.cur.g, u = this.cur.u, a = this.cur.a, P = this.paths;
      const aim = [0, 1, 2, 3].map(j => a.mU[j] + this.steer * (a.mW[j] - a.mU[j]));
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.save(); ctx.globalAlpha = this.alpha;

      // the look-aheads kept from the moment guidance switched on (the paper's still), shown again at the end
      if (this.snap > 0.02) {
        const sh = this.shot;
        this.drawFan(ctx, sh.st, T_ON, sh.a.mW, sh.ends, { grow: 1, glow: this.active() ? 1 : 0, alpha: 0.8 * this.snap });
      }
      // live look-aheads from the current state
      const fade = 1 - smooth((t - 0.9) / 0.1);
      const geoms = fade > 0.01 ? this.drawFan(ctx, g, t, aim, this.ends, { grow: smooth(this.fan), glow: this.glow, alpha: fade }) : [];

      // where the flow time stands on the plane
      const tl = this.alpha * (1 - smooth((t - 0.84) / 0.08));
      if (tl > 0.02 && t > 0.03) {
        const q0 = PL(-YR, t), q1 = PL(YR, t);
        ctx.globalAlpha = tl; ctx.strokeStyle = "rgba(29,45,53,0.30)"; ctx.lineWidth = 1 * px;
        ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.stroke();
        mtext(ctx, [["t", "i"]], q0[0] - 0.5 * fs, q0[1] + 0.15 * fs, 0.95 * fs, C.ink, "right");
        ctx.globalAlpha = this.alpha;
      }
      // paths so far: shared stem, then the base flow (dashed) and the steered flow (red)
      const upto = (path, proj, i0, tEnd) => {
        const n = Math.min(NSTEP, Math.floor(tEnd * NSTEP));
        ctx.beginPath();
        for (let i = i0; i <= n; i++) { const q = proj(path[i], i / NSTEP); if (i === i0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]); }
        const q = proj(this.at(path, tEnd), tEnd); ctx.lineTo(q[0], q[1]); ctx.stroke();
      };
      const projL = (s, tt) => PL(s[0], tt), projR = s => PR(s.slice(1));
      for (const proj of [projL, projR]) {
        ctx.setLineDash([]); ctx.strokeStyle = C.ink; ctx.lineWidth = 2.3 * px;
        upto(P.U, proj, 0, Math.min(t, T_ON));
        if (t > T_ON) {
          // the base flow is the control: draw where it has reached, then ghost the rest of its path to t = 1
          ctx.setLineDash([5.5 * px, 4.5 * px]); ctx.lineWidth = 2.5 * px;
          ctx.strokeStyle = rgba([29, 45, 53], 0.30 * this.alpha); upto(P.U, proj, I_ON, 1);
          ctx.strokeStyle = C.ink; upto(P.U, proj, I_ON, t);
          ctx.setLineDash([]); ctx.strokeStyle = C.red; ctx.lineWidth = 3.1 * px; upto(P.G, proj, I_ON, t);
        }
      }
      // the reward signal coming back
      if (this.pulse >= 0 && geoms.length) {
        const s = 1 - smooth(this.pulse), fadeP = Math.sin(Math.PI * clamp01(this.pulse));
        ctx.fillStyle = rgba([220, 165, 31], 0.95 * fadeP);
        for (let i = 0; i < NLOOK; i++) {
          if (this.ends[i].rho < 0.5 || !geoms[i]) continue;
          for (const tri of [[geoms[i].lS, geoms[i].lC, geoms[i].lQ], [geoms[i].rS, geoms[i].rC, geoms[i].rQ]]) {
            const x = (1 - s) * (1 - s) * tri[0][0] + 2 * s * (1 - s) * tri[1][0] + s * s * tri[2][0];
            const y = (1 - s) * (1 - s) * tri[0][1] + 2 * s * (1 - s) * tri[1][1] + s * s * tri[2][1];
            ctx.beginPath(); ctx.arc(x, y, 2.7 * px, 0, 2 * Math.PI); ctx.fill();
          }
        }
      }
      // p0(y0) stands in front of the plane
      ctx.setLineDash([]);
      const pts = [PL(-YR, 0)];
      for (let i = 0; i <= 160; i++) { const y = -YR + 2 * YR * i / 160; pts.push(PL(y, 0, P0_H * gaussPdf(y, 0, 1))); }
      pts.push(PL(YR, 0));
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
      ctx.globalAlpha = 1; ctx.fillStyle = C.p0Fill; ctx.fill();
      ctx.beginPath(); pts.slice(1, -1).forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.strokeStyle = C.p0Line; ctx.lineWidth = 1.5 * px; ctx.stroke();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.7 * px; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]); ctx.stroke();
      let q = PL(0.9, 0); mtext(ctx, TX.p0y, q[0], q[1] + 1.35 * fs, 1.14 * fs, C.p0Text, "center");
      ctx.globalAlpha = this.alpha;

      // x_t = (y_t, a_t): one state, seen in both panels
      const lS = PL(g[0], t), rS = PR(g.slice(1)), live = t < 0.985;
      if (live) {
        ctx.setLineDash([4 * px, 4 * px]); ctx.strokeStyle = rgba([35, 137, 201], 0.9); ctx.lineWidth = 1.2 * px;
        ctx.beginPath(); ctx.moveTo(lS[0], lS[1]); ctx.lineTo(rS[0], rS[1]); ctx.stroke(); ctx.setLineDash([]);
        const la = this.alpha * (1 - smooth((t - 0.78) / 0.1)) * (this.wide ? 1 : 0);   // the label steps aside before the views meet
        if (la > 0.02) {
          // sit the label clear of its own line, on the perpendicular, with a halo behind it
          const dx = rS[0] - lS[0], dy = rS[1] - lS[1], n = Math.hypot(dx, dy) || 1;
          const mx = lS[0] + 0.52 * dx, my = lS[1] + 0.52 * dy, off = 0.95 * fs;
          ctx.globalAlpha = la;
          mtext(ctx, TX.xt, mx - dy / n * off, my + dx / n * off + 0.32 * fs, 1.02 * fs, C.blue, "center", C.halo);
          ctx.globalAlpha = this.alpha;
        }
      }
      // reward gradient
      if (this.snap > 0.02) this.drawArrows(ctx, this.shot.st, T_ON, this.shot.a, 1, this.snap, false);
      this.drawArrows(ctx, g, t, a, this.steer * fade, 1, this.wide);

      // the state, and where the base flow would be by now
      const dot = (p, r, fill) => { ctx.beginPath(); ctx.arc(p[0], p[1], r * px, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = C.paper; ctx.lineWidth = 1.4 * px; ctx.stroke(); };
      if (this.snap > 0.02) { ctx.globalAlpha = this.alpha * this.snap; dot(PL(this.shot.st[0], T_ON), 3.8, C.ink); dot(PR(this.shot.st.slice(1)), 3.8, C.ink); ctx.globalAlpha = this.alpha; }
      if (t > T_ON + 0.004) { dot(PL(u[0], t), 3.6, C.ink); dot(PR(u.slice(1)), 3.6, C.ink); }
      const pop = this.phase === "intro" ? smooth(this.tau / 0.3) : 1, steered = t > T_ON + 0.004 && this.active();
      dot(lS, 5 * pop, steered ? C.red : C.ink); dot(rS, 5 * pop, steered ? C.red : C.ink);
      ctx.restore();
    }

    readout() {
      const a = this.cur.a, e = "e" + "₁₂₃₄"[this.target], P = this.paths, live = this.active();
      const tt = "flow time t = " + this.t.toFixed(2) + " · ";
      let s;
      if (this.phase === "intro" || this.phase === "pre") s = tt + "following the base generator; " + a.hits + " of " + NLOOK + " look-aheads end on " + e;
      else if (this.phase === "hold") s = live ? tt + "the reward scores the look-aheads: " + a.hits + " of " + NLOOK + " end on " + e
        : tt + (this.lam > 0 ? "no look-ahead ends on " + e + ", so there is nothing to steer with" : "reward strength is zero: the two paths coincide");
      else if (this.phase === "post") s = tt + (live ? "steering; " : "") + a.hits + " of " + NLOOK + " look-aheads now end on " + e;
      else { const names = k => "e" + "₁₂₃₄"[k]; s = "base sample ends on " + names(nearestVertex(P.U[NSTEP])) + ", steered sample on " + names(nearestVertex(P.G[NSTEP])); }
      if (s !== this.lastRead) { this.read.textContent = s; this.lastRead = s; }
    }
  }

  function init() {
    const root = document.getElementById("steer-anim");
    if (!root) return;
    const anim = new Steer(root);
    Toy.anim = anim;
    const playBtn = root.querySelector("#steer-play"), newBtn = root.querySelector("#steer-new");
    const lam = root.querySelector("#steer-lam"), lamVal = root.querySelector("#steer-lam-val");
    const segs = root.querySelectorAll("#steer-target button");
    const reduced = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let visible = true, last = null, raf = 0, acc = 0;
    function frame(now) {
      raf = 0;
      const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.05); last = now;
      anim.tick(dt); anim.draw();
      acc += dt; if (acc > 0.12 || dt === 0) { anim.readout(); acc = 0; }
      if (visible && !document.hidden) raf = requestAnimationFrame(frame);
    }
    function wake() { if (!raf && visible && !document.hidden) { last = null; raf = requestAnimationFrame(frame); } }
    function setPlaying(p) { anim.playing = p; playBtn.textContent = p ? "Pause" : "Play"; wake(); }

    playBtn.addEventListener("click", () => setPlaying(!anim.playing));
    newBtn.addEventListener("click", () => { anim.newNoise(); setPlaying(true); });
    function applyLam() { anim.lam = parseFloat(lam.value); lamVal.textContent = anim.lam.toFixed(1); anim.recompute(); anim.drawStatic(); wake(); }
    lam.addEventListener("input", applyLam);
    segs.forEach(b => b.addEventListener("click", () => {
      anim.target = parseInt(b.dataset.k, 10);
      segs.forEach(o => { const on = o === b; o.classList.toggle("on", on); o.setAttribute("aria-pressed", on ? "true" : "false"); });
      anim.recompute(); anim.drawStatic(); wake();
    }));

    if (reduced) {           // a still, composed like the paper's figure
      anim.playing = false; playBtn.textContent = "Play";
      anim.phase = "post"; anim.u = 0.42; anim.t = T_ON + (1 - T_ON) * (0.5 * anim.u + 0.5 * anim.u * anim.u);
      anim.glow = 1; anim.steer = 1; anim.fan = 1; anim.alpha = 1; anim.snapEnds = true;
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
