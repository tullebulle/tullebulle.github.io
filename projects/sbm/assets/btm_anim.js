/* What a Beckmann map is: three live panels on one five-atom toy, same noise in each.
   Source mu_0 = N(0, I_2); target mu_1 = sum_j p_j delta_{z_j}; straight interpolant I_t = (1-t) x0 + t x1 (t = flow time).
   (1) Flow matching: the time-dependent field b_t(x) = (E[x1 | I_t = x] - x) / (1 - t), integrated as t runs 0 -> 1.
   (2) Autonomous flow: b = j / nu with j = int rho_t b_t dt, nu = int rho_t dt (quadrature in alpha = 1 - t, on a grid);
       its basins are the one-step map T, and each basin carries the mass of its atom.
   (3) Equilibrium Matching: the same regression with the target scaled by c_t = (1-t)^kappa (BTM paper, Sec. 2.7 and Fig. 3):
       b_kappa = J_kappa / nu with J_kappa = int c_t rho_t b_t dt. For kappa = 0 this is BTM; otherwise div J_kappa != mu_0 - mu_1
       and the basins carry the wrong mass (the bias BTM points out). Its background is the Coulomb energy of that current,
       E = -phi with Laplace phi = div J_kappa = mu_0 - kappa int (1-t)^(kappa-1) rho_t dt (= mu_0 - mu_1 at kappa = 0).
   The clock of an autonomous field is free (the paper: it changes the speed, not the flow lines or the map), so panels
   (2) and (3) move with a clamped speed along the exact flow lines. */
(function (global) {
  "use strict";
  const TAU = 2 * Math.PI;
  const SPEC = [[25, 1.55, 0.30, "#1B8A6B"], [100, 1.35, 0.15, "#E8862E"], [160, 1.50, 0.30, "#2F6FDE"], [235, 1.30, 0.10, "#B23A8C"], [320, 1.45, 0.15, "#C9961A"]];
  const ATOMS = SPEC.map(([deg, r, p, c]) => ({ z: [r * Math.cos(deg * Math.PI / 180), r * Math.sin(deg * Math.PI / 180)], p, c }));
  const J = ATOMS.length;

  function hex2rgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  function mix(c, k, base) { const a = hex2rgb(c), b = hex2rgb(base); return a.map((v, i) => Math.round(k * v + (1 - k) * b[i])); }
  const CREAM = "#FBF6E9";
  const TINTS = ATOMS.map(a => mix(a.c, 0.26, CREAM));
  const EDGES = ATOMS.map(a => mix(a.c, 0.55, CREAM));

  function nearestAtom(x, y) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < J; k++) { const d = Math.hypot(x - ATOMS[k].z[0], y - ATOMS[k].z[1]); if (d < bd) { bd = d; best = k; } }
    return best;
  }
  const ARRIVE = 0.05;
  function arrivedAt(x, y) {
    for (let k = 0; k < J; k++) if (Math.hypot(x - ATOMS[k].z[0], y - ATOMS[k].z[1]) < ARRIVE) return k;
    return -1;
  }

  // ---------- (1) flow matching ----------
  const LOGP = ATOMS.map(a => Math.log(a.p));
  function zhat(x, y, s) {
    const a = 1 - s, inv2 = 1 / (2 * a * a);
    let m = -Infinity; const L = new Array(J);
    for (let k = 0; k < J; k++) { const z = ATOMS[k].z, dx = x - s * z[0], dy = y - s * z[1]; L[k] = LOGP[k] - (dx * dx + dy * dy) * inv2; if (L[k] > m) m = L[k]; }
    let W = 0, zx = 0, zy = 0;
    for (let k = 0; k < J; k++) { const w = Math.exp(L[k] - m); W += w; zx += w * ATOMS[k].z[0]; zy += w * ATOMS[k].z[1]; }
    return [zx / W, zy / W];
  }
  function bFM(x, y, s) { const z = zhat(x, y, s); const a = Math.max(1 - s, 1e-6); return [(z[0] - x) / a, (z[1] - y) / a]; }
  function fmStep(p, s, ds) { const z = zhat(p.x, p.y, s); const f = Math.min(1, ds / (1 - s)); p.x += f * (z[0] - p.x); p.y += f * (z[1] - p.y); }
  function destFM(x, y) {
    const p = { x, y }, n = 400;
    for (let k = 0; k < n; k++) fmStep(p, k / n, 1 / n);
    return nearestAtom(p.x, p.y);
  }

  // ---------- (2)+(3) autonomous fields on a grid: BTM (kappa = 0) and EqM for c_s = (1-s)^kappa ----------
  const KAPPAS = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const GRID = { n: 129, lo: -3.25, hi: 3.25 };
  GRID.h = (GRID.hi - GRID.lo) / (GRID.n - 1);
  function newGrid(kappa) { return { kappa, n: GRID.n, lo: GRID.lo, hi: GRID.hi, h: GRID.h, bx: new Float32Array(GRID.n * GRID.n), by: new Float32Array(GRID.n * GRID.n), ready: false }; }
  const FIELDS = KAPPAS.map(newGrid);
  const FG = FIELDS[0];
  const NA = 220, LA0 = Math.log(1e-4), DL = -LA0 / (NA - 1);
  const AL = []; for (let k = 0; k < NA; k++) AL.push(Math.exp(LA0 + k * DL));
  const AK = KAPPAS.map(kap => AL.map(a => Math.pow(a, kap)));   // c_s = alpha^kappa at the quadrature nodes
  function autoFields(x, y) {   // returns [[bx, by] for each kappa]: J_kappa / nu with J_kappa = int alpha^kappa rho_s b_s ds
    let nu = 0; const jx = new Float64Array(KAPPAS.length), jy = new Float64Array(KAPPAS.length);
    for (let q = 0; q < NA; q++) {
      const a = AL[q], inv2 = 1 / (2 * a * a), norm = DL / (TAU * a);   // (1/(2 pi a^2)) * (a dlog) weight
      for (let k = 0; k < J; k++) {
        const z = ATOMS[k].z, dx = x - (1 - a) * z[0], dy = y - (1 - a) * z[1], q2 = (dx * dx + dy * dy) * inv2;
        if (q2 > 40) continue;
        const rho = ATOMS[k].p * norm * Math.exp(-q2), vx = rho * (z[0] - x) / a, vy = rho * (z[1] - y) / a;
        nu += rho;
        for (let m = 0; m < KAPPAS.length; m++) { const c = AK[m][q]; jx[m] += c * vx; jy[m] += c * vy; }
      }
    }
    return KAPPAS.map((_, m) => nu > 0 ? [jx[m] / nu, jy[m] / nu] : [0, 0]);
  }
  function autoField(x, y) { return autoFields(x, y)[0]; }
  function computeAutoGrid(rows) {   // rows = [i0, i1): compute a block of rows of every grid
    for (let i = rows[0]; i < rows[1]; i++) for (let j = 0; j < GRID.n; j++) {
      const bs = autoFields(GRID.lo + i * GRID.h, GRID.lo + j * GRID.h);
      for (let m = 0; m < FIELDS.length; m++) { FIELDS[m].bx[i * GRID.n + j] = bs[m][0]; FIELDS[m].by[i * GRID.n + j] = bs[m][1]; }
    }
  }
  function gridField(G) {
    return function (x, y) {
      let fi = (x - G.lo) / G.h, fj = (y - G.lo) / G.h;
      fi = Math.min(Math.max(fi, 0), G.n - 1 - 1e-9); fj = Math.min(Math.max(fj, 0), G.n - 1 - 1e-9);
      const i = Math.floor(fi), j = Math.floor(fj), ai = fi - i, aj = fj - j, k = i * G.n + j, n = G.n;
      const bx = (1 - ai) * ((1 - aj) * G.bx[k] + aj * G.bx[k + 1]) + ai * ((1 - aj) * G.bx[k + n] + aj * G.bx[k + n + 1]);
      const by = (1 - ai) * ((1 - aj) * G.by[k] + aj * G.by[k + 1]) + ai * ((1 - aj) * G.by[k + n] + aj * G.by[k + n + 1]);
      return [bx, by];
    };
  }
  for (const G of FIELDS) G.field = gridField(G);
  const bAuto = FG.field;

  // ---------- Coulomb energy of the EqM current: Laplace phi = div J_kappa ----------
  // Continuity gives div(rho_t b_t) = -d/dt rho_t, so div J_kappa = c_0 rho_0 - c_1 rho_1 + int c'_t rho_t dt
  //   = mu_0 - kappa int_0^1 (1-t)^(kappa-1) rho_t dt   for c_t = (1-t)^kappa, kappa > 0   (= mu_0 - mu_1 at kappa = 0).
  // The sink is a unit charge smeared along the interpolation paths; its potential is the (1-t)^(kappa-1)-weighted average
  // of the potentials of the Gaussians N(t z_j, (1-t)^2), g(|x - t z_j| / (1-t)), with u = (1-t)^kappa as the quadrature variable.
  const GT = (function () {   // g(r) = (1/2pi) int_0^r (1 - e^{-u^2/2}) / u du: potential of a unit Gaussian charge, table on [0, 8]
    const n = 1600, h = 8 / n, t = new Float64Array(n + 2);
    let acc = 0, prev = 0;
    for (let k = 1; k <= n; k++) { const r = k * h, f = (1 - Math.exp(-r * r / 2)) / r; acc += 0.5 * (prev + f) * h; prev = f; t[k] = acc / TAU; }
    t[n + 1] = t[n];
    return { t, n, h, inv: n / 8, g8: t[n] };
  })();
  function PG(r) { if (r >= 8) return GT.g8 + Math.log(r / 8) / TAU; const u = r * GT.inv, k = u | 0, a = u - k; return GT.t[k] + a * (GT.t[k + 1] - GT.t[k]); }
  const NU = 120, ALPHAU = KAPPAS.map(kap => { const o = new Float64Array(NU); for (let q = 0; q < NU; q++) o[q] = kap > 0 ? Math.pow((q + 0.5) / NU, 1 / kap) : 0; return o; });
  function phiAt(x, y, m) {
    let v = PG(Math.sqrt(x * x + y * y));
    if (KAPPAS[m] === 0) { for (let k = 0; k < J; k++) { const dx = x - ATOMS[k].z[0], dy = y - ATOMS[k].z[1]; v -= ATOMS[k].p / TAU * Math.log(Math.max(Math.sqrt(dx * dx + dy * dy), 1e-4)); } return v; }
    const AU = ALPHAU[m], t = GT.t, inv = GT.inv, g8 = GT.g8;
    for (let k = 0; k < J; k++) {
      const z0 = ATOMS[k].z[0], z1 = ATOMS[k].z[1]; let acc = 0;
      for (let q = 0; q < NU; q++) {
        const a = AU[q], dx = x - (1 - a) * z0, dy = y - (1 - a) * z1, r = Math.sqrt(dx * dx + dy * dy) / a;
        if (r >= 8) acc += g8 + Math.log(r / 8) / TAU;
        else { const u = r * inv, i = u | 0, f = u - i; acc += t[i] + f * (t[i + 1] - t[i]); }
      }
      v -= ATOMS[k].p * acc / NU;
    }
    return v;
  }
  function phiGrid(m) {   // phi_kappa on the field grid, cached
    const G = FIELDS[m]; if (G.phi) return G.phi;
    const n = GRID.n, P = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = phiAt(GRID.lo + i * GRID.h, GRID.lo + j * GRID.h, m);
    G.phi = P; return P;
  }
  function phiInterp(P, x, y) {
    const n = GRID.n; let fi = (x - GRID.lo) / GRID.h, fj = (y - GRID.lo) / GRID.h;
    fi = Math.min(Math.max(fi, 0), n - 1 - 1e-9); fj = Math.min(Math.max(fj, 0), n - 1 - 1e-9);
    const i = Math.floor(fi), j = Math.floor(fj), ai = fi - i, aj = fj - j, k = i * n + j;
    return (1 - ai) * ((1 - aj) * P[k] + aj * P[k + 1]) + ai * ((1 - aj) * P[k + n] + aj * P[k + n + 1]);
  }

  // ---------- clocks: clamped speed along the exact flow lines ----------
  function stepDir(p, field, dt, vmin, vmax) {
    const b = field(p.x, p.y), m = Math.hypot(b[0], b[1]);
    if (m < 1e-12) return;
    const sp = Math.min(Math.max(m, vmin), vmax) / m;
    const mx = p.x + 0.5 * dt * sp * b[0], my = p.y + 0.5 * dt * sp * b[1];
    const b2 = field(mx, my), m2 = Math.hypot(b2[0], b2[1]);
    if (m2 < 1e-12) return;
    const sp2 = Math.min(Math.max(m2, vmin), vmax) / m2;
    p.x += dt * sp2 * b2[0]; p.y += dt * sp2 * b2[1];
  }
  const CLK = { auto: { vmin: 0.6, vmax: 2.2 } };
  function destOf(field, clk, x, y) {
    const p = { x, y };
    for (let k = 0; k < 3000; k++) {
      stepDir(p, field, 0.02, clk.vmin, clk.vmax);
      const a = arrivedAt(p.x, p.y); if (a >= 0) return a;
    }
    return nearestAtom(p.x, p.y);
  }
  const destAuto = (x, y) => destOf(bAuto, CLK.auto, x, y);
  const destIn = G => (x, y) => destOf(G.field, CLK.auto, x, y);

  // ---------- basins and masses of the maps (display grid), one label grid per kappa ----------
  const BOX = 2.5, NB = 360;   // resolution of the live fallback; the shipped grids are refined to 1440^2 (see refineBasin)
  const MG = { box: 3.25, n: 120 };   // wider, coarser label grid used only for the basin masses (99.8 % of the Gaussian mass lies inside it)
  function newBasin() { return { idx: new Int8Array(NB * NB).fill(-1), nb: NB, row: 0, done: false, masses: null, seps: null, midx: null }; }
  const BASINS = KAPPAS.map(newBasin);
  const BAS = BASINS[0];
  function basinCell(G, i, j) { const x = -BOX + (i + 0.5) * 2 * BOX / NB, y = -BOX + (j + 0.5) * 2 * BOX / NB; return destIn(G)(x, y); }
  function computeBasinRows(k, m = 0) {
    const B = BASINS[m], G = FIELDS[m];
    for (let r = 0; r < k && B.row < NB; r++, B.row++) for (let j = 0; j < NB; j++) B.idx[B.row * NB + j] = basinCell(G, B.row, j);
    B.done = B.row >= NB;
    if (B.done) { B.masses = massesOf(B); B.seps = null; }
  }
  function sepsOf(B) { if (B.done && !B.seps) B.seps = separatrices(B); return B.seps; }
  function refineBasin(m) {   // double the resolution of the label grid: recompute the cells near a label change, inherit the label elsewhere
    const B = BASINS[m], f = destIn(FIELDS[m]), n0 = B.nb, n1 = 2 * n0, out = new Int8Array(n1 * n1); let recomputed = 0;
    for (let i = 0; i < n0; i++) for (let j = 0; j < n0; j++) {
      const k = B.idx[i * n0 + j]; let mixed = false;
      for (let di = -1; di <= 1 && !mixed; di++) for (let dj = -1; dj <= 1; dj++) {
        const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= n0 || jj >= n0) continue;
        if (B.idx[ii * n0 + jj] !== k) { mixed = true; break; }
      }
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
        const i1 = 2 * i + a, j1 = 2 * j + b;
        if (mixed) { out[i1 * n1 + j1] = f(-BOX + (i1 + 0.5) * 2 * BOX / n1, -BOX + (j1 + 0.5) * 2 * BOX / n1); recomputed++; } else out[i1 * n1 + j1] = k;
      }
    }
    B.idx = out; B.nb = n1; B.row = n1; B.seps = null;
    return recomputed;
  }
  function computeMassGrid(m = 0) {
    const B = BASINS[m], f = destIn(FIELDS[m]); B.midx = new Int8Array(MG.n * MG.n);
    for (let i = 0; i < MG.n; i++) for (let j = 0; j < MG.n; j++) B.midx[i * MG.n + j] = f(-MG.box + (i + 0.5) * 2 * MG.box / MG.n, -MG.box + (j + 0.5) * 2 * MG.box / MG.n);
    B.masses = massesOf(B);
  }
  function massesOf(B) {   // Gaussian mass of each basin: from the wide grid when present, else the display grid normalised to the frame
    if (B.midx) {
      const m = new Array(J).fill(0), cell = (2 * MG.box / MG.n) ** 2;
      for (let i = 0; i < MG.n; i++) for (let j = 0; j < MG.n; j++) {
        const k = B.midx[i * MG.n + j]; if (k < 0) continue;
        const x = -MG.box + (i + 0.5) * 2 * MG.box / MG.n, y = -MG.box + (j + 0.5) * 2 * MG.box / MG.n;
        m[k] += Math.exp(-(x * x + y * y) / 2) / TAU * cell;
      }
      return m;
    }
    const m = basinMasses(B), tot = m.reduce((a, b) => a + b, 0) || 1;
    return m.map(v => v / tot);
  }
  function basinMasses(B = BAS) {   // Gaussian mass of each basin inside the box (the remainder lies outside it)
    const m = new Array(J).fill(0), nb = B.nb, cell = (2 * BOX / nb) ** 2;
    for (let i = 0; i < nb; i++) for (let j = 0; j < nb; j++) {
      const k = B.idx[i * nb + j]; if (k < 0) continue;
      const x = -BOX + (i + 0.5) * 2 * BOX / nb, y = -BOX + (j + 0.5) * 2 * BOX / nb;
      m[k] += Math.exp(-(x * x + y * y) / 2) / TAU * cell;
    }
    return m;
  }
  function massError(m) { let e = 0; for (let k = 0; k < J; k++) e += Math.abs(m[k] - ATOMS[k].p); return e / J; }
  // separatrices: polylines through the midpoints of label changes between neighbouring cells, smoothed (Chaikin)
  function separatrices(B) {
    const NB = B.nb;   // resolution of this label grid
    const cellXY = (i, j) => [-BOX + (i + 0.5) * 2 * BOX / NB, -BOX + (j + 0.5) * 2 * BOX / NB];
    const segs = new Map();   // node key -> neighbours; nodes are midpoints of cell edges: ("h",i,j) between (i,j)-(i+1,j), ("v",i,j) between (i,j)-(i,j+1)
    const key = (t, i, j) => t + ":" + i + ":" + j;
    const pos = new Map();
    const addNode = (t, i, j) => { const k = key(t, i, j); if (!pos.has(k)) { const a = cellXY(i, j), b = t === "h" ? cellXY(i + 1, j) : cellXY(i, j + 1); pos.set(k, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]); segs.set(k, []); } return k; };
    const link = (a, b) => { segs.get(a).push(b); segs.get(b).push(a); };
    // every 2x2 block of cells: connect the edge midpoints that separate different labels
    for (let i = 0; i < NB - 1; i++) for (let j = 0; j < NB - 1; j++) {
      const l00 = B.idx[i * NB + j], l10 = B.idx[(i + 1) * NB + j], l01 = B.idx[i * NB + j + 1], l11 = B.idx[(i + 1) * NB + j + 1];
      const e = [];   // crossing midpoints on the four edges of the block
      if (l00 !== l10) e.push(addNode("h", i, j));
      if (l01 !== l11) e.push(addNode("h", i, j + 1));
      if (l00 !== l01) e.push(addNode("v", i, j));
      if (l10 !== l11) e.push(addNode("v", i + 1, j));
      if (e.length === 2) link(e[0], e[1]);
      else if (e.length >= 3) {   // junction: join through the block centre
        const c = addNode("c", i, j); pos.set(c, [cellXY(i, j)[0] + BOX / NB, cellXY(i, j)[1] + BOX / NB]);
        for (const k of e) link(k, c);
      }
    }
    // walk the graph into polylines, starting at endpoints / junctions
    const seen = new Set(), lines = [];
    const walk = (start, next) => { const pts = [pos.get(start)]; let prev = start, cur = next; seen.add(prev + ">" + cur); seen.add(cur + ">" + prev);
      while (true) { pts.push(pos.get(cur)); const nb = segs.get(cur); if (nb.length !== 2) break; const nx = nb[0] === prev ? nb[1] : nb[0]; if (seen.has(cur + ">" + nx)) break; seen.add(cur + ">" + nx); seen.add(nx + ">" + cur); prev = cur; cur = nx; }
      return pts; };
    for (const [k, nb] of segs) if (nb.length !== 2) for (const n of nb) if (!seen.has(k + ">" + n)) lines.push(walk(k, n));
    for (const [k, nb] of segs) if (nb.length === 2) for (const n of nb) if (!seen.has(k + ">" + n)) lines.push(walk(k, n));   // closed loops
    const rounds = NB >= 1000 ? 1 : 2;
    const smooth = pts => { for (let r = 0; r < rounds; r++) { const o = [pts[0]]; for (let q = 0; q < pts.length - 1; q++) { const a = pts[q], b = pts[q + 1]; o.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]], [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]); } o.push(pts[pts.length - 1]); pts = o; } return pts; };
    return lines.filter(l => l.length > 3).map(smooth);
  }

  // precomputed grids and basins (assets/btm_fields.js, generated by the same code in node); computed live if absent
  function b64bytes(b64) { const bin = atob(b64), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return u8; }
  function b64i16(b64) { const u8 = b64bytes(b64); return new Int16Array(u8.buffer, 0, u8.length >> 1); }
  function rleDecode(b64, n) {   // runs of (label int8, length uint16 little-endian)
    const u8 = b64bytes(b64), out = new Int8Array(n); let o = 0;
    for (let k = 0; k + 2 < u8.length; k += 3) { const lab = (u8[k] << 24) >> 24, len = u8[k + 1] | (u8[k + 2] << 8); out.fill(lab, o, o + len); o += len; }
    return out;
  }
  function loadFields(D) {
    if (!D || D.n !== GRID.n || D.lo !== GRID.lo || D.hi !== GRID.hi || !(D.nb >= NB) || D.box !== BOX || !D.kappas || D.kappas.length !== KAPPAS.length) return false;
    for (let m = 0; m < KAPPAS.length; m++) {
      if (Math.abs(D.kappas[m] - KAPPAS[m]) > 1e-9) return false;
      const F = D.fields[m], G = FIELDS[m], qx = b64i16(F.bx), qy = b64i16(F.by);
      for (let k = 0; k < qx.length; k++) { G.bx[k] = qx[k] * F.scale; G.by[k] = qy[k] * F.scale; }
      G.ready = true;
      const B = BASINS[m]; B.nb = D.nb; B.idx = rleDecode(D.basins[m], D.nb * D.nb); B.row = D.nb; B.done = true;
      if (D.massgrid && D.mg === MG.n && D.mbox === MG.box) B.midx = rleDecode(D.massgrid[m], MG.n * MG.n);
      B.masses = massesOf(B); B.seps = null;   // separatrices are extracted when the grid is first drawn
      if (D.phi && D.phi[m]) { const Q = D.phi[m], q = b64i16(Q.data), P = new Float32Array(q.length); for (let k = 0; k < q.length; k++) P[k] = q[k] * Q.scale + Q.off; G.phi = P; }   // Coulomb potential of the EqM current (kappa = 0 is analytic, computed live)
    }
    return true;
  }

  const Toy = { ATOMS, zhat, bFM, destFM, autoField, autoFields, computeAutoGrid, bAuto, FG, FIELDS, KAPPAS, GRID, destAuto, destIn, stepDir, CLK, phiAt, phiGrid, phiInterp, computeBasinRows, refineBasin, computeMassGrid, basinMasses, massesOf, massError, separatrices, sepsOf, BAS, BASINS, NB, BOX, MG, arrivedAt, loadFields };
  global.BTMToy = Toy;
  if (typeof document === "undefined") return;

  // ---------- animation ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const rng = mulberry32(20260801);
  function gauss() { const r = Math.sqrt(-2 * Math.log(1 - rng())), th = TAU * rng(); return [r * Math.cos(th), r * Math.sin(th)]; }
  const N = 300, DRAWS = []; for (let i = 0; i < N; i++) DRAWS.push(gauss());
  const EP = { sweep: 4.8, hold: 1.8, kAuto: 0.55 };   // model time per real second; p95 arrival ~ 4 s
  const INK = "rgba(60,58,54,0.30)", SEP_INK = "#3B3833", BTM_SEP = "#4A4A4A";

  function rgba(c, a) { const [r, g, b] = typeof c === "string" ? hex2rgb(c) : c; return `rgba(${r},${g},${b},${a})`; }

  class Panel {
    constructor(stage, kind) {   // kind: "fm" | "auto" (BTM, kappa = 0) | "eqm" (kappa from the slider)
      this.stage = stage; this.kind = kind;
      this.bg = stage.querySelector("canvas.bg"); this.fg = stage.querySelector("canvas.fg"); this.hl = stage.querySelector("canvas.hl");
      this.read = stage.parentElement.querySelector(".anim-read");
      this.parts = DRAWS.map((d, i) => ({ x: d[0], y: d[1], px: d[0], py: d[1], dest: -1, done: false, i }));
      this.m = kind === "eqm" ? KAPPAS.indexOf(0.8) : 0;
      this.s = 0; this.streams = null; this.layer = null; this.elayer = null; this.counts = new Array(J).fill(0);
      this.resize();
    }
    get G() { return FIELDS[this.m]; }
    get B() { return BASINS[this.m]; }
    get kappa() { return KAPPAS[this.m]; }
    reset() {
      for (const p of this.parts) { const d = DRAWS[p.i]; p.x = d[0]; p.y = d[1]; p.px = d[0]; p.py = d[1]; p.done = false; }
      this.counts.fill(0); this.s = 0; this.clearTrails();
    }
    assignDest() {   // destination of every particle from where it is now (so a kappa change recolours the ones in flight)
      const f = this.kind === "fm" ? destFM : destIn(this.G);
      for (const p of this.parts) if (!p.done) p.dest = f(p.x, p.y);
    }
    setKappa(m) {   // eqm panel: switch the field and basins, keep the episode running
      if (m === this.m) return;
      this.m = m; this.layer = null; this.elayer = null; this.streams = null;
      if (this.G.ready) this.assignDest();
      this.drawBackground();
    }
    advance(dt, t) {
      if (this.kind === "fm") {
        const s1 = Math.min(1, t / EP.sweep), ds = s1 - this.s;
        if (ds > 0) for (const p of this.parts) { if (p.done) continue; p.px = p.x; p.py = p.y; fmStep(p, this.s, ds); if (s1 >= 1) { p.done = true; this.counts[nearestAtom(p.x, p.y)]++; } }
        this.s = s1;
      } else {
        const field = this.G.field, clk = CLK.auto;
        for (const p of this.parts) {
          if (p.done) continue;
          p.px = p.x; p.py = p.y; stepDir(p, field, EP.kAuto * dt, clk.vmin, clk.vmax);
          const a = arrivedAt(p.x, p.y);
          if (a >= 0) { p.done = true; p.x = ATOMS[a].z[0]; p.y = ATOMS[a].z[1]; this.counts[a]++; }
        }
      }
    }
    // --- geometry: the canvases are rendered at >= 2 device pixels per CSS pixel so lines stay smooth on any screen
    resize() {
      const dpr = Math.min(Math.max(global.devicePixelRatio || 1, 2), 3), w = this.stage.clientWidth, h = this.stage.clientHeight;
      if (!w || !h) return;
      for (const c of [this.bg, this.fg, this.hl]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      this.dpr = dpr; this.sx = this.bg.width / (2 * BOX); this.sy = this.bg.height / (2 * BOX);
      this.layer = null; this.elayer = null; this.drawBackground(); this.clearTrails();
    }
    X(u) { return (u + BOX) * this.sx; }
    Y(v) { return (BOX - v) * this.sy; }
    path(ctx, pts) { for (let k = 0; k < pts.length; k++) { const q = pts[k]; if (k) ctx.lineTo(this.X(q[0]), this.Y(q[1])); else ctx.moveTo(this.X(q[0]), this.Y(q[1])); } }
    // --- static layers
    streamlines(field, vmin, vmax) {
      const seeds = [];
      for (let k = 0; k < 13; k++) { const u = -2.35 + 4.7 * k / 12; seeds.push([u, BOX - 0.02], [u, -BOX + 0.02]); }
      for (let k = 0; k < 9; k++) { const v = -2.0 + 4.0 * k / 8; seeds.push([-BOX + 0.02, v], [BOX - 0.02, v]); }
      const lines = [];
      for (const s of seeds) {
        const p = { x: s[0], y: s[1] }, pts = [[p.x, p.y]];
        for (let k = 0; k < 1500; k++) { stepDir(p, field, 0.025, vmin, vmax); pts.push([p.x, p.y]); if (arrivedAt(p.x, p.y) >= 0) break; }
        lines.push(pts);
      }
      return lines;
    }
    makeLayer() {   // basin fill at the label grid's own resolution (1440^2 when shipped); the canvas downsamples it with smoothing, the separatrices go on top as vector lines
      const B = this.B, NB = B.nb, c = document.createElement("canvas"); c.width = NB; c.height = NB;
      const ctx = c.getContext("2d"), img = ctx.createImageData(NB, NB), d = img.data, cr = hex2rgb(CREAM);
      for (let i = 0; i < NB; i++) for (let j = 0; j < NB; j++) {
        const k = B.idx[i * NB + j], col = k >= 0 ? TINTS[k] : cr, o = ((NB - 1 - j) * NB + i) * 4;   // row = screen y (v decreasing), col = screen x
        d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      return c;
    }
    makeEnergyLayer() {   // E = -phi_kappa at device resolution, 18 contour bands, band edges darkened (the energy view)
      const W = this.bg.width, H = this.bg.height, P = phiGrid(this.m), c = document.createElement("canvas"); c.width = W; c.height = H;
      const ctx = c.getContext("2d"), img = ctx.createImageData(W, H), d = img.data;
      const E = new Float32Array(W * H); let hi = -Infinity;
      for (let py = 0; py < H; py++) { const y = BOX - (py + 0.5) / this.sy; for (let px = 0; px < W; px++) { const e = -phiInterp(P, -BOX + (px + 0.5) / this.sx, y); E[py * W + px] = e; if (e > hi) hi = e; } }
      let lo = Infinity;   // floor the wells a short distance from the atoms (log-deep at kappa = 0), so the bands cover the landscape
      for (const a of ATOMS) lo = Math.min(lo, -phiInterp(P, a.z[0] + 0.12, a.z[1]), -phiInterp(P, a.z[0], a.z[1] + 0.12));
      const stops = [[0, [38, 70, 83]], [0.35, [27, 138, 107]], [0.62, [157, 187, 167]], [0.84, [232, 217, 181]], [1, [251, 246, 233]]];
      const NBANDS = 18, cols = [];
      for (let b = 0; b < NBANDS; b++) { const t = (b + 0.5) / NBANDS; let col = stops[stops.length - 1][1]; for (let q = 1; q < stops.length; q++) if (t <= stops[q][0]) { const a = (t - stops[q - 1][0]) / (stops[q][0] - stops[q - 1][0]); col = stops[q - 1][1].map((v, i) => Math.round(v + a * (stops[q][1][i] - v))); break; } cols.push(col); }
      const band = new Uint8Array(W * H);
      for (let q = 0; q < W * H; q++) { const t = Math.min(1, Math.max(0, (E[q] - lo) / (hi - lo))); band[q] = Math.min(NBANDS - 1, Math.floor(Math.pow(t, 0.7) * NBANDS)); }
      const edge = new Uint8Array(W * H), r = Math.max(1, Math.round(this.dpr * 0.6));   // edge lines ~1 CSS px wide
      for (let py = 1; py < H; py++) for (let px = 1; px < W; px++) { const q = py * W + px; if (band[q] !== band[q - 1] || band[q] !== band[q - W]) for (let dy = 0; dy < r; dy++) for (let dx = 0; dx < r; dx++) if (px + dx < W && py + dy < H) edge[q + dy * W + dx] = 1; }
      for (let q = 0; q < W * H; q++) { const col = cols[band[q]], k = edge[q] ? 0.72 : 1, o = 4 * q; d[o] = Math.round(col[0] * k); d[o + 1] = Math.round(col[1] * k); d[o + 2] = Math.round(col[2] * k); d[o + 3] = 255; }
      ctx.putImageData(img, 0, 0);
      return c;
    }
    drawSeps(ctx, seps, dotted, light) {
      const dpr = this.dpr;
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      if (dotted) {
        ctx.setLineDash([1.2 * dpr, 3.2 * dpr]); ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = light ? "rgba(30,28,26,0.85)" : BTM_SEP;
        for (const pts of seps) { ctx.beginPath(); this.path(ctx, pts); ctx.stroke(); }
        ctx.setLineDash([]);
        return;
      }
      const styles = light ? [[3.6, "rgba(40,38,34,0.35)"], [2.0, "rgba(255,255,255,0.96)"]] : [[3.4, "rgba(255,255,255,0.95)"], [1.3, SEP_INK]];
      for (const [w, col] of styles) {
        ctx.lineWidth = w * dpr; ctx.strokeStyle = col;
        for (const pts of seps) { ctx.beginPath(); this.path(ctx, pts); ctx.stroke(); }
      }
    }
    windLines(ctx, s) {   // arrows along b_s on a lattice, redrawn as s moves: a short flow segment with a filled head
      const dpr = this.dpr, n = 9, L = 0.36, HEAD = 6.5 * dpr;
      ctx.lineWidth = 1.5 * dpr; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(78,76,70,0.55)"; ctx.fillStyle = "rgba(78,76,70,0.70)";
      const ss = Math.min(s, 0.97);
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        let x = -2.3 + 4.6 * a / (n - 1), y = -2.3 + 4.6 * b / (n - 1);
        const pts = [[x, y]];
        for (let k = 0; k < 12; k++) { const v = bFM(x, y, ss), m = Math.hypot(v[0], v[1]) || 1; x += (L / 12) * v[0] / m; y += (L / 12) * v[1] / m; pts.push([x, y]); }
        const e = pts[pts.length - 1], f = pts[pts.length - 2], ang = Math.atan2(-(e[1] - f[1]), e[0] - f[0]);
        const ex = this.X(e[0]), ey = this.Y(e[1]);
        // shaft stops where the head begins
        const bx = ex - 0.8 * HEAD * Math.cos(ang), by = ey - 0.8 * HEAD * Math.sin(ang);
        ctx.beginPath(); this.path(ctx, pts.slice(0, -2)); ctx.lineTo(bx, by); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ex, ey);
        ctx.lineTo(ex - HEAD * Math.cos(ang - 0.42), ey - HEAD * Math.sin(ang - 0.42));
        ctx.lineTo(ex - HEAD * Math.cos(ang + 0.42), ey - HEAD * Math.sin(ang + 0.42)); ctx.closePath(); ctx.fill();
      }
    }
    drawAtoms(ctx) {
      const dpr = this.dpr;
      for (const a of ATOMS) {
        const r = (3.5 + 9 * Math.sqrt(a.p)) * dpr;
        ctx.beginPath(); ctx.arc(this.X(a.z[0]), this.Y(a.z[1]), r, 0, TAU);
        ctx.fillStyle = a.c; ctx.fill(); ctx.lineWidth = 1.6 * dpr; ctx.strokeStyle = "#ffffff"; ctx.stroke();
      }
    }
    drawBackground() {
      const ctx = this.bg.getContext("2d"), W = this.bg.width, H = this.bg.height, dpr = this.dpr;
      ctx.clearRect(0, 0, W, H);
      if (this.kind === "fm") {
        ctx.fillStyle = CREAM; ctx.fillRect(0, 0, W, H);
        this.windLines(ctx, this.s);
      } else {
        const G = this.G, B = this.B, energy = this.kind === "eqm";
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        if (energy) {   // the landscape needs the potential grid: wait for the data (or the live field computation) rather than computing it twice
          if (G.ready) { if (!this.elayer) this.elayer = this.makeEnergyLayer(); ctx.drawImage(this.elayer, 0, 0, W, H); }
          else { ctx.fillStyle = CREAM; ctx.fillRect(0, 0, W, H); }
        }
        if (B.done) {
          if (!this.layer) this.layer = this.makeLayer();
          if (energy) ctx.globalAlpha = 0.22;   // the basins of the EqM map, marked on the energy landscape
          ctx.drawImage(this.layer, 0, 0, W, H); ctx.globalAlpha = 1;
        } else if (!energy) { ctx.fillStyle = CREAM; ctx.fillRect(0, 0, W, H); }
        if (!this.streams) this.streams = G.ready ? this.streamlines(G.field, CLK.auto.vmin, CLK.auto.vmax) : [];
        ctx.lineWidth = 0.9 * dpr; ctx.lineCap = "round"; ctx.strokeStyle = energy ? "rgba(255,255,255,0.45)" : INK;
        for (const pts of this.streams) { ctx.beginPath(); this.path(ctx, pts); ctx.stroke(); }
        if (energy && this.m !== 0 && sepsOf(BASINS[0])) this.drawSeps(ctx, BASINS[0].seps, true, true);   // BTM separatrices for comparison
        if (sepsOf(B)) this.drawSeps(ctx, B.seps, false, energy);
      }
      this.drawAtoms(ctx);
      ctx.strokeStyle = "#B8AA95"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(0.5 * dpr, 0.5 * dpr, W - dpr, H - dpr);
    }
    clearTrails() { const ctx = this.fg.getContext("2d"); ctx.clearRect(0, 0, this.fg.width, this.fg.height); }
    drawParticles() {
      const ctx = this.fg.getContext("2d"), dpr = this.dpr;
      ctx.globalCompositeOperation = "destination-out"; ctx.fillStyle = "rgba(0,0,0,0.14)"; ctx.fillRect(0, 0, this.fg.width, this.fg.height);
      ctx.globalCompositeOperation = "source-over"; ctx.lineCap = "round"; ctx.lineWidth = 2.3 * dpr;
      for (const p of this.parts) {
        const c = p.dest >= 0 ? ATOMS[p.dest].c : "#8F8E88";
        ctx.strokeStyle = c; ctx.fillStyle = c;
        if (!p.done || p.px !== p.x) { ctx.beginPath(); ctx.moveTo(this.X(p.px), this.Y(p.py)); ctx.lineTo(this.X(p.x), this.Y(p.y)); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(this.X(p.x), this.Y(p.y), 1.7 * dpr, 0, TAU); ctx.fill();
      }
    }
    chips(vals, fmt) { return vals.map((v, k) => `<span class="chip" style="background:${ATOMS[k].c}"></span>${fmt(v)}`).join(" "); }
    readout() {
      const tot = this.counts.reduce((a, b) => a + b, 0), B = this.B;
      if (this.kind === "fm") {
        this.read.innerHTML = tot ? `arrived at each atom: ${this.chips(this.counts, v => (v / tot).toFixed(2))}` : `flow time t = ${this.s.toFixed(2)}`;
      } else if (!B.done) {
        this.read.innerHTML = !this.G.ready ? `computing the field …` : `computing the basins of T … ${(100 * B.row / B.nb).toFixed(0)} %`;
      } else if (this.kind === "auto") {
        this.read.innerHTML = `basin mass: ${this.chips(B.masses, v => v.toFixed(3))} · mean |error| ${massError(B.masses).toFixed(3)}`;
      } else {
        const ref = BASINS[0].masses ? ` (BTM ${massError(BASINS[0].masses).toFixed(3)})` : "";
        this.read.innerHTML = `basin mass: ${this.chips(B.masses, v => v.toFixed(3))} · mean |error| <b>${massError(B.masses).toFixed(3)}</b>${ref}`;
      }
    }
    // --- hover: the flow line through the pointer, and T along it
    hover(mx, my) {
      const ctx = this.hl.getContext("2d"); ctx.clearRect(0, 0, this.hl.width, this.hl.height);
      if (mx === null || this.kind === "fm" || !this.G.ready) return;
      const field = this.G.field, clk = CLK.auto, dpr = this.dpr;
      const fwd = [[mx, my]], p = { x: mx, y: my }; let dest = -1;
      for (let k = 0; k < 3000; k++) { stepDir(p, field, 0.015, clk.vmin, clk.vmax); fwd.push([p.x, p.y]); const a = arrivedAt(p.x, p.y); if (a >= 0) { dest = a; break; } }
      const back = [[mx, my]], q = { x: mx, y: my }; const neg = (x, y) => { const b = field(x, y); return [-b[0], -b[1]]; };
      for (let k = 0; k < 1200; k++) { stepDir(q, neg, 0.015, clk.vmin, clk.vmax); if (Math.abs(q.x) > BOX + 0.3 || Math.abs(q.y) > BOX + 0.3) break; back.push([q.x, q.y]); }
      if (dest < 0) dest = nearestAtom(p.x, p.y);
      const col = ATOMS[dest].c;
      for (const [pts, w, c] of [[back, 5, "rgba(255,255,255,0.9)"], [fwd, 5, "rgba(255,255,255,0.9)"], [back, 2.4, col], [fwd, 2.4, col]]) {
        ctx.lineWidth = w * dpr; ctx.strokeStyle = c; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath(); this.path(ctx, pts); ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(this.X(mx), this.Y(my), 4.5 * dpr, 0, TAU); ctx.fillStyle = "#ffffff"; ctx.fill(); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = col; ctx.stroke();
      const za = ATOMS[dest].z, r = (3.5 + 9 * Math.sqrt(ATOMS[dest].p)) * dpr + 4 * dpr;
      ctx.beginPath(); ctx.arc(this.X(za[0]), this.Y(za[1]), r, 0, TAU); ctx.lineWidth = 2.5 * dpr; ctx.strokeStyle = "#ffffff"; ctx.stroke();
      ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = col; ctx.stroke();
      ctx.font = `italic ${14 * dpr}px "Latin Modern Roman", "Times New Roman", serif`; ctx.fillStyle = "#2F2A24"; ctx.textBaseline = "bottom"; ctx.textAlign = "left";
      const label = "T(x) is the same at every point of this line";
      const tx = Math.min(this.X(mx) + 8 * dpr, this.bg.width - ctx.measureText(label).width - 6 * dpr), ty = Math.max(this.Y(my) - 8 * dpr, 18 * dpr);
      ctx.lineWidth = 4 * dpr; ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.strokeText(label, tx, ty); ctx.fillText(label, tx, ty);
    }
  }

  function init() {
    const root = document.getElementById("btm-anim");
    if (!root) return;
    const stages = root.querySelectorAll(".anim-stage");
    const panels = [new Panel(stages[0], "fm"), new Panel(stages[1], "auto"), new Panel(stages[2], "eqm")];
    const [fm, au, eq] = panels;
    const playBtn = root.querySelector("#btm-play"), replayBtn = root.querySelector("#btm-replay");
    const slider = root.querySelector("#kappa"), kapVal = root.querySelector("#kappa-val"), eqTitle = root.querySelector("#eqm-title");
    const reduced = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let playing = !reduced, last = null, t = 0, acc = 0, gridRow = 0;
    const loaded = loadFields(global.BTM_FIELDS);
    if (loaded) for (const p of [au, eq]) { p.streams = null; p.layer = null; p.elayer = null; p.drawBackground(); }
    for (const p of panels) if (p.kind === "fm" || p.G.ready) p.assignDest();
    playBtn.textContent = playing ? "Pause" : "Play";
    function startEpisode() { t = 0; for (const p of panels) p.reset(); fm.drawBackground(); for (const p of panels) p.drawParticles(); }
    playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; last = null; if (playing) requestAnimationFrame(frame); });
    replayBtn.addEventListener("click", () => { startEpisode(); if (!playing) { playing = true; playBtn.textContent = "Pause"; last = null; requestAnimationFrame(frame); } });
    // kappa slider: snaps to the precomputed schedules
    function applyKappa() {
      const v = parseFloat(slider.value); let m = 0;
      for (let k = 0; k < KAPPAS.length; k++) if (Math.abs(KAPPAS[k] - v) < Math.abs(KAPPAS[m] - v)) m = k;
      eq.setKappa(m);
      if (kapVal) kapVal.textContent = KAPPAS[m].toFixed(1);
      if (eqTitle) eqTitle.innerHTML = m === 0 ? "Equilibrium Matching, &kappa; = 0 (BTM)" : `Equilibrium Matching, <i>c<sub>t</sub></i> = (1 &minus; <i>t</i>)<sup>${KAPPAS[m].toFixed(1)}</sup>`;
      eq.readout(); eq.drawParticles();
    }
    if (slider) { slider.addEventListener("input", applyKappa); applyKappa(); }
    // hover on the two autonomous panels
    for (const p of [au, eq]) {
      const toModel = ev => { const r = p.stage.getBoundingClientRect(); return [-BOX + 2 * BOX * (ev.clientX - r.left) / r.width, BOX - 2 * BOX * (ev.clientY - r.top) / r.height]; };
      p.stage.addEventListener("pointermove", ev => { const m = toModel(ev); p.hover(m[0], m[1]); });
      p.stage.addEventListener("pointerleave", () => p.hover(null, null));
    }
    function frame(now) {
      if (!playing) return;
      if (last === null) last = now;
      const dt = Math.min((now - last) / 1000, 0.05); last = now;
      // background work when the precomputed data is missing: the field grids (a few rows per frame), then the basins
      if (!FG.ready) {
        const r1 = Math.min(GRID.n, gridRow + 4); computeAutoGrid([gridRow, r1]); gridRow = r1;
        if (gridRow >= GRID.n) { for (const G of FIELDS) G.ready = true; for (const p of [au, eq]) { p.assignDest(); p.streams = null; p.elayer = null; p.drawBackground(); } }
      } else {
        const pend = [au, eq].find(p => !p.B.done);
        if (pend) { computeBasinRows(4, pend.m); if (pend.B.done) { for (const p of [au, eq]) { p.layer = null; p.drawBackground(); p.readout(); } } }
      }
      t += dt;
      if (t <= EP.sweep + 0.3) {
        for (const p of panels) if (p.kind === "fm" || p.G.ready) p.advance(dt, t);
        fm.drawBackground();
      }
      for (const p of panels) p.drawParticles();
      if (t > EP.sweep + EP.hold) startEpisode();
      acc += dt;
      if (acc > 0.25) { for (const p of panels) p.readout(); acc = 0; }
      requestAnimationFrame(frame);
    }
    for (const p of panels) { p.drawParticles(); p.readout(); }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { for (const p of panels) p.drawBackground(); });
    if (playing) requestAnimationFrame(frame);
    let rt = null;
    global.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { for (const p of panels) p.resize(); }, 120); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})(typeof window !== "undefined" ? window : globalThis);
