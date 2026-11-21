(() => {
  "use strict";

  const player = document.getElementById("steering-player");
  const figure = document.getElementById("steering-animation");
  const play = document.getElementById("steer-play");
  const replay = document.getElementById("steer-replay");
  const slider = document.getElementById("steer-progress");
  const phase = document.getElementById("steer-phase");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");

  // K indexes one joint endpoint: a Gaussian position and a fixed category vertex.
  const labels = ["A", "B", "C"];
  const prior = [0.31, 0.38, 0.31];
  const means = [-1.35, 0, 1.35];
  const vertices = [[-0.85, -0.45], [0, 0.95], [0.85, -0.45]];
  const variance = 0.23 ** 2;
  const steps = 250;
  const branch = 75;
  const logReward = 3.6;
  const start = [0.06, -0.09, 0.13];

  // For Z_t = (1-t) noise + t endpoint, the class likelihood is Gaussian.
  function posterior(t, state, reward = -1) {
    const noiseVar = (1 - t) ** 2;
    const positionVar = noiseVar + t * t * variance;
    const classLogs = prior.map((mass, k) => {
      const dy = state[0] - t * means[k];
      const du = state[1] - t * vertices[k][0];
      const dv = state[2] - t * vertices[k][1];
      return Math.log(mass) + (reward === k ? logReward : 0)
        - 0.5 * (dy * dy / positionVar + (du * du + dv * dv) / noiseVar);
    });
    const max = Math.max(...classLogs);
    const weights = classLogs.map(value => Math.exp(value - max));
    const total = weights.reduce((sum, w) => sum + w, 0);
    return {
      probabilities: weights.map(w => w / total),
      positionMeans: means.map(m => m + t * variance * (state[0] - t * m) / positionVar),
      positionSD: Math.sqrt(variance * noiseVar / positionVar)
    };
  }

  function trajectory(reward) {
    const states = [start];
    for (let i = 0; i < steps; i++) {
      const state = states[i];
      const { probabilities, positionMeans } = posterior(i / steps, state, i < branch ? -1 : reward);
      const mean = [0, 0, 0];
      probabilities.forEach((w, k) => {
        mean[0] += w * positionMeans[k];
        mean[1] += w * vertices[k][0];
        mean[2] += w * vertices[k][1];
      });
      // The conditional velocity is (E[endpoint | state] - state) / (1 - t).
      states.push(state.map((value, d) => value + (mean[d] - value) / (steps - i)));
    }
    return states;
  }
  const unguided = trajectory(-1);
  const guided = { 0: trajectory(0), 2: trajectory(2) };

  // A fixed seed keeps the same 18 paired look-aheads when the reward changes.
  function drawEndpoints() {
    let seed = 84313;
    function uniform() {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return (seed + 0.5) / 4294967296;
    }
    const { probabilities, positionMeans, positionSD } = posterior(branch / steps, unguided[branch]);
    return Array.from({ length: 18 }, () => {
      const u = uniform();
      const k = u < probabilities[0] ? 0 : u < probabilities[0] + probabilities[1] ? 1 : 2;
      const normal = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
      return { k, state: [positionMeans[k] + positionSD * normal, ...vertices[k]] };
    });
  }
  const endpoints = drawEndpoints();

  const svgNS = "http://www.w3.org/2000/svg";
  function node(parent, tag, attrs, label) {
    const el = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attrs)) el.setAttribute(name, value);
    if (label !== undefined) el.textContent = label;
    parent.appendChild(el);
    return el;
  }
  function coords(points) {
    return points.map(([x, y], i) => (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2)).join(" ");
  }
  function place(circle, [x, y]) {
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
  }
  const continuous = (s, t) => [175 + 58 * s[0], 169 - 105 * t];
  const categorical = (s, t) => [172 + 112 * s[1], 167 - 99 * t - 38 * s[2]];
  const terminalState = k => [means[k], ...vertices[k]];
  function densityShape(plot, t, density, height, className) {
    const points = [];
    for (let i = 0; i <= 168; i++) {
      const y = -2.35 + 4.7 * i / 168;
      const [x, baseline] = continuous([y], t);
      points.push([x, baseline - height * density(y)]);
    }
    const baseline = continuous([0], t)[1];
    return node(plot, "path", {d: coords([[points[0][0], baseline], ...points, [points.at(-1)[0], baseline]]) + "Z", class: className});
  }

  function makePanel(id, project, isContinuous) {
    const plot = document.getElementById(id);
    if (isContinuous) {
      densityShape(plot, 0, y => Math.exp(-0.5 * y * y), 29, "start-density");
      densityShape(plot, 1, y => means.reduce((p, m, k) => p + prior[k] * Math.exp(-0.5 * (y - m) ** 2 / variance), 0), 78, "end-density");
      node(plot, "text", {x: 4, y: 68, class: "time-label"}, "t = 1");
      node(plot, "text", {x: 4, y: 173, class: "time-label"}, "t = 0");
    } else {
      node(plot, "path", {d: coords(vertices.map((v, k) => project(terminalState(k), 1))) + "Z", class: "simplex"});
    }
    const markers = labels.map((label, k) => {
      const [x, y] = project(terminalState(k), 1);
      if (!isContinuous) node(plot, "circle", {cx:x, cy:y, r:5.8, class:"vertex"});
      return node(plot, "text", {x, y:isContinuous ? 22 : y + (k === 1 ? -13 : -12),
        class:"class-label", "text-anchor":"middle"}, label);
    });
    const atBranch = project(unguided[branch], branch / steps);
    const fan = endpoints.map(({state, k}) => {
      const end = project(state, 1);
      const line = node(plot, "path", {d: coords([atBranch, end]), pathLength:1, "stroke-dasharray":"1", class:"look-line"});
      const dot = node(plot, "circle", {cx:end[0], cy:end[1], r:isContinuous ? 2.7 : 3.7, class:"look-dot"});
      return {line, dot, k};
    });
    const baseLine = node(plot, "path", {class:"base-path"});
    const rewardLine = node(plot, "path", {class:"guided-path"});
    const branchPoint = node(plot, "circle", {cx:atBranch[0], cy:atBranch[1], r:3.8, class:"branch-point"});
    const baseDot = node(plot, "circle", {r:3.9, class:"base-dot"});
    const rewardDot = node(plot, "circle", {r:4.5, class:"guided-dot"});
    const origin = project(start, 0);
    node(plot, "text", {x:origin[0], y:192, "text-anchor":"middle", class:"origin-label"}, "Initial noise");

    const projected = states => states.map((state, i) => project(state, i / steps));
    const basePoints = projected(unguided);
    const rewardPoints = { 0:projected(guided[0]), 2:projected(guided[2]) };
    return function draw(index, count, emphasis, reward) {
      markers.forEach((mark, k) => mark.classList.toggle("reward-label", k === reward));
      fan.forEach(({line, dot, k}, i) => {
        const reveal = Math.max(0, Math.min(1, count - i));
        const active = k === reward && emphasis > 0;
        const opacity = active ? 0.44 + 0.4 * emphasis : 0.5 - 0.27 * emphasis;
        line.classList.toggle("reward-look", active);
        dot.classList.toggle("reward-look", active);
        line.setAttribute("stroke-dashoffset", 1 - reveal);
        line.style.opacity = reveal > 0 ? opacity : 0;
        dot.style.opacity = reveal >= 1 ? 0.92 : 0;
      });
      baseLine.setAttribute("d", coords(basePoints.slice(0, index + 1)));
      place(baseDot, basePoints[index]);
      branchPoint.style.opacity = count > 0 ? 1 : 0;
      rewardLine.setAttribute("d", index > branch ? coords(rewardPoints[reward].slice(branch, index + 1)) : "");
      place(rewardDot, rewardPoints[reward][index]);
      rewardDot.style.opacity = index > branch ? 1 : 0;
    };
  }

  const panels = [
    makePanel("steer-continuous", continuous, true),
    makePanel("steer-discrete", categorical, false)
  ];
  let reward = 2;
  let progress = motion.matches ? 1 : 0;
  let playing = !motion.matches;
  let onScreen = false;
  let frame = 0;
  let lastTime = null;
  function render() {
    // Hold outer time fixed while the posterior draws and their rewards are shown.
    const departure = Math.max(0, Math.min(1, (progress - 0.62) / 0.34));
    const index = progress < 0.23 ? Math.round(branch * progress / 0.23)
      : branch + Math.round((steps - branch) * departure);
    const count = Math.min(1, Math.max(0, (progress - 0.23) / 0.25)) * endpoints.length;
    const emphasis = Math.max(0, Math.min(1, (progress - 0.48) / 0.1));
    panels.forEach(draw => draw(index, count, emphasis, reward));
    const message = progress < 0.23 ? "Follow the base flow to a shared state."
      : progress < 0.48 ? "Draw paired endpoints from this state."
      : progress < 0.62 ? "Give higher reward to endpoints in category " + labels[reward] + "."
      : progress < 0.96 ? "Continue with and without the reward."
      : "The unguided path reaches B. The guided path reaches " + labels[reward] + ".";
    phase.textContent = "t = " + (index / steps).toFixed(2) + " · " + message;
    slider.value = Math.round(progress * 1000);
    slider.setAttribute("aria-valuetext", Math.round(progress * 100) + "%, " + phase.textContent);
    play.textContent = playing ? "Pause" : "Play";
  }

  function tick(time) {
    frame = 0;
    if (lastTime !== null) progress = Math.min(1, progress + (time - lastTime) / 11200);
    lastTime = time;
    if (progress >= 1) playing = false;
    render();
    if (playing) frame = requestAnimationFrame(tick);
  }
  function syncPlayback() {
    cancelAnimationFrame(frame);
    frame = 0;
    lastTime = null;
    render();
    if (playing && onScreen && !document.hidden) frame = requestAnimationFrame(tick);
  }
  play.addEventListener("click", () => {
    if (progress >= 1) progress = 0;
    playing = !playing;
    syncPlayback();
  });
  replay.addEventListener("click", () => {
    progress = 0;
    playing = true;
    syncPlayback();
  });
  slider.addEventListener("input", () => {
    playing = false;
    progress = Number(slider.value) / 1000;
    syncPlayback();
  });
  player.querySelectorAll('input[name="steer-reward"]').forEach(input => {
    input.addEventListener("change", () => {
      reward = Number(input.value);
      syncPlayback();
    });
  });
  motion.addEventListener("change", () => {
    if (motion.matches) {
      playing = false;
      progress = 1;
      syncPlayback();
    }
  });
  document.addEventListener("visibilitychange", syncPlayback);
  document.getElementById("steering-poster").hidden = true;
  player.hidden = false;
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(([entry]) => {
      onScreen = entry.isIntersecting;
      syncPlayback();
    });
    observer.observe(player);
  } else {
    onScreen = true;
  }
  syncPlayback();
})();
