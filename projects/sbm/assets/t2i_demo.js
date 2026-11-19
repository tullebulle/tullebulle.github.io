/* Live text-to-image demo: calls the SBM-XL Space on Hugging Face (ZeroGPU) through Gradio's HTTP API and shows the
   eight images side by side. The call is anonymous (per-address GPU allowance); the Space page is linked for more. */
(function () {
  "use strict";
  var HOST = "https://tullebulle-sbm-t2i.hf.space";
  var SPACE = "https://huggingface.co/spaces/tullebulle/sbm-t2i";
  var N = 8, LABEL_SCALE = 1.125, WAKE_TRIES = 20, WAKE_WAIT_MS = 9000, POST_TIMEOUT_MS = 30000, RESULT_TIMEOUT_MS = 300000;
  var EXAMPLES = [
    "a red fox standing in a snowy forest",
    "a corgi wearing sunglasses sitting on a beach at sunset",
    "a bowl of ramen with a soft-boiled egg and green onions, top view",
    "ukiyo-e style woodblock print of a great wave",
    "a stained-glass window of a phoenix",
    "a cute penguin wearing a scarf holding a mug, 3D render",
    "a man playing an electric guitar on stage, concert lighting, photo",
    "a donkey with the head of an elephant, standing in a field, photo"
  ];
  var form = document.getElementById("t2i-form"), promptEl = document.getElementById("t2i-prompt"), seedEl = document.getElementById("t2i-seed");
  var go = document.getElementById("t2i-go"), status = document.getElementById("t2i-status"), strip = document.getElementById("t2i-strip");
  var examples = document.getElementById("t2i-examples");
  if (!form || !strip) return;

  function setStatus(text, isError) {
    status.textContent = "";
    status.className = "demo-status" + (isError ? " err" : "");
    if (typeof text === "string") { status.textContent = text; return; }
    text.forEach(function (part) { status.appendChild(typeof part === "string" ? document.createTextNode(part) : part); });
  }
  function link(href, label) { var a = document.createElement("a"); a.href = href; a.textContent = label; a.target = "_blank"; a.rel = "noopener"; return a; }
  function showImages(urls, alt) {
    strip.innerHTML = "";
    urls.forEach(function (u, k) {
      var a = document.createElement("a"); a.href = u; a.target = "_blank"; a.rel = "noopener"; a.title = alt + " (image " + (k + 1) + ", opens full size)";
      var im = document.createElement("img"); im.src = u; im.alt = alt + ", image " + (k + 1); im.decoding = "async";
      a.appendChild(im); strip.appendChild(a);
    });
  }
  function showPrecomputed() {
    var urls = []; for (var k = 0; k < N; k++) urls.push("seeds/samples/p0_s" + k + ".png");
    showImages(urls, EXAMPLES[0]);
    setStatus(["Precomputed draws, seeds 0 to 7, for the placeholder prompt. Type a prompt and generate; the ", link("seeds/", "seed gallery"), " has more."]);
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function timed(ms) { var c = new AbortController(); var t = setTimeout(function () { c.abort(); }, ms); return { signal: c.signal, clear: function () { clearTimeout(t); } }; }
  function fail(step, e) { var err = new Error((e && e.message) || String(e)); err.step = step; err.name = (e && e.name) || "Error"; return err; }

  async function startCall(prompt, seed) {
    // the Space sleeps after idle time; while it boots, the API answers with an error page or not at all: retry
    var lastErr = null;
    for (var attempt = 1; attempt <= WAKE_TRIES; attempt++) {
      var res = null, t = timed(POST_TIMEOUT_MS);
      try {
        res = await fetch(HOST + "/gradio_api/call/generate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data: [prompt, N, seed, LABEL_SCALE] }), signal: t.signal
        });
      } catch (e) { res = null; lastErr = e; }
      t.clear();
      if (res && res.ok) {
        var j = null; try { j = await res.json(); } catch (e) { j = null; }
        if (j && j.event_id) return j.event_id;
      }
      if (res && res.status === 429) throw fail("request", new Error("busy"));
      setStatus("Waking up the Space, this takes about a minute (try " + attempt + (res ? ", HTTP " + res.status : (lastErr ? ", " + lastErr.name : "")) + ")…");
      await sleep(WAKE_WAIT_MS);
    }
    throw fail("request", new Error("The Space did not wake up in three minutes" + (lastErr ? " (" + lastErr.name + ": " + lastErr.message + ")" : "") + "."));
  }

  async function waitResult(eventId) {
    var t = timed(RESULT_TIMEOUT_MS), res;
    try {
      try {
        res = await fetch(HOST + "/gradio_api/call/generate/" + eventId, { signal: t.signal, headers: { "Accept": "text/event-stream" } });
      } catch (e) { throw fail("result stream", e); }
      if (!res.ok) throw fail("result stream", new Error("The Space answered HTTP " + res.status + "."));
      var event = "", buf = "", result = null;
      function handle(line) {
        if (line.indexOf("event:") === 0) { event = line.slice(6).trim(); return null; }
        if (line.indexOf("data:") !== 0) return null;
        var raw = line.slice(5).trim(), data; try { data = JSON.parse(raw); } catch (e) { data = raw; }
        if (event === "complete") return { ok: true, data: data };
        if (event === "error") return { ok: false, data: data };
        if (event === "heartbeat") setStatus("In the GPU queue…");
        return null;
      }
      try {
        if (res.body && res.body.getReader) {
          var reader = res.body.getReader(), dec = new TextDecoder();
          while (!result) {
            var chunk = await reader.read();
            if (chunk.done) break;
            buf += dec.decode(chunk.value, { stream: true });
            var idx;
            while (!result && (idx = buf.indexOf("\n")) >= 0) { var line = buf.slice(0, idx).replace(/\r$/, ""); buf = buf.slice(idx + 1); result = handle(line); }
          }
          try { reader.cancel(); } catch (e) {}
        } else {
          var text = await res.text();
          text.split(/\r?\n/).forEach(function (l) { if (!result) result = handle(l); });
        }
      } catch (e) { throw fail("reading the result", e); }
      if (!result) throw fail("result stream", new Error("The result stream ended without a result."));
      if (!result.ok) {
        var d = result.data, msg = typeof d === "string" ? d : (d && (d.error || d.message)) || "";
        throw fail("generation", new Error(msg || "The Space reported an error."));
      }
      if (!Array.isArray(result.data) || result.data.length < 2) throw fail("result stream", new Error("Unexpected reply from the Space."));
      return result.data;
    } finally { t.clear(); }
  }

  function describeError(e) {
    var m = (e && e.message) || String(e), step = e && e.step ? e.step : "request", name = (e && e.name) || "Error";
    if (/quota|runs limit|GPU limit/i.test(m)) {
      return ["This address has used its free GPU allowance for today. Open the demo on ", link(SPACE, "Hugging Face"), " and sign in to keep going."];
    }
    if (/^busy$/.test(m)) return ["The Space is busy; try again in a moment, or use it on ", link(SPACE, "Hugging Face"), "."];
    if (/please wait|per minute|hourly limit|per address/i.test(m)) return [m];
    if (/Failed to fetch|NetworkError|abort|Load failed/i.test(m) || name === "AbortError" || name === "TypeError") {
      var why = name === "AbortError" ? "timed out" : "the browser could not reach it";
      return ["No answer from the Space at the " + step + " step (" + why + ", " + name + ": " + m + "). Try again, or open it on ", link(SPACE, "Hugging Face"), "."];
    }
    return [m + " (" + step + ") ", link(SPACE, "Open the demo on Hugging Face"), "."];
  }

  var running = false;
  async function generate() {
    if (running) return;
    var prompt = (promptEl.value || "").trim() || promptEl.placeholder;
    var seedRaw = (seedEl.value || "").trim(), seed = seedRaw === "" ? -1 : Math.max(0, Math.floor(Number(seedRaw)) || 0);
    running = true; go.disabled = true; go.textContent = "Generating…";
    strip.classList.add("busy");
    setStatus("Sending the prompt…");
    try {
      var t0 = Date.now();
      var ev = await startCall(prompt, seed);
      setStatus("In the GPU queue…");
      var out = await waitResult(ev);
      var items = out[0] || [], info = typeof out[1] === "string" ? out[1] : "";
      if (/^ERROR:/.test(info)) throw fail("generation", new Error(info.replace(/^ERROR:\s*/, "")));
      var urls = items.map(function (it) { var im = it && (it.image || it); return im && im.url; }).filter(Boolean);
      if (!urls.length) throw fail("generation", new Error("The checker withheld every image for this prompt."));
      showImages(urls, prompt);
      var secs = ((Date.now() - t0) / 1000).toFixed(1);
      var m = /([\d.]+) s total/.exec(info), gpu = m ? m[1] + " s on the GPU, " : "", sm = /seed (\d+)/.exec(info);
      setStatus(urls.length + " images, one network evaluation each: " + gpu + secs + " s round trip" + (sm ? ", seed " + sm[1] : "") + ".");
    } catch (e) {
      if (window.console && console.warn) console.warn("SBM demo:", e && e.step, e);
      setStatus(describeError(e), true);
    } finally {
      running = false; go.disabled = false; go.textContent = "Generate 8 images"; strip.classList.remove("busy");
    }
  }

  EXAMPLES.forEach(function (p) {
    var b = document.createElement("button"); b.type = "button"; b.textContent = p;
    b.addEventListener("click", function () { promptEl.value = p; promptEl.focus(); });
    examples.appendChild(b);
  });
  form.addEventListener("submit", function (e) { e.preventDefault(); generate(); });
  promptEl.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); generate(); } });
  showPrecomputed();
})();
