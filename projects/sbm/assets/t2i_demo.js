/* Live text-to-image demo: calls the SBM-XL Space on Hugging Face (ZeroGPU) through Gradio's HTTP API and shows the
   eight images side by side. The call is anonymous (per-address GPU allowance); the Space page is linked for more. */
(function () {
  "use strict";
  var HOST = "https://tullebulle-sbm-t2i.hf.space";
  var SPACE = "https://huggingface.co/spaces/tullebulle/sbm-t2i";
  var N = 8, LABEL_SCALE = 1.125, WAKE_TRIES = 20, WAKE_WAIT_MS = 9000, CALL_TIMEOUT_MS = 240000;
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
  function withTimeout(ms) { var c = new AbortController(); setTimeout(function () { c.abort(); }, ms); return c.signal; }

  async function startCall(prompt, seed) {
    // the Space sleeps after idle time; while it boots, the API answers with an error page or not at all: retry
    for (var attempt = 1; attempt <= WAKE_TRIES; attempt++) {
      var res = null;
      try {
        res = await fetch(HOST + "/gradio_api/call/generate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data: [prompt, N, seed, LABEL_SCALE] }), signal: withTimeout(30000)
        });
      } catch (e) { res = null; }
      if (res && res.ok) {
        var j = await res.json();
        if (j && j.event_id) return j.event_id;
      }
      if (res && res.status === 429) throw new Error("busy");
      setStatus("Waking up the Space, this takes about a minute (try " + attempt + ")…");
      await sleep(WAKE_WAIT_MS);
    }
    throw new Error("The Space did not wake up. Open it on Hugging Face and try again there.");
  }
  async function waitResult(eventId) {
    var res = await fetch(HOST + "/gradio_api/call/generate/" + eventId, { signal: withTimeout(CALL_TIMEOUT_MS) });
    if (!res.ok) throw new Error("The Space answered HTTP " + res.status + ".");
    var text = await res.text(), lines = text.split(/\r?\n/), event = "", lastData = null, lastEvent = "";
    for (var i = 0; i < lines.length; i++) {
      var l = lines[i];
      if (l.indexOf("event:") === 0) event = l.slice(6).trim();
      else if (l.indexOf("data:") === 0) { lastEvent = event; try { lastData = JSON.parse(l.slice(5).trim()); } catch (e) { lastData = l.slice(5).trim(); } }
    }
    if (lastEvent === "error") {
      var msg = typeof lastData === "string" ? lastData : (lastData && lastData.message) || "";
      throw new Error(msg || "The Space reported an error.");
    }
    if (lastEvent !== "complete" || !Array.isArray(lastData) || lastData.length < 2) throw new Error("Unexpected reply from the Space.");
    return lastData;
  }
  function describeError(e) {
    var m = (e && e.message) || String(e);
    if (/quota|runs limit|GPU limit/i.test(m)) {
      return ["This address has used its free GPU allowance for today. Open the demo on ", link(SPACE, "Hugging Face"), " and sign in to keep going."];
    }
    if (/busy/.test(m)) return ["The Space is busy; try again in a moment, or use it on ", link(SPACE, "Hugging Face"), "."];
    if (/Failed to fetch|NetworkError|aborted/i.test(m)) return ["No answer from the Space. Try again, or open it on ", link(SPACE, "Hugging Face"), "."];
    return [m + " ", link(SPACE, "Open the demo on Hugging Face"), "."];
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
      var urls = items.map(function (it) { var im = it && (it.image || it); return im && im.url; }).filter(Boolean);
      if (!urls.length) throw new Error("The checker withheld every image for this prompt.");
      showImages(urls, prompt);
      var secs = ((Date.now() - t0) / 1000).toFixed(1);
      var m = /([\d.]+) s total/.exec(info), gpu = m ? m[1] + " s on the GPU, " : "", sm = /seed (\d+)/.exec(info);
      setStatus(urls.length + " images, one network evaluation each: " + gpu + secs + " s round trip" + (sm ? ", seed " + sm[1] : "") + ".");
    } catch (e) {
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
