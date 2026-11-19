/* Page behaviour: sections fade in as they scroll into view, and a thin bar at the top tracks reading progress.
   Both respect the reduced-motion preference and degrade to a static page without JavaScript. */
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var blocks = document.querySelectorAll(".reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    for (var i = 0; i < blocks.length; i++) blocks[i].classList.add("in");
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    for (var j = 0; j < blocks.length; j++) io.observe(blocks[j]);
    // anything already in view at load shows at once
    setTimeout(function () { for (var k = 0; k < blocks.length; k++) { var r = blocks[k].getBoundingClientRect(); if (r.top < window.innerHeight) blocks[k].classList.add("in"); } }, 50);
  }
  var bar = document.getElementById("progress");
  if (bar) {
    var ticking = false;
    function update() {
      var h = document.documentElement, max = h.scrollHeight - h.clientHeight;
      bar.style.transform = "scaleX(" + (max > 0 ? Math.min(1, h.scrollTop / max) : 0) + ")"; ticking = false;
    }
    window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }
})();
