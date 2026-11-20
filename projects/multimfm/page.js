const copyButton = document.getElementById("copy");
const citation = document.getElementById("bibtex");
const status = document.getElementById("copy-status");

if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      status.textContent = "Citation copied.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Citation selected. Use your browser’s Copy command.";
    }
  });
}
