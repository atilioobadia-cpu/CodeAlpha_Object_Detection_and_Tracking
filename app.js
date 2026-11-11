const $ = (selector) => document.querySelector(selector);

const feed = $("#video-feed");
const emptyState = $("#empty-state");
const startButton = $("#start-button");
const stopButton = $("#stop-button");
const sourceTabs = [...document.querySelectorAll(".source-tab")];
const fileInput = $("#video-file");
const errorMessage = $("#error-message");
const confidence = $("#confidence");
let sourceMode = "webcam";
let polling = false;

function showError(message = "") {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function setSource(mode) {
  sourceMode = mode;
  sourceTabs.forEach((tab) => {
    const selected = tab.dataset.source === mode;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
  $("#webcam-choice").hidden = mode !== "webcam";
  $("#file-choice").hidden = mode !== "video";
  showError();
}

function setRunLabel(status) {
  const labels = {
    idle: "Ready", starting: "Connecting", loading_model: "Loading YOLO", running: "Tracking live",
    stopping: "Stopping", stopped: "Stopped", ended: "Video ended", error: "Needs attention",
  };
  const live = status === "running";
  $("#run-label").textContent = labels[status] || "Ready";
  $("#run-pill").classList.toggle("is-live", live);
  $("#service-label").textContent = status === "error" ? "Check the session" : "Local processing";
  startButton.disabled = ["starting", "loading_model", "running", "stopping"].includes(status);
  stopButton.disabled = !["starting", "loading_model", "running", "stopping"].includes(status);
}

function renderObjects(classes) {
  const list = $("#object-list");
  const entries = Object.entries(classes || {}).sort((a, b) => b[1] - a[1]);
  $("#summary-count").textContent = String(entries.reduce((sum, [, count]) => sum + count, 0));
  if (!entries.length) {
    list.innerHTML = '<p class="empty-list">No objects detected in the current frame.</p>';
    return;
  }
  list.replaceChildren(...entries.slice(0, 8).map(([label, count]) => {
    const row = document.createElement("div");
    row.className = "object-row";
    const name = document.createElement("span");
    name.className = "object-name";
    const dot = document.createElement("i");
    dot.className = "object-dot";
    const labelText = document.createElement("span");
    labelText.textContent = label;
    name.append(dot, labelText);
    const quantity = document.createElement("span");
    quantity.className = "object-track";
    quantity.textContent = `${count} ${count === 1 ? "track" : "tracks"}`;
    row.append(name, quantity);
    return row;
  }));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || "The video service could not process this request.");
  return payload;
}

function startFeed() {
  feed.hidden = false;
  emptyState.hidden = true;
  feed.src = `/api/stream?session=${Date.now()}`;
}

async function startTracking() {
  showError();
  startButton.disabled = true;
  stopButton.disabled = false;
  setRunLabel("starting");
  const confidenceValue = Number(confidence.value) / 100;
  try {
    if (sourceMode === "webcam") {
      await requestJson("/api/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence: confidenceValue }),
      });
    } else {
      const file = fileInput.files[0];
      if (!file) throw new Error("Choose a video file first.");
      const form = new FormData();
      form.append("video", file);
      form.append("confidence", String(confidenceValue));
      await requestJson("/api/upload", { method: "POST", body: form });
    }
    startFeed();
    pollStatus();
  } catch (error) {
    setRunLabel("error");
    showError(error.message);
  }
}

async function stopTracking() {
  stopButton.disabled = true;
  try {
    const state = await requestJson("/api/stop", { method: "POST" });
    setRunLabel(state.status);
  } catch (error) {
    showError(error.message);
  }
}

async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    const state = await requestJson("/api/status");
    setRunLabel(state.status);
    $("#object-count").textContent = String(state.objects || 0);
    $("#fps-count").textContent = Number(state.fps || 0).toFixed(1);
    $("#frame-count").textContent = Number(state.frames || 0).toLocaleString();
    renderObjects(state.classes);
    if (state.error) showError(state.error);
    else showError();
    if (["starting", "loading_model", "running", "stopping"].includes(state.status)) {
      window.setTimeout(() => { polling = false; pollStatus(); }, 700);
    } else {
      polling = false;
      if (state.status === "ended" || state.status === "stopped" || state.status === "error") {
        feed.hidden = true;
        emptyState.hidden = false;
        emptyState.querySelector("strong").textContent = state.status === "ended" ? "Video playback finished" : "Your preview appears here";
        emptyState.querySelector("span:last-child").textContent = state.status === "ended" ? "Choose the video again to process it from the beginning." : "Start your webcam or choose a video to begin tracking.";
        feed.src = "";
      }
    }
  } catch (error) {
    polling = false;
    setRunLabel("error");
    showError(error.message);
  }
}

sourceTabs.forEach((tab) => tab.addEventListener("click", () => setSource(tab.dataset.source)));
startButton.addEventListener("click", startTracking);
stopButton.addEventListener("click", stopTracking);
confidence.addEventListener("input", () => { $("#confidence-value").value = `${confidence.value}%`; });
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  $("#file-label").textContent = file ? file.name : "Choose a video";
  showError();
});

requestJson("/api/health")
  .then(() => { $("#service-label").textContent = "Local processing ready"; })
  .catch(() => { $("#service-label").textContent = "Service unavailable"; });
