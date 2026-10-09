"use strict";

const CLASS_NAMES = ["Normal", "Overhang fault", "Underhang fault"];
const CLASS_COLORS = ["#28775d", "#c17b13", "#a64943"];
const MAX_SAMPLES = 1_000_000;

const elements = Object.fromEntries([
  "api-base",
  "api-token",
  "connect-button",
  "connection-indicator",
  "connection-label",
  "sample-rate",
  "csv-file",
  "analyze-button",
  "input-message",
  "result-empty",
  "result-content",
  "result-banner",
  "result-condition",
  "result-time",
  "window-count",
  "result-rate",
  "sample-count",
  "tail-note",
  "score-chart",
  "signal-chart",
  "signal-caption",
  "result-rows",
].map((id) => [id, document.getElementById(id)]));

let apiBase = "";
let apiToken = "";
let pollTimer = null;
let polling = false;
let latestTimestamp = "";

function setConnection(state, label) {
  elements["connection-indicator"].dataset.state = state;
  elements["connection-label"].textContent = label;
}

function setMessage(text, state = "") {
  elements["input-message"].textContent = text;
  elements["input-message"].dataset.state = state;
}

function normalizedApiBase(value) {
  const parsed = new URL(value);
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("Enter a base URL without credentials, query string or fragment.");
  }
  const isLocal = ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname);
  if (parsed.protocol !== "https:" && !(isLocal && parsed.protocol === "http:")) {
    throw new Error("Use HTTPS for the Pi API. HTTP is allowed only for localhost testing.");
  }
  return parsed.toString().replace(/\/+$/, "");
}

async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${apiToken}`);
  if (options.body) headers.set("Content-Type", "application/json");
  let response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...options,
      headers,
      cache: "no-store",
    });
  } catch (error) {
    throw new Error(`Could not reach the Pi API. Check its HTTPS address and CORS settings. ${error.message}`);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The Pi API returned a non-JSON response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `Pi API request failed (HTTP ${response.status}).`);
  }
  return payload;
}

function disconnect() {
  if (pollTimer !== null) window.clearInterval(pollTimer);
  pollTimer = null;
  polling = false;
  apiBase = "";
  apiToken = "";
  elements["csv-file"].disabled = true;
  elements["analyze-button"].disabled = true;
  elements["connect-button"].textContent = "Connect to Raspberry Pi";
  latestTimestamp = "";
  setConnection("offline", "Pi not connected");
}

async function connect() {
  apiBase = normalizedApiBase(elements["api-base"].value.trim());
  apiToken = elements["api-token"].value;
  if (apiToken.length < 32) throw new Error("Enter the API token configured on the Pi.");
  setConnection("offline", "Connecting…");
  await apiRequest("/api/health");
  elements["csv-file"].disabled = false;
  elements["analyze-button"].disabled = false;
  elements["connect-button"].textContent = "Disconnect";
  setConnection("online", "Pi API connected");
  setMessage("Connected to the Pi service. Waiting for sensor data or a CSV recording.", "success");
  await refreshLatest();
  pollTimer = window.setInterval(refreshLatest, 2000);
}

async function refreshLatest() {
  if (polling || !apiToken) return;
  polling = true;
  try {
    const payload = await apiRequest("/api/latest");
    if (payload.available && payload.received_at !== latestTimestamp) {
      latestTimestamp = payload.received_at;
      renderResults(payload);
      setMessage("Received a new prediction from the Pi.", "success");
    } else if (!payload.available) {
      setConnection("online", "Connected · waiting for data");
    }
  } catch (error) {
    setConnection("error", "Pi connection lost");
    setMessage(error.message, "error");
  } finally {
    polling = false;
  }
}

function parseFirstColumn(csvText) {
  const lines = csvText.split(/\r?\n/);
  const samples = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    const field = line.split(",", 1)[0].trim();
    if (!field) throw new Error("A row has no value in the first CSV column.");
    const value = Number(field);
    if (!Number.isFinite(value)) {
      throw new Error("The first CSV column must contain only finite numeric samples and no header.");
    }
    samples.push(value);
    if (samples.length > MAX_SAMPLES) {
      throw new Error(`The recording exceeds the ${MAX_SAMPLES.toLocaleString()} sample limit.`);
    }
  }
  if (samples.length === 0) throw new Error("The selected CSV contains no numeric samples.");
  return samples;
}

function formatScore(value) {
  return Number(value).toFixed(4);
}

function renderResults(payload) {
  if (!Array.isArray(payload.windows) || payload.windows.length === 0) {
    throw new Error("The Pi API returned no analyzed windows.");
  }
  const windows = payload.windows;
  const latest = windows[windows.length - 1];
  const isNormal = latest.label === CLASS_NAMES[0];

  elements["result-empty"].hidden = true;
  elements["result-content"].hidden = false;
  elements["result-banner"].dataset.condition = isNormal ? "normal" : "fault";
  elements["result-condition"].textContent = isNormal
    ? "Model class: Normal"
    : `Model-indicated pattern: ${latest.label}`;
  elements["result-time"].textContent = payload.received_at
    ? new Date(payload.received_at).toLocaleString()
    : "";
  elements["window-count"].textContent = String(windows.length);
  elements["result-rate"].textContent = `${Number(payload.sample_rate_hz).toLocaleString()} Hz`;
  elements["sample-count"].textContent = Number(payload.sample_count).toLocaleString();

  const analyzedSeconds = windows.length * 3;
  const sourceDuration = Number(payload.sample_count) / Number(payload.sample_rate_hz);
  const omittedSeconds = Math.max(0, sourceDuration - analyzedSeconds);
  elements["tail-note"].textContent = omittedSeconds > 0.05
    ? `Trailing ${omittedSeconds.toFixed(2)} s is shorter than a full window and was not analyzed.`
    : "";

  renderTable(windows);
  drawScores(elements["score-chart"], windows);
}

function renderTable(windows) {
  const body = elements["result-rows"];
  body.replaceChildren();
  for (const item of windows) {
    const row = document.createElement("tr");
    const values = [
      String(item.window_number),
      `${Number(item.start_seconds).toFixed(0)}–${Number(item.end_seconds).toFixed(0)} s`,
      item.label,
      formatScore(item.decision_scores[CLASS_NAMES[0]]),
      formatScore(item.decision_scores[CLASS_NAMES[1]]),
      formatScore(item.decision_scores[CLASS_NAMES[2]]),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index >= 3) cell.className = "score";
      row.appendChild(cell);
    });
    body.appendChild(row);
  }
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(150, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawScores(canvas, windows) {
  const { context: ctx, width, height } = prepareCanvas(canvas);
  const pad = { left: 48, right: 14, top: 14, bottom: 30 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const series = CLASS_NAMES.map((name) => windows.map((item) => Number(item.decision_scores[name])));
  const allScores = series.flat();
  let minimum = Math.min(...allScores);
  let maximum = Math.max(...allScores);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return;
  if (maximum === minimum) { maximum += 1; minimum -= 1; }
  const range = maximum - minimum;

  ctx.font = "11px Segoe UI, Arial, sans-serif";
  ctx.strokeStyle = "#dce5e3";
  ctx.fillStyle = "#71817f";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = pad.top + (chartHeight * tick) / 4;
    const value = maximum - (range * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(1), pad.left - 8, y + 4);
  }
  const xFor = (index) => pad.left + (windows.length < 2 ? chartWidth / 2 : (chartWidth * index) / (windows.length - 1));
  const yFor = (value) => pad.top + chartHeight * (maximum - value) / range;
  series.forEach((values, seriesIndex) => {
    ctx.strokeStyle = CLASS_COLORS[seriesIndex];
    ctx.fillStyle = CLASS_COLORS[seriesIndex];
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    values.forEach((value, index) => {
      ctx.beginPath();
      ctx.arc(xFor(index), yFor(value), 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  });
  ctx.fillStyle = "#71817f";
  ctx.textAlign = "center";
  ctx.fillText("Window start time (seconds)", pad.left + chartWidth / 2, height - 7);
  ctx.textAlign = "left";
  windows.forEach((item, index) => {
    if (windows.length <= 12 || index === 0 || index === windows.length - 1 || index % Math.ceil(windows.length / 8) === 0) {
      ctx.fillText(String(Math.round(Number(item.start_seconds))), xFor(index) - 3, height - 19);
    }
  });
}

function drawSignal(canvas, samples, sampleRate) {
  const { context: ctx, width, height } = prepareCanvas(canvas);
  const pad = { left: 45, right: 12, top: 14, bottom: 25 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const stride = Math.max(1, Math.ceil(samples.length / 3000));
  const points = [];
  for (let index = 0; index < samples.length; index += stride) points.push(Number(samples[index]));
  const peak = Math.max(1e-8, ...points.map((value) => Math.abs(value)));
  ctx.strokeStyle = "#dce5e3";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + plotHeight / 2);
  ctx.lineTo(width - pad.right, pad.top + plotHeight / 2);
  ctx.stroke();
  ctx.strokeStyle = "#267b72";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  points.forEach((value, index) => {
    const x = pad.left + (points.length < 2 ? 0 : (plotWidth * index) / (points.length - 1));
    const y = pad.top + plotHeight * (0.5 - value / (2 * peak));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#71817f";
  ctx.font = "11px Segoe UI, Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`Time (seconds) · ${samples.length / sampleRate}`, pad.left + plotWidth / 2, height - 5);
}

elements["connect-button"].addEventListener("click", async () => {
  if (apiToken) {
    disconnect();
    setMessage("Disconnected from Raspberry Pi.");
    return;
  }
  elements["connect-button"].disabled = true;
  try {
    await connect();
  } catch (error) {
    disconnect();
    setConnection("error", "Connection failed");
    setMessage(error.message, "error");
  } finally {
    elements["connect-button"].disabled = false;
  }
});

elements["analyze-button"].addEventListener("click", async () => {
  const file = elements["csv-file"].files[0];
  const sampleRate = Number(elements["sample-rate"].value);
  if (!file) { setMessage("Choose a CSV recording first.", "error"); return; }
  if (!Number.isInteger(sampleRate) || sampleRate < 1_000 || sampleRate > 384_000) {
    setMessage("Enter an integer sample rate from 1000 to 384000 Hz.", "error");
    return;
  }
  elements["analyze-button"].disabled = true;
  setMessage(`Reading ${file.name}…`);
  try {
    const samples = parseFirstColumn(await file.text());
    drawSignal(elements["signal-chart"], samples, sampleRate);
    elements["signal-caption"].textContent = `${file.name} · column 1`;
    setMessage(`Sending ${samples.length.toLocaleString()} samples to the Pi…`);
    const payload = await apiRequest("/api/predict", {
      method: "POST",
      body: JSON.stringify({ samples, sample_rate_hz: sampleRate }),
    });
    latestTimestamp = payload.received_at;
    renderResults(payload);
    setConnection("online", "Pi API connected");
    setMessage("Recording processed by the hybrid model on the Pi.", "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    elements["analyze-button"].disabled = false;
  }
});

window.addEventListener("resize", () => {
  if (!elements["result-content"].hidden && latestTimestamp) {
    refreshLatest();
  }
});
