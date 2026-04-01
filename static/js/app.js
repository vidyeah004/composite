const PROPS   = ["Tensile","Youngs","Hardness","Buckling"];
const COLORS  = {"Tensile":"#00e5a0","Youngs":"#3d9eff","Hardness":"#f5a623","Buckling":"#b06aff"};
const LABELS  = {"Tensile":"Tensile Strength","Youngs":"Young's Modulus","Hardness":"Hardness","Buckling":"Buckling"};

let chartBN    = null;
let chartAO    = null;
let activeProp = "Tensile";
let lastPrediction = null;
let chatHistory    = [];

// ── Check if models are already loaded ────────────────────────────────────────
fetch("/api/status").then(r => r.json()).then(d => {
  if (d.models_loaded) showApp();
});

// ── File input listeners ───────────────────────────────────────────────────────
["theory","fea","exp"].forEach(key => {
  document.getElementById("file-" + key).addEventListener("change", function() {
    if (this.files[0]) {
      document.getElementById("name-" + key).textContent = this.files[0].name;
      document.getElementById("drop-" + key).classList.add("has-file");
    }
    checkFilesReady();
  });
});

function checkFilesReady() {
  const ready = ["theory","fea","exp"].every(k => document.getElementById("file-"+k).files[0]);
  document.getElementById("train-btn").disabled = !ready;
}

// ── Train models ───────────────────────────────────────────────────────────────
async function trainModels() {
  const btn = document.getElementById("train-btn");
  const status = document.getElementById("train-status");
  btn.disabled = true;
  btn.textContent = "Training...";
  status.textContent = "Uploading datasets and training models — this takes ~20 seconds...";

  const form = new FormData();
  form.append("theory", document.getElementById("file-theory").files[0]);
  form.append("fea",    document.getElementById("file-fea").files[0]);
  form.append("exp",    document.getElementById("file-exp").files[0]);

  try {
    const res  = await fetch("/api/train", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) {
      status.textContent = "Error: " + data.error;
      btn.disabled = false; btn.textContent = "Train Models";
      return;
    }
    const r2s = data.r2_scores;
    status.textContent = "Done! " + Object.entries(r2s).map(([p,v]) =>
      p + ": GPR=" + v.GPR + " GBM=" + v.GBM).join(" | ");
    setTimeout(showApp, 1200);
  } catch(e) {
    status.textContent = "Error: " + e.message;
    btn.disabled = false; btn.textContent = "Train Models";
  }
}

function showApp() {
  document.getElementById("upload-section").style.display = "none";
  document.getElementById("app-section").style.display = "block";
  document.getElementById("model-status").textContent = "● Models loaded";
  document.getElementById("model-status").classList.add("loaded");
  buildTabs();
}

// ── Sliders ────────────────────────────────────────────────────────────────────
function updateSlider(id) {
  const val = parseFloat(document.getElementById(id+"-slider").value);
  document.getElementById(id+"-display").textContent = val.toFixed(1) + "%";
}

// ── Predict ────────────────────────────────────────────────────────────────────
async function runPrediction() {
  const bn  = parseFloat(document.getElementById("bn-slider").value);
  const ao  = parseFloat(document.getElementById("ao-slider").value);
  const btn = document.getElementById("predict-btn");
  btn.disabled = true; btn.textContent = "Predicting...";

  // Loading state
  PROPS.forEach(p => {
    document.querySelector("#res-"+p+" .res-value").textContent = "...";
    document.querySelector("#res-"+p+" .res-value").classList.add("loading");
  });

  try {
    const res  = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({bn, ao})
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    lastPrediction = data;
    renderResults(data.results);
    renderCharts(data.curves, data.exp_data, bn, ao);
    document.getElementById("charts-row").style.display = "grid";

  } catch(e) {
    alert("Prediction error: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Predict Properties";
  }
}

function getConfidence(uncertainty, value) {
  if (uncertainty === null || uncertainty === undefined) return { label: "High", cls: "conf-high" };
  const cv = Math.abs(uncertainty / value);
  if (cv < 0.02) return { label: "High", cls: "conf-high" };
  if (cv < 0.06) return { label: "Medium-High", cls: "conf-medhi" };
  return { label: "Medium", cls: "conf-med" };
}

function renderResults(results) {
  PROPS.forEach(prop => {
    const r   = results[prop];
    const el  = document.getElementById("res-"+prop);
    const val = el.querySelector(".res-value");
    const meta= el.querySelector(".res-meta");
    val.textContent = r.value.toFixed(5);
    val.classList.remove("loading");
    el.classList.add("active");
    const conf = getConfidence(r.uncertainty, r.value);
    let metaHtml = `<span class="res-tag ${r.model.toLowerCase()}">${r.model}</span>`;
    metaHtml += `<span class="res-tag conf-badge ${conf.cls}">${conf.label} confidence</span>`;
    if (r.uncertainty !== null) {
      metaHtml += `<span class="res-uncert">±${r.uncertainty.toFixed(5)}</span>`;
    }
    meta.innerHTML = metaHtml;
  });
}

// ── Charts ─────────────────────────────────────────────────────────────────────
function buildTabs() {
  ["bn","ao"].forEach(axis => {
    const el = document.getElementById("tabs-"+axis);
    el.innerHTML = "";
    PROPS.forEach(p => {
      const btn = document.createElement("button");
      btn.className = "prop-tab" + (p === activeProp ? " active" : "");
      btn.textContent = LABELS[p];
      btn.onclick = () => {
        activeProp = p;
        document.querySelectorAll(".prop-tab").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".prop-tab").forEach(b => {
          if (b.textContent === LABELS[p]) b.classList.add("active");
        });
        if (lastPrediction) renderCharts(lastPrediction.curves, lastPrediction.exp_data,
          parseFloat(document.getElementById("bn-slider").value),
          parseFloat(document.getElementById("ao-slider").value));
      };
      el.appendChild(btn);
    });
  });
}

function makeChartData(curves, expData, bn, ao, axis) {
  const c    = curves[activeProp];
  const col  = COLORS[activeProp];
  const sweep = c.sweep;
  const mu   = axis === "bn" ? c.bn_mu : c.ao_mu;
  const sd   = axis === "bn" ? c.bn_sd : c.ao_sd;
  const xkey = axis === "bn" ? "BN" : "AO";

  const upper = mu.map((v,i) => v + sd[i]);
  const lower = mu.map((v,i) => v - sd[i]);

  const expPoints = expData.map(d => ({x: d[xkey], y: d[activeProp]}));
  const myX = axis === "bn" ? bn : ao;
  const myY = lastPrediction ? lastPrediction.results[activeProp].value : null;

  return {
    labels: sweep,
    datasets: [
      {
        label: "Upper bound",
        data: upper, borderWidth: 0,
        backgroundColor: col.replace(")", ",0.12)").replace("rgb","rgba"),
        pointRadius: 0, fill: "+1", tension: 0.4,
      },
      {
        label: LABELS[activeProp],
        data: mu, borderColor: col, borderWidth: 2,
        backgroundColor: "transparent",
        pointRadius: 0, tension: 0.4, fill: false,
      },
      {
        label: "Lower bound",
        data: lower, borderWidth: 0,
        backgroundColor: col.replace(")", ",0.12)").replace("rgb","rgba"),
        pointRadius: 0, fill: "-1", tension: 0.4,
      },
      {
        label: "Experimental",
        data: expPoints, type: "scatter",
        backgroundColor: "#fff", borderColor: "#333", borderWidth: 1.5,
        pointRadius: 5, pointStyle: "circle",
      },
      ...(myY !== null ? [{
        label: "Your input",
        data: [{x: myX, y: myY}], type: "scatter",
        backgroundColor: "#ffed4a", borderColor: "#000", borderWidth: 1.5,
        pointRadius: 8, pointStyle: "star",
      }] : []),
    ]
  };
}

function chartOpts(xLabel) {
  return {
    responsive: true, maintainAspectRatio: false, animation: {duration: 300},
    interaction: {mode: "index", intersect: false},
    plugins: {
      legend: {display: false},
      tooltip: {
        backgroundColor: "#141c24", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1,
        titleColor: "#e8edf2", bodyColor: "#6b7a8d",
        callbacks: {
          title: items => xLabel + " = " + parseFloat(items[0].label).toFixed(1) + "%",
          label: item => item.dataset.label + ": " + (typeof item.raw === "object" ? item.raw.y.toFixed(5) : parseFloat(item.raw).toFixed(5)),
        }
      }
    },
    scales: {
      x: {
        type: "linear", title: {display: true, text: xLabel, color: "#6b7a8d"},
        ticks: {color: "#6b7a8d", maxTicksLimit: 6,
                callback: v => parseFloat(v).toFixed(1) + "%"},
        grid: {color: "rgba(255,255,255,0.04)"},
      },
      y: {
        ticks: {color: "#6b7a8d", maxTicksLimit: 5},
        grid: {color: "rgba(255,255,255,0.04)"},
      }
    }
  };
}

function renderCharts(curves, expData, bn, ao) {
  if (chartBN) { chartBN.destroy(); chartBN = null; }
  if (chartAO) { chartAO.destroy(); chartAO = null; }

  chartBN = new Chart(document.getElementById("chart-bn"), {
    type: "line",
    data: makeChartData(curves, expData, bn, ao, "bn"),
    options: chartOpts("BN %")
  });
  chartAO = new Chart(document.getElementById("chart-ao"), {
    type: "line",
    data: makeChartData(curves, expData, bn, ao, "ao"),
    options: chartOpts("AO %")
  });
}

// ── Chat ───────────────────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById("chat-input");
  const text  = input.value.trim();
  if (!text) return;
  input.value = "";
  appendMsg("user", text);
  chatHistory.push({role: "user", content: text});
  document.getElementById("chat-suggestions").style.display = "none";

  const typingId = appendMsg("ai", "Thinking...", true);

  const body = {
    messages: chatHistory,
    prediction_context: lastPrediction ? {
      bn: lastPrediction.bn,
      ao: lastPrediction.ao,
      results: lastPrediction.results,
    } : null
  };

  try {
    const res  = await fetch("/api/chat", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    removeMsg(typingId);
    if (data.error) {
      appendMsg("ai", "Error: " + data.error);
      return;
    }
    appendMsg("ai", data.reply);
    chatHistory.push({role: "assistant", content: data.reply});
    if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
  } catch(e) {
    removeMsg(typingId);
    appendMsg("ai", "Connection error. Please try again.");
  }
}

function quickAsk(text) {
  document.getElementById("chat-input").value = text;
  sendChat();
}

let msgId = 0;
function appendMsg(role, text, typing=false) {
  const id  = "msg-" + (++msgId);
  const box = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.id = id;
  div.className = "msg msg-" + role + (typing ? " msg-typing" : "");
  div.innerHTML = `<div class="msg-bubble">${text.replace(/\n/g,"<br>")}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return id;
}

function removeMsg(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Mode Tab Switching ─────────────────────────────────────────────────────────
function switchMode(mode) {
  document.getElementById("pane-forward").style.display  = mode === "forward" ? "block" : "none";
  document.getElementById("pane-inverse").style.display  = mode === "inverse" ? "block" : "none";
  document.getElementById("tab-forward-btn").classList.toggle("active", mode === "forward");
  document.getElementById("tab-inverse-btn").classList.toggle("active", mode === "inverse");
}

// ── Inverse Prediction ─────────────────────────────────────────────────────────
async function runInverse() {
  const tensile  = document.getElementById("inv-tensile").value.trim();
  const youngs   = document.getElementById("inv-youngs").value.trim();
  const hardness = document.getElementById("inv-hardness").value.trim();
  const buckling = document.getElementById("inv-buckling").value.trim();

  const errEl = document.getElementById("inv-error");
  errEl.textContent = "";

  if (!tensile && !youngs && !hardness && !buckling) {
    errEl.textContent = "Enter at least one target property.";
    return;
  }

  const btn = document.getElementById("inv-run-btn");
  btn.disabled = true;
  btn.textContent = "Optimising...";

  try {
    const res = await fetch("/api/inverse_predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tensile:  tensile  || null,
        youngs:   youngs   || null,
        hardness: hardness || null,
        buckling: buckling || null,
      })
    });
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; return; }

    renderInverseResult(data, {
      Tensile:  tensile  ? parseFloat(tensile)  : null,
      Youngs:   youngs   ? parseFloat(youngs)   : null,
      Hardness: hardness ? parseFloat(hardness) : null,
      Buckling: buckling ? parseFloat(buckling) : null,
    });
    renderInverseHeatmap(data);

  } catch (e) {
    errEl.textContent = "Network error: " + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Find Optimal Composition";
  }
}

function renderInverseResult(d, targets) {
  document.getElementById("inv-empty").style.display = "none";
  document.getElementById("inv-result").style.display = "block";

  document.getElementById("res-inv-bn").textContent = d.bn.toFixed(2);
  document.getElementById("res-inv-ao").textContent = d.ao.toFixed(2);
  document.getElementById("inv-total-err-val").textContent = d.total_error.toExponential(3);

  const grid = document.getElementById("inv-achieved-grid");
  grid.innerHTML = PROPS.map(prop => {
    const a = d.achieved[prop];
    const target = targets[prop];
    let valClass = "";
    let pctStr = "";
    if (target !== null) {
      const pct = Math.abs((a.value - target) / target) * 100;
      valClass = pct < 2 ? "match" : pct < 8 ? "close" : "off";
      pctStr = `<div class="inv-achieved-pct">Δ ${pct.toFixed(1)}%</div>`;
    }
    const unc = a.uncertainty
      ? `<div class="inv-achieved-unc">±${a.uncertainty}</div>` : "";
    const tgt = target !== null
      ? `<div class="inv-achieved-target">target: ${target}</div>` : "";
    const label = {"Tensile":"Tensile","Youngs":"Young's Mod.","Hardness":"Hardness","Buckling":"Buckling"}[prop];
    const unit = a.unit ? ` ${a.unit}` : "";
    return `
      <div class="inv-achieved-item">
        <div class="inv-achieved-prop">${label}${unit}</div>
        <div class="inv-achieved-val ${valClass}">${a.value}</div>
        ${unc}${pctStr}${tgt}
      </div>`;
  }).join("");
}

function renderInverseHeatmap(d) {
  const card = document.getElementById("inv-heatmap-card");
  card.style.display = "block";

  const canvas = document.getElementById("inv-heatmap");
  const surface = d.error_surface;
  const rows = surface.length;
  const cols = surface[0].length;

  let minE = Infinity, maxE = -Infinity;
  for (const row of surface) for (const v of row) {
    if (v < minE) minE = v;
    if (v > maxE) maxE = v;
  }

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 560;
  const H = 200;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const cellW = W / cols;
  const cellH = H / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (surface[r][c] - minE) / (maxE - minE + 1e-12);
      // teal (low error) → amber/red (high error)
      const hue = 160 - t * 130;
      const sat = 55 + t * 30;
      const lig = 18 + t * 28;
      ctx.fillStyle = `hsl(${hue},${sat}%,${lig}%)`;
      ctx.fillRect(c * cellW, (rows - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // Mark best point
  const bx = Math.round((d.bn - 2.5) / 5 * (cols - 1));
  const by = Math.round((d.ao - 2.5) / 5 * (rows - 1));
  const px = bx * cellW + cellW / 2;
  const py = (rows - 1 - by) * cellH + cellH / 2;
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${Math.max(12, Math.round(cellW * 1.8))}px sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("★", px, py);
}
