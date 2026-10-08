/**
 * TruthLens Web Application — Interactive Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Preset Samples for quick evaluation
  const SAMPLES = {
    nasa: {
      title: "NASA Launches Perseverance Rover to Mars",
      text: "NASA launched the exploration rover to Mars successfully from Cape Canaveral. The mission aims to seek signs of ancient microbial life and collect rock samples for future return to Earth. Engineers confirmed all telemetry systems are nominal and the spacecraft is on a direct trajectory.",
    },
    hollow_moon: {
      title: "Hollow Moon Artificial Megastructure",
      text: "The moon is an artificially hollow titanium sphere manufactured by extraterrestrial intelligences and placed into Earth's orbit millions of years ago. Ancient secret documents prove underground military personnel have entered the inner lunar chambers.",
    },
    solar_subsidy: {
      title: "Federal Solar Subsidy Changes",
      text: "New federal solar subsidy legislation announced today will completely eliminate 100 percent of home electricity bills for every household in the country starting next month without any installation costs or grid utility fees.",
    },
    pill_cure: {
      title: "Miracle Universal Health Capsule",
      text: "Medical researchers discover that a single herbal extract pill cures all human illnesses within ten minutes. Pharmaceutical companies have banded together to suppress this miracle formula from hospital pharmacies.",
    },
  };

  // DOM Elements
  const tabButtons = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const sampleChips = document.querySelectorAll(".chip");
  
  const form = document.getElementById("analyze-form");
  const inputTitle = document.getElementById("input-title");
  const inputText = document.getElementById("input-text");
  const selectModel = document.getElementById("select-model");
  const inputUrl = document.getElementById("input-url");
  const charCount = document.getElementById("char-count");
  const btnClear = document.getElementById("btn-clear");
  const btnSubmit = document.getElementById("btn-submit");

  const placeholderState = document.getElementById("placeholder-state");
  const loadingState = document.getElementById("loading-state");
  const assessmentContainer = document.getElementById("assessment-container");

  const verdictBanner = document.getElementById("verdict-banner");
  const verdictBadge = document.getElementById("verdict-badge");
  const valConfidence = document.getElementById("val-confidence");
  const barConfidence = document.getElementById("bar-confidence");
  const valSufficiency = document.getElementById("val-sufficiency");
  const barSufficiency = document.getElementById("bar-sufficiency");
  const valConflict = document.getElementById("val-conflict");
  const barConflict = document.getElementById("bar-conflict");
  const statLatency = document.getElementById("stat-latency");
  const statModel = document.getElementById("stat-model");

  const explanationSummary = document.getElementById("explanation-summary");
  const explanationBreakdown = document.getElementById("explanation-breakdown");
  const claimsList = document.getElementById("claims-list");
  const claimsCountBadge = document.getElementById("claims-count-badge");
  const evidenceList = document.getElementById("evidence-list");

  // Diagnostics DOM
  const systemStatusPill = document.getElementById("system-status-pill");
  const systemStatusText = document.getElementById("system-status-text");
  const diagDbStatus = document.getElementById("diag-db-status");
  const diagDbDialect = document.getElementById("diag-db-dialect");
  const diagDbLatency = document.getElementById("diag-db-latency");
  const diagTorchStatus = document.getElementById("diag-torch-status");
  const diagTorchVer = document.getElementById("diag-torch-ver");
  const diagTorchDevice = document.getElementById("diag-torch-device");
  const diagSklearnVer = document.getElementById("diag-sklearn-ver");
  const diagTransformersVer = document.getElementById("diag-transformers-ver");
  const rawHealthJson = document.getElementById("raw-health-json");
  const btnRefreshHealth = document.getElementById("btn-refresh-health");

  // Benchmarks DOM
  const benchmarkTableBody = document.getElementById("benchmark-table-body");

  // 1. Tab Switching
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const activePane = document.getElementById(targetTab);
      if (activePane) activePane.classList.add("active");

      if (targetTab === "tab-benchmarks") loadBenchmarks();
      if (targetTab === "tab-diagnostics") loadDiagnostics();
    });
  });

  // 2. Character Counter
  inputText.addEventListener("input", () => {
    const len = inputText.value.length;
    charCount.textContent = `${len.toLocaleString()} / 50,000 chars`;
  });

  // 3. Sample Chips
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const sampleKey = chip.getAttribute("data-sample");
      if (SAMPLES[sampleKey]) {
        inputTitle.value = SAMPLES[sampleKey].title;
        inputText.value = SAMPLES[sampleKey].text;
        charCount.textContent = `${inputText.value.length.toLocaleString()} / 50,000 chars`;
        // Trigger auto submit
        form.dispatchEvent(new Event("submit"));
      }
    });
  });

  // 4. Clear Form
  btnClear.addEventListener("click", () => {
    inputTitle.value = "";
    inputText.value = "";
    inputUrl.value = "";
    charCount.textContent = "0 / 50,000 chars";
    placeholderState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    assessmentContainer.classList.add("hidden");
  });

  // 5. Submit Form
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = inputText.value.trim();
    if (!text || text.length < 20) {
      alert("Please provide at least 20 characters of news text or statement.");
      return;
    }

    const payload = {
      text: text,
      title: inputTitle.value.trim() || null,
      url: inputUrl.value.trim() || null,
    };
    const model = selectModel.value;

    // Show loading
    placeholderState.classList.add("hidden");
    assessmentContainer.classList.add("hidden");
    loadingState.classList.remove("hidden");
    btnSubmit.disabled = true;

    try {
      const res = await fetch(`/api/v1/analyze?model=${encodeURIComponent(model)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail?.[0]?.msg || `HTTP Error ${res.status}`);
      }

      const data = await res.json();
      renderAssessment(data);
    } catch (err) {
      alert(`Analysis failed: ${err.message}`);
      placeholderState.classList.remove("hidden");
    } finally {
      loadingState.classList.add("hidden");
      btnSubmit.disabled = false;
    }
  });

  // Render Assessment Output
  function renderAssessment(data) {
    const { assessment, claims, evidence, explanation, execution_stats } = data;

    // Clear class modifiers
    verdictBanner.className = "verdict-banner";

    let stateClass = "state-true";
    let stateLabel = "LIKELY TRUE";

    if (assessment.state === "LIKELY_FALSE") {
      stateClass = "state-false";
      stateLabel = "LIKELY FALSE";
    } else if (assessment.state === "MISLEADING_OR_PARTIALLY_SUPPORTED") {
      stateClass = "state-misleading";
      stateLabel = "MISLEADING / PARTIALLY SUPPORTED";
    } else if (assessment.state === "INSUFFICIENT_EVIDENCE") {
      stateClass = "state-insufficient";
      stateLabel = "INSUFFICIENT EVIDENCE";
    }

    verdictBanner.classList.add(stateClass);
    verdictBadge.textContent = stateLabel;

    // Metrics
    const confPct = Math.round(assessment.calibrated_confidence * 100);
    const suffPct = Math.round(assessment.evidence_sufficiency * 100);
    const conflictPct = Math.round(assessment.conflict_score * 100);

    valConfidence.textContent = `${confPct}%`;
    barConfidence.style.width = `${confPct}%`;

    valSufficiency.textContent = `${suffPct}%`;
    barSufficiency.style.width = `${suffPct}%`;

    valConflict.textContent = `${conflictPct}%`;
    barConflict.style.width = `${conflictPct}%`;

    statLatency.textContent = `⚡ ${execution_stats.processing_time_ms}ms`;
    statModel.textContent = `Model: ${execution_stats.model_used.toUpperCase()}`;

    // Explanation
    explanationSummary.textContent = explanation.summary;
    explanationBreakdown.textContent = explanation.evidence_breakdown;

    // Claims
    claimsCountBadge.textContent = `${claims.length} Claims Extracted`;
    claimsList.innerHTML = claims
      .map(
        (c) => `
        <div class="claim-item">
          <div class="claim-text">"${escapeHtml(c.claim_text)}"</div>
          <div class="claim-meta">
            <span class="salience-badge">Salience: ${Math.round(c.salience_score * 100)}%</span>
            <span class="span-info">[${c.start_char}:${c.end_char}]</span>
          </div>
        </div>
      `
      )
      .join("");

    // Evidence Cards
    evidenceList.innerHTML = evidence
      .map((e) => {
        let stanceClass = "stance-neutral";
        if (e.stance === "SUPPORTS") stanceClass = "stance-supports";
        if (e.stance === "CONTRADICTS") stanceClass = "stance-contradicts";

        return `
        <div class="evidence-item">
          <div class="evidence-header">
            <div class="evidence-source">
              <span>${escapeHtml(e.title || "Independent Verification")}</span>
              <span class="domain-badge">${escapeHtml(e.source_domain)}</span>
            </div>
            <span class="stance-badge ${stanceClass}">${e.stance} (${Math.round(e.stance_confidence * 100)}%)</span>
          </div>
          <p class="evidence-snippet">${escapeHtml(e.snippet)}</p>
          <div class="evidence-footer">
            <span>Provider: ${escapeHtml(e.retrieval_provider)}</span>
            <span>Relevance: ${Math.round(e.relevance_score * 100)}%</span>
          </div>
        </div>
      `;
      })
      .join("");

    assessmentContainer.classList.remove("hidden");
  }

  // Load Benchmarks
  async function loadBenchmarks() {
    try {
      const res = await fetch("/api/v1/benchmarks");
      if (!res.ok) throw new Error("Could not load benchmarks");
      const list = await res.json();

      if (list.length === 0) {
        benchmarkTableBody.innerHTML = `<tr><td colspan="7" class="loading-cell">No benchmark metrics found.</td></tr>`;
        return;
      }

      benchmarkTableBody.innerHTML = list
        .map((row) => {
          const isTop = row.macro_f1 > 0.42;
          return `
          <tr class="${isTop ? "highlight" : ""}">
            <td>${escapeHtml(row.model_name)}</td>
            <td>${(row.accuracy * 100).toFixed(2)}%</td>
            <td>${(row.macro_f1 * 100).toFixed(2)}%</td>
            <td>${(row.macro_precision * 100).toFixed(2)}%</td>
            <td>${(row.macro_recall * 100).toFixed(2)}%</td>
            <td>${(row.weighted_f1 * 100).toFixed(2)}%</td>
            <td>${row.inference_time_s ? (row.inference_time_s * 1000).toFixed(1) + " ms" : "--"}</td>
          </tr>
        `;
        })
        .join("");
    } catch (e) {
      benchmarkTableBody.innerHTML = `<tr><td colspan="7" class="loading-cell">Failed to fetch benchmarks: ${e.message}</td></tr>`;
    }
  }

  // Load Diagnostics
  async function loadDiagnostics() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      rawHealthJson.textContent = JSON.stringify(data, null, 2);

      // Populate Telemetry Cards
      diagDbStatus.textContent = data.components?.database?.status === "up" ? "Connected" : "Degraded";
      diagDbDialect.textContent = data.components?.database?.dialect || "sqlite";
      diagDbLatency.textContent = `${data.components?.database?.latency_ms || 0} ms`;

      diagTorchStatus.textContent = data.components?.ml_runtime?.status === "available" ? "Operational" : "Unavailable";
      diagTorchVer.textContent = data.components?.ml_runtime?.torch_version || "2.13.0+cpu";
      diagTorchDevice.textContent = (data.components?.ml_runtime?.device || "cpu").toUpperCase();

      diagSklearnVer.textContent = data.components?.ml_runtime?.scikit_learn_status || "Active";
      diagTransformersVer.textContent = data.components?.ml_runtime?.transformers_status || "Active";
    } catch (e) {
      rawHealthJson.textContent = `Telemetry error: ${e.message}`;
    }
  }

  if (btnRefreshHealth) {
    btnRefreshHealth.addEventListener("click", loadDiagnostics);
  }

  // Initial Health Ping
  fetch("/health")
    .then((r) => r.json())
    .then((data) => {
      if (data.status === "healthy") {
        systemStatusText.textContent = "All Models & DB Online";
      }
    })
    .catch(() => {
      systemStatusText.textContent = "Connecting to Backend...";
    });

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
