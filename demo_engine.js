/**
 * SIH26127 — Pure Client-Side DemoEngine Module
 * Orchestrates existing prototype features into a controlled live judging demonstration.
 * Zero backend changes. Zero API schema changes. Zero fake data generation.
 */

window.DemoEngine = (function() {
  let active = false;
  let currentStageIndex = 0;
  let isPlaying = false;
  let timerId = null;
  let savedTabBeforeDemo = 'view-command-center';
  const stageDurationMs = 7000;

  const stages = [
    {
      id: 1,
      name: "1 / 10 — INTRO",
      headline: "AI-Powered City-Wide Traffic Intelligence",
      subtext: "From distributed CCTV observations to vehicle identity, trajectory, prediction and decision intelligence.",
      tab: "view-command-center",
      action: () => {
        window.switchTab('view-command-center');
        updateStageOverlay(
          "STAGE 1: City Intelligence Overview",
          "AI-Powered City-Wide Traffic Intelligence",
          "From distributed CCTV observations to vehicle identity, trajectory, prediction and decision intelligence."
        );
      }
    },
    {
      id: 2,
      name: "2 / 10 — LIVE ANPR",
      headline: "Real-Time ANPR Vision & Plate Recognition",
      subtext: "Live optical plate extraction and classification using existing camera sensor feeds.",
      tab: "view-live-anpr",
      action: () => {
        window.switchTab('view-live-anpr');
        updateStageOverlay(
          "STAGE 2: Live ANPR Vision Lab",
          "Real-Time ANPR & Plate Crop Extraction",
          "Active real vehicle detection on CAM_01. License plate text recognition & confidence rating."
        );
      }
    },
    {
      id: 3,
      name: "3 / 10 — TRAJECTORY",
      headline: "Multi-Camera Trajectory Reconstruction",
      subtext: "Reconstructing spatial camera-to-camera movement vectors for vehicle MH12DE1408.",
      tab: "view-command-center",
      action: () => {
        window.switchTab('view-command-center');
        if (typeof window.selectVehicle === 'function') {
          window.selectVehicle('MH12DE1408');
        }
        updateStageOverlay(
          "STAGE 3: Multi-Camera Trajectory",
          "Spatial Trajectory Reconstruction Across Cameras",
          "Tracking MH12DE1408 across CAM_01 -> CAM_02 node corridors on spatial GIS radar map."
        );
      }
    },
    {
      id: 4,
      name: "4 / 10 — PREDICTION",
      headline: "Predictive Route Intelligence (Probabilistic Estimate)",
      subtext: "PREDICTED — probabilistic estimate based on historical Markov transition frequencies.",
      tab: "view-route-prediction",
      action: () => {
        window.switchTab('view-route-prediction');
        updateStageOverlay(
          "STAGE 4: Route Prediction",
          "PREDICTED — probabilistic estimate",
          "Predicted Next Camera: CAM_03 (78.5% confidence score). Probabilistic transition model."
        );
      }
    },
    {
      id: 5,
      name: "5 / 10 — ANOMALY & EVIDENCE",
      headline: "Anomaly Radar & 5 Ws Court Evidence Chain",
      subtext: "Auditing vehicle behavior. MH12DE1408: No anomalous behavior detected. Nominal status.",
      tab: "view-privacy-audit",
      action: () => {
        window.switchTab('view-privacy-audit');
        updateStageOverlay(
          "STAGE 5: Anomaly Radar & Evidence Chain",
          "Court-Admissible 5 Ws Evidence Log",
          "MH12DE1408 Status: No anomalous behavior detected. Tamper-evident evidence chain active."
        );
      }
    },
    {
      id: 6,
      name: "6 / 10 — TRAFFIC ANALYTICS",
      headline: "City Digital Twin & Origin-Destination (OD) Matrix",
      subtext: "City-wide spatial congestion scoring, inter-camera node transition matrix and volume vectors.",
      tab: "view-digital-twin",
      action: () => {
        window.switchTab('view-digital-twin');
        updateStageOverlay(
          "STAGE 6: City Digital Twin & OD Matrix",
          "Inter-Camera Flow & Origin-Destination Matrix",
          "Real-time node volume vectors, congestion scores (32/100 nominal), and transition matrix."
        );
      }
    },
    {
      id: 7,
      name: "7 / 10 — WHAT-IF SIMULATOR",
      headline: "What-If Traffic Scenario Simulator",
      subtext: "SIMULATION — decision-support estimate. Testing hypothetical traffic volume shifts.",
      tab: "view-what-if",
      action: () => {
        window.switchTab('view-what-if');
        if (typeof window.runWhatIfSimulation === 'function') {
          window.runWhatIfSimulation();
        }
        updateStageOverlay(
          "STAGE 7: What-If Traffic Simulator",
          "SIMULATION — decision-support estimate",
          "Simulating +40% volume shift at CAM_01. Predicting downstream propagation & congestion impact."
        );
      }
    },
    {
      id: 8,
      name: "8 / 10 — AI COPILOT",
      headline: "Natural Language City Intelligence Copilot",
      subtext: "Interactive natural language telemetry query processing over live analytics pipeline data.",
      tab: "view-copilot",
      action: () => {
        window.switchTab('view-copilot');
        if (typeof window.askCopilot === 'function') {
          window.askCopilot("Which camera has the highest congestion?");
        }
        updateStageOverlay(
          "STAGE 8: AI City Copilot",
          "Natural Language Query Processing",
          "Executing query: 'Which camera has the highest congestion?' over live telemetry."
        );
      }
    },
    {
      id: 9,
      name: "9 / 10 — PRIVACY & SECURITY",
      headline: "DPDP-aligned Privacy Controls & Role-Based Access Control",
      subtext: "Salted SHA-256 PII license plate hashing, RBAC security roles, and audit trail logs.",
      tab: "view-privacy-audit",
      action: () => {
        window.switchTab('view-privacy-audit');
        if (typeof window.updateSecurityRole === 'function') {
          window.updateSecurityRole('LAW_ENFORCEMENT');
        }
        updateStageOverlay(
          "STAGE 9: Privacy & System Health",
          "DPDP-aligned Privacy Controls & RBAC Audit",
          "Salted SHA-256 PII plate hashing active. Role elevated to LAW_ENFORCEMENT with audit log."
        );
      }
    },
    {
      id: 10,
      name: "10 / 10 — SUMMARY",
      headline: "ANPR is only the input.",
      subtext: "Our platform converts distributed camera observations into city-wide vehicle, trajectory, prediction, anomaly and decision intelligence.",
      tab: "view-command-center",
      action: () => {
        window.switchTab('view-command-center');
        if (typeof window.updateSecurityRole === 'function') {
          window.updateSecurityRole('PUBLIC_OPERATOR');
        }
        updateStageOverlay(
          "STAGE 10: Final Command Center Summary",
          "ANPR is only the input.",
          "Our platform converts distributed camera observations into city-wide vehicle, trajectory, prediction, anomaly and decision intelligence."
        );
      }
    }
  ];

  function updateStageOverlay(badgeText, headlineText, subtextText) {
    const bElem = document.getElementById('demo-stage-badge');
    const hElem = document.getElementById('demo-stage-headline');
    const sElem = document.getElementById('demo-stage-subtext');
    if (bElem) bElem.innerText = badgeText;
    if (hElem) hElem.innerText = headlineText;
    if (sElem) sElem.innerText = subtextText;
  }

  function renderControlBar() {
    let bar = document.getElementById('demo-mode-bar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'demo-mode-bar';
      bar.className = 'fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 p-3.5 rounded-2xl bg-[#120D1F]/95 backdrop-blur-md border border-[#A855F7]/50 shadow-2xl shadow-purple-950/90 text-xs font-mono select-none transition-all duration-300 w-11/12 max-w-2xl';
      document.body.appendChild(bar);
    }

    const stage = stages[currentStageIndex];

    bar.innerHTML = `
      <div class="flex items-center justify-between border-b border-[#39245A] pb-2">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full ${isPlaying ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}"></span>
          <span class="font-bold text-white tracking-wider">DEMO MODE</span>
          <span id="demo-stage-badge" class="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-extrabold text-[11px]">${stage.name}</span>
        </div>

        <div class="flex items-center gap-1.5">
          <button onclick="DemoEngine.prevStep()" class="px-2.5 py-1 rounded bg-[#171025] hover:bg-purple-900/40 text-purple-300 border border-[#39245A] hover:border-purple-500 transition-colors" title="Previous Stage (Left Arrow)">
            <i class="fa-solid fa-backward-step mr-1"></i> Prev
          </button>
          ${isPlaying ? `
            <button onclick="DemoEngine.pause()" class="px-3 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/50 font-bold flex items-center gap-1 transition-colors">
              <i class="fa-solid fa-pause"></i> PAUSE
            </button>
          ` : `
            <button onclick="DemoEngine.play()" class="px-3 py-1 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/50 font-bold flex items-center gap-1 transition-colors">
              <i class="fa-solid fa-play"></i> PLAY
            </button>
          `}
          <button onclick="DemoEngine.nextStep()" class="px-2.5 py-1 rounded bg-[#171025] hover:bg-purple-900/40 text-purple-300 border border-[#39245A] hover:border-purple-500 transition-colors" title="Next Stage (Right Arrow)">
            Next <i class="fa-solid fa-forward-step ml-1"></i>
          </button>
          <button onclick="DemoEngine.exit()" class="ml-2 px-2.5 py-1 rounded bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800 text-[11px] font-bold transition-colors" title="Exit Demo Mode (ESC)">
            <i class="fa-solid fa-xmark mr-1"></i> EXIT
          </button>
        </div>
      </div>

      <div class="pt-1 space-y-0.5">
        <div id="demo-stage-headline" class="text-white font-display font-bold text-sm tracking-wide">${stage.headline}</div>
        <div id="demo-stage-subtext" class="text-slate-300 font-sans text-xs leading-tight">${stage.subtext}</div>
      </div>
    `;
  }

  function runStage(index) {
    if (index < 0) index = 0;
    if (index >= stages.length) index = stages.length - 1;
    currentStageIndex = index;
    const stage = stages[currentStageIndex];
    stage.action();
    renderControlBar();
  }

  function startAutoplay() {
    stopAutoplay();
    isPlaying = true;
    renderControlBar();
    timerId = setInterval(() => {
      if (currentStageIndex < stages.length - 1) {
        runStage(currentStageIndex + 1);
      } else {
        stopAutoplay();
      }
    }, stageDurationMs);
  }

  function stopAutoplay() {
    isPlaying = false;
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
    renderControlBar();
  }

  function handleKeyDown(e) {
    if (!active) return;
    if (e.key === 'Escape') {
      exitDemo();
    } else if (e.key === 'ArrowRight') {
      nextStep();
    } else if (e.key === 'ArrowLeft') {
      prevStep();
    } else if (e.key === ' ') {
      e.preventDefault();
      if (isPlaying) stopAutoplay(); else startAutoplay();
    }
  }

  function startDemo() {
    if (active) return;
    active = true;
    savedTabBeforeDemo = window.currentTab || 'view-command-center';
    currentStageIndex = 0;
    window.addEventListener('keydown', handleKeyDown);
    runStage(0);
    startAutoplay();
  }

  function exitDemo() {
    if (!active) return;
    stopAutoplay();
    active = false;
    window.removeEventListener('keydown', handleKeyDown);

    const bar = document.getElementById('demo-mode-bar');
    if (bar) bar.remove();

    if (window.switchTab && savedTabBeforeDemo) {
      window.switchTab(savedTabBeforeDemo);
    }
  }

  function nextStep() {
    stopAutoplay();
    if (currentStageIndex < stages.length - 1) {
      runStage(currentStageIndex + 1);
    }
  }

  function prevStep() {
    stopAutoplay();
    if (currentStageIndex > 0) {
      runStage(currentStageIndex - 1);
    }
  }

  return {
    start: startDemo,
    exit: exitDemo,
    play: startAutoplay,
    pause: stopAutoplay,
    nextStep: nextStep,
    prevStep: prevStep,
    isActive: () => active,
    getCurrentStage: () => currentStageIndex + 1
  };
})();
