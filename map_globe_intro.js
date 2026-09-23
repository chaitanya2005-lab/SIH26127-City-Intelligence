import * as THREE from 'three';

/**
 * SIH26127 — 3D Earth Globe Transition Overlay
 * Completely independent visual intro overlay above existing Leaflet GIS Map.
 */

window.__sih_globe_initializing = false;

function initGlobeTransition() {
  if (window.__sih_globe_initializing || window.__sih_globe_played) {
    return;
  }
  window.__sih_globe_initializing = true;

  console.log('[GLOBE] page initialization');

  // Reduced motion check
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    console.log('[GLOBE] prefers-reduced-motion enabled. Skipping overlay.');
    window.__sih_globe_played = true;
    return;
  }

  try {
    const container = document.getElementById('command-center-map-container') || document.getElementById('command-center-map');
    if (!container) {
      console.warn('[GLOBE] Map container element not found yet.');
      window.__sih_globe_initializing = false;
      setTimeout(initGlobeTransition, 100);
      return;
    }

    const rect = container.getBoundingClientRect();
    const width = rect.width || container.clientWidth || 650;
    const height = rect.height || container.clientHeight || 420;

    if (width <= 0 || height <= 0) {
      console.warn('[GLOBE] Map container bounds not ready.');
      window.__sih_globe_initializing = false;
      setTimeout(initGlobeTransition, 100);
      return;
    }

    console.log('[GLOBE] map container found');
    console.log('[GLOBE] existing Leaflet map preserved');

    // WebGL support check
    const canvasCheck = document.createElement('canvas');
    const gl = canvasCheck.getContext('webgl') || canvasCheck.getContext('experimental-webgl');
    if (!gl) {
      console.error('[GLOBE ERROR] WebGL context unavailable.');
      window.__sih_globe_played = true;
      return;
    }

    // Remove existing overlay if any
    const oldOverlay = document.getElementById('globe3d-overlay');
    if (oldOverlay) oldOverlay.remove();

    // Set position context on container without touching its innerHTML or children
    if (getComputedStyle(container).position === 'static') {
      container.style.position = 'relative';
    }

    // Create absolute overlay container
    const overlay = document.createElement('div');
    overlay.id = 'globe3d-overlay';
    overlay.style.cssText = `
      position: absolute !important;
      top: 0 !important;
      left: 0 !important;
      width: 100% !important;
      height: 100% !important;
      z-index: 99999 !important;
      background: #08060F !important;
      border-radius: 0.75rem !important;
      overflow: hidden !important;
      pointer-events: none !important;
      opacity: 1;
      transition: opacity 0.05s linear;
    `;

    // Status Badge
    const badge = document.createElement('div');
    badge.style.cssText = `
      position: absolute;
      bottom: 14px;
      left: 14px;
      z-index: 100000;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 800;
      color: #00e5ff;
      background: rgba(8, 6, 15, 0.92);
      border: 1px solid #00e5ff;
      padding: 6px 14px;
      border-radius: 8px;
      letter-spacing: 0.1em;
      box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
      display: flex;
      align-items: center;
      gap: 8px;
    `;
    badge.innerHTML = '<i class="fa-solid fa-earth-asia fa-spin text-purple-400"></i> <span id="globe-badge-status">3D EARTH INTRO</span>';
    overlay.appendChild(badge);

    container.appendChild(overlay);
    console.log('[GLOBE] globe overlay mounted');
    window.__sih_globe_played = true;

    // Three.js Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08060F);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 240;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.cssText = 'width: 100% !important; height: 100% !important; display: block;';
    overlay.appendChild(renderer.domElement);
    console.log('[GLOBE] renderer initialized');

    // Starfield Background
    const starGeo = new THREE.BufferGeometry();
    const starCount = 350;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
      starPos[i] = (Math.random() - 0.5) * 600;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xf5f3ff, size: 1.2, transparent: true, opacity: 0.7 });
    const starField = new THREE.Points(starGeo, starMat);
    scene.add(starField);

    // Procedural Earth Equirectangular Texture (1024x512)
    const mapCanvas = document.createElement('canvas');
    mapCanvas.width = 1024;
    mapCanvas.height = 512;
    const mCtx = mapCanvas.getContext('2d');

    // Ocean Shading
    const oceanGrad = mCtx.createLinearGradient(0, 0, 0, 512);
    oceanGrad.addColorStop(0, '#0c1b38');
    oceanGrad.addColorStop(0.5, '#122b54');
    oceanGrad.addColorStop(1, '#081226');
    mCtx.fillStyle = oceanGrad;
    mCtx.fillRect(0, 0, 1024, 512);

    // Grid Lines
    mCtx.strokeStyle = 'rgba(34, 211, 238, 0.25)';
    mCtx.lineWidth = 1;
    for (let x = 0; x <= 1024; x += 64) {
      mCtx.beginPath(); mCtx.moveTo(x, 0); mCtx.lineTo(x, 512); mCtx.stroke();
    }
    for (let y = 0; y <= 512; y += 64) {
      mCtx.beginPath(); mCtx.moveTo(0, y); mCtx.lineTo(1024, y); mCtx.stroke();
    }

    // Continents
    const continents = [
      // India & South Asia
      [{x: 720, y: 160}, {x: 700, y: 180}, {x: 710, y: 220}, {x: 730, y: 240}, {x: 750, y: 220}, {x: 760, y: 180}],
      // Eurasia
      [{x: 520, y: 50}, {x: 750, y: 65}, {x: 880, y: 120}, {x: 800, y: 160}, {x: 650, y: 170}, {x: 560, y: 130}],
      // Africa
      [{x: 480, y: 160}, {x: 580, y: 170}, {x: 610, y: 240}, {x: 580, y: 350}, {x: 500, y: 320}, {x: 460, y: 220}],
      // Americas
      [{x: 200, y: 80}, {x: 320, y: 100}, {x: 280, y: 200}, {x: 350, y: 260}, {x: 320, y: 380}, {x: 250, y: 420}, {x: 220, y: 240}, {x: 150, y: 120}],
      // Australia
      [{x: 840, y: 300}, {x: 900, y: 310}, {x: 920, y: 370}, {x: 850, y: 380}]
    ];

    mCtx.fillStyle = '#2e7d32';
    mCtx.strokeStyle = '#a855f7';
    mCtx.lineWidth = 3;

    continents.forEach(poly => {
      mCtx.beginPath();
      poly.forEach((pt, idx) => {
        if (idx === 0) mCtx.moveTo(pt.x, pt.y);
        else mCtx.lineTo(pt.x, pt.y);
      });
      mCtx.closePath();
      mCtx.fill();
      mCtx.stroke();
    });

    // India Pin Beacon
    mCtx.fillStyle = '#00e5ff';
    mCtx.beginPath();
    mCtx.arc(730, 200, 6, 0, Math.PI * 2);
    mCtx.fill();
    mCtx.strokeStyle = '#ffffff';
    mCtx.lineWidth = 2;
    mCtx.stroke();

    const earthTexture = new THREE.CanvasTexture(mapCanvas);

    // 3D Earth Sphere Mesh
    const sphereRadius = Math.min(width, height) * 0.28;
    const globeGeo = new THREE.SphereGeometry(sphereRadius, 64, 64);
    const globeMat = new THREE.MeshPhongMaterial({
      map: earthTexture,
      shininess: 30,
      specular: new THREE.Color(0x22d3ee),
      emissive: new THREE.Color(0x0a0618)
    });
    const globeMesh = new THREE.Mesh(globeGeo, globeMat);
    scene.add(globeMesh);

    // Outer Atmosphere Glow Mesh
    const atmosGeo = new THREE.SphereGeometry(sphereRadius * 1.12, 64, 64);
    const atmosMat = new THREE.MeshBasicMaterial({
      color: 0xa855f7,
      transparent: true,
      opacity: 0.25,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending
    });
    const atmosMesh = new THREE.Mesh(atmosGeo, atmosMat);
    scene.add(atmosMesh);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x22d3ee, 1.4);
    dirLight.position.set(150, 100, 200);
    scene.add(dirLight);

    // Easing helper
    function easeInOutCubic(x) {
      return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    }

    const TARGET_ROT_X = 0.35;
    const TARGET_ROT_Y = 3.37;

    let startTime = null;
    let animId = null;
    let loggedEarthVisible = false;
    let loggedIndiaZoom = false;
    let loggedMapReveal = false;

    let startRotY = 0;
    let bRotYDistance = 0;
    let phaseBInit = false;

    function animate(timestamp) {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;

      if (elapsed < 1500) {
        // --- 0.0s - 1.5s: GLOBAL VIEW ---
        if (!loggedEarthVisible) {
          loggedEarthVisible = true;
          console.log('[GLOBE] Earth visible');
        }
        globeMesh.rotation.y += 0.004;
        atmosMesh.rotation.y += 0.004;
        starField.rotation.y += 0.0003;

      } else if (elapsed < 2800) {
        // --- 1.5s - 2.8s: INDIA ZOOM ---
        if (!loggedIndiaZoom) {
          loggedIndiaZoom = true;
          console.log('[GLOBE] India zoom started');
        }

        if (!phaseBInit) {
          phaseBInit = true;
          startRotY = globeMesh.rotation.y;
          let curRotYNorm = startRotY % (Math.PI * 2);
          if (curRotYNorm < 0) curRotYNorm += Math.PI * 2;
          let diff = TARGET_ROT_Y - curRotYNorm;
          while (diff > Math.PI) diff -= Math.PI * 2;
          while (diff < -Math.PI) diff += Math.PI * 2;
          bRotYDistance = diff;
        }

        const pB = (elapsed - 1500) / 1300;
        const eB = easeInOutCubic(Math.min(1, Math.max(0, pB)));

        globeMesh.rotation.y = startRotY + bRotYDistance * eB;
        globeMesh.rotation.x = TARGET_ROT_X * eB;
        atmosMesh.rotation.y = globeMesh.rotation.y;
        atmosMesh.rotation.x = globeMesh.rotation.x;

        camera.position.z = 240 - (240 - 65) * eB;
        starField.rotation.y += 0.0001;

      } else if (elapsed < 3800) {
        // --- 2.8s - 3.8s: MAP REVEAL ---
        if (!loggedMapReveal) {
          loggedMapReveal = true;
          console.log('[GLOBE] Leaflet map reveal started');
        }

        const pC = (elapsed - 2800) / 1000;
        const eC = easeInOutCubic(Math.min(1, Math.max(0, pC)));

        camera.position.z = 65 - (65 - 30) * eC;
        overlay.style.opacity = Math.max(0, (1 - pC)).toFixed(3);

      } else {
        // --- AFTER 3.8s: CLEANUP & REMOVE OVERLAY ---
        if (animId) cancelAnimationFrame(animId);
        overlay.remove();
        console.log('[GLOBE] globe overlay removed');
        renderer.dispose();

        if (window.map && typeof window.map.invalidateSize === 'function') {
          window.map.invalidateSize();
        }
        return;
      }

      renderer.render(scene, camera);
      animId = requestAnimationFrame(animate);
    }

    animId = requestAnimationFrame(animate);

    // Window Resize Handler
    window.addEventListener('resize', () => {
      if (!container || !renderer) return;
      const r = container.getBoundingClientRect();
      const w = r.width || container.clientWidth;
      const h = r.height || container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

  } catch (err) {
    console.error('[GLOBE ERROR] Exception during globe overlay setup:', err);
    const oldOverlay = document.getElementById('globe3d-overlay');
    if (oldOverlay) oldOverlay.remove();
    window.__sih_globe_played = true;
  }
}

// Global invocation helper
window.initThreeGlobe = initGlobeTransition;

// Execution trigger - single listener
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  setTimeout(initGlobeTransition, 10);
} else {
  window.addEventListener('DOMContentLoaded', initGlobeTransition, { once: true });
}
