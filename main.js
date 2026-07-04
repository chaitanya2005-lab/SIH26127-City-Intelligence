// Configuration Constants
const TOTAL_FRAMES = 300;
const LERP_FACTOR = 0.08; // Control animation smoothness (lower = smoother/more inertia)
const BASE_FRAME_URL = '/frames/ezgif-frame-';
const FRAME_EXTENSION = '.jpg';

// State Variables
let images = [];
let currentFrame = 0;
let targetFrame = 0;
let isPreloaded = false;
let isAutoPlaying = false;
let autoPlaySpeed = 1; // Frames per animation tick
let showTextOverlays = true;
let activeFilter = 'normal';

// Drag Scrubber State
let isDraggingScrubber = false;

// DOM Elements
const canvas = document.getElementById('animation-canvas');
const ctx = canvas.getContext('2d');
const loaderOverlay = document.getElementById('loader-overlay');
const circleProgress = document.getElementById('circle-progress-bar');
const loaderPercentageText = document.getElementById('loader-percentage-text');
const loaderStatus = document.getElementById('loader-status');

const playPauseBtn = document.getElementById('play-pause-btn');
const playSvg = document.getElementById('play-svg');
const pauseSvg = document.getElementById('pause-svg');
const restartBtn = document.getElementById('restart-btn');
const textToggleBtn = document.getElementById('text-toggle-btn');
const scrubberTrack = document.getElementById('scrubber-track');
const scrubberProgress = document.getElementById('scrubber-progress');
const scrubberFrameTxt = document.getElementById('scrubber-frame-txt');
const scrubberPercentTxt = document.getElementById('scrubber-percent-txt');
const speedLabel = document.getElementById('speed-label');
const scrollPrompt = document.getElementById('scroll-prompt');

const filterBtns = document.querySelectorAll('.filter-btn');
const scrollDots = document.querySelectorAll('.scroll-dot');
const textCards = document.querySelectorAll('.text-card');
const sections = document.querySelectorAll('.scroll-section');
const restartScrollBtn = document.getElementById('restart-scroll-btn');

// Preloading Image Sequence
function preloadImages() {
  return new Promise((resolve) => {
    let loadedCount = 0;
    const totalCircumference = 283; // 2 * PI * r (r=45)

    for (let i = 1; i <= TOTAL_FRAMES; i++) {
      const img = new Image();
      const paddedNum = String(i).padStart(3, '0');
      img.src = `${BASE_FRAME_URL}${paddedNum}${FRAME_EXTENSION}`;

      img.onload = () => {
        images.push(img);
        loadedCount++;

        // Calculate progress percentage
        const progress = loadedCount / TOTAL_FRAMES;
        const offset = totalCircumference - progress * totalCircumference;
        circleProgress.style.strokeDashoffset = offset;
        loaderPercentageText.textContent = `${Math.round(progress * 100)}%`;
        loaderStatus.textContent = `Loading frame ${loadedCount} of ${TOTAL_FRAMES}...`;

        if (loadedCount === TOTAL_FRAMES) {
          isPreloaded = true;
          // Sort images to make sure they are in order (in case of async load finishing out of order)
          images.sort((a, b) => {
            const aNum = parseInt(a.src.match(/frame-(\d+)/)[1]);
            const bNum = parseInt(b.src.match(/frame-(\d+)/)[1]);
            return aNum - bNum;
          });
          onPreloadComplete();
          resolve();
        }
      };

      img.onerror = () => {
        console.error(`Failed to load frame: ${img.src}`);
        // Increment anyway to prevent blocking
        loadedCount++;
        if (loadedCount === TOTAL_FRAMES) {
          isPreloaded = true;
          onPreloadComplete();
          resolve();
        }
      };
    }
  });
}

function onPreloadComplete() {
  // Hide loader
  loaderOverlay.classList.add('fade-out');

  // Trigger initial resize & draw first frame
  resizeCanvas();
  renderFrame(0);

  // Start Animation Loop
  requestAnimationFrame(animationLoop);
}

// Draw Frame on Canvas with Cover Style Fit
function renderFrame(frameIndex) {
  if (!images[frameIndex]) return;

  const img = images[frameIndex];
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Calculate cover dimensions
  const canvasRatio = canvas.width / canvas.height;
  const imageRatio = img.naturalWidth / img.naturalHeight;

  let drawWidth = canvas.width;
  let drawHeight = canvas.height;
  let drawX = 0;
  let drawY = 0;

  if (canvasRatio > imageRatio) {
    // Canvas is wider than image
    drawHeight = canvas.width / imageRatio;
    drawY = (canvas.height - drawHeight) / 2;
  } else {
    // Canvas is taller than image
    drawWidth = canvas.height * imageRatio;
    drawX = (canvas.width - drawWidth) / 2;
  }

  // Draw image with high quality smoothing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
}

// Update Scroll-driven Frame logic
function updateScrollTarget() {
  if (isAutoPlaying || isDraggingScrubber) return;

  const scrollY = window.scrollY;
  const animationScrollRange = window.innerHeight * 5;
  if (animationScrollRange <= 0) return;

  const scrollFraction = scrollY / animationScrollRange;
  targetFrame = Math.floor(scrollFraction * (TOTAL_FRAMES - 1));
  targetFrame = Math.max(0, Math.min(TOTAL_FRAMES - 1, targetFrame));
}

// Animation Loop (Lerp + Render + UI Sync)
function animationLoop() {
  if (isAutoPlaying) {
    // In autoplay, increment targetFrame
    targetFrame += autoPlaySpeed;
    if (targetFrame >= TOTAL_FRAMES) {
      targetFrame = 0;
    }

    // Scroll window dynamically to match the autoplay state within animation range
    const animationScrollRange = window.innerHeight * 5;
    const scrollTarget = (targetFrame / (TOTAL_FRAMES - 1)) * animationScrollRange;
    window.scrollTo(0, scrollTarget);
  }

  // Linear Interpolation (Lerp) for smooth frames
  const diff = targetFrame - currentFrame;
  if (Math.abs(diff) > 0.02) {
    currentFrame += diff * LERP_FACTOR;
  } else {
    currentFrame = targetFrame;
  }

  const roundedFrameIndex = Math.round(currentFrame);
  renderFrame(roundedFrameIndex);

  // Update UI Elements
  updateUI(roundedFrameIndex);

  requestAnimationFrame(animationLoop);
}

// Update Controls and Text Overlay Opacities
function updateUI(frameIndex) {
  // Update scrubber text & track width
  const percent = (frameIndex / (TOTAL_FRAMES - 1)) * 100;
  
  scrubberFrameTxt.textContent = `FRM ${String(frameIndex + 1).padStart(3, '0')}`;
  scrubberPercentTxt.textContent = `${percent.toFixed(1)}%`;
  
  if (!isDraggingScrubber) {
    scrubberProgress.style.width = `${percent}%`;
  }

  // Hide scroll indicator mouse prompt if scrolled past 2%
  if (percent > 2) {
    scrollPrompt.classList.add('hide');
  } else {
    scrollPrompt.classList.remove('hide');
  }

  // Handle active scroll navigation dots and text overlays depending on scroll percent
  const sectionPercentage = percent / 100;
  
  // We have 6 sections. Ranges can be mapped as:
  // 0: [0 - 0.15]
  // 1: [0.15 - 0.35]
  // 2: [0.35 - 0.55]
  // 3: [0.55 - 0.75]
  // 4: [0.75 - 0.90]
  // 5: [0.90 - 1.0]
  const ranges = [
    { start: 0.0, end: 0.15 },
    { start: 0.15, end: 0.35 },
    { start: 0.35, end: 0.55 },
    { start: 0.55, end: 0.75 },
    { start: 0.75, end: 0.90 },
    { start: 0.90, end: 1.01 }
  ];

  let currentSectionIdx = 0;
  ranges.forEach((range, idx) => {
    const card = document.getElementById(`card-${idx}`);
    if (idx === 0) {
      // Dynamic fade and luxury upward drift for Uday Jewellers title (card-0)
      if (card) {
        if (showTextOverlays && percent <= 30) {
          const fadeOpacity = 1 - (percent / 30);
          card.style.opacity = fadeOpacity;
          card.style.transform = `translateY(${-percent * 1.5}px)`;
          card.style.visibility = 'visible';
        } else {
          card.style.opacity = 0;
          card.style.visibility = 'hidden';
        }
      }
      return;
    }
    
    if (sectionPercentage >= range.start && sectionPercentage < range.end) {
      currentSectionIdx = idx;
      if (showTextOverlays && card) {
        card.classList.add('visible');
      }
    } else {
      if (card && idx !== currentSectionIdx) {
        card.classList.remove('visible');
      }
    }
  });

  // Update Dots indicator
  scrollDots.forEach((dot, idx) => {
    if (idx === currentSectionIdx) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

// Scrubber Click & Drag Logic
function initScrubberEvents() {
  function handleScrubberMove(e) {
    const rect = scrubberTrack.getBoundingClientRect();
    let clientX = e.clientX;
    if (e.touches && e.touches[0]) {
      clientX = e.touches[0].clientX;
    }
    
    let fraction = (clientX - rect.left) / rect.width;
    fraction = Math.max(0, Math.min(1, fraction));
    
    scrubberProgress.style.width = `${fraction * 100}%`;
    
    // Stop autoplay
    if (isAutoPlaying) {
      toggleAutoPlay(false);
    }

    // Instantly scroll window to match scrubbed position within animation sections
    const animationScrollRange = window.innerHeight * 5;
    const targetScrollY = fraction * animationScrollRange;
    
    window.scrollTo(0, targetScrollY);
    targetFrame = Math.floor(fraction * (TOTAL_FRAMES - 1));
  }

  scrubberTrack.addEventListener('pointerdown', (e) => {
    isDraggingScrubber = true;
    scrubberTrack.setPointerCapture(e.pointerId);
    handleScrubberMove(e);
  });

  scrubberTrack.addEventListener('pointermove', (e) => {
    if (isDraggingScrubber) {
      handleScrubberMove(e);
    }
  });

  scrubberTrack.addEventListener('pointerup', (e) => {
    if (isDraggingScrubber) {
      isDraggingScrubber = false;
      scrubberTrack.releasePointerCapture(e.pointerId);
    }
  });
}

// Autoplay loop controls
function toggleAutoPlay(playState) {
  isAutoPlaying = playState;
  if (isAutoPlaying) {
    playSvg.style.display = 'none';
    pauseSvg.style.display = 'block';
    playPauseBtn.classList.add('active');
    speedLabel.textContent = 'Auto Scrolling';
  } else {
    playSvg.style.display = 'block';
    pauseSvg.style.display = 'none';
    playPauseBtn.classList.remove('active');
    speedLabel.textContent = 'Interactive Scroll';
  }
}

// Window resize listener
function resizeCanvas() {
  // Support device pixel ratio for sharp screens
  const dpr = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  
  // Set display size
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;

  // Re-render immediately if loaded
  if (isPreloaded) {
    renderFrame(Math.round(currentFrame));
  }
}

// Wire Up Listeners
window.addEventListener('resize', resizeCanvas);
window.addEventListener('scroll', updateScrollTarget);

// Play Pause toggle
playPauseBtn.addEventListener('click', () => {
  toggleAutoPlay(!isAutoPlaying);
});

// Restart handler
restartBtn.addEventListener('click', () => {
  toggleAutoPlay(false);
  window.scrollTo({ top: 0, behavior: 'smooth' });
  targetFrame = 0;
});

// Restart CTA scroll btn
restartScrollBtn.addEventListener('click', (e) => {
  e.preventDefault();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Text Visibility Toggle
textToggleBtn.addEventListener('click', () => {
  showTextOverlays = !showTextOverlays;
  textToggleBtn.classList.toggle('active', showTextOverlays);
  
  textCards.forEach((card) => {
    if (!showTextOverlays) {
      card.classList.remove('visible');
    }
  });
});

// Color filters toggle
filterBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    filterBtns.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');

    const filterType = btn.getAttribute('data-filter');
    canvas.className = ''; // Reset classes
    if (filterType !== 'normal') {
      canvas.classList.add(`filter-${filterType}`);
    }
  });
});

// Sidebar Dot navigation click handler
scrollDots.forEach((dot) => {
  dot.addEventListener('click', () => {
    const dotIndex = parseInt(dot.getAttribute('data-index'));
    const animationScrollRange = window.innerHeight * 5;
    
    // Calculate target frame range
    const ranges = [0.0, 0.25, 0.45, 0.65, 0.82, 0.98];
    const targetFraction = ranges[dotIndex];
    
    window.scrollTo({
      top: targetFraction * animationScrollRange,
      behavior: 'smooth'
    });
  });
});

// Main init
document.addEventListener('DOMContentLoaded', () => {
  preloadImages();
  initScrubberEvents();
});
