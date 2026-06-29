/* =========================================================
   COFFEE MONITOR PRO - CCTV CAMERA V43 ULTRA
   New in V43 (on top of V35/V35.2 fixes):
   - 📸 Screenshot button (captures video + AI overlay as PNG)
   - 🆔 Basic person tracking (persistent IDs across frames using
     nearest-centroid matching, so "Person 1" stays Person 1 while
     they move, instead of IDs reshuffling every frame)
   - ⚡ Android-optimized performance:
       - AI detection decoupled from rendering: runs on its own
         throttled interval (not every single animation frame),
         which saves battery/CPU on phones
       - Detection automatically pauses when the tab/app is in the
         background (Page Visibility API) - no wasted battery
       - Adaptive throttling based on device core count
         (navigator.hardwareConcurrency) - slower phones detect
         less often automatically
   ========================================================= */

/* ---------- Safe element getter ---------- */
function $(id) {
    const el = document.getElementById(id);
    if (!el) console.warn(`[CoffeeMonitor] Missing element: #${id}`);
    return el;
}

/* ---------- DOM Elements ---------- */
const video   = $("video");
const overlay = $("overlayCanvas");
const ctx     = overlay ? overlay.getContext("2d") : null;

const uploadInput = $("uploadInput");
const btnUpload      = $("btnUpload");
const btnLive        = $("btnLive");
const btnSwitchCam   = $("btnSwitchCam");
const btnNight       = $("btnNight");
const btnRecord      = $("btnRecord");
const btnScreenshot  = $("btnScreenshot");
const btnStart       = $("btnStart");
const btnStop        = $("btnStop");

const totalCountEl    = $("totalCount");
const standingCountEl = $("standingCount");
const sittingCountEl  = $("sittingCount");

const statusText = $("statusText");
const alertText  = $("alertText");

const livePersonsWrap = $("livePersonsWrap");
const logBox = $("logBox");

const fpsValueEl   = $("fpsValue");
const speedValueEl = $("speedValue");
const loadValueEl  = $("loadValue");
const confValueEl  = $("confValue");
const frameValueEl = $("frameValue");
const perfFillEl   = $("perfFill");

const browserNameEl  = $("browserName");
const platformNameEl = $("platformName");
const cpuCoresEl     = $("cpuCores");
const deviceRamEl    = $("deviceRam");
const screenSizeEl   = $("screenSize");

const totalDetectionsEl = $("totalDetections");
const currentPersonsEl  = $("currentPersons");
const maxPersonsEl      = $("maxPersons");
const aiAccuracyEl      = $("aiAccuracy");
const aiStatusEl        = $("aiStatus");

const cameraResolutionEl = $("cameraResolution");
const cameraFPSEl        = $("cameraFPS");
const cameraStatusEl     = $("cameraStatus");
const modelNameEl        = $("modelName");

const liveTimeEl = $("liveTime");
const liveDateEl = $("liveDate");
const liveDayEl  = $("liveDay");

const installBtn = $("installBtn");

/* ---------- State ---------- */
let stream = null;
let currentFacingMode = "environment";
let detectModel = null;     // coco-ssd (boxes)
let poseModel = null;       // pose-detection (standing/sitting accuracy)
let usingPoseModel = false;
let aiRunning = false;
let nightVision = false;
let recorder = null;
let recordedChunks = [];
let isRecording = false;
let currentObjectUrl = null;

let lastFrameTime = performance.now();
let frameCount = 0;
let totalDetections = 0;
let maxPersons = 0;
let confidenceSum = 0;
let confidenceSamples = 0;
let previousPersonCount = 0;

const FPS_WINDOW = 12;
let fpsHistory = [];

const LOG_LIMIT = 60;

/* ---------- Basic Person Tracking (V43) ----------
   Assigns a persistent ID to each detected person by matching the
   current frame's bounding-box centroids to the previous frame's
   tracked centroids (nearest-neighbor within a distance threshold).
   This is intentionally lightweight (no extra model) so it stays
   fast on phones, while still keeping "Person 1" attached to the
   same person as they move around instead of IDs reshuffling. ---*/
let tracks = [];           // [{ id, cx, cy, lastSeen }]
let nextTrackId = 1;
const TRACK_MAX_MISSES = 10;     // frames a track can go unmatched before being dropped
const TRACK_MATCH_RATIO = 0.18;  // max match distance, as a fraction of frame diagonal

function updateTracks(boxes) {
    const diag = overlay ? Math.hypot(overlay.width, overlay.height) : 1000;
    const maxDist = diag * TRACK_MATCH_RATIO;

    const assigned = new Array(boxes.length).fill(null);
    const usedTrackIdx = new Set();

    // Greedy nearest-centroid matching
    boxes.forEach((box, i) => {
        const [x, y, w, h] = box;
        const cx = x + w / 2;
        const cy = y + h / 2;

        let bestIdx = -1;
        let bestDist = Infinity;

        tracks.forEach((t, ti) => {
            if (usedTrackIdx.has(ti)) return;
            const d = Math.hypot(t.cx - cx, t.cy - cy);
            if (d < bestDist && d < maxDist) {
                bestDist = d;
                bestIdx = ti;
            }
        });

        if (bestIdx >= 0) {
            tracks[bestIdx].cx = cx;
            tracks[bestIdx].cy = cy;
            tracks[bestIdx].lastSeen = 0;
            assigned[i] = tracks[bestIdx].id;
            usedTrackIdx.add(bestIdx);
        } else {
            const newTrack = { id: nextTrackId++, cx, cy, lastSeen: 0 };
            tracks.push(newTrack);
            assigned[i] = newTrack.id;
            usedTrackIdx.add(tracks.length - 1);
        }
    });

    // Age out tracks that weren't matched this frame
    tracks.forEach((t, ti) => {
        if (!usedTrackIdx.has(ti)) t.lastSeen++;
    });
    tracks = tracks.filter(t => t.lastSeen <= TRACK_MAX_MISSES);

    return assigned; // array of IDs, same order/length as `boxes`
}

/* ---------- Android-optimized detection throttling (V43) ----------
   Running full AI inference at the browser's native paint rate
   (~60 times/sec) is wasteful on phones and drains battery fast.
   Instead, AI inference runs on its own timer at a target interval,
   while the video itself keeps playing smoothly. Lower-core devices
   automatically get a longer interval (less frequent AI, smoother UI).--*/
const cores = navigator.hardwareConcurrency || 4;
const DETECTION_INTERVAL_MS = cores <= 4 ? 200 : (cores <= 6 ? 130 : 90);
let detectionTimer = null;
let tabVisible = true;

document.addEventListener("visibilitychange", () => {
    tabVisible = document.visibilityState === "visible";
    if (!tabVisible) {
        addLog("⏸ Paused AI (app in background) - saving battery");
    } else if (aiRunning) {
        addLog("▶ Resumed AI (app in foreground)");
    }
});

/* ---------- Logging ---------- */
function addLog(text) {
    if (!logBox) return;
    const time = new Date().toLocaleTimeString();
    const entries = logBox.innerHTML.split("<br>").filter(Boolean);
    entries.unshift(`[${time}] ${text}`);
    logBox.innerHTML = entries.slice(0, LOG_LIMIT).join("<br>");
}

function setStatus(text) {
    if (statusText) statusText.textContent = text;
}

function setBusy(btn, busy) {
    if (btn) btn.disabled = busy;
}

/* ---------- Tiny beep (no audio file needed) ---------- */
let audioCtx = null;
function beep() {
    try {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.value = 880;
        gain.gain.value = 0.06;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
    } catch (e) {
        // Audio not available - silently ignore, never break detection
    }
}

/* ---------- Camera ---------- */
async function startCamera() {
    if (!video) return;
    setBusy(btnLive, true);
    try {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;
        video.muted = true;
        await video.play();

        video.onloadedmetadata = () => {
            if (overlay) {
                overlay.width = video.videoWidth;
                overlay.height = video.videoHeight;
            }
            if (cameraResolutionEl) {
                cameraResolutionEl.textContent = `${video.videoWidth} × ${video.videoHeight}`;
            }
        };

        if (cameraStatusEl) cameraStatusEl.textContent = "Running";
        setStatus("Camera Ready");
        addLog("📷 Camera Started");

    } catch (err) {
        console.error(err);
        setStatus("Camera Error");
        if (cameraStatusEl) cameraStatusEl.textContent = "Error";
        addLog("❌ Camera Permission Denied");
        alert("Camera permission denied or not available.");
    } finally {
        setBusy(btnLive, false);
    }
}

function stopCamera() {
    if (!stream) return;

    stream.getTracks().forEach(track => track.stop());
    if (video) video.srcObject = null;
    stream = null;

    if (cameraStatusEl) cameraStatusEl.textContent = "Stopped";
    setStatus("Camera Stopped");
    addLog("⏹ Camera Stopped");
}

async function switchCamera() {
    setBusy(btnSwitchCam, true);
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    addLog("🔄 Camera Switched");
    await startCamera();
    setBusy(btnSwitchCam, false);
}

/* ---------- Video Upload ---------- */
function handleVideoUpload(file) {
    if (!file || !video) return;

    stopCamera();

    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
    }

    currentObjectUrl = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = currentObjectUrl;
    video.muted = false;
    video.loop = true;
    video.play();

    video.onloadedmetadata = () => {
        if (overlay) {
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;
        }
        if (cameraResolutionEl) {
            cameraResolutionEl.textContent = `${video.videoWidth} × ${video.videoHeight}`;
        }
    };

    if (cameraStatusEl) cameraStatusEl.textContent = "Playing File";
    setStatus("Video File Loaded");
    addLog("📂 Video Uploaded: " + file.name);
}

/* ---------- Night Vision ---------- */
function toggleNightVision() {
    if (!video) return;
    nightVision = !nightVision;

    if (nightVision) {
        video.style.filter = "brightness(1.8) contrast(1.3) grayscale(0.4) sepia(0.2) hue-rotate(80deg)";
        if (btnNight) btnNight.textContent = "🌙 Night Vision: ON";
        addLog("🌙 Night Vision Enabled");
    } else {
        video.style.filter = "none";
        if (btnNight) btnNight.textContent = "🌙 Night Vision";
        addLog("🌙 Night Vision Disabled");
    }
}

/* ---------- Recording ---------- */
function toggleRecording() {
    if (!isRecording) {
        if (!stream) {
            alert("Start the camera before recording.");
            return;
        }

        recordedChunks = [];

        try {
            recorder = new MediaRecorder(stream);
        } catch (e) {
            console.error(e);
            addLog("❌ Recording not supported on this browser");
            return;
        }

        recorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        recorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `recording-${Date.now()}.webm`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 5000);
            addLog("💾 Recording Saved");
        };

        recorder.start();
        isRecording = true;
        if (btnRecord) btnRecord.textContent = "⏹ Stop Recording";
        addLog("⏺ Recording Started");

    } else {
        if (recorder && recorder.state !== "inactive") recorder.stop();
        isRecording = false;
        if (btnRecord) btnRecord.textContent = "⏺ Recording";
        addLog("⏹ Recording Stopped");
    }
}

/* ---------- Screenshot (V43) ----------
   Captures the current video frame together with the AI overlay
   (bounding boxes / labels) as a single PNG, since they're rendered
   in two separate layers on screen (<video> + <canvas>). ---*/
function takeScreenshot() {
    if (!video || video.readyState < 2) {
        alert("Camera/video isn't ready yet.");
        return;
    }

    const shot = document.createElement("canvas");
    shot.width = video.videoWidth;
    shot.height = video.videoHeight;
    const shotCtx = shot.getContext("2d");

    // Layer 1: the raw video frame (respects night-vision CSS filter)
    shotCtx.filter = video.style.filter || "none";
    shotCtx.drawImage(video, 0, 0, shot.width, shot.height);
    shotCtx.filter = "none";

    // Layer 2: AI overlay (boxes/labels), if present and same size
    if (overlay && overlay.width > 0 && overlay.height > 0) {
        shotCtx.drawImage(overlay, 0, 0, shot.width, shot.height);
    }

    shot.toBlob((blob) => {
        if (!blob) {
            addLog("❌ Screenshot failed");
            return;
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `screenshot-${Date.now()}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        addLog("📸 Screenshot Saved");
    }, "image/png");
}
async function loadAI() {
    if (detectModel) return;

    setStatus("Loading AI...");
    if (aiStatusEl) aiStatusEl.textContent = "Loading...";

    // --- Guard: required libraries must exist before we touch them ---
    if (typeof tf === "undefined") {
        addLog("❌ TensorFlow.js not found - check your internet connection / script tags");
        setStatus("AI Error: tfjs missing");
        if (aiStatusEl) aiStatusEl.textContent = "Error";
        throw new Error("tfjs not loaded");
    }

    if (typeof cocoSsd === "undefined") {
        addLog("❌ COCO-SSD library not found - check script tags / internet connection");
        setStatus("AI Error: coco-ssd missing");
        if (aiStatusEl) aiStatusEl.textContent = "Error";
        throw new Error("coco-ssd not loaded");
    }

    // Explicitly set + warm up the WebGL backend before loading models.
    // This avoids silent slowdowns/crashes if the backend wasn't ready yet.
    try {
        await tf.setBackend("webgl");
        await tf.ready();
        addLog("⚙️ TensorFlow Backend: " + tf.getBackend());
    } catch (e) {
        console.warn("WebGL backend unavailable, using default backend.", e);
    }

    addLog("🤖 Loading COCO-SSD Model...");
    detectModel = await cocoSsd.load();
    addLog("✅ COCO-SSD Model Loaded");

    // Try to also load a pose model for accurate standing/sitting detection.
    // If the library wasn't loaded, or loading fails for any reason, we
    // silently fall back to the bbox heuristic - never breaks the app.
    if (typeof poseDetection === "undefined") {
        addLog("ℹ️ Pose-detection library not found - using bbox heuristic for posture");
        usingPoseModel = false;
    } else {
        try {
            addLog("🤖 Loading Pose Model (MoveNet)...");
            poseModel = await poseDetection.createDetector(
                poseDetection.SupportedModels.MoveNet,
                { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
            );
            usingPoseModel = true;
            addLog("✅ Pose Model Loaded (accurate posture detection)");
        } catch (e) {
            console.warn("Pose model unavailable, using bbox heuristic instead.", e);
            usingPoseModel = false;
        }
    }

    setStatus("AI Ready");
    if (aiStatusEl) aiStatusEl.textContent = "Ready";
    if (modelNameEl) modelNameEl.textContent = usingPoseModel ? "COCO-SSD + MoveNet" : "COCO-SSD";
}

/* ---------- Posture classification (performance-friendly) ----------
   Running pose estimation once PER PERSON, every frame, is expensive and
   tanks FPS. Instead we:
   1. Run estimatePoses() at most ONCE per frame (not per person)
   2. Only re-run it every POSE_INTERVAL frames - in between, we reuse the
      last cached result
   3. If no pose model is available (or it fails), we instantly fall back
      to the cheap bbox aspect-ratio heuristic - this never blocks detection
------------------------------------------------------------------- */
const POSE_INTERVAL = 8; // run MoveNet every 8 frames (~ a few times per second)
let poseFrameCounter = 0;
let cachedPoses = [];

async function refreshPoseCache() {
    if (!usingPoseModel || !poseModel) return;

    poseFrameCounter++;
    if (poseFrameCounter % POSE_INTERVAL !== 0) return; // reuse cache

    try {
        cachedPoses = await poseModel.estimatePoses(video);
    } catch (e) {
        console.warn("Pose estimation failed for this frame, reusing cache.", e);
    }
}

function classifyPostureFromBbox(bbox) {
    const [, , w, h] = bbox;
    const aspect = h / w;
    return aspect >= 1.4 ? "standing" : "sitting";
}

function classifyPosture(bbox) {
    if (usingPoseModel && cachedPoses && cachedPoses.length > 0) {
        const pose = cachedPoses[0]; // single-pose model -> most prominent person
        const kp = pose.keypoints;
        const hip = kp.find(p => p.name === "left_hip" || p.name === "right_hip");
        const knee = kp.find(p => p.name === "left_knee" || p.name === "right_knee");
        const shoulder = kp.find(p => p.name === "left_shoulder" || p.name === "right_shoulder");

        if (hip && knee && shoulder && hip.score > 0.3 && knee.score > 0.3) {
            const torsoLen = Math.abs(hip.y - shoulder.y);
            const legLen = Math.abs(knee.y - hip.y);
            return legLen > torsoLen * 0.6 ? "standing" : "sitting";
        }
    }

    // Fallback - cheap and instant, always available
    return classifyPostureFromBbox(bbox);
}

/* ---------- AI Detection ---------- */
async function startDetection() {
    if (!stream && (!video || !video.src)) {
        await startCamera();
    }

    setBusy(btnStart, true);
    try {
        if (!detectModel) {
            await loadAI();
        }
    } finally {
        setBusy(btnStart, false);
    }

    aiRunning = true;
    if (aiStatusEl) aiStatusEl.textContent = "Running";
    setStatus("AI Running");
    addLog("🟢 AI Detection Started");

    scheduleNextDetection();
}

function stopDetection() {
    aiRunning = false;

    if (detectionTimer) {
        clearTimeout(detectionTimer);
        detectionTimer = null;
    }

    if (ctx && overlay) ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (totalCountEl) totalCountEl.textContent = "0";
    if (standingCountEl) standingCountEl.textContent = "0";
    if (sittingCountEl) sittingCountEl.textContent = "0";
    if (currentPersonsEl) currentPersonsEl.textContent = "0";
    if (alertText) {
        alertText.textContent = "No Person";
        alertText.style.color = "#ffffff";
    }
    if (livePersonsWrap) livePersonsWrap.innerHTML = "";

    tracks = [];
    nextTrackId = 1;

    if (aiStatusEl) aiStatusEl.textContent = "Stopped";
    setStatus("Stopped");
    addLog("⏹ AI Detection Stopped");
}

// Throttled scheduler: keeps AI detection at a sane, battery-friendly
// rate (see DETECTION_INTERVAL_MS) instead of running flat-out every
// animation frame. Also auto-pauses while the tab/app is hidden.
function scheduleNextDetection() {
    if (!aiRunning) return;

    detectionTimer = setTimeout(async () => {
        if (!aiRunning) return;

        if (!tabVisible) {
            // Don't waste battery running AI in the background -
            // just keep checking until the app is visible again.
            scheduleNextDetection();
            return;
        }

        await detectFrame();
        scheduleNextDetection();
    }, DETECTION_INTERVAL_MS);
}

async function detectFrame() {
    if (!detectModel || !video || !ctx || !overlay) return;

    // Skip if the video has no data yet (e.g. paused/buffering)
    if (video.readyState < 2) return;

    const frameStart = performance.now();

    let predictions = [];
    try {
        predictions = await detectModel.detect(video);
    } catch (e) {
        console.error(e);
        return;
    }

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Refresh pose cache at most once per frame (throttled internally)
    await refreshPoseCache();

    let personCount = 0;
    let standingCount = 0;
    let sittingCount = 0;
    let bestScore = 0;
    if (livePersonsWrap) livePersonsWrap.innerHTML = "";

    const persons = predictions.filter(p => p.class === "person");

    // Assign persistent tracking IDs by matching centroids to previous frame
    const trackIds = updateTracks(persons.map(p => p.bbox));

    for (let i = 0; i < persons.length; i++) {
        const item = persons[i];
        personCount++;

        if (item.score > bestScore) bestScore = item.score;

        const [x, y, w, h] = item.bbox;
        const posture = classifyPosture(item.bbox); // synchronous, cheap
        const isStanding = posture === "standing";
        const trackId = trackIds[i];

        if (isStanding) standingCount++;
        else sittingCount++;

        ctx.strokeStyle = isStanding ? "#00ff00" : "#ff33aa";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = isStanding ? "#00ff00" : "#ff33aa";
        ctx.font = "18px Arial";
        ctx.fillText(
            `ID ${trackId} - ${Math.round(item.score * 100)}%`,
            x,
            y > 20 ? y - 8 : y + 18
        );

        if (livePersonsWrap) {
            const card = document.createElement("div");
            card.className = "personCard";
            card.innerHTML = `
                <h3>Person #${trackId}</h3>
                <p>Confidence: ${Math.round(item.score * 100)}%</p>
                <span class="badge ${isStanding ? "standing" : "sitting"}">
                    ${isStanding ? "Standing" : "Sitting"}
                </span>
            `;
            livePersonsWrap.appendChild(card);
        }

        totalDetections++;
    }

    if (personCount > maxPersons) maxPersons = personCount;
    if (bestScore > 0) {
        confidenceSum += bestScore;
        confidenceSamples++;
    }

    // Sound alert only when a NEW person appears (not every single frame)
    if (personCount > previousPersonCount) {
        beep();
    }
    previousPersonCount = personCount;

    // Update stat cards
    if (totalCountEl) totalCountEl.textContent = personCount;
    if (standingCountEl) standingCountEl.textContent = standingCount;
    if (sittingCountEl) sittingCountEl.textContent = sittingCount;
    if (currentPersonsEl) currentPersonsEl.textContent = personCount;
    if (maxPersonsEl) maxPersonsEl.textContent = maxPersons;
    if (totalDetectionsEl) totalDetectionsEl.textContent = totalDetections;

    const avgConfidence = confidenceSamples > 0
        ? Math.round((confidenceSum / confidenceSamples) * 100)
        : 0;
    if (aiAccuracyEl) aiAccuracyEl.textContent = avgConfidence + "%";
    if (confValueEl) confValueEl.textContent = Math.round(bestScore * 100) + "%";

    if (alertText) {
        if (personCount > 0) {
            alertText.textContent = "⚠ PERSON DETECTED";
            alertText.style.color = "#00e676";
        } else {
            alertText.textContent = "No Person";
            alertText.style.color = "#ffffff";
        }
    }

    // Performance metrics (smoothed FPS so the number doesn't jitter)
    const frameEnd = performance.now();
    const frameTime = frameEnd - frameStart;
    const now = performance.now();
    const instantFps = 1000 / (now - lastFrameTime);
    lastFrameTime = now;
    frameCount++;

    fpsHistory.push(instantFps);
    if (fpsHistory.length > FPS_WINDOW) fpsHistory.shift();
    const smoothedFps = Math.round(fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length);

    if (fpsValueEl) fpsValueEl.textContent = smoothedFps;
    if (cameraFPSEl) cameraFPSEl.textContent = smoothedFps;
    if (speedValueEl) speedValueEl.textContent = Math.round(frameTime) + " ms";
    if (frameValueEl) frameValueEl.textContent = frameCount;

    const loadPct = Math.min(100, Math.round(frameTime / 2));
    if (loadValueEl) loadValueEl.textContent = loadPct + "%";
    if (perfFillEl) perfFillEl.style.width = loadPct + "%";
}

/* ---------- Live Clock ---------- */
function updateClock() {
    const now = new Date();
    if (liveTimeEl) liveTimeEl.textContent = now.toLocaleTimeString();
    if (liveDateEl) liveDateEl.textContent = now.toLocaleDateString();
    if (liveDayEl) liveDayEl.textContent = now.toLocaleDateString(undefined, { weekday: "long" });
}

setInterval(updateClock, 1000);
updateClock();

/* ---------- System Info ---------- */
function loadSystemInfo() {
    if (browserNameEl) {
        const ua = navigator.userAgent;
        let name = "Unknown";
        if (ua.includes("Edg/")) name = "Edge";
        else if (ua.includes("Chrome/")) name = "Chrome";
        else if (ua.includes("Firefox/")) name = "Firefox";
        else if (ua.includes("Safari/")) name = "Safari";
        browserNameEl.textContent = name;
    }
    if (platformNameEl) platformNameEl.textContent = navigator.platform || "Unknown";
    if (cpuCoresEl) cpuCoresEl.textContent = navigator.hardwareConcurrency || "Unknown";
    if (deviceRamEl) deviceRamEl.textContent = navigator.deviceMemory ? navigator.deviceMemory + " GB" : "Unknown";
    if (screenSizeEl) screenSizeEl.textContent = `${window.screen.width} x ${window.screen.height}`;
}

loadSystemInfo();

/* ---------- PWA Install Prompt ---------- */
let deferredPrompt;

window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (installBtn) installBtn.hidden = false;
});

if (installBtn) {
    installBtn.addEventListener("click", async () => {
        if (!deferredPrompt) return;
        installBtn.hidden = true;
        deferredPrompt.prompt();
        await deferredPrompt.userChoice;
        deferredPrompt = null;
    });
}

/* ---------- Button Events ---------- */
if (btnUpload && uploadInput) {
    btnUpload.addEventListener("click", () => uploadInput.click());
    uploadInput.addEventListener("change", (e) => handleVideoUpload(e.target.files[0]));
}

if (btnLive) {
    btnLive.addEventListener("click", async () => {
        if (video) {
            video.removeAttribute("src");
            video.srcObject = null;
        }
        await startCamera();
    });
}

if (btnSwitchCam) btnSwitchCam.addEventListener("click", switchCamera);
if (btnNight) btnNight.addEventListener("click", toggleNightVision);
if (btnRecord) btnRecord.addEventListener("click", toggleRecording);
if (btnScreenshot) btnScreenshot.addEventListener("click", takeScreenshot);

if (btnStart) {
    btnStart.addEventListener("click", async () => {
        if (!aiRunning) await startDetection();
    });
}

if (btnStop) {
    btnStop.addEventListener("click", () => {
        if (aiRunning) stopDetection();
    });
}

/* ---------- Cleanup on page unload ---------- */
window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
});

/* ---------- Auto Start ---------- */
window.addEventListener("load", async () => {
    addLog("🚀 Coffee Monitor PRO V43 ULTRA Started");
    try {
        await startCamera();
    } catch (e) {
        console.error(e);
    }
});
