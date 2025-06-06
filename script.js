// script.js  (must be loaded with type="module")

// ——————————————————————————————————————————————
// 1. IMPORT MediaPipe (ES module)
// ——————————————————————————————————————————————
import { FilesetResolver, PoseLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";
// 

// ——————————————————————————————————————————————
// 2. GET REFERENCES to HTML elements
// ——————————————————————————————————————————————
const dancerVideo   = document.getElementById("dancerVideo");
const webcamVideo   = document.getElementById("webcamVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const scoreValue    = document.getElementById("scoreValue");
const startButton   = document.getElementById("startButton");
const canvasCtx     = overlayCanvas.getContext("2d");

// ——————————————————————————————————————————————
// 3. SET UP two PoseLandmarker instances & offscreen canvases
// ——————————————————————————————————————————————
let poseLandmarkerUser   = null; // IMAGE mode for webcam
let poseLandmarkerDancer = null; // IMAGE mode for dancer video

let totalScore   = 0;
let frameCount   = 0;
let dancerResult = null;

// Offscreen canvas for capturing the webcam frame
const webcamCaptureCanvas = document.createElement("canvas");
webcamCaptureCanvas.width  = 480;
webcamCaptureCanvas.height = 360;
const webcamCtx = webcamCaptureCanvas.getContext("2d");

// Offscreen canvas for capturing the dancer frame
const dancerCaptureCanvas = document.createElement("canvas");
dancerCaptureCanvas.width  = 480;
dancerCaptureCanvas.height = 360;
const dancerCtx = dancerCaptureCanvas.getContext("2d");

// ——————————————————————————————————————————————
// 4. INITIALIZE the user’s webcam
// ——————————————————————————————————————————————
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360 },
      audio: false,
    });
    webcamVideo.srcObject = stream;
    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => {
        webcamVideo.play();
        resolve();
      };
    });
    console.log("Webcam initialized.");
  } catch (err) {
    console.error("Error accessing webcam:", err);
    alert("Could not access webcam. Please allow camera permission.");
  }
}

// ——————————————————————————————————————————————
// 5. LOAD both PoseLandmarker models in IMAGE mode
// ——————————————————————————————————————————————
async function loadPoseModel() {
  // 5.1. Point to the WASM binaries (so PoseLandmarker can load)
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  // 5.2. USER detector in IMAGE mode
  poseLandmarkerUser = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "IMAGE",
    numPoses: 1,
  });
  console.log("MediaPipe PoseLandmarker (USER/IMAGE) loaded.");

  // 5.3. DANCER detector in IMAGE mode
  poseLandmarkerDancer = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "IMAGE",
    numPoses: 1,
  });
  console.log("MediaPipe PoseLandmarker (DANCER/IMAGE) loaded.");
}

// ——————————————————————————————————————————————
// 6. COMPUTE per-frame similarity (upper body + hips only)
// ——————————————————————————————————————————————
const COMPARE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];

function computeFrameScore(userLM, dancerLM) {
  let totalDist = 0;
  const n = COMPARE_INDICES.length;
  for (let idx of COMPARE_INDICES) {
    if (userLM[idx] && dancerLM[idx]) {
      const dx = userLM[idx].x - dancerLM[idx].x;
      const dy = userLM[idx].y - dancerLM[idx].y;
      totalDist += Math.hypot(dx, dy);
    } else {
      // If a landmark is missing (off-camera), treat as maximum distance
      totalDist += 1.0;
    }
  }
  const avgDist = totalDist / n;
  const threshold = 0.3;
  let score = 0;
  if (avgDist < threshold) {
    score = (1 - avgDist / threshold) * 100;
  } else {
    score = 0;
  }
  return Math.max(0, Math.min(100, Math.round(score)));
}

// ——————————————————————————————————————————————
// 7. DRAW the user’s pose on the overlay canvas
// ——————————————————————————————————————————————
function drawUserPose(result) {
  canvasCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!result || !result.landmarks || result.landmarks.length < 1) return;
  const landmarks = result.landmarks[0];

  // Draw a simple stick‐figure using upper body + hips connections
  const connections = [
    [0, 1], [0, 2],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
    [11, 12], [23, 24],
    [11, 23], [12, 24],
  ];

  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = "cyan";
  connections.forEach(([i, j]) => {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * overlayCanvas.width, p1.y * overlayCanvas.height);
    canvasCtx.lineTo(p2.x * overlayCanvas.width, p2.y * overlayCanvas.height);
    canvasCtx.stroke();
  });

  // Draw each joint as a red circle
  landmarks.forEach((pt) => {
    canvasCtx.beginPath();
    const x = pt.x * overlayCanvas.width;
    const y = pt.y * overlayCanvas.height;
    canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "red";
    canvasCtx.fill();
  });
}

// ——————————————————————————————————————————————
// 8. MAIN loop: detect both webcam & dancer via IMAGE mode
// ——————————————————————————————————————————————
function startProcessingLoop() {
  async function onFrame() {
    // 8.1. Capture and detect the USER’s pose
    webcamCtx.drawImage(webcamVideo, 0, 0, 480, 360);
    const userResult = poseLandmarkerUser.detect(webcamCaptureCanvas);
    console.log("User landmarks:", userResult.landmarks.length);
    drawUserPose(userResult);

    // 8.2. Only start comparing once the dancer’s intro is done
    const DANCE_START = 10.0;
    if (dancerVideo.currentTime >= DANCE_START) {
      // Capture and detect the DANCER’s pose
      dancerCtx.drawImage(dancerVideo, 0, 0, 480, 360);
      dancerResult = poseLandmarkerDancer.detect(dancerCaptureCanvas);
      console.log("Dancer landmarks:", dancerResult.landmarks.length);

      // If both found at least one pose, compute this frame’s score
      if (
        userResult.landmarks.length > 0 &&
        dancerResult.landmarks.length > 0
      ) {
        const frameScore = computeFrameScore(
          userResult.landmarks[0],
          dancerResult.landmarks[0]
        );
        console.log("Frame score:", frameScore);
        totalScore += frameScore;
        frameCount++;
        scoreValue.textContent = `${totalScore}`;

      }
    }

    // 8.3. Continue looping while dancer video is playing
    if (!dancerVideo.paused && !dancerVideo.ended) {
      requestAnimationFrame(onFrame);
    } else {
      startButton.disabled = false;
      startButton.textContent = "Start Again";
      console.log("Dance finished. Final score:", scoreValue.textContent);
    }
  }

  requestAnimationFrame(onFrame);
}

// ——————————————————————————————————————————————
// 9. START button logic
// ——————————————————————————————————————————————
startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  startButton.textContent = "Loading…";

  // 9.1. Initialize webcam
  await initWebcam();

  // 9.2. Load both PoseLandmarker models in IMAGE mode
  await loadPoseModel();

  // 9.3. Reset state
  totalScore = 0;
  frameCount = 0;
  scoreValue.textContent = "0";
  dancerResult = null;

  // 9.4. Seek dancer video to the very start
  dancerVideo.currentTime = 0;

  // 9.5. Play dancer video (must be muted to allow autoplay)
  try {
    await dancerVideo.play();
    console.log("Dancer video playing:", !dancerVideo.paused);
  } catch (e) {
    console.error("Could not autoplay dancer video:", e);
    alert(
      "Autoplay blocked. Please click ▶ on the dancer video, then press Start Again."
    );
    startButton.disabled = false;
    startButton.textContent = "Start Game";
    return;
  }

  // 9.6. Begin the pose-processing loop
  startProcessingLoop();
  startButton.textContent = "Game Running…";
});
