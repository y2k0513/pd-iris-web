import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './style.css';
import { evaluateQuality, measurePd } from './measurement.js';

const OFFICIAL_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';

const elements = {
  video: document.querySelector('#video'),
  captureCanvas: document.querySelector('#captureCanvas'),
  resultCanvas: document.querySelector('#resultCanvas'),
  mediaStage: document.querySelector('#mediaStage'),
  emptyState: document.querySelector('#emptyState'),
  resultPlaceholder: document.querySelector('#resultPlaceholder'),
  modelStatus: document.querySelector('#modelStatus'),
  cameraMeta: document.querySelector('#cameraMeta'),
  qualityBadge: document.querySelector('#qualityBadge'),
  startCameraButton: document.querySelector('#startCameraButton'),
  captureButton: document.querySelector('#captureButton'),
  fileInput: document.querySelector('#fileInput'),
  pdValue: document.querySelector('#pdValue'),
  pdPxValue: document.querySelector('#pdPxValue'),
  irisPxValue: document.querySelector('#irisPxValue'),
  irisDiffValue: document.querySelector('#irisDiffValue'),
  poseValue: document.querySelector('#poseValue'),
  resolutionValue: document.querySelector('#resolutionValue'),
  qualityScoreValue: document.querySelector('#qualityScoreValue'),
  messageBox: document.querySelector('#messageBox'),
  irisReferenceInput: document.querySelector('#irisReferenceInput'),
  maxYawInput: document.querySelector('#maxYawInput'),
  maxPitchInput: document.querySelector('#maxPitchInput'),
  maxRollInput: document.querySelector('#maxRollInput'),
};

let faceLandmarker = null;
let cameraStream = null;
let lastAnalysisSource = null;

function setStatus(element, text, kind) {
  element.textContent = text;
  element.className = `status-pill ${kind}`;
}

function setMessage(text, kind = 'info') {
  elements.messageBox.textContent = text;
  elements.messageBox.className = `message-box ${kind}`;
}

function assetUrl(relativePath) {
  return new URL(relativePath, document.baseURI).href;
}

async function localModelExists() {
  const localModelUrl = assetUrl('models/face_landmarker.task');
  try {
    const response = await fetch(localModelUrl, { method: 'HEAD', cache: 'no-store' });
    const length = Number(response.headers.get('content-length') || 0);
    const type = response.headers.get('content-type') || '';
    return response.ok && length > 1_000_000 && !type.includes('text/html');
  } catch {
    return false;
  }
}

async function initializeModel() {
  setStatus(elements.modelStatus, '모델 불러오는 중', 'pending');
  try {
    const wasmRoot = assetUrl('wasm');
    const vision = await FilesetResolver.forVisionTasks(wasmRoot);
    const useLocalModel = await localModelExists();
    const modelAssetPath = useLocalModel ? assetUrl('models/face_landmarker.task') : OFFICIAL_MODEL_URL;

    const options = (delegate) => ({
      baseOptions: {
        modelAssetPath,
        delegate,
      },
      runningMode: 'IMAGE',
      numFaces: 1,
      minFaceDetectionConfidence: 0.6,
      minFacePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    });

    let activeDelegate = 'GPU';
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options('GPU'));
    } catch (gpuError) {
      console.warn('GPU delegate initialization failed. Falling back to CPU.', gpuError);
      activeDelegate = 'CPU';
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options('CPU'));
    }

    const sourceText = useLocalModel ? '로컬 모델' : '모델';
    setStatus(elements.modelStatus, `${sourceText} 준비됨 · ${activeDelegate}`, 'success');
    setMessage('정면 사진을 촬영하거나 선택하세요.', 'info');
  } catch (error) {
    console.error(error);
    setStatus(elements.modelStatus, '모델 로드 실패', 'danger');
    setMessage(`모델을 불러오지 못했습니다: ${error.message}`, 'danger');
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage('이 브라우저는 카메라 API를 지원하지 않습니다.', 'danger');
    return;
  }

  stopCamera();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'user' },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
    elements.video.srcObject = cameraStream;
    await elements.video.play();
    elements.video.hidden = false;
    elements.emptyState.hidden = true;
    elements.captureButton.disabled = false;
    elements.startCameraButton.textContent = '카메라 다시 시작';

    const settings = cameraStream.getVideoTracks()[0]?.getSettings() || {};
    elements.cameraMeta.textContent = `${settings.width || elements.video.videoWidth}×${settings.height || elements.video.videoHeight}`;
    setMessage('얼굴을 가이드 중앙에 두고 렌즈를 바라본 뒤 촬영하세요.', 'info');
  } catch (error) {
    console.error(error);
    setMessage(`카메라를 시작하지 못했습니다: ${error.message}`, 'danger');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
}

function captureFrame() {
  const width = elements.video.videoWidth;
  const height = elements.video.videoHeight;
  if (!(width > 0 && height > 0)) {
    setMessage('카메라 프레임이 아직 준비되지 않았습니다.', 'danger');
    return;
  }

  const canvas = elements.captureCanvas;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });

  // 미리보기와 같은 방향으로 저장한다. 비율 측정에는 미러링이 영향을 주지 않는다.
  context.save();
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(elements.video, 0, 0, width, height);
  context.restore();

  showCapturedCanvas();
  analyzeCanvas(canvas);
}

function showCapturedCanvas() {
  elements.video.hidden = true;
  elements.captureCanvas.hidden = false;
  elements.emptyState.hidden = true;
}

async function loadFile(file) {
  if (!file) return;
  const image = new Image();
  image.decoding = 'async';
  const objectUrl = URL.createObjectURL(file);
  image.src = objectUrl;

  try {
    await image.decode();
    const maxDimension = 4096;
    const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.round(image.naturalWidth * scale);
    const height = Math.round(image.naturalHeight * scale);
    const canvas = elements.captureCanvas;
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d', { willReadFrequently: true }).drawImage(image, 0, 0, width, height);
    showCapturedCanvas();
    elements.cameraMeta.textContent = `${width}×${height} 업로드`;
    await analyzeCanvas(canvas);
  } catch (error) {
    console.error(error);
    setMessage(`사진을 열지 못했습니다: ${error.message}`, 'danger');
  } finally {
    URL.revokeObjectURL(objectUrl);
    elements.fileInput.value = '';
  }
}

function getConfig() {
  return {
    irisReferenceMm: Number(elements.irisReferenceInput.value) || 11.7,
    maxYaw: Number(elements.maxYawInput.value) || 8,
    maxPitch: Number(elements.maxPitchInput.value) || 8,
    maxRoll: Number(elements.maxRollInput.value) || 5,
    maxIrisDifferenceRatio: 0.12,
    minIrisPixels: 14,
  };
}

async function analyzeCanvas(canvas) {
  if (!faceLandmarker) {
    setMessage('모델이 아직 준비되지 않았습니다.', 'danger');
    return;
  }

  elements.captureButton.disabled = true;
  setStatus(elements.qualityBadge, '분석 중', 'pending');
  setMessage('얼굴과 홍채 랜드마크를 분석하고 있습니다.', 'info');

  try {
    // detect()는 동기식이지만 UI 상태가 먼저 그려지도록 한 프레임 양보한다.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const result = faceLandmarker.detect(canvas);

    if (result.faceLandmarks.length !== 1) {
      throw new Error(result.faceLandmarks.length === 0 ? '얼굴을 찾지 못했습니다.' : '한 명의 얼굴만 촬영해야 합니다.');
    }

    const landmarks = result.faceLandmarks[0];
    const matrix = result.facialTransformationMatrixes?.[0];
    const config = getConfig();
    const measurement = measurePd({
      landmarks,
      matrix,
      width: canvas.width,
      height: canvas.height,
      irisReferenceMm: config.irisReferenceMm,
    });
    const quality = evaluateQuality(measurement, config);

    lastAnalysisSource = canvas;
    renderResult(canvas, landmarks, measurement, quality);
    updateMetrics(canvas, measurement, quality);

    if (quality.accepted) {
      setStatus(elements.qualityBadge, '측정 가능', 'success');
      const warningText = quality.warnings.length ? ` ${quality.warnings.join(' ')}` : '';
      setMessage(`품질 조건을 통과했습니다.${warningText}`, quality.warnings.length ? 'warning' : 'success');
    } else {
      setStatus(elements.qualityBadge, '재촬영 권장', 'danger');
      setMessage(quality.reasons.join(' · '), 'danger');
    }
  } catch (error) {
    console.error(error);
    resetMetrics();
    setStatus(elements.qualityBadge, '분석 실패', 'danger');
    setMessage(error.message, 'danger');
  } finally {
    elements.captureButton.disabled = !cameraStream;
  }
}

function renderResult(sourceCanvas, landmarks, measurement, quality) {
  const canvas = elements.resultCanvas;
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  const context = canvas.getContext('2d');
  context.drawImage(sourceCanvas, 0, 0);

  const scale = Math.max(1, Math.min(canvas.width, canvas.height) / 700);
  const { leftCenter, rightCenter, leftBoundary, rightBoundary } = measurement.points;

  context.lineCap = 'round';
  context.lineWidth = 3 * scale;
  context.strokeStyle = quality.accepted ? '#12b76a' : '#f04438';
  context.beginPath();
  context.moveTo(leftCenter.x, leftCenter.y);
  context.lineTo(rightCenter.x, rightCenter.y);
  context.stroke();

  drawPoint(context, leftCenter, '#2e90fa', 6 * scale);
  drawPoint(context, rightCenter, '#2e90fa', 6 * scale);
  [...leftBoundary, ...rightBoundary].forEach((point) => drawPoint(context, point, '#f79009', 4 * scale));

  // 눈 주변 랜드마크 일부를 얇게 표시한다.
  context.fillStyle = 'rgba(255,255,255,0.55)';
  for (const index of [33, 133, 159, 145, 362, 263, 386, 374]) {
    const point = landmarks[index];
    if (!point) continue;
    context.beginPath();
    context.arc(point.x * canvas.width, point.y * canvas.height, 2.2 * scale, 0, Math.PI * 2);
    context.fill();
  }

  elements.resultPlaceholder.hidden = true;
  canvas.hidden = false;
}

function drawPoint(context, point, color, radius) {
  context.fillStyle = color;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
}

function updateMetrics(canvas, measurement, quality) {
  elements.pdValue.textContent = measurement.pdMm.toFixed(1);
  elements.pdPxValue.textContent = `${measurement.pdPx.toFixed(1)} px`;
  elements.irisPxValue.textContent = `${measurement.meanIrisPx.toFixed(1)} px`;
  elements.irisDiffValue.textContent = `${(measurement.irisDifferenceRatio * 100).toFixed(1)}%`;

  const { yaw, pitch, roll } = measurement.pose;
  elements.poseValue.textContent = [yaw, pitch, roll].every(Number.isFinite)
    ? `${yaw.toFixed(1)}° / ${pitch.toFixed(1)}° / ${roll.toFixed(1)}°`
    : '계산 불가';
  elements.resolutionValue.textContent = `${canvas.width}×${canvas.height}`;
  elements.qualityScoreValue.textContent = `${quality.score}/100`;
}

function resetMetrics() {
  elements.pdValue.textContent = '--';
  elements.pdPxValue.textContent = '-- px';
  elements.irisPxValue.textContent = '-- px';
  elements.irisDiffValue.textContent = '--';
  elements.poseValue.textContent = '--';
  elements.resolutionValue.textContent = '--';
  elements.qualityScoreValue.textContent = '--';
}

async function reanalyzeLastImage() {
  if (lastAnalysisSource) await analyzeCanvas(lastAnalysisSource);
}

for (const input of [
  elements.irisReferenceInput,
  elements.maxYawInput,
  elements.maxPitchInput,
  elements.maxRollInput,
]) {
  input.addEventListener('change', reanalyzeLastImage);
}

elements.startCameraButton.addEventListener('click', startCamera);
elements.captureButton.addEventListener('click', captureFrame);
elements.fileInput.addEventListener('change', (event) => loadFile(event.target.files?.[0]));
window.addEventListener('beforeunload', stopCamera);

initializeModel();
