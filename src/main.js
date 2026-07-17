import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './style.css';
import { evaluateQuality, measurePd } from './measurement.js';

const OFFICIAL_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';

const elements = {
  captureCanvas: document.querySelector('#captureCanvas'),
  resultCanvas: document.querySelector('#resultCanvas'),
  emptyState: document.querySelector('#emptyState'),
  resultPlaceholder: document.querySelector('#resultPlaceholder'),
  modelStatus: document.querySelector('#modelStatus'),
  cameraMeta: document.querySelector('#cameraMeta'),
  qualityBadge: document.querySelector('#qualityBadge'),
  cameraInput: document.querySelector('#cameraInput'),
  galleryInput: document.querySelector('#galleryInput'),
  pdValue: document.querySelector('#pdValue'),
  pdPxValue: document.querySelector('#pdPxValue'),
  irisPxValue: document.querySelector('#irisPxValue'),
  irisDiffValue: document.querySelector('#irisDiffValue'),
  poseValue: document.querySelector('#poseValue'),
  framingValue: document.querySelector('#framingValue'),
  resolutionValue: document.querySelector('#resolutionValue'),
  qualityScoreValue: document.querySelector('#qualityScoreValue'),
  messageBox: document.querySelector('#messageBox'),
  irisReferenceInput: document.querySelector('#irisReferenceInput'),
  maxYawInput: document.querySelector('#maxYawInput'),
  maxPitchInput: document.querySelector('#maxPitchInput'),
  maxRollInput: document.querySelector('#maxRollInput'),
};

let faceLandmarker = null;
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
      baseOptions: { modelAssetPath, delegate },
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
    setMessage('카메라 앱으로 정면 사진을 촬영하세요.', 'info');
  } catch (error) {
    console.error(error);
    setStatus(elements.modelStatus, '모델 로드 실패', 'danger');
    setMessage(`모델을 불러오지 못했습니다: ${error.message}`, 'danger');
  }
}

async function decodeImageFile(file) {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch (error) {
      console.warn('createImageBitmap failed, using Image fallback.', error);
    }
  }

  const image = new Image();
  image.decoding = 'async';
  const objectUrl = URL.createObjectURL(file);
  try {
    image.src = objectUrl;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function loadFile(file, sourceLabel) {
  if (!file) return;

  setMessage('사진을 불러오고 있습니다.', 'info');
  let source = null;

  try {
    source = await decodeImageFile(file);
    const sourceWidth = source.width || source.naturalWidth;
    const sourceHeight = source.height || source.naturalHeight;
    const maxDimension = 4096;
    const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    const canvas = elements.captureCanvas;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height);

    elements.emptyState.hidden = true;
    canvas.hidden = false;
    elements.cameraMeta.textContent = `${sourceLabel} · ${width}×${height}`;
    await analyzeCanvas(canvas);
  } catch (error) {
    console.error(error);
    setMessage(`사진을 열지 못했습니다: ${error.message}`, 'danger');
  } finally {
    if (typeof source?.close === 'function') source.close();
    elements.cameraInput.value = '';
    elements.galleryInput.value = '';
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
    minFaceHeightRatio: 0.42,
    maxFaceHeightRatio: 0.92,
    maxFaceCenterOffsetX: 0.13,
    maxFaceCenterOffsetY: 0.16,
  };
}

async function analyzeCanvas(canvas) {
  if (!faceLandmarker) {
    setMessage('모델이 아직 준비되지 않았습니다.', 'danger');
    return;
  }

  setStatus(elements.qualityBadge, '분석 중', 'pending');
  setMessage('얼굴, 홍채, 촬영 프레이밍을 분석하고 있습니다.', 'info');

  try {
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

  drawTargetFrame(context, canvas.width, canvas.height, scale);

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

function drawTargetFrame(context, width, height, scale) {
  const frameHeight = height * 0.78;
  const frameWidth = Math.min(width * 0.72, frameHeight * 0.78);
  const centerX = width / 2;
  const centerY = height / 2;

  context.save();
  context.strokeStyle = 'rgba(132, 202, 255, 0.78)';
  context.lineWidth = 2.2 * scale;
  context.setLineDash([10 * scale, 8 * scale]);
  context.beginPath();
  context.ellipse(centerX, centerY, frameWidth / 2, frameHeight / 2, 0, 0, Math.PI * 2);
  context.stroke();
  context.restore();
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

  const framing = measurement.framing;
  elements.framingValue.textContent = Number.isFinite(framing.faceHeightRatio)
    ? `얼굴 ${(framing.faceHeightRatio * 100).toFixed(0)}% · 중심 오차 X ${(framing.centerOffsetX * 100).toFixed(0)}% / Y ${(framing.centerOffsetY * 100).toFixed(0)}%`
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
  elements.framingValue.textContent = '--';
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

elements.cameraInput.addEventListener('change', (event) => loadFile(event.target.files?.[0], '카메라 촬영'));
elements.galleryInput.addEventListener('change', (event) => loadFile(event.target.files?.[0], '앨범 사진'));

initializeModel();
