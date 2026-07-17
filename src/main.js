import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './style.css';
import { applySexPdPrior, evaluateQuality, measurePd, SEX_PD_PRIORS } from './measurement.js';

const OFFICIAL_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const LIVE_INTERVAL_MS = 170;
const STABLE_FRAME_TARGET = 4;
const MAX_PD_VARIATION_RATIO = 0.025;
const AUTO_CAPTURE_DELAY_MS = 1000;

const elements = {
  mediaStage: document.querySelector('#mediaStage'),
  cameraVideo: document.querySelector('#cameraVideo'),
  captureCanvas: document.querySelector('#captureCanvas'),
  resultCanvas: document.querySelector('#resultCanvas'),
  emptyState: document.querySelector('#emptyState'),
  liveOverlay: document.querySelector('#liveOverlay'),
  liveGuide: document.querySelector('#liveGuide'),
  liveStatus: document.querySelector('#liveStatus'),
  resultPlaceholder: document.querySelector('#resultPlaceholder'),
  modelStatus: document.querySelector('#modelStatus'),
  cameraMeta: document.querySelector('#cameraMeta'),
  qualityBadge: document.querySelector('#qualityBadge'),
  startCameraButton: document.querySelector('#startCameraButton'),
  captureButton: document.querySelector('#captureButton'),
  stopCameraButton: document.querySelector('#stopCameraButton'),
  galleryInput: document.querySelector('#galleryInput'),
  sexInput: document.querySelector('#sexInput'),
  sexPriorHint: document.querySelector('#sexPriorHint'),
  poseCheck: document.querySelector('#poseCheck'),
  perspectiveCheck: document.querySelector('#perspectiveCheck'),
  gazeCheck: document.querySelector('#gazeCheck'),
  stabilityCheck: document.querySelector('#stabilityCheck'),
  pdValue: document.querySelector('#pdValue'),
  rawPdValue: document.querySelector('#rawPdValue'),
  priorCenterValue: document.querySelector('#priorCenterValue'),
  priorLossValue: document.querySelector('#priorLossValue'),
  priorWeightValue: document.querySelector('#priorWeightValue'),
  pdPxValue: document.querySelector('#pdPxValue'),
  irisPxValue: document.querySelector('#irisPxValue'),
  irisDiffValue: document.querySelector('#irisDiffValue'),
  poseValue: document.querySelector('#poseValue'),
  perspectiveValue: document.querySelector('#perspectiveValue'),
  depthEstimateValue: document.querySelector('#depthEstimateValue'),
  depthAgreementValue: document.querySelector('#depthAgreementValue'),
  framingValue: document.querySelector('#framingValue'),
  resolutionValue: document.querySelector('#resolutionValue'),
  qualityScoreValue: document.querySelector('#qualityScoreValue'),
  messageBox: document.querySelector('#messageBox'),
  irisReferenceInput: document.querySelector('#irisReferenceInput'),
  priorStrengthInput: document.querySelector('#priorStrengthInput'),
  maxYawInput: document.querySelector('#maxYawInput'),
  maxPitchInput: document.querySelector('#maxPitchInput'),
  maxRollInput: document.querySelector('#maxRollInput'),
};

let faceLandmarker = null;
let activeMode = 'VIDEO';
let cameraStream = null;
let animationFrameId = null;
let liveBusy = false;
let livePaused = false;
let lastLiveAnalysisAt = 0;
let lastVideoTime = -1;
let stableFrames = 0;
let pdHistory = [];
let latestLiveState = null;
let lastAnalysisSource = null;
let captureArmed = false;
let validHoldStartedAt = null;
let autoCaptureInProgress = false;

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

async function setModelMode(mode) {
  if (!faceLandmarker || activeMode === mode) return;
  await faceLandmarker.setOptions({ runningMode: mode });
  activeMode = mode;
}

async function initializeModel() {
  setStatus(elements.modelStatus, '모델 불러오는 중', 'pending');
  try {
    const vision = await FilesetResolver.forVisionTasks(assetUrl('wasm'));
    const useLocalModel = await localModelExists();
    const modelAssetPath = useLocalModel ? assetUrl('models/face_landmarker.task') : OFFICIAL_MODEL_URL;
    const options = (delegate) => ({
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.65,
      minFacePresenceConfidence: 0.65,
      minTrackingConfidence: 0.65,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    });

    let activeDelegate = 'GPU';
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options('GPU'));
    } catch (gpuError) {
      console.warn('GPU delegate failed; using CPU.', gpuError);
      activeDelegate = 'CPU';
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options('CPU'));
    }

    activeMode = 'VIDEO';
    setStatus(elements.modelStatus, `${useLocalModel ? '로컬 모델' : '모델'} 준비됨 · ${activeDelegate}`, 'success');
    setMessage('카메라를 시작한 뒤 얼굴을 프레임 중앙에 맞추세요.', 'info');
  } catch (error) {
    console.error(error);
    setStatus(elements.modelStatus, '모델 로드 실패', 'danger');
    setMessage(`모델을 불러오지 못했습니다: ${error.message}`, 'danger');
  }
}

function getConfig() {
  return {
    irisReferenceMm: Number(elements.irisReferenceInput.value) || 11.7,
    priorStrength: Math.min(1, Math.max(0, Number(elements.priorStrengthInput.value) || 0)),
    maxYaw: Number(elements.maxYawInput.value) || 6,
    maxPitch: Number(elements.maxPitchInput.value) || 7,
    maxRoll: Number(elements.maxRollInput.value) || 5,
    requirePose: true,
    maxIrisDifferenceRatio: 0.10,
    minIrisPixels: 14,
    minFaceHeightRatio: 0.40,
    maxFaceHeightRatio: 0.72,
    maxFaceCenterOffsetX: 0.10,
    maxFaceCenterOffsetY: 0.12,
    maxGazeOffset: 0.28,
    minEyeOpeningRatio: 0.06,
    maxPerspectiveAsymmetryRatio: 0.16,
    maxEyeWidthDifferenceRatio: 0.20,
    maxIrisDepthDifference: 0.025,
    max2D3DDisagreementRatio: 0.10,
    warn2D3DDisagreementRatio: 0.06,
  };
}

function getSelectedSex() {
  return elements.sexInput.value;
}

function updateSexPriorHint() {
  const prior = SEX_PD_PRIORS[getSelectedSex()];
  if (!prior) {
    elements.sexPriorHint.textContent = '성별 분포를 선택하면 중심값 기반 soft prior를 적용합니다.';
    return;
  }
  elements.sexPriorHint.textContent = `${prior.label} 기준 ${prior.minMm}–${prior.maxMm}mm · 중심 ${prior.centerMm}mm · 범위 끝에서 loss 1.0`;
}

async function requestHighResolutionCamera() {
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: 'user' },
      width: { ideal: 3840 },
      height: { ideal: 2160 },
      frameRate: { ideal: 30, max: 30 },
    },
  };

  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    console.warn('High-resolution constraints failed; falling back.', error);
    return navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'user' },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 24, max: 30 },
      },
    });
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage('이 브라우저는 웹 카메라 촬영을 지원하지 않습니다.', 'danger');
    return;
  }
  if (!window.isSecureContext && location.hostname !== 'localhost') {
    setMessage('모바일 카메라는 HTTPS 주소에서만 사용할 수 있습니다.', 'danger');
    return;
  }
  if (!faceLandmarker) {
    setMessage('모델이 아직 준비되지 않았습니다.', 'danger');
    return;
  }

  stopCamera();
  elements.startCameraButton.disabled = true;
  setMessage('고해상도 전면 카메라를 요청하고 있습니다.', 'info');

  try {
    cameraStream = await requestHighResolutionCamera();
    elements.cameraVideo.srcObject = cameraStream;
    await elements.cameraVideo.play();
    await new Promise((resolve) => {
      if (elements.cameraVideo.videoWidth > 0) resolve();
      else elements.cameraVideo.addEventListener('loadedmetadata', resolve, { once: true });
    });

    const track = cameraStream.getVideoTracks()[0];
    const settings = track.getSettings();
    const width = settings.width || elements.cameraVideo.videoWidth;
    const height = settings.height || elements.cameraVideo.videoHeight;
    const fps = settings.frameRate ? ` · ${Math.round(settings.frameRate)}fps` : '';

    elements.mediaStage.style.aspectRatio = `${elements.cameraVideo.videoWidth} / ${elements.cameraVideo.videoHeight}`;
    elements.cameraMeta.textContent = `${width}×${height}${fps}`;
    elements.emptyState.hidden = true;
    elements.cameraVideo.hidden = false;
    elements.liveOverlay.hidden = false;
    elements.stopCameraButton.disabled = false;
    elements.startCameraButton.textContent = '카메라 재시작';
    elements.startCameraButton.disabled = false;
    resetLiveGate('얼굴을 프레임에 맞추세요');
    await setModelMode('VIDEO');
    runLiveLoop();
  } catch (error) {
    console.error(error);
    elements.startCameraButton.disabled = false;
    const message = error.name === 'NotAllowedError'
      ? '카메라 권한이 거부되었습니다. 브라우저 사이트 권한에서 카메라를 허용하세요.'
      : `카메라를 시작하지 못했습니다: ${error.message}`;
    setMessage(message, 'danger');
  }
}

function stopCamera() {
  autoCaptureInProgress = false;
  livePaused = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  animationFrameId = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  elements.cameraVideo.srcObject = null;
  elements.cameraVideo.hidden = true;
  elements.liveOverlay.hidden = true;
  elements.emptyState.hidden = false;
  elements.stopCameraButton.disabled = true;
  elements.captureButton.disabled = true;
  elements.cameraMeta.textContent = '카메라 대기';
  resetLiveGate('카메라를 시작하세요');
}

function runLiveLoop() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  const step = async (timestamp) => {
    animationFrameId = requestAnimationFrame(step);
    if (!cameraStream || livePaused || liveBusy || !faceLandmarker) return;
    if (elements.cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (timestamp - lastLiveAnalysisAt < LIVE_INTERVAL_MS) return;
    if (elements.cameraVideo.currentTime === lastVideoTime) return;

    liveBusy = true;
    lastLiveAnalysisAt = timestamp;
    lastVideoTime = elements.cameraVideo.currentTime;
    try {
      await setModelMode('VIDEO');
      const result = faceLandmarker.detectForVideo(elements.cameraVideo, timestamp);
      processLiveResult(result, timestamp);
    } catch (error) {
      console.error('Live analysis failed', error);
      resetLiveGate('실시간 분석 오류', { preserveArm: true });
    } finally {
      liveBusy = false;
    }
  };
  animationFrameId = requestAnimationFrame(step);
}

function variationRatio(values) {
  if (values.length < 2) return Number.POSITIVE_INFINITY;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return (Math.max(...values) - Math.min(...values)) / mean;
}

function processLiveResult(result, timestamp = performance.now()) {
  if (result.faceLandmarks.length !== 1) {
    resetLiveGate(
      result.faceLandmarks.length === 0 ? '얼굴을 찾는 중' : '한 명만 화면에 나오게 하세요',
      { preserveArm: true },
    );
    return;
  }

  const landmarks = result.faceLandmarks[0];
  const matrix = result.facialTransformationMatrixes?.[0];
  const config = getConfig();
  const measurement = measurePd({
    landmarks,
    matrix,
    width: elements.cameraVideo.videoWidth,
    height: elements.cameraVideo.videoHeight,
    irisReferenceMm: config.irisReferenceMm,
  });
  const quality = evaluateQuality(measurement, config);

  if (quality.accepted) {
    pdHistory.push(measurement.pdMm);
    if (pdHistory.length > STABLE_FRAME_TARGET + 2) pdHistory.shift();

    let variation = variationRatio(pdHistory);
    const variationAccepted = pdHistory.length < 3 || variation <= MAX_PD_VARIATION_RATIO;

    if (!variationAccepted) {
      pdHistory = [measurement.pdMm];
      stableFrames = 1;
      validHoldStartedAt = null;
      variation = Number.POSITIVE_INFINITY;
    } else {
      stableFrames = Math.min(STABLE_FRAME_TARGET, pdHistory.length);
    }

    const ready = pdHistory.length >= STABLE_FRAME_TARGET
      && variationRatio(pdHistory) <= MAX_PD_VARIATION_RATIO;

    latestLiveState = {
      measurement,
      quality,
      landmarks,
      ready,
      variation: variationRatio(pdHistory),
    };

    let remainingMs = null;
    if (captureArmed) {
      if (validHoldStartedAt === null) validHoldStartedAt = timestamp;
      remainingMs = Math.max(0, AUTO_CAPTURE_DELAY_MS - (timestamp - validHoldStartedAt));

      if (remainingMs <= 0 && ready && !autoCaptureInProgress) {
        autoCaptureInProgress = true;
        captureArmed = false;
        syncCaptureButton();
        updateLiveIndicators(measurement, quality, ready, variationRatio(pdHistory), 0);
        void captureCurrentFrame();
        return;
      }
    } else {
      validHoldStartedAt = null;
    }

    updateLiveIndicators(measurement, quality, ready, variationRatio(pdHistory), remainingMs);
  } else {
    latestLiveState = { measurement, quality, landmarks, ready: false, variation: Number.NaN };
    stableFrames = 0;
    pdHistory = [];
    validHoldStartedAt = null;
    updateLiveIndicators(measurement, quality, false, Number.NaN, null);
  }
}

function updateLiveIndicators(measurement, quality, ready, variation, remainingMs) {
  const config = getConfig();
  const { pose, eyeAndPerspective } = measurement;
  const poseOk = [pose.yaw, pose.pitch, pose.roll].every(Number.isFinite)
    && Math.abs(pose.yaw) <= config.maxYaw
    && Math.abs(pose.pitch) <= config.maxPitch
    && Math.abs(pose.roll) <= config.maxRoll;
  const perspectiveOk = eyeAndPerspective.perspectiveAsymmetryRatio <= config.maxPerspectiveAsymmetryRatio
    && measurement.irisDifferenceRatio <= config.maxIrisDifferenceRatio
    && measurement.depthAware.disagreementRatio <= config.max2D3DDisagreementRatio;
  const gazeOk = eyeAndPerspective.gazeOffset <= config.maxGazeOffset;

  setCheck(elements.poseCheck, poseOk, `자세 ${pose.yaw.toFixed(1)}/${pose.pitch.toFixed(1)}/${pose.roll.toFixed(1)}°`);
  setCheck(
    elements.perspectiveCheck,
    perspectiveOk,
    `원근 ${(eyeAndPerspective.perspectiveAsymmetryRatio * 100).toFixed(1)}% · 3D검증 ${(measurement.depthAware.disagreementRatio * 100).toFixed(1)}%`,
  );
  setCheck(elements.gazeCheck, gazeOk, `시선 ${(eyeAndPerspective.gazeOffset * 100).toFixed(0)}%`);
  setCheck(
    elements.stabilityCheck,
    ready,
    Number.isFinite(variation) ? `안정성 ${(variation * 100).toFixed(1)}%` : `안정성 ${stableFrames}/${STABLE_FRAME_TARGET}`,
  );

  syncCaptureButton();

  if (!quality.accepted) {
    elements.liveGuide.className = 'face-guide live-guide invalid';
    const reason = quality.reasons[0] || '자세를 다시 맞추세요';
    elements.liveStatus.textContent = captureArmed
      ? `${reason} · 맞추면 1초 카운트 재시작`
      : reason;
    return;
  }

  if (captureArmed) {
    if (remainingMs !== null) {
      elements.liveGuide.className = ready
        ? 'face-guide live-guide ready'
        : 'face-guide live-guide checking';
      elements.liveStatus.textContent = `자세 유지 — ${(remainingMs / 1000).toFixed(1)}초 후 자동촬영`;
    } else {
      elements.liveGuide.className = 'face-guide live-guide checking';
      elements.liveStatus.textContent = '움직이지 말고 자세를 유지하세요';
    }
    return;
  }

  if (ready) {
    elements.liveGuide.className = 'face-guide live-guide ready';
    elements.liveStatus.textContent = '자세 양호 — 촬영 준비 버튼을 누르세요';
  } else {
    elements.liveGuide.className = 'face-guide live-guide checking';
    elements.liveStatus.textContent = `좋습니다. 잠시 자세를 유지하세요 (${stableFrames}/${STABLE_FRAME_TARGET})`;
  }
}

function setCheck(element, ok, text) {
  element.textContent = text;
  element.className = ok ? 'ok' : 'bad';
}

function syncCaptureButton() {
  elements.captureButton.disabled = !cameraStream || !getSelectedSex() || autoCaptureInProgress;
  elements.captureButton.classList.toggle('armed', captureArmed);

  if (autoCaptureInProgress) {
    elements.captureButton.textContent = '촬영 및 분석 중';
  } else if (captureArmed) {
    elements.captureButton.textContent = '자동촬영 취소';
  } else {
    elements.captureButton.textContent = '촬영 준비';
  }
}

function resetLiveGate(message, { preserveArm = false } = {}) {
  stableFrames = 0;
  pdHistory = [];
  latestLiveState = null;
  validHoldStartedAt = null;
  if (!preserveArm) captureArmed = false;

  elements.liveGuide.className = 'face-guide live-guide waiting';
  elements.liveStatus.textContent = captureArmed
    ? `${message} · 조건 충족 후 1초 자동촬영`
    : message;
  for (const [element, label] of [
    [elements.poseCheck, '자세 --'],
    [elements.perspectiveCheck, '원근 --'],
    [elements.gazeCheck, '시선 --'],
    [elements.stabilityCheck, '안정성 --'],
  ]) {
    element.textContent = label;
    element.className = '';
  }
  syncCaptureButton();
}

function toggleAutoCapture() {
  if (!cameraStream || autoCaptureInProgress) {
    setMessage('먼저 카메라를 시작하세요.', 'danger');
    return;
  }
  if (!getSelectedSex()) {
    setMessage('촬영 전에 성별을 선택하세요.', 'danger');
    return;
  }

  captureArmed = !captureArmed;
  validHoldStartedAt = null;

  if (captureArmed) {
    setMessage('자세 조건을 맞춘 채 유지하면 1초 뒤 자동으로 촬영합니다.', 'info');
    elements.liveStatus.textContent = '얼굴을 맞추세요 — 조건 충족 후 1초 자동촬영';
  } else {
    setMessage('자동촬영을 취소했습니다.', 'info');
    elements.liveStatus.textContent = latestLiveState?.ready
      ? '자세 양호 — 촬영 준비 버튼을 누르세요'
      : '촬영 준비 버튼을 누른 뒤 자세를 맞추세요';
  }

  syncCaptureButton();
}

async function captureCurrentFrame() {
  if (!cameraStream || !latestLiveState?.ready) {
    autoCaptureInProgress = false;
    captureArmed = true;
    validHoldStartedAt = null;
    syncCaptureButton();
    setMessage('촬영 직전 자세가 흔들렸습니다. 자세를 다시 유지하세요.', 'warning');
    return;
  }

  livePaused = true;
  autoCaptureInProgress = true;
  captureArmed = false;
  validHoldStartedAt = null;
  syncCaptureButton();

  try {
    const width = elements.cameraVideo.videoWidth;
    const height = elements.cameraVideo.videoHeight;
    const canvas = elements.captureCanvas;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(elements.cameraVideo, 0, 0, width, height);
    context.restore();

    elements.cameraMeta.textContent = `촬영 · ${width}×${height}`;
    await analyzeCanvas(canvas, { strictCapture: true });
  } finally {
    autoCaptureInProgress = false;
    resetLiveGate('다음 측정은 촬영 준비 버튼을 누르세요');
    livePaused = false;
  }
}

async function decodeImageFile(file) {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch (error) {
      console.warn('createImageBitmap failed; using Image fallback.', error);
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

async function loadFile(file) {
  if (!file) return;
  if (!getSelectedSex()) {
    setMessage('사진 분석 전에 성별을 선택하세요.', 'danger');
    elements.galleryInput.value = '';
    return;
  }
  let source = null;
  try {
    source = await decodeImageFile(file);
    const sourceWidth = source.width || source.naturalWidth;
    const sourceHeight = source.height || source.naturalHeight;
    const maxDimension = 4096;
    const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);
    const canvas = elements.captureCanvas;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(source, 0, 0, width, height);
    elements.cameraMeta.textContent = `사진 테스트 · ${width}×${height}`;
    await analyzeCanvas(canvas, { strictCapture: false });
  } catch (error) {
    console.error(error);
    setMessage(`사진을 열지 못했습니다: ${error.message}`, 'danger');
  } finally {
    if (typeof source?.close === 'function') source.close();
    elements.galleryInput.value = '';
  }
}

async function analyzeCanvas(canvas, { strictCapture }) {
  if (!faceLandmarker) return;
  const selectedSex = getSelectedSex();
  if (!selectedSex) {
    setMessage('분석 전에 성별을 선택하세요.', 'danger');
    return;
  }
  const wasLivePaused = livePaused;
  livePaused = true;
  setStatus(elements.qualityBadge, '분석 중', 'pending');
  setMessage('고해상도 프레임을 다시 분석하고 있습니다.', 'info');

  try {
    await setModelMode('IMAGE');
    const result = faceLandmarker.detect(canvas);
    if (result.faceLandmarks.length !== 1) {
      throw new Error(result.faceLandmarks.length === 0 ? '얼굴을 찾지 못했습니다.' : '한 명의 얼굴만 촬영해야 합니다.');
    }

    const landmarks = result.faceLandmarks[0];
    const measurement = measurePd({
      landmarks,
      matrix: result.facialTransformationMatrixes?.[0],
      width: canvas.width,
      height: canvas.height,
      irisReferenceMm: getConfig().irisReferenceMm,
    });
    const config = getConfig();
    const quality = evaluateQuality(measurement, config);
    const sexPrior = applySexPdPrior({
      rawPdMm: measurement.pdMm,
      sex: selectedSex,
      qualityScore: quality.score,
      strength: config.priorStrength,
    });

    if (strictCapture && !quality.accepted) {
      resetMetrics();
      setStatus(elements.qualityBadge, '촬영 무효', 'danger');
      setMessage(`촬영 순간 자세가 바뀌었습니다: ${quality.reasons.join(' · ')}`, 'danger');
      return;
    }

    lastAnalysisSource = canvas;
    renderResult(canvas, landmarks, measurement, quality);
    updateMetrics(canvas, measurement, quality, sexPrior);
    setStatus(elements.qualityBadge, quality.accepted ? '측정 가능' : '재촬영 권장', quality.accepted ? 'success' : 'danger');
    const priorSummary = sexPrior.withinTypicalRange
      ? `${sexPrior.label} 분포 안 · raw ${sexPrior.rawPdMm.toFixed(1)} → 보정 ${sexPrior.adjustedPdMm.toFixed(1)}mm`
      : `${sexPrior.label} 기준 ${sexPrior.minMm}–${sexPrior.maxMm}mm 밖 · raw ${sexPrior.rawPdMm.toFixed(1)} → 보정 ${sexPrior.adjustedPdMm.toFixed(1)}mm`;
    setMessage(
      quality.accepted ? `품질 조건을 통과했습니다. ${priorSummary}` : `${quality.reasons.join(' · ')} · ${priorSummary}`,
      quality.accepted ? (sexPrior.withinTypicalRange ? 'success' : 'warning') : 'danger',
    );
  } catch (error) {
    console.error(error);
    resetMetrics();
    setStatus(elements.qualityBadge, '분석 실패', 'danger');
    setMessage(error.message, 'danger');
  } finally {
    await setModelMode('VIDEO');
    livePaused = wasLivePaused;
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

  context.fillStyle = 'rgba(255,255,255,0.55)';
  for (const index of [33, 133, 159, 145, 362, 263, 386, 374, 1]) {
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

function updateMetrics(canvas, measurement, quality, sexPrior) {
  elements.pdValue.textContent = sexPrior.adjustedPdMm.toFixed(1);
  elements.rawPdValue.textContent = `${measurement.pdMm.toFixed(1)} mm`;
  elements.priorCenterValue.textContent = `${sexPrior.label} ${sexPrior.minMm}–${sexPrior.maxMm}mm · 중심 ${sexPrior.centerMm}mm`;
  elements.priorLossValue.textContent = `${sexPrior.priorLoss.toFixed(2)} (${Math.abs(sexPrior.normalizedDistance).toFixed(2)}× 반폭)`;
  elements.priorWeightValue.textContent = `${(sexPrior.priorWeight * 100).toFixed(1)}% · 측정 σ≈${sexPrior.measurementSigmaMm.toFixed(1)}mm`;
  elements.pdPxValue.textContent = `${measurement.pdPx.toFixed(1)} px`;
  elements.irisPxValue.textContent = `${measurement.meanIrisPx.toFixed(1)} px`;
  elements.irisDiffValue.textContent = `${(measurement.irisDifferenceRatio * 100).toFixed(1)}%`;
  const { yaw, pitch, roll } = measurement.pose;
  elements.poseValue.textContent = [yaw, pitch, roll].every(Number.isFinite)
    ? `${yaw.toFixed(1)}° / ${pitch.toFixed(1)}° / ${roll.toFixed(1)}°`
    : '계산 불가';
  const perspective = measurement.eyeAndPerspective;
  elements.perspectiveValue.textContent = `대칭 ${(perspective.perspectiveAsymmetryRatio * 100).toFixed(1)}% · 시선 ${(perspective.gazeOffset * 100).toFixed(0)}%`;
  elements.depthEstimateValue.textContent = `2D ${measurement.pdMm2D.toFixed(1)} mm · 3D ${measurement.pdMm3D.toFixed(1)} mm`;
  elements.depthAgreementValue.textContent = `${(measurement.depthAware.disagreementRatio * 100).toFixed(1)}% · 최종값 반영 0%`;
  const framing = measurement.framing;
  elements.framingValue.textContent = `얼굴 ${(framing.faceHeightRatio * 100).toFixed(0)}% · 중심 X ${(framing.centerOffsetX * 100).toFixed(0)}% / Y ${(framing.centerOffsetY * 100).toFixed(0)}%`;
  elements.resolutionValue.textContent = `${canvas.width}×${canvas.height}`;
  elements.qualityScoreValue.textContent = `${quality.score}/100`;
}

function resetMetrics() {
  elements.pdValue.textContent = '--';
  elements.rawPdValue.textContent = '-- mm';
  elements.priorCenterValue.textContent = '--';
  elements.priorLossValue.textContent = '--';
  elements.priorWeightValue.textContent = '--';
  elements.pdPxValue.textContent = '-- px';
  elements.irisPxValue.textContent = '-- px';
  elements.irisDiffValue.textContent = '--';
  elements.poseValue.textContent = '--';
  elements.perspectiveValue.textContent = '--';
  elements.depthEstimateValue.textContent = '--';
  elements.depthAgreementValue.textContent = '--';
  elements.framingValue.textContent = '--';
  elements.resolutionValue.textContent = '--';
  elements.qualityScoreValue.textContent = '--';
}

async function reanalyzeLastImage() {
  if (lastAnalysisSource) await analyzeCanvas(lastAnalysisSource, { strictCapture: false });
}

for (const input of [elements.irisReferenceInput, elements.priorStrengthInput, elements.maxYawInput, elements.maxPitchInput, elements.maxRollInput]) {
  input.addEventListener('change', reanalyzeLastImage);
}


elements.sexInput.addEventListener('change', () => {
  updateSexPriorHint();
  captureArmed = false;
  validHoldStartedAt = null;
  syncCaptureButton();
  void reanalyzeLastImage();
});

elements.startCameraButton.addEventListener('click', startCamera);
elements.stopCameraButton.addEventListener('click', stopCamera);
elements.captureButton.addEventListener('click', toggleAutoCapture);
elements.galleryInput.addEventListener('change', (event) => loadFile(event.target.files?.[0]));
window.addEventListener('pagehide', stopCamera);

updateSexPriorHint();
initializeModel();
