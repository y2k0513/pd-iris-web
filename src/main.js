import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './style.css';
import {
  applySexPdPrior,
  evaluateQuality,
  measurePd,
  median,
  SEX_PD_PRIORS,
} from './measurement.js';
import {
  ensureOpenCvReady,
  measureEyeImageQuality,
  refinePupilCenters,
} from './pupil.js';

const OFFICIAL_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const OFFICIAL_OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';
const LIVE_INTERVAL_MS = 170;
const STABLE_FRAME_TARGET = 4;
const MAX_PD_VARIATION_RATIO = 0.025;
const AUTO_CAPTURE_DELAY_MS = 1000;
const VIDEO_CAPTURE_DURATION_MS = 1000;
const VIDEO_CAPTURE_TARGET_FPS = 30;
const VIDEO_CAPTURE_MAX_FRAMES = 30;
const VIDEO_ANALYSIS_FRAME_COUNT = 8;
const MIN_VIDEO_VALID_FRAMES = 5;
const VIDEO_FRAME_MAX_DIMENSION = 1920;
const MAX_CAPTURE_DIMENSION = 4096;

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
  opencvStatus: document.querySelector('#opencvStatus'),
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
  eyeQualityCheck: document.querySelector('#eyeQualityCheck'),
  stabilityCheck: document.querySelector('#stabilityCheck'),
  pdValue: document.querySelector('#pdValue'),
  rawPdValue: document.querySelector('#rawPdValue'),
  priorCenterValue: document.querySelector('#priorCenterValue'),
  priorLossValue: document.querySelector('#priorLossValue'),
  priorWeightValue: document.querySelector('#priorWeightValue'),
  pdPxValue: document.querySelector('#pdPxValue'),
  irisPxValue: document.querySelector('#irisPxValue'),
  irisDiffValue: document.querySelector('#irisDiffValue'),
  pupilValue: document.querySelector('#pupilValue'),
  eyeQualityValue: document.querySelector('#eyeQualityValue'),
  burstValue: document.querySelector('#burstValue'),
  captureMethodValue: document.querySelector('#captureMethodValue'),
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
  minIrisPixelsInput: document.querySelector('#minIrisPixelsInput'),
  minEyeSharpnessInput: document.querySelector('#minEyeSharpnessInput'),
  minEyeBrightnessInput: document.querySelector('#minEyeBrightnessInput'),
  maxEyeBrightnessInput: document.querySelector('#maxEyeBrightnessInput'),
  showProcessingInput: document.querySelector('#showProcessingInput'),
  processingPlaceholder: document.querySelector('#processingPlaceholder'),
  processingContent: document.querySelector('#processingContent'),
  processingSummary: document.querySelector('#processingSummary'),
  processingRight: document.querySelector('#processingRight'),
  processingLeft: document.querySelector('#processingLeft'),
  processingRightMeta: document.querySelector('#processingRightMeta'),
  processingLeftMeta: document.querySelector('#processingLeftMeta'),
  processingDialog: document.querySelector('#processingDialog'),
  processingDialogTitle: document.querySelector('#processingDialogTitle'),
  processingDialogCanvas: document.querySelector('#processingDialogCanvas'),
  processingDialogDescription: document.querySelector('#processingDialogDescription'),
  processingDialogClose: document.querySelector('#processingDialogClose'),
};

let faceLandmarker = null;
let activeMode = 'VIDEO';
let cameraStream = null;
let imageCapture = null;
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
let openCvInstance = null;
let openCvLoadError = null;
let openCvPromise = null;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(element, text, kind) {
  if (!element) return;
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

async function initializeOpenCv() {
  setStatus(elements.opencvStatus, '동공 보정 준비 중', 'pending');
  openCvPromise = ensureOpenCvReady([
    assetUrl('opencv/opencv.js'),
    OFFICIAL_OPENCV_URL,
  ]);
  try {
    openCvInstance = await openCvPromise;
    setStatus(elements.opencvStatus, 'OpenCV 동공 보정 준비됨', 'success');
    return openCvInstance;
  } catch (error) {
    openCvLoadError = error;
    console.warn('OpenCV unavailable; MediaPipe fallback will be used.', error);
    setStatus(elements.opencvStatus, 'OpenCV 미사용 · fallback', 'danger');
    return null;
  }
}

async function getOpenCvOptional() {
  if (openCvInstance?.Mat) return openCvInstance;
  if (openCvLoadError) return null;
  if (!openCvPromise) void initializeOpenCv();
  try {
    return await openCvPromise;
  } catch {
    return null;
  }
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
    setMessage('카메라를 시작한 뒤 휴대폰을 얼굴에서 약 30~40cm 떨어뜨리고 얼굴을 프레임 중앙에 맞추세요.', 'info');
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
    minIrisPixels: Number(elements.minIrisPixelsInput.value) || 20,
    minFaceHeightRatio: 0.31,
    maxFaceHeightRatio: 0.58,
    maxFaceCenterOffsetX: 0.10,
    maxFaceCenterOffsetY: 0.12,
    maxGazeOffset: 0.28,
    minEyeOpeningRatio: 0.06,
    maxPerspectiveAsymmetryRatio: 0.16,
    maxEyeWidthDifferenceRatio: 0.20,
    maxIrisDepthDifference: 0.025,
    max2D3DDisagreementRatio: 0.10,
    warn2D3DDisagreementRatio: 0.06,
    minEyeSharpness: Number(elements.minEyeSharpnessInput.value) || 35,
    minEyeBrightness: Number(elements.minEyeBrightnessInput.value) || 35,
    maxEyeBrightness: Number(elements.maxEyeBrightnessInput.value) || 225,
    minPupilConfidence: 0.50,
    showProcessing: Boolean(elements.showProcessingInput?.checked),
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

async function maximizeCameraTrack(track) {
  if (!track?.getCapabilities || !track?.applyConstraints) return;
  try {
    const capabilities = track.getCapabilities();
    const width = Math.min(capabilities.width?.max || 3840, 4096);
    const height = Math.min(capabilities.height?.max || 2160, 3072);
    await track.applyConstraints({
      width: { ideal: width },
      height: { ideal: height },
      frameRate: { ideal: Math.min(capabilities.frameRate?.max || 30, 30) },
    });
  } catch (error) {
    console.warn('Maximum camera constraint could not be applied.', error);
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
    const track = cameraStream.getVideoTracks()[0];
    await maximizeCameraTrack(track);
    elements.cameraVideo.srcObject = cameraStream;
    await elements.cameraVideo.play();
    await new Promise((resolve) => {
      if (elements.cameraVideo.videoWidth > 0) resolve();
      else elements.cameraVideo.addEventListener('loadedmetadata', resolve, { once: true });
    });

    const settings = track.getSettings();
    const width = settings.width || elements.cameraVideo.videoWidth;
    const height = settings.height || elements.cameraVideo.videoHeight;
    const fps = settings.frameRate ? ` · ${Math.round(settings.frameRate)}fps` : '';
    imageCapture = null;
    if ('ImageCapture' in window) {
      try {
        imageCapture = new ImageCapture(track);
      } catch (error) {
        console.warn('ImageCapture initialization failed.', error);
      }
    }

    elements.mediaStage.style.aspectRatio = `${elements.cameraVideo.videoWidth} / ${elements.cameraVideo.videoHeight}`;
    elements.cameraMeta.textContent = `${width}×${height}${fps}${imageCapture ? ' · 고해상도 사진 지원' : ' · 비디오 프레임 캡처'}`;
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
  imageCapture = null;
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

  try {
    measurement.eyeImageQuality = measureEyeImageQuality(
      elements.cameraVideo,
      landmarks,
      elements.cameraVideo.videoWidth,
      elements.cameraVideo.videoHeight,
    );
  } catch (error) {
    console.warn('Live eye quality measurement failed.', error);
  }

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
    if (captureArmed && ready) {
      if (validHoldStartedAt === null) validHoldStartedAt = timestamp;
      remainingMs = Math.max(0, AUTO_CAPTURE_DELAY_MS - (timestamp - validHoldStartedAt));

      if (remainingMs <= 0 && !autoCaptureInProgress) {
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
  const { pose, eyeAndPerspective, eyeImageQuality } = measurement;
  const poseOk = [pose.yaw, pose.pitch, pose.roll].every(Number.isFinite)
    && Math.abs(pose.yaw) <= config.maxYaw
    && Math.abs(pose.pitch) <= config.maxPitch
    && Math.abs(pose.roll) <= config.maxRoll;
  const perspectiveOk = eyeAndPerspective.perspectiveAsymmetryRatio <= config.maxPerspectiveAsymmetryRatio
    && measurement.irisDifferenceRatio <= config.maxIrisDifferenceRatio
    && measurement.depthAware.disagreementRatio <= config.max2D3DDisagreementRatio;
  const gazeOk = eyeAndPerspective.gazeOffset <= config.maxGazeOffset;
  const eyeQualityOk = Boolean(eyeImageQuality)
    && eyeImageQuality.minSharpness >= config.minEyeSharpness
    && eyeImageQuality.minBrightness >= config.minEyeBrightness
    && eyeImageQuality.maxBrightness <= config.maxEyeBrightness;

  setCheck(elements.poseCheck, poseOk, `자세 ${pose.yaw.toFixed(1)}/${pose.pitch.toFixed(1)}/${pose.roll.toFixed(1)}°`);
  setCheck(
    elements.perspectiveCheck,
    perspectiveOk,
    `원근 ${(eyeAndPerspective.perspectiveAsymmetryRatio * 100).toFixed(1)}% · 3D검증 ${(measurement.depthAware.disagreementRatio * 100).toFixed(1)}%`,
  );
  setCheck(elements.gazeCheck, gazeOk, `시선 ${(eyeAndPerspective.gazeOffset * 100).toFixed(0)}%`);
  setCheck(
    elements.eyeQualityCheck,
    eyeQualityOk,
    eyeImageQuality
      ? `눈 초점 ${eyeImageQuality.minSharpness.toFixed(0)} · 밝기 ${eyeImageQuality.meanBrightness.toFixed(0)}`
      : '눈 품질 --',
  );
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
    if (ready && remainingMs !== null) {
      elements.liveGuide.className = 'face-guide live-guide ready';
      elements.liveStatus.textContent = `자세 유지 — ${(remainingMs / 1000).toFixed(1)}초 후 3장 자동촬영`;
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
    elements.captureButton.textContent = '3장 촬영 및 분석 중';
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
    [elements.eyeQualityCheck, '눈 품질 --'],
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
    setMessage('휴대폰을 얼굴에서 약 30~40cm 떨어뜨리고 조건을 1초 유지하면, 이어서 1초간 영상 프레임을 수집합니다.', 'info');
    elements.liveStatus.textContent = '30~40cm 거리에서 얼굴을 맞추세요 — 조건 충족 후 1초 영상수집';
  } else {
    setMessage('자동촬영을 취소했습니다.', 'info');
    elements.liveStatus.textContent = latestLiveState?.ready
      ? '자세 양호 — 촬영 준비 버튼을 누르세요'
      : '촬영 준비 버튼을 누른 뒤 자세를 맞추세요';
  }

  syncCaptureButton();
}

function createCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function drawSourceToCanvas(source, sourceWidth, sourceHeight, maxDimension = MAX_CAPTURE_DIMENSION) {
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = createCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(source, 0, 0, width, height);
  return canvas;
}

async function captureWithImageCapture() {
  if (!imageCapture?.takePhoto) return null;
  const blob = await imageCapture.takePhoto();
  const bitmap = await decodeImageFile(blob);
  try {
    const width = bitmap.width || bitmap.naturalWidth;
    const height = bitmap.height || bitmap.naturalHeight;
    return {
      canvas: drawSourceToCanvas(bitmap, width, height),
      method: 'ImageCapture.takePhoto',
    };
  } finally {
    if (typeof bitmap.close === 'function') bitmap.close();
  }
}

function captureVideoFrame() {
  const width = elements.cameraVideo.videoWidth;
  const height = elements.cameraVideo.videoHeight;

  // CSS 미러링은 프리뷰에만 적용한다.
  // 분석 이미지는 카메라 원본 좌표계를 유지한다.
  const canvas = drawSourceToCanvas(
    elements.cameraVideo,
    width,
    height,
    VIDEO_FRAME_MAX_DIMENSION,
  );

  return {
    canvas,
    method: '1s video stream frame',
  };
}

async function captureHighResolutionFrame() {
  if (imageCapture) {
    try {
      return await captureWithImageCapture();
    } catch (error) {
      console.warn('takePhoto failed; using video frame.', error);
      imageCapture = null;
    }
  }
  return captureVideoFrame();
}

function waitForNextDecodedVideoFrame() {
  const video = elements.cameraVideo;
  const fallbackDelayMs =
    1000 / VIDEO_CAPTURE_TARGET_FPS;

  if (
    typeof video.requestVideoFrameCallback
    !== 'function'
  ) {
    return delay(fallbackDelayMs);
  }

  return new Promise((resolve) => {
    let settled = false;
    let callbackId = null;
    let timeoutId = null;

    const finish = (
      timestamp,
      cancelPendingCallback,
    ) => {
      if (settled) return;
      settled = true;

      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }

      if (
        cancelPendingCallback
        && callbackId !== null
        && typeof video.cancelVideoFrameCallback
          === 'function'
      ) {
        video.cancelVideoFrameCallback(
          callbackId,
        );
      }

      resolve(timestamp);
    };

    callbackId =
      video.requestVideoFrameCallback(
        (timestamp) => {
          finish(timestamp, false);
        },
      );

    // 저프레임 카메라나 브라우저 오류 시 무한대기 방지
    timeoutId = setTimeout(
      () => {
        finish(
          performance.now(),
          true,
        );
      },
      120,
    );
  });
}

function scoreVideoCandidate(
  canvas,
  landmarks,
) {
  if (!landmarks) return 0;

  try {
    const eyeQuality =
      measureEyeImageQuality(
        canvas,
        landmarks,
        canvas.width,
        canvas.height,
      );

    const exposurePenalty =
      (
        eyeQuality.maxUnderexposedRatio
        + eyeQuality.maxOverexposedRatio
      ) * 500;

    return (
      eyeQuality.minSharpness
      - exposurePenalty
    );
  } catch {
    return 0;
  }
}

function releaseCaptureCanvas(capture) {
  if (!capture?.canvas) return;

  // 선택되지 않은 Canvas backing store를 즉시 축소한다.
  capture.canvas.width = 1;
  capture.canvas.height = 1;
}

async function captureOneSecondVideoWindow() {
  const referenceLandmarks =
    latestLiveState?.landmarks;

  if (!referenceLandmarks) {
    throw new Error(
      '영상 수집 직전 얼굴 랜드마크가 없습니다.',
    );
  }

  const selectedByBucket = new Map();

  const bucketDurationMs =
    VIDEO_CAPTURE_DURATION_MS
    / VIDEO_ANALYSIS_FRAME_COUNT;

  const startedAt = performance.now();
  let sampledFrameCount = 0;

  const addCurrentFrame = () => {
    const elapsedMs = Math.max(
      0,
      Math.min(
        VIDEO_CAPTURE_DURATION_MS - 0.001,
        performance.now() - startedAt,
      ),
    );

    const bucketIndex = Math.min(
      VIDEO_ANALYSIS_FRAME_COUNT - 1,
      Math.floor(
        elapsedMs / bucketDurationMs,
      ),
    );

    const capture =
      captureVideoFrame();

    capture.method =
      '1s video stream frame '
      + (sampledFrameCount + 1);

    const score =
      scoreVideoCandidate(
        capture.canvas,
        referenceLandmarks,
      );

    const existing =
      selectedByBucket.get(
        bucketIndex,
      );

    if (
      !existing
      || score > existing.score
    ) {
      if (existing) {
        releaseCaptureCanvas(
          existing.capture,
        );
      }

      selectedByBucket.set(
        bucketIndex,
        {
          bucketIndex,
          score,
          capture,
        },
      );
    } else {
      releaseCaptureCanvas(capture);
    }

    sampledFrameCount += 1;
  };

  // 시점 0의 프레임도 포함한다.
  addCurrentFrame();

  while (
    sampledFrameCount
      < VIDEO_CAPTURE_MAX_FRAMES
    && performance.now() - startedAt
      < VIDEO_CAPTURE_DURATION_MS
  ) {
    await waitForNextDecodedVideoFrame();

    if (
      performance.now() - startedAt
      >= VIDEO_CAPTURE_DURATION_MS
    ) {
      break;
    }

    addCurrentFrame();
  }

  const captures =
    [...selectedByBucket.values()]
      .sort(
        (a, b) =>
          a.bucketIndex
          - b.bucketIndex,
      )
      .map(
        (entry) => entry.capture,
      );

  if (
    captures.length
    < MIN_VIDEO_VALID_FRAMES
  ) {
    throw new Error(
      '1초 영상에서 분석 가능한 구간 프레임이 '
      + captures.length
      + '장뿐입니다. 최소 '
      + MIN_VIDEO_VALID_FRAMES
      + '장이 필요합니다.',
    );
  }

  return {
    captures,
    sampledFrameCount,
  };
}

function filterPdOutliers(analyses) {
  if (
    analyses.length
    < MIN_VIDEO_VALID_FRAMES
  ) {
    return analyses;
  }

  const values = analyses.map(
    (analysis) =>
      analysis.measurement.pdMm,
  );

  const center = median(values);

  const absoluteDeviations =
    values.map(
      (value) =>
        Math.abs(value - center),
    );

  const mad =
    median(absoluteDeviations);

  /*
   * MAD가 매우 작아도 최소 0.75mm 여유를 둔다.
   * 2.5×MAD 밖의 큰 fitting 오류만 제거한다.
   */
  const limitMm = Math.max(
    0.75,
    2.5 * (
      Number.isFinite(mad)
        ? mad
        : 0
    ),
  );

  const filtered = analyses.filter(
    (analysis) =>
      Math.abs(
        analysis.measurement.pdMm
        - center,
      ) <= limitMm,
  );

  /*
   * 이상치 제거 후 5장 미만이면
   * 지나치게 공격적인 제거로 보고 원본을 유지한다.
   */
  return filtered.length
    >= MIN_VIDEO_VALID_FRAMES
    ? filtered
    : analyses;
}

async function captureCurrentFrame() {
  if (
    !cameraStream
    || !latestLiveState?.ready
  ) {
    autoCaptureInProgress = false;
    captureArmed = true;
    validHoldStartedAt = null;
    syncCaptureButton();

    setMessage(
      '촬영 직전 자세가 흔들렸습니다. 자세를 다시 유지하세요.',
      'warning',
    );

    return;
  }

  livePaused = true;
  autoCaptureInProgress = true;
  captureArmed = false;
  validHoldStartedAt = null;
  syncCaptureButton();

  setMessage(
    '1초간 영상 프레임을 수집하고 있습니다. 자세를 유지하세요.',
    'info',
  );

  try {
    const {
      captures,
      sampledFrameCount,
    } =
      await captureOneSecondVideoWindow();

    setMessage(
      sampledFrameCount
      + '프레임을 수집했습니다. 선명한 '
      + captures.length
      + '프레임을 정밀 분석하고 있습니다.',
      'info',
    );

    await analyzeBurst(
      captures,
      {
        strictCapture: true,
        sampledFrameCount,
      },
    );
  } finally {
    autoCaptureInProgress = false;

    resetLiveGate(
      '다음 측정은 촬영 준비 버튼을 누르세요',
    );

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
    const canvas = drawSourceToCanvas(source, sourceWidth, sourceHeight);
    elements.cameraMeta.textContent = `사진 테스트 · ${canvas.width}×${canvas.height}`;
    await analyzeCanvas(canvas, { strictCapture: false, captureMethod: 'gallery image' });
  } catch (error) {
    console.error(error);
    setMessage(`사진을 열지 못했습니다: ${error.message}`, 'danger');
  } finally {
    if (typeof source?.close === 'function') source.close();
    elements.galleryInput.value = '';
  }
}

async function analyzeFrame(canvas, cv, captureMethod = 'unknown') {
  const result = faceLandmarker.detect(canvas);
  if (result.faceLandmarks.length !== 1) {
    throw new Error(result.faceLandmarks.length === 0 ? '얼굴을 찾지 못했습니다.' : '한 명의 얼굴만 촬영해야 합니다.');
  }

  const landmarks = result.faceLandmarks[0];
  const matrix = result.facialTransformationMatrixes?.[0];
  const config = getConfig();
  const eyeImageQuality = measureEyeImageQuality(canvas, landmarks, canvas.width, canvas.height);
  const pupilRefinement = refinePupilCenters({
    source: canvas,
    landmarks,
    width: canvas.width,
    height: canvas.height,
    cv,
    includeDebug: config.showProcessing,
  });
  /*
   * pupilRefinement.*.ellipse는 원본 이미지 좌표계의
   * 3.5% 확대된 보라색 fitting 원이다.
   */
  const rightPurpleCircle =
    pupilRefinement.right.ellipse;

  const leftPurpleCircle =
    pupilRefinement.left.ellipse;

  const validPurpleCircle = (circle) => (
    circle
    && Number.isFinite(circle.x)
    && Number.isFinite(circle.y)
    && Number.isFinite(circle.width)
    && Number.isFinite(circle.height)
    && circle.width > 0
    && circle.height > 0
  );

  if (
    !validPurpleCircle(rightPurpleCircle)
    || !validPurpleCircle(leftPurpleCircle)
  ) {
    throw new Error(
      '양쪽 보라색 기준 원을 모두 검출하지 못했습니다.',
    );
  }

  const rightPurpleDiameter =
    (
      rightPurpleCircle.width
      + rightPurpleCircle.height
    ) / 2;

  const leftPurpleDiameter =
    (
      leftPurpleCircle.width
      + leftPurpleCircle.height
    ) / 2;

  const measurement = measurePd({
    landmarks,
    matrix,
    width: canvas.width,
    height: canvas.height,

    // 보라색 원 평균 지름을 실제 11.7mm로 가정
    irisReferenceMm:
      config.irisReferenceMm,

    // 보라색 원 자체의 중심 사용
    centerOverrides: {
      right: {
        x: rightPurpleCircle.x,
        y: rightPurpleCircle.y,
      },

      left: {
        x: leftPurpleCircle.x,
        y: leftPurpleCircle.y,
      },
    },

    // 3.5% 확대된 보라색 원 자체의 지름 사용
    diameterOverrides: {
      right: rightPurpleDiameter,
      left: leftPurpleDiameter,
    },
  });
  measurement.eyeImageQuality = eyeImageQuality;
  measurement.pupilRefinement = pupilRefinement;
  measurement.captureMeta = { method: captureMethod };
  const quality = evaluateQuality(measurement, config);
  return { canvas, landmarks, measurement, quality };
}

function aggregateBurst(analyses) {
  const values = analyses.map((analysis) => analysis.measurement.pdMm);
  const medianPdMm = median(values);
  const selected = [...analyses].sort((a, b) => {
    const aDistance = Math.abs(a.measurement.pdMm - medianPdMm);
    const bDistance = Math.abs(b.measurement.pdMm - medianPdMm);
    if (aDistance !== bDistance) return aDistance - bDistance;
    const aSharpness = a.measurement.eyeImageQuality?.minSharpness || 0;
    const bSharpness = b.measurement.eyeImageQuality?.minSharpness || 0;
    return bSharpness - aSharpness;
  })[0];
  const selectedFrameIndex = analyses.indexOf(selected);

  const singleFramePdMm = selected.measurement.pdMm;
  const meanQualityScore = Math.round(
    analyses.reduce((sum, analysis) => sum + analysis.quality.score, 0) / analyses.length,
  );
  const pd3D = selected.measurement.pdMm3D;
  const estimateMean = (medianPdMm + pd3D) / 2;
  const measurement = {
    ...selected.measurement,
    pdMm: medianPdMm,
    pdMm2D: medianPdMm,
    pdPx: medianPdMm / selected.measurement.mmPerPixel,
    depthAware: {
      ...selected.measurement.depthAware,
      disagreementRatio: Math.abs(pd3D - medianPdMm) / estimateMean,
    },
    burst: {
      frameCount: analyses.length,
      values,
      medianPdMm,
      singleFramePdMm,
      spreadMm: Math.max(...values) - Math.min(...values),
      selectedFrameIndex,
    },
  };
  const quality = {
    ...selected.quality,
    score: meanQualityScore,
  };
  return { ...selected, measurement, quality };
}

async function analyzeBurst(captures, { strictCapture, sampledFrameCount = null } = {}) {
  if (!faceLandmarker) return;
  const selectedSex = getSelectedSex();
  if (!selectedSex) {
    setMessage('분석 전에 성별을 선택하세요.', 'danger');
    return;
  }

  setStatus(elements.qualityBadge, captures.length + '프레임 정밀 분석 중', 'pending');
  try {
    await setModelMode('IMAGE');
    const cv = await getOpenCvOptional();
    const analyses = [];
    const failures = [];

    for (const capture of captures) {
      try {
        analyses.push(await analyzeFrame(capture.canvas, cv, capture.method));
      } catch (error) {
        failures.push(error.message);
      }
    }

    const qualityAccepted =
      strictCapture
        ? analyses.filter(
          (analysis) =>
            analysis.quality.accepted,
        )
        : analyses;

    const requiredFrames =
      strictCapture
        ? MIN_VIDEO_VALID_FRAMES
        : 1;

    if (
      qualityAccepted.length
      < requiredFrames
    ) {
      const qualityReasons =
        analyses.flatMap(
          (analysis) =>
            analysis.quality.reasons,
        );

      throw new Error(
        qualityReasons[0]
        || failures[0]
        || (
          '유효한 프레임이 '
          + qualityAccepted.length
          + '장뿐입니다. 최소 '
          + requiredFrames
          + '장이 필요합니다.'
        ),
      );
    }

    const usable =
      strictCapture
        ? filterPdOutliers(
          qualityAccepted,
        )
        : qualityAccepted;

    const final =
      aggregateBurst(usable);

    final.measurement.burst.sampledFrameCount =
      Number.isFinite(
        sampledFrameCount,
      )
        ? sampledFrameCount
        : captures.length;

    final.measurement.burst.analyzedFrameCount =
      captures.length;

    final.measurement.burst.qualityAcceptedFrameCount =
      qualityAccepted.length;

    final.measurement.burst.outlierRemovedCount =
      qualityAccepted.length
      - usable.length;
    const sexPrior = applySexPdPrior({
      rawPdMm: final.measurement.pdMm,
      sex: selectedSex,
      qualityScore: final.quality.score,
      strength: getConfig().priorStrength,
    });

    copyCanvas(final.canvas, elements.captureCanvas);
    lastAnalysisSource = elements.captureCanvas;
    renderResult(final.canvas, final.landmarks, final.measurement, final.quality);
    renderProcessingDebug(final.measurement);
    updateMetrics(final.canvas, final.measurement, final.quality, sexPrior);
    setStatus(elements.qualityBadge, '측정 가능', 'success');
    const priorSummary = sexPrior.withinTypicalRange
      ? `${sexPrior.label} 분포 안 · raw ${sexPrior.rawPdMm.toFixed(1)} → 보정 ${sexPrior.adjustedPdMm.toFixed(1)}mm`
      : `${sexPrior.label} 기준 ${sexPrior.minMm}–${sexPrior.maxMm}mm 밖 · raw ${sexPrior.rawPdMm.toFixed(1)} → 보정 ${sexPrior.adjustedPdMm.toFixed(1)}mm`;
    const fallbackText = final.measurement.pupilRefinement.fallbackCount
      ? ` · ${final.measurement.pupilRefinement.fallbackCount}개 눈 MediaPipe fallback`
      : ' · 양쪽 OpenCV 동공 보정';
    const captureSummary =
      Number.isFinite(
        sampledFrameCount,
      )
        ? (
          '1초 영상 '
          + sampledFrameCount
          + '프레임 수집 → '
          + captures.length
          + '프레임 선별 → '
        )
        : '';

    setMessage(
      captureSummary
      + usable.length
      + '프레임 중앙값을 사용했습니다'
      + fallbackText
      + '. '
      + priorSummary,
      sexPrior.withinTypicalRange
        ? 'success'
        : 'warning',
    );
  } catch (error) {
    console.error(error);
    resetMetrics();
    setStatus(elements.qualityBadge, '촬영 무효', 'danger');
    setMessage(error.message, 'danger');
  } finally {
    await setModelMode('VIDEO');
  }
}

function copyCanvas(source, target) {
  if (source === target) return;
  target.width = source.width;
  target.height = source.height;
  target.getContext('2d', { willReadFrequently: true }).drawImage(source, 0, 0);
}

async function analyzeCanvas(canvas, { strictCapture, captureMethod = 'single image' }) {
  if (!faceLandmarker) return;
  const selectedSex = getSelectedSex();
  if (!selectedSex) {
    setMessage('분석 전에 성별을 선택하세요.', 'danger');
    return;
  }
  const wasLivePaused = livePaused;
  livePaused = true;
  setStatus(elements.qualityBadge, '분석 중', 'pending');
  setMessage('눈 crop에서 동공 중심과 초점 품질을 분석하고 있습니다.', 'info');

  try {
    await setModelMode('IMAGE');
    const cv = await getOpenCvOptional();
    const analysis = await analyzeFrame(canvas, cv, captureMethod);
    const { measurement, quality, landmarks } = analysis;
    const config = getConfig();
    const sexPrior = applySexPdPrior({
      rawPdMm: measurement.pdMm,
      sex: selectedSex,
      qualityScore: quality.score,
      strength: config.priorStrength,
    });

    if (strictCapture && !quality.accepted) {
      resetMetrics();
      setStatus(elements.qualityBadge, '촬영 무효', 'danger');
      setMessage(`촬영 순간 조건이 바뀌었습니다: ${quality.reasons.join(' · ')}`, 'danger');
      return;
    }

    copyCanvas(canvas, elements.captureCanvas);
    lastAnalysisSource = elements.captureCanvas;
    renderResult(canvas, landmarks, measurement, quality);
    renderProcessingDebug(measurement);
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

function resetProcessingDebug(message = '촬영하거나 사진을 분석하면 좌우 눈 crop부터 동공 타원 fitting까지 각 단계를 표시합니다.') {
  if (!elements.processingPlaceholder) return;
  elements.processingPlaceholder.textContent = message;
  elements.processingPlaceholder.hidden = false;
  elements.processingContent.hidden = true;
  elements.processingRight.replaceChildren();
  elements.processingLeft.replaceChildren();
  elements.processingRightMeta.textContent = '--';
  elements.processingLeftMeta.textContent = '--';
}

function copyStageToDialog(stage) {
  if (!elements.processingDialog || !stage?.canvas) return;
  elements.processingDialogTitle.textContent = stage.label;
  elements.processingDialogDescription.textContent = stage.description || '';
  const target = elements.processingDialogCanvas;
  target.width = stage.canvas.width;
  target.height = stage.canvas.height;
  target.getContext('2d').drawImage(stage.canvas, 0, 0);
  if (typeof elements.processingDialog.showModal === 'function') {
    elements.processingDialog.showModal();
  } else {
    elements.processingDialog.setAttribute('open', '');
  }
}

function renderEyeProcessing(container, metaElement, pupil) {
  container.replaceChildren();
  if (!pupil?.debug?.stages?.length) {
    const unavailable = document.createElement('div');
    unavailable.className = 'processing-placeholder';
    unavailable.textContent = pupil?.reason || '이 눈의 처리과정 이미지가 없습니다.';
    container.append(unavailable);
    metaElement.textContent = `${pupil?.source || 'unknown'} · 신뢰도 ${((pupil?.confidence || 0) * 100).toFixed(0)}%`;
    return;
  }

  const diagnostics = pupil.diagnostics;
  const threshold = pupil.debug.darkThreshold;
  metaElement.textContent = [
    `${pupil.source} · 신뢰도 ${(pupil.confidence * 100).toFixed(0)}%`,
    Number.isFinite(threshold) ? `임계값 ${threshold}` : null,
    diagnostics?.fitType ? `fit ${diagnostics.fitType}` : null,
    diagnostics && Number.isFinite(diagnostics.iou) ? `IoU ${(diagnostics.iou * 100).toFixed(0)}%` : null,
    diagnostics ? `대비 ${(diagnostics.contrast * 100).toFixed(0)}% · 원형도 ${(diagnostics.circularity * 100).toFixed(0)}%` : null,
  ].filter(Boolean).join(' · ');

  for (const stage of pupil.debug.stages) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'processing-stage';
    button.setAttribute('aria-label', `${stage.label} 확대`);
    const title = document.createElement('strong');
    title.textContent = stage.label;
    const description = document.createElement('small');
    description.textContent = stage.description || '';
    button.append(stage.canvas, title, description);
    button.addEventListener('click', () => copyStageToDialog(stage));
    container.append(button);
  }
}

function renderProcessingDebug(measurement) {
  if (!elements.processingPlaceholder) return;
  const config = getConfig();
  if (!config.showProcessing) {
    resetProcessingDebug('측정 설정의 “처리과정 표시”를 켜고 다시 분석하면 각 단계를 볼 수 있습니다.');
    return;
  }

  const pupilRefinement = measurement?.pupilRefinement;
  if (!pupilRefinement) {
    resetProcessingDebug('동공 보정 결과가 없어 처리과정을 표시할 수 없습니다.');
    return;
  }

  elements.processingPlaceholder.hidden = true;
  elements.processingContent.hidden = false;
  const burst = measurement.burst;
  const selectedFrameText = burst
    ? `버스트 ${burst.frameCount}장 중 ${burst.selectedFrameIndex + 1}번째 프레임이 중앙값에 가장 가까워 시각화에 선택됨`
    : '단일 프레임 처리과정';
  elements.processingSummary.textContent = `${selectedFrameText} · 실제 PD에는 파란 최종 중심이 사용됩니다. OpenCV 신뢰도가 낮으면 흰색 MediaPipe 중심으로 fallback합니다.`;
  renderEyeProcessing(elements.processingRight, elements.processingRightMeta, pupilRefinement.right);
  renderEyeProcessing(elements.processingLeft, elements.processingLeftMeta, pupilRefinement.left);
}

function renderResult(sourceCanvas, landmarks, measurement, quality) {
  const canvas = elements.resultCanvas;
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  const context = canvas.getContext('2d');
  context.drawImage(sourceCanvas, 0, 0);
  const scale = Math.max(1, Math.min(canvas.width, canvas.height) / 700);
  const {
    leftCenter,
    rightCenter,
    leftBoundary,
    rightBoundary,
    mediaPipeLeftCenter,
    mediaPipeRightCenter,
  } = measurement.points;

  context.lineCap = 'round';
  context.lineWidth = 3 * scale;
  context.strokeStyle = quality.accepted ? '#12b76a' : '#f04438';
  context.beginPath();
  context.moveTo(leftCenter.x, leftCenter.y);
  context.lineTo(rightCenter.x, rightCenter.y);
  context.stroke();

  drawPoint(context, mediaPipeLeftCenter, 'rgba(255,255,255,.72)', 3.2 * scale);
  drawPoint(context, mediaPipeRightCenter, 'rgba(255,255,255,.72)', 3.2 * scale);
  drawPoint(context, leftCenter, '#2e90fa', 6 * scale);
  drawPoint(context, rightCenter, '#2e90fa', 6 * scale);
  [...leftBoundary, ...rightBoundary].forEach((point) => drawPoint(context, point, '#f79009', 4 * scale));

  for (const pupil of [measurement.pupilRefinement?.left, measurement.pupilRefinement?.right]) {
    if (!pupil?.ellipse) continue;
    drawEllipse(context, pupil.ellipse, pupil.source === 'opencv' ? '#7f56d9' : '#fdb022', 2.5 * scale);
  }

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

function drawEllipse(context, ellipse, color, lineWidth) {
  context.save();
  context.translate(ellipse.x, ellipse.y);
  context.rotate((ellipse.angle || 0) * Math.PI / 180);
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.beginPath();
  context.ellipse(0, 0, ellipse.width / 2, ellipse.height / 2, 0, 0, Math.PI * 2);
  context.stroke();
  context.restore();
}

function pupilSummary(pupilRefinement) {
  if (!pupilRefinement) return '--';
  const labels = {
    opencv: 'OpenCV',
    blended: '융합',
    mediapipe: 'MediaPipe',
  };
  const left = pupilRefinement.left;
  const right = pupilRefinement.right;
  return `좌 ${labels[left.source]} ${(left.confidence * 100).toFixed(0)}% · 우 ${labels[right.source]} ${(right.confidence * 100).toFixed(0)}%`;
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
  elements.pupilValue.textContent = pupilSummary(measurement.pupilRefinement);
  const eyeQuality = measurement.eyeImageQuality;
  elements.eyeQualityValue.textContent = eyeQuality
    ? `초점 ${eyeQuality.left.sharpness.toFixed(0)} / ${eyeQuality.right.sharpness.toFixed(0)} · 밝기 ${eyeQuality.meanBrightness.toFixed(0)}`
    : '--';
  const burstSamplePrefix =
    measurement.burst?.sampledFrameCount
      ? (
        measurement.burst.sampledFrameCount
        + '프레임 수집 → '
      )
      : '';

  elements.burstValue.textContent =
    measurement.burst
      ? (
        burstSamplePrefix
        + measurement.burst.frameCount
        + '장 중앙값 · '
        + measurement.burst.values
          .map(
            (value) =>
              value.toFixed(1),
          )
          .join(' / ')
        + 'mm · 범위 '
        + measurement.burst.spreadMm
          .toFixed(2)
        + 'mm'
      )
      : '단일 프레임';
  elements.captureMethodValue.textContent = measurement.captureMeta?.method || '--';
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
  elements.pupilValue.textContent = '--';
  elements.eyeQualityValue.textContent = '--';
  elements.burstValue.textContent = '--';
  elements.captureMethodValue.textContent = '--';
  elements.poseValue.textContent = '--';
  elements.perspectiveValue.textContent = '--';
  elements.depthEstimateValue.textContent = '--';
  elements.depthAgreementValue.textContent = '--';
  elements.framingValue.textContent = '--';
  elements.resolutionValue.textContent = '--';
  elements.qualityScoreValue.textContent = '--';
  resetProcessingDebug();
}

async function reanalyzeLastImage() {
  if (lastAnalysisSource) await analyzeCanvas(lastAnalysisSource, { strictCapture: false, captureMethod: 'reanalyzed image' });
}

for (const input of [
  elements.irisReferenceInput,
  elements.priorStrengthInput,
  elements.maxYawInput,
  elements.maxPitchInput,
  elements.maxRollInput,
  elements.minIrisPixelsInput,
  elements.minEyeSharpnessInput,
  elements.minEyeBrightnessInput,
  elements.maxEyeBrightnessInput,
  elements.showProcessingInput,
]) {
  input.addEventListener('change', reanalyzeLastImage);
}

elements.sexInput.addEventListener('change', () => {
  updateSexPriorHint();
  captureArmed = false;
  validHoldStartedAt = null;
  syncCaptureButton();
  void reanalyzeLastImage();
});

elements.processingDialogClose?.addEventListener('click', () => elements.processingDialog.close());
elements.processingDialog?.addEventListener('click', (event) => {
  if (event.target === elements.processingDialog) elements.processingDialog.close();
});

elements.startCameraButton.addEventListener('click', startCamera);
elements.stopCameraButton.addEventListener('click', stopCamera);
elements.captureButton.addEventListener('click', toggleAutoCapture);
elements.galleryInput.addEventListener('change', (event) => loadFile(event.target.files?.[0]));
window.addEventListener('pagehide', stopCamera);

updateSexPriorHint();
void initializeOpenCv();
void initializeModel();
