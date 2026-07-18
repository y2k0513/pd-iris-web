import { fillSmallBlackHoles4Connected } from './binary.js';
import { chooseBestPupilFit, scorePupilFitMetrics } from './fitting.js';

const EYE_DEFINITIONS = Object.freeze({
  right: Object.freeze({
    outer: 33,
    inner: 133,
    upper: 159,
    lower: 145,
    irisCenter: 468,
    irisBoundary: [469, 470, 471, 472],
  }),
  left: Object.freeze({
    outer: 263,
    inner: 362,
    upper: 386,
    lower: 374,
    irisCenter: 473,
    irisBoundary: [474, 475, 476, 477],
  }),
});

const DEFAULT_OPENCV_URLS = [
  'opencv/opencv.js',
  'https://docs.opencv.org/4.x/opencv.js',
];

// Pupil segmentation tuning values.
// Tone correction uses a dark-side pivot. Pixels farther from the pivot are
// pushed progressively toward black or white with a sigmoid curve.
const PUPIL_TONE_PIVOT_PERCENTILE = 0.18;
const PUPIL_TONE_SOFTNESS = 18;
// A higher binary percentile/offset includes moderately dark pupil pixels,
// rather than keeping only the very darkest pixels.
const PUPIL_THRESHOLD_PERCENTILE = 0.45;
const PUPIL_THRESHOLD_OFFSET = 10;
const PUPIL_THRESHOLD_MIN = 12;
const PUPIL_THRESHOLD_MAX = 180;
const PUPIL_OPEN_KERNEL_SIZE = 3;
const PUPIL_CLOSE_KERNEL_SIZE = 7;
const PUPIL_HOLE_MAX_AREA_RATIO = 0.08;
const PUPIL_HOLE_MAX_AREA_MIN = 12;
const PUPIL_HOLE_MAX_AREA_MAX = 800;
const PUPIL_FIT_MIN_CONFIDENCE = 0.38;
const PUPIL_FIT_FULL_CONFIDENCE = 0.72;
const PUPIL_PRIMARY_COMPONENT_MIN_AREA_RATIO = 0.008;
// Fill larger enclosed dark regions after closing. Border-connected black
// pixels are still preserved as exterior background by the 4-neighbour
// connected-component labelling step.

let openCvReadyPromise = null;
const scratchCanvas = document.createElement('canvas');
const scratchContext = scratchCanvas.getContext('2d', { willReadFrequently: true });

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function pointToPixel(point, width, height) {
  return { x: point.x * width, y: point.y * height };
}

function pointDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function maxPairwiseDistance(points) {
  let maximum = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      maximum = Math.max(maximum, pointDistance(points[i], points[j]));
    }
  }
  return maximum;
}

function normalizeOpenCvCandidate(candidate) {
  if (candidate?.Mat) return Promise.resolve(candidate);
  if (candidate && typeof candidate.then === 'function') {
    return candidate.then((resolved) => {
      if (!resolved?.Mat) throw new Error('OpenCV.js 런타임 객체가 올바르지 않습니다.');
      window.cv = resolved;
      return resolved;
    });
  }
  return null;
}

async function waitForOpenCv(timeoutMs = 25_000) {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    const candidate = normalizeOpenCvCandidate(window.cv);
    if (candidate) {
      try {
        return await candidate;
      } catch {
        // Runtime may still be starting. Continue polling.
      }
    }
    await delay(60);
  }
  throw new Error('OpenCV.js 초기화 시간이 초과되었습니다.');
}

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const existing = [...document.scripts].find((script) => script.dataset.opencvSource === url);
    if (existing) {
      if (existing.dataset.loaded === 'true') resolve();
      else {
        existing.addEventListener('load', resolve, { once: true });
        existing.addEventListener('error', reject, { once: true });
      }
      return;
    }

    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.dataset.opencvSource = url;
    script.addEventListener('load', () => {
      script.dataset.loaded = 'true';
      resolve();
    }, { once: true });
    script.addEventListener('error', () => reject(new Error(`OpenCV.js 로드 실패: ${url}`)), { once: true });
    document.head.append(script);
  });
}

export function ensureOpenCvReady(urls = DEFAULT_OPENCV_URLS) {
  if (window.cv?.Mat) return Promise.resolve(window.cv);
  if (openCvReadyPromise) return openCvReadyPromise;

  openCvReadyPromise = (async () => {
    let lastError = null;
    for (const url of urls) {
      try {
        await loadScript(url);
        return await waitForOpenCv(url.includes('opencv/opencv.js') && !url.startsWith('http') ? 5_000 : 25_000);
      } catch (error) {
        lastError = error;
        console.warn(error);
      }
    }
    throw lastError || new Error('OpenCV.js를 불러오지 못했습니다.');
  })();

  return openCvReadyPromise;
}

export function calculateEyeRegion(landmarks, side, width, height, {
  paddingXRatio = 0.35,
  paddingYRatio = 0.30,
} = {}) {
  const definition = EYE_DEFINITIONS[side];
  if (!definition) throw new Error(`알 수 없는 눈 방향: ${side}`);

  const indices = [
    definition.outer,
    definition.inner,
    definition.upper,
    definition.lower,
    definition.irisCenter,
    ...definition.irisBoundary,
  ];
  const points = indices.map((index) => pointToPixel(landmarks[index], width, height));
  const outer = pointToPixel(landmarks[definition.outer], width, height);
  const inner = pointToPixel(landmarks[definition.inner], width, height);
  const eyeWidth = Math.max(1, pointDistance(outer, inner));
  const padX = eyeWidth * paddingXRatio;
  const padY = eyeWidth * paddingYRatio;
  const minX = Math.min(...points.map((point) => point.x)) - padX;
  const maxX = Math.max(...points.map((point) => point.x)) + padX;
  const minY = Math.min(...points.map((point) => point.y)) - padY;
  const maxY = Math.max(...points.map((point) => point.y)) + padY;
  const x = Math.floor(clamp(minX, 0, width - 2));
  const y = Math.floor(clamp(minY, 0, height - 2));
  const regionWidth = Math.max(2, Math.ceil(clamp(maxX, x + 2, width) - x));
  const regionHeight = Math.max(2, Math.ceil(clamp(maxY, y + 2, height) - y));
  const irisCenter = pointToPixel(landmarks[definition.irisCenter], width, height);
  const irisBoundary = definition.irisBoundary.map((index) => pointToPixel(landmarks[index], width, height));

  return {
    side,
    x,
    y,
    width: regionWidth,
    height: regionHeight,
    eyeWidth,
    irisCenter,
    irisBoundary,
    irisDiameter: maxPairwiseDistance(irisBoundary),
  };
}

function drawRegion(source, region, targetWidth) {
  const width = Math.max(80, Math.round(targetWidth));
  const height = Math.max(40, Math.round(region.height * width / region.width));
  scratchCanvas.width = width;
  scratchCanvas.height = height;
  scratchContext.clearRect(0, 0, width, height);
  scratchContext.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    width,
    height,
  );
  return {
    canvas: scratchCanvas,
    context: scratchContext,
    scaleX: width / region.width,
    scaleY: height / region.height,
  };
}

function grayscaleFromImageData(imageData) {
  const { data, width, height } = imageData;
  const gray = new Uint8Array(width * height);
  let sum = 0;
  let underexposed = 0;
  let overexposed = 0;
  for (let sourceIndex = 0, grayIndex = 0; sourceIndex < data.length; sourceIndex += 4, grayIndex += 1) {
    const value = Math.round(0.299 * data[sourceIndex] + 0.587 * data[sourceIndex + 1] + 0.114 * data[sourceIndex + 2]);
    gray[grayIndex] = value;
    sum += value;
    if (value < 20) underexposed += 1;
    if (value > 245) overexposed += 1;
  }
  const count = gray.length;
  return {
    gray,
    width,
    height,
    brightness: sum / count,
    underexposedRatio: underexposed / count,
    overexposedRatio: overexposed / count,
  };
}

export function calculateLaplacianVariance(gray, width, height) {
  if (!gray || width < 3 || height < 3) return 0;
  let sum = 0;
  let sumSquares = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    const row = y * width;
    for (let x = 1; x < width - 1; x += 1) {
      const index = row + x;
      const laplacian = gray[index - width] + gray[index + width]
        + gray[index - 1] + gray[index + 1] - 4 * gray[index];
      sum += laplacian;
      sumSquares += laplacian * laplacian;
      count += 1;
    }
  }
  if (!count) return 0;
  const mean = sum / count;
  return Math.max(0, sumSquares / count - mean * mean);
}

function measureSingleEyeImageQuality(source, landmarks, side, width, height) {
  const region = calculateEyeRegion(landmarks, side, width, height);
  const crop = drawRegion(source, region, 180);
  const imageData = crop.context.getImageData(0, 0, crop.canvas.width, crop.canvas.height);
  const grayData = grayscaleFromImageData(imageData);
  return {
    side,
    region,
    sharpness: calculateLaplacianVariance(grayData.gray, grayData.width, grayData.height),
    brightness: grayData.brightness,
    underexposedRatio: grayData.underexposedRatio,
    overexposedRatio: grayData.overexposedRatio,
  };
}

export function measureEyeImageQuality(source, landmarks, width, height) {
  const right = measureSingleEyeImageQuality(source, landmarks, 'right', width, height);
  const left = measureSingleEyeImageQuality(source, landmarks, 'left', width, height);
  return {
    right,
    left,
    minSharpness: Math.min(right.sharpness, left.sharpness),
    meanSharpness: (right.sharpness + left.sharpness) / 2,
    minBrightness: Math.min(right.brightness, left.brightness),
    maxBrightness: Math.max(right.brightness, left.brightness),
    meanBrightness: (right.brightness + left.brightness) / 2,
    maxUnderexposedRatio: Math.max(right.underexposedRatio, left.underexposedRatio),
    maxOverexposedRatio: Math.max(right.overexposedRatio, left.overexposedRatio),
  };
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.round(clamp(fraction, 0, 1) * (sorted.length - 1));
  return sorted[index];
}

function meanEllipseAndRing(gray, width, height, ellipse, irisCenter, irisRadius) {
  const angle = -(ellipse.angle || 0) * Math.PI / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const rx = Math.max(1, ellipse.width / 2);
  const ry = Math.max(1, ellipse.height / 2);
  let insideSum = 0;
  let insideCount = 0;
  let ringSum = 0;
  let ringCount = 0;

  const xStart = Math.max(0, Math.floor(irisCenter.x - irisRadius));
  const xEnd = Math.min(width - 1, Math.ceil(irisCenter.x + irisRadius));
  const yStart = Math.max(0, Math.floor(irisCenter.y - irisRadius));
  const yEnd = Math.min(height - 1, Math.ceil(irisCenter.y + irisRadius));

  for (let y = yStart; y <= yEnd; y += 1) {
    for (let x = xStart; x <= xEnd; x += 1) {
      const irisDistance = Math.hypot(x - irisCenter.x, y - irisCenter.y);
      if (irisDistance > irisRadius) continue;
      const dx = x - ellipse.x;
      const dy = y - ellipse.y;
      const rotatedX = dx * cos - dy * sin;
      const rotatedY = dx * sin + dy * cos;
      const normalized = (rotatedX * rotatedX) / (rx * rx) + (rotatedY * rotatedY) / (ry * ry);
      const value = gray[y * width + x];
      if (normalized <= 1) {
        insideSum += value;
        insideCount += 1;
      } else if (normalized <= 2.3) {
        ringSum += value;
        ringCount += 1;
      }
    }
  }

  return {
    insideMean: insideCount ? insideSum / insideCount : 255,
    ringMean: ringCount ? ringSum / ringCount : 255,
  };
}

export function selectRefinedPupilCenter(mediaPipeCenter, detectedCenter, confidence) {
  if (!detectedCenter || !Number.isFinite(confidence) || confidence < PUPIL_FIT_MIN_CONFIDENCE) {
    return { center: { ...mediaPipeCenter }, source: 'mediapipe', detectedWeight: 0 };
  }
  if (confidence >= PUPIL_FIT_FULL_CONFIDENCE) {
    return { center: { ...detectedCenter }, source: 'opencv', detectedWeight: 1 };
  }
  const progress = clamp(
    (confidence - PUPIL_FIT_MIN_CONFIDENCE)
      / (PUPIL_FIT_FULL_CONFIDENCE - PUPIL_FIT_MIN_CONFIDENCE),
    0,
    1,
  );
  const detectedWeight = 0.62 + progress * 0.28;
  return {
    center: {
      x: detectedCenter.x * detectedWeight + mediaPipeCenter.x * (1 - detectedWeight),
      y: detectedCenter.y * detectedWeight + mediaPipeCenter.y * (1 - detectedWeight),
    },
    source: 'blended',
    detectedWeight,
  };
}

function gammaCorrect(cv, source, gamma = 0.85) {
  if (typeof cv.LUT !== 'function') return source.clone();
  const lut = new cv.Mat(1, 256, cv.CV_8UC1);
  for (let index = 0; index < 256; index += 1) {
    lut.data[index] = clamp(Math.round(255 * ((index / 255) ** gamma)), 0, 255);
  }
  const output = new cv.Mat();
  cv.LUT(source, lut, output);
  lut.delete();
  return output;
}

function collectCircularValues(mat, center, radius) {
  const values = [];
  const pixels = mat.data;
  const xStart = Math.max(0, Math.floor(center.x - radius));
  const xEnd = Math.min(mat.cols - 1, Math.ceil(center.x + radius));
  const yStart = Math.max(0, Math.floor(center.y - radius));
  const yEnd = Math.min(mat.rows - 1, Math.ceil(center.y + radius));

  for (let y = yStart; y <= yEnd; y += 1) {
    for (let x = xStart; x <= xEnd; x += 1) {
      if (Math.hypot(x - center.x, y - center.y) <= radius) {
        values.push(pixels[y * mat.cols + x]);
      }
    }
  }
  return values;
}

function pivotContrast(cv, source, pivot, softness = PUPIL_TONE_SOFTNESS) {
  if (typeof cv.LUT !== 'function') return source.clone();
  const safeSoftness = Math.max(1, softness);
  const lut = new cv.Mat(1, 256, cv.CV_8UC1);
  for (let index = 0; index < 256; index += 1) {
    // Sigmoid contrast: the pivot maps near mid-gray. Values increasingly
    // farther below/above it converge smoothly toward black/white.
    const mapped = 255 / (1 + Math.exp(-(index - pivot) / safeSoftness));
    lut.data[index] = clamp(Math.round(mapped), 0, 255);
  }
  const output = new cv.Mat();
  cv.LUT(source, lut, output);
  lut.delete();
  return output;
}

function cloneCanvas(sourceCanvas) {
  const canvas = document.createElement('canvas');
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  canvas.getContext('2d').drawImage(sourceCanvas, 0, 0);
  return canvas;
}

function matToCanvas(cv, mat) {
  const canvas = document.createElement('canvas');
  canvas.width = mat.cols;
  canvas.height = mat.rows;
  cv.imshow(canvas, mat);
  return canvas;
}

function drawDebugOverlay(cropCanvas, {
  localIrisCenter,
  localIrisRadius,
  mediaPipeCenter,
  detectedCenter,
  finalCenter,
  ellipse,
  scaleX,
  scaleY,
  region,
  confidence,
  source,
}) {
  const canvas = cloneCanvas(cropCanvas);
  const context = canvas.getContext('2d');
  const toLocal = (point) => point ? ({
    x: (point.x - region.x) * scaleX,
    y: (point.y - region.y) * scaleY,
  }) : null;
  const localMediaPipe = toLocal(mediaPipeCenter);
  const localDetected = toLocal(detectedCenter);
  const localFinal = toLocal(finalCenter);

  context.save();
  context.lineWidth = Math.max(2, canvas.width / 180);
  context.strokeStyle = '#f79009';
  context.beginPath();
  context.arc(localIrisCenter.x, localIrisCenter.y, localIrisRadius, 0, Math.PI * 2);
  context.stroke();

  const drawPoint = (point, color, radius) => {
    if (!point) return;
    context.fillStyle = color;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
  };

  drawPoint(localMediaPipe, '#ffffff', Math.max(3, canvas.width / 85));
  drawPoint(localDetected, '#7f56d9', Math.max(3, canvas.width / 75));
  drawPoint(localFinal, '#2e90fa', Math.max(4, canvas.width / 65));

  if (ellipse) {
    context.save();
    context.translate(ellipse.x, ellipse.y);
    context.rotate((ellipse.angle || 0) * Math.PI / 180);
    context.strokeStyle = '#7f56d9';
    context.lineWidth = Math.max(2, canvas.width / 150);
    context.beginPath();
    context.ellipse(0, 0, ellipse.width / 2, ellipse.height / 2, 0, 0, Math.PI * 2);
    context.stroke();
    context.restore();
  }

  context.font = `700 ${Math.max(12, Math.round(canvas.width / 24))}px sans-serif`;
  context.textBaseline = 'top';
  const label = `${source} · 신뢰도 ${(confidence * 100).toFixed(0)}%`;
  const textWidth = context.measureText(label).width;
  context.fillStyle = 'rgba(2,8,18,.76)';
  context.fillRect(6, 6, textWidth + 12, Math.max(22, canvas.width / 18));
  context.fillStyle = '#ffffff';
  context.fillText(label, 12, 9);
  context.restore();
  return canvas;
}


function contourGeometry(cv, contour) {
  const area = Math.max(0, cv.contourArea(contour, false));
  const perimeter = Math.max(0, cv.arcLength(contour, true));
  const moments = cv.moments(contour, false);
  let center = null;
  if (Math.abs(moments.m00) > 1e-6) {
    center = {
      x: moments.m10 / moments.m00,
      y: moments.m01 / moments.m00,
    };
  } else {
    const rect = cv.boundingRect(contour);
    center = {
      x: rect.x + rect.width / 2,
      y: rect.y + rect.height / 2,
    };
  }
  return {
    area,
    perimeter,
    center,
    circularity: perimeter > 0
      ? clamp(4 * Math.PI * area / (perimeter * perimeter), 0, 1)
      : 0,
  };
}

function selectPrimaryContour(cv, contours, localIrisCenter, irisArea, localIrisRadius) {
  let best = null;
  for (let index = 0; index < contours.size(); index += 1) {
    const contour = contours.get(index);
    try {
      const geometry = contourGeometry(cv, contour);
      const areaRatio = geometry.area / Math.max(1, irisArea);
      if (areaRatio < PUPIL_PRIMARY_COMPONENT_MIN_AREA_RATIO || areaRatio > 1.12) continue;
      const centerDistanceRatio = Math.hypot(
        geometry.center.x - localIrisCenter.x,
        geometry.center.y - localIrisCenter.y,
      ) / Math.max(1, localIrisRadius);
      if (centerDistanceRatio > 1.05) continue;

      // The component must be substantial and close to the MediaPipe iris
      // center. This prevents eyelashes or isolated reflections from winning.
      const proximity = clamp(1 - centerDistanceRatio / 1.05, 0, 1);
      const componentScore = geometry.area * (0.30 + proximity * 0.70);
      if (!best || componentScore > best.componentScore) {
        best = {
          index,
          componentScore,
          areaRatio,
          centerDistanceRatio,
          ...geometry,
        };
      }
    } finally {
      contour.delete();
    }
  }
  return best;
}

function buildFitCandidates(cv, contour, geometry) {
  const candidates = [];
  const equivalentRadius = Math.sqrt(Math.max(1, geometry.area) / Math.PI);
  candidates.push({
    type: 'equivalent-circle',
    x: geometry.center.x,
    y: geometry.center.y,
    width: equivalentRadius * 2,
    height: equivalentRadius * 2,
    angle: 0,
  });

  if (typeof cv.minEnclosingCircle === 'function') {
    try {
      const enclosing = cv.minEnclosingCircle(contour);
      if (enclosing?.center && Number.isFinite(enclosing.radius) && enclosing.radius > 0) {
        candidates.push({
          type: 'enclosing-circle',
          x: enclosing.center.x,
          y: enclosing.center.y,
          width: enclosing.radius * 2,
          height: enclosing.radius * 2,
          angle: 0,
        });
      }
    } catch (error) {
      console.debug('minEnclosingCircle unavailable for this contour.', error);
    }
  }

  if (contour.rows >= 5) {
    const hull = new cv.Mat();
    try {
      cv.convexHull(contour, hull, false, true);
      if (hull.rows >= 5) {
        const rotatedRect = cv.fitEllipse(hull);
        candidates.push({
          type: 'ellipse',
          x: rotatedRect.center.x,
          y: rotatedRect.center.y,
          width: rotatedRect.size.width,
          height: rotatedRect.size.height,
          angle: rotatedRect.angle,
        });
      }
    } finally {
      hull.delete();
    }
  }

  return candidates;
}

function drawCandidateMask(cv, rows, cols, candidate) {
  const shapeMask = cv.Mat.zeros(rows, cols, cv.CV_8UC1);
  const center = new cv.Point(Math.round(candidate.x), Math.round(candidate.y));
  if (candidate.type.endsWith('circle')) {
    cv.circle(
      shapeMask,
      center,
      Math.max(1, Math.round(candidate.width / 2)),
      new cv.Scalar(255),
      -1,
    );
  } else {
    cv.ellipse(
      shapeMask,
      center,
      new cv.Size(
        Math.max(1, Math.round(candidate.width / 2)),
        Math.max(1, Math.round(candidate.height / 2)),
      ),
      candidate.angle || 0,
      0,
      360,
      new cv.Scalar(255),
      -1,
    );
  }
  return shapeMask;
}

function evaluateFitCandidate(
  cv,
  candidate,
  componentMask,
  blurred,
  localIrisCenter,
  localIrisRadius,
  localIrisDiameter,
  contourGeometryResult,
) {
  const shapeMask = drawCandidateMask(cv, componentMask.rows, componentMask.cols, candidate);
  const intersection = new cv.Mat();
  const union = new cv.Mat();
  try {
    cv.bitwise_and(componentMask, shapeMask, intersection);
    cv.bitwise_or(componentMask, shapeMask, union);
    const intersectionArea = cv.countNonZero(intersection);
    const unionArea = cv.countNonZero(union);
    const componentArea = cv.countNonZero(componentMask);
    const shapeArea = cv.countNonZero(shapeMask);
    const iou = unionArea > 0 ? intersectionArea / unionArea : 0;
    const coverage = componentArea > 0 ? intersectionArea / componentArea : 0;
    const precision = shapeArea > 0 ? intersectionArea / shapeArea : 0;
    const centerDistanceRatio = Math.hypot(
      candidate.x - localIrisCenter.x,
      candidate.y - localIrisCenter.y,
    ) / Math.max(1, localIrisRadius);
    const majorAxis = Math.max(candidate.width, candidate.height);
    const minorAxis = Math.min(candidate.width, candidate.height);
    const axisRatio = minorAxis / Math.max(1, majorAxis);
    const diameterRatio = majorAxis / Math.max(1, localIrisDiameter);
    const fitScore = scorePupilFitMetrics({
      iou,
      coverage,
      precision,
      centerDistanceRatio,
      axisRatio,
      diameterRatio,
    });
    const intensity = meanEllipseAndRing(
      blurred.data,
      blurred.cols,
      blurred.rows,
      candidate,
      localIrisCenter,
      localIrisRadius,
    );
    const contrast = clamp((intensity.ringMean - intensity.insideMean) / 70, 0, 1);
    const confidence = clamp(
      Math.max(fitScore, fitScore * 0.90 + contrast * 0.10),
      0,
      1,
    );

    return {
      ...candidate,
      score: fitScore,
      confidence,
      iou,
      coverage,
      precision,
      centerDistanceRatio,
      axisRatio,
      diameterRatio,
      contrast,
      circularity: contourGeometryResult.circularity,
      areaRatio: contourGeometryResult.areaRatio,
    };
  } finally {
    shapeMask.delete();
    intersection.delete();
    union.delete();
  }
}

function buildFallbackDebug(source, landmarks, side, width, height, reason) {
  const region = calculateEyeRegion(landmarks, side, width, height);
  const targetWidth = clamp(Math.round(region.width * 4), 220, 480);
  const crop = drawRegion(source, region, targetWidth);
  const definition = EYE_DEFINITIONS[side];
  const mediaPipeCenter = pointToPixel(landmarks[definition.irisCenter], width, height);
  const localCenter = {
    x: (mediaPipeCenter.x - region.x) * crop.scaleX,
    y: (mediaPipeCenter.y - region.y) * crop.scaleY,
  };
  const overlay = cloneCanvas(crop.canvas);
  const context = overlay.getContext('2d');
  context.fillStyle = '#ffffff';
  context.beginPath();
  context.arc(localCenter.x, localCenter.y, Math.max(4, overlay.width / 65), 0, Math.PI * 2);
  context.fill();
  return {
    darkThreshold: null,
    targetWidth,
    reason,
    stages: [
      {
        key: 'crop',
        label: '1. 원본 눈 crop',
        description: 'MediaPipe 눈 랜드마크 기준으로 고해상도 원본에서 잘라낸 영역',
        canvas: cloneCanvas(crop.canvas),
      },
      {
        key: 'fallback',
        label: '2. MediaPipe fallback',
        description: `OpenCV 정밀 검출을 사용하지 못해 흰 점의 홍채 중심을 사용함 · ${reason}`,
        canvas: overlay,
      },
    ],
  };
}

function detectPupilWithOpenCv(cv, source, landmarks, side, width, height, { includeDebug = false } = {}) {
  const region = calculateEyeRegion(landmarks, side, width, height);
  const targetWidth = clamp(Math.round(region.width * 4), 220, 480);
  const crop = drawRegion(source, region, targetWidth);
  const definition = EYE_DEFINITIONS[side];
  const mediaPipeCenter = pointToPixel(landmarks[definition.irisCenter], width, height);
  const scaleX = crop.scaleX;
  const scaleY = crop.scaleY;
  const localIrisCenter = {
    x: (mediaPipeCenter.x - region.x) * scaleX,
    y: (mediaPipeCenter.y - region.y) * scaleY,
  };
  const localIrisDiameter = region.irisDiameter * ((scaleX + scaleY) / 2);
  const localIrisRadius = Math.max(5, localIrisDiameter * 0.55);

  let sourceMat;
  let gray;
  let gamma;
  let enhanced;
  let blurred;
  let mask;
  let binary;
  let maskedBinary;
  let opened;
  let closed;
  let holeFilled;
  let kernelSmall;
  let kernelLarge;
  let contours;
  let hierarchy;

  try {
    sourceMat = cv.imread(crop.canvas);
    gray = new cv.Mat();
    cv.cvtColor(sourceMat, gray, cv.COLOR_RGBA2GRAY);
    gamma = gammaCorrect(cv, gray, 0.85);

    // Replace global histogram equalization, which can over-darken eyelid
    // shadows, with a dark-side adaptive pivot and sigmoid contrast curve.
    const gammaIrisValues = collectCircularValues(gamma, localIrisCenter, localIrisRadius);
    const tonePivot = clamp(
      percentile(gammaIrisValues, PUPIL_TONE_PIVOT_PERCENTILE),
      8,
      110,
    );
    enhanced = pivotContrast(cv, gamma, tonePivot, PUPIL_TONE_SOFTNESS);

    blurred = new cv.Mat();
    cv.GaussianBlur(enhanced, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    mask = cv.Mat.zeros(blurred.rows, blurred.cols, cv.CV_8UC1);
    cv.circle(
      mask,
      new cv.Point(Math.round(localIrisCenter.x), Math.round(localIrisCenter.y)),
      Math.round(localIrisRadius),
      new cv.Scalar(255),
      -1,
    );

    const irisValues = collectCircularValues(blurred, localIrisCenter, localIrisRadius);
    const darkThreshold = clamp(
      percentile(irisValues, PUPIL_THRESHOLD_PERCENTILE) + PUPIL_THRESHOLD_OFFSET,
      PUPIL_THRESHOLD_MIN,
      PUPIL_THRESHOLD_MAX,
    );

    binary = new cv.Mat();
    cv.threshold(blurred, binary, darkThreshold, 255, cv.THRESH_BINARY_INV);
    maskedBinary = new cv.Mat();
    cv.bitwise_and(binary, mask, maskedBinary);
    kernelSmall = cv.getStructuringElement(
      cv.MORPH_ELLIPSE,
      new cv.Size(PUPIL_OPEN_KERNEL_SIZE, PUPIL_OPEN_KERNEL_SIZE),
    );
    kernelLarge = cv.getStructuringElement(
      cv.MORPH_ELLIPSE,
      new cv.Size(PUPIL_CLOSE_KERNEL_SIZE, PUPIL_CLOSE_KERNEL_SIZE),
    );
    opened = new cv.Mat();
    closed = new cv.Mat();
    cv.morphologyEx(maskedBinary, opened, cv.MORPH_OPEN, kernelSmall);
    cv.morphologyEx(opened, closed, cv.MORPH_CLOSE, kernelLarge);

    const irisArea = Math.PI * localIrisRadius * localIrisRadius;
    const maxHolePixels = clamp(
      Math.round(irisArea * PUPIL_HOLE_MAX_AREA_RATIO),
      PUPIL_HOLE_MAX_AREA_MIN,
      PUPIL_HOLE_MAX_AREA_MAX,
    );
    const holeFill = fillSmallBlackHoles4Connected(
      closed.data,
      closed.cols,
      closed.rows,
      maxHolePixels,
    );
    holeFilled = closed.clone();
    holeFilled.data.set(holeFill.data);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(holeFilled, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

    const primary = selectPrimaryContour(
      cv,
      contours,
      localIrisCenter,
      irisArea,
      localIrisRadius,
    );
    let componentMask = null;
    let selectedContour = null;
    let fitSelection = { best: null, accepted: false, reason: 'no-component' };
    let displayCandidate = null;
    let best = null;

    if (primary) {
      componentMask = cv.Mat.zeros(holeFilled.rows, holeFilled.cols, cv.CV_8UC1);
      cv.drawContours(
        componentMask,
        contours,
        primary.index,
        new cv.Scalar(255),
        -1,
      );
      selectedContour = contours.get(primary.index);
      const candidates = buildFitCandidates(cv, selectedContour, primary);
      const evaluatedCandidates = candidates.map((candidate) => evaluateFitCandidate(
        cv,
        candidate,
        componentMask,
        blurred,
        localIrisCenter,
        localIrisRadius,
        localIrisDiameter,
        primary,
      ));
      fitSelection = chooseBestPupilFit(evaluatedCandidates);
      displayCandidate = fitSelection.best;

      if (fitSelection.accepted && fitSelection.best) {
        const fitted = fitSelection.best;
        const detectedCenter = {
          x: region.x + fitted.x / scaleX,
          y: region.y + fitted.y / scaleY,
        };
        best = {
          ...fitted,
          detectedCenter,
          localEllipse: {
            x: fitted.x,
            y: fitted.y,
            width: fitted.width,
            height: fitted.height,
            angle: fitted.angle || 0,
          },
          ellipseGlobal: {
            x: detectedCenter.x,
            y: detectedCenter.y,
            width: fitted.width / scaleX,
            height: fitted.height / scaleY,
            angle: fitted.angle || 0,
          },
          fitType: fitted.type,
          componentScore: primary.componentScore,
          accepted: true,
        };
      }
    }

    try { selectedContour?.delete(); } catch { /* Ignore OpenCV cleanup failures. */ }
    selectedContour = null;
    try { componentMask?.delete(); } catch { /* Ignore OpenCV cleanup failures. */ }
    componentMask = null;

    const selected = selectRefinedPupilCenter(mediaPipeCenter, best?.detectedCenter, best?.confidence ?? 0);
    let debug = null;
    if (includeDebug) {
      const fitTypeLabels = {
        'equivalent-circle': '중심·면적 원',
        'enclosing-circle': '외접원',
        ellipse: '타원',
      };
      const shownFit = best ?? displayCandidate;
      const shownDetectedCenter = shownFit ? {
        x: region.x + shownFit.x / scaleX,
        y: region.y + shownFit.y / scaleY,
      } : null;
      const shownEllipse = shownFit ? {
        x: shownFit.x,
        y: shownFit.y,
        width: shownFit.width,
        height: shownFit.height,
        angle: shownFit.angle || 0,
      } : null;
      const fitLabel = shownFit ? fitTypeLabels[shownFit.type] || shownFit.type : '후보 없음';
      const sourceLabel = selected.source === 'opencv'
        ? `OpenCV ${fitLabel}`
        : selected.source === 'blended'
          ? `OpenCV ${fitLabel}+MediaPipe 융합`
          : shownFit ? `${fitLabel} 후보 거절` : 'MediaPipe fallback';
      const finalOverlay = drawDebugOverlay(crop.canvas, {
        localIrisCenter,
        localIrisRadius,
        mediaPipeCenter,
        detectedCenter: shownDetectedCenter,
        finalCenter: selected.center,
        ellipse: shownEllipse,
        scaleX,
        scaleY,
        region,
        confidence: best?.confidence ?? displayCandidate?.confidence ?? 0,
        source: sourceLabel,
      });
      debug = {
        darkThreshold,
        tonePivot,
        toneSoftness: PUPIL_TONE_SOFTNESS,
        targetWidth,
        localIrisDiameter,
        contourCount: contours.size(),
        holeFill,
        stages: [
          {
            key: 'crop',
            label: '1. 원본 눈 crop',
            description: `원본에서 눈 주변을 잘라 ${crop.canvas.width}×${crop.canvas.height}px로 확대`,
            canvas: cloneCanvas(crop.canvas),
          },
          {
            key: 'gray',
            label: '2. 회색조 변환',
            description: 'RGB 색상 정보를 제거하고 밝기 정보만 유지',
            canvas: matToCanvas(cv, gray),
          },
          {
            key: 'gamma',
            label: '3. Gamma 보정',
            description: '어두운 갈색 홍채와 동공의 명암 차이를 강화',
            canvas: matToCanvas(cv, gamma),
          },
          {
            key: 'equalized',
            label: '4. 피벗 대비 보정',
            description: `홍채의 어두운 쪽 ${Math.round(PUPIL_TONE_PIVOT_PERCENTILE * 100)}백분위(${tonePivot})를 기준으로, 멀어질수록 검정·흰색에 가깝게 분리`,
            canvas: matToCanvas(cv, enhanced),
          },
          {
            key: 'blurred',
            label: '5. Gaussian blur',
            description: '속눈썹·센서 잡음 같은 고주파 노이즈를 완화',
            canvas: matToCanvas(cv, blurred),
          },
          {
            key: 'mask',
            label: '6. 홍채 탐색 마스크',
            description: 'MediaPipe 홍채 중심 주변만 남겨 눈꺼풀과 속눈썹 후보를 제한',
            canvas: matToCanvas(cv, mask),
          },
          {
            key: 'threshold',
            label: '7. 어두운 영역 이진화',
            description: `홍채 내부 ${Math.round(PUPIL_THRESHOLD_PERCENTILE * 100)}백분위 + ${PUPIL_THRESHOLD_OFFSET} 기준 임계값 ${darkThreshold}로 중간 정도의 어두운 픽셀까지 흰색 후보로 포함`,
            canvas: matToCanvas(cv, maskedBinary),
          },
          {
            key: 'morphology',
            label: '8. 형태학 보정',
            description: '열기·닫기 연산으로 작은 잡음을 제거하고 끊어진 동공 후보를 연결',
            canvas: matToCanvas(cv, closed),
          },
          {
            key: 'hole-fill',
            label: '9. 작은 내부 구멍 제거',
            description: `4방향 연결요소 라벨링으로 ${holeFill.filledComponentCount}개 내부 검은 영역, ${holeFill.filledPixelCount}px를 채움 · 홍채 면적의 ${Math.round(PUPIL_HOLE_MAX_AREA_RATIO * 100)}% 범위, 최대 ${maxHolePixels}px까지 채움`,
            canvas: matToCanvas(cv, holeFilled),
          },
          {
            key: 'result',
            label: '10. 원·타원 fitting 결과',
            description: displayCandidate
              ? `${displayCandidate.type} 선택 · IoU ${(displayCandidate.iou * 100).toFixed(0)}% · 적합도 ${(displayCandidate.score * 100).toFixed(0)}% · 주황=탐색영역, 흰 점=MediaPipe, 보라=OpenCV 후보, 파랑=최종 중심`
              : '유효한 fitting 후보가 없어 MediaPipe 홍채 중심을 사용',
            canvas: finalOverlay,
          },
        ],
      };
    }

    return {
      side,
      region,
      mediaPipeCenter,
      detectedCenter: best?.detectedCenter ?? null,
      finalCenter: selected.center,
      source: selected.source,
      detectedWeight: selected.detectedWeight,
      confidence: best?.confidence ?? 0,
      ellipse: best?.ellipseGlobal ?? null,
      diagnostics: best,
      debug,
    };
  } finally {
    for (const mat of [
      sourceMat, gray, gamma, enhanced, blurred, mask, binary, maskedBinary,
      opened, closed, holeFilled, kernelSmall, kernelLarge, contours, hierarchy,
    ]) {
      try { mat?.delete(); } catch { /* Ignore OpenCV cleanup failures. */ }
    }
  }
}

export function fallbackPupilResult(landmarks, side, width, height, reason = 'opencv-unavailable') {
  const definition = EYE_DEFINITIONS[side];
  const center = pointToPixel(landmarks[definition.irisCenter], width, height);
  return {
    side,
    mediaPipeCenter: center,
    detectedCenter: null,
    finalCenter: center,
    source: 'mediapipe',
    detectedWeight: 0,
    confidence: 0,
    ellipse: null,
    reason,
  };
}

export function refinePupilCenters({ source, landmarks, width, height, cv = window.cv, includeDebug = false }) {
  if (!cv?.Mat) {
    return {
      right: {
        ...fallbackPupilResult(landmarks, 'right', width, height),
        debug: includeDebug ? buildFallbackDebug(source, landmarks, 'right', width, height, 'OpenCV 사용 불가') : null,
      },
      left: {
        ...fallbackPupilResult(landmarks, 'left', width, height),
        debug: includeDebug ? buildFallbackDebug(source, landmarks, 'left', width, height, 'OpenCV 사용 불가') : null,
      },
      available: false,
      meanConfidence: 0,
      minConfidence: 0,
      fallbackCount: 2,
    };
  }

  let right;
  let left;
  try {
    right = detectPupilWithOpenCv(cv, source, landmarks, 'right', width, height, { includeDebug });
  } catch (error) {
    console.warn('Right pupil refinement failed.', error);
    right = {
      ...fallbackPupilResult(landmarks, 'right', width, height, error.message),
      debug: includeDebug ? buildFallbackDebug(source, landmarks, 'right', width, height, error.message) : null,
    };
  }
  try {
    left = detectPupilWithOpenCv(cv, source, landmarks, 'left', width, height, { includeDebug });
  } catch (error) {
    console.warn('Left pupil refinement failed.', error);
    left = {
      ...fallbackPupilResult(landmarks, 'left', width, height, error.message),
      debug: includeDebug ? buildFallbackDebug(source, landmarks, 'left', width, height, error.message) : null,
    };
  }

  const confidences = [right.confidence, left.confidence];
  return {
    right,
    left,
    available: true,
    meanConfidence: (confidences[0] + confidences[1]) / 2,
    minConfidence: Math.min(...confidences),
    fallbackCount: [right, left].filter((item) => item.source === 'mediapipe').length,
  };
}
