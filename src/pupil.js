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
  if (!detectedCenter || !Number.isFinite(confidence) || confidence < 0.5) {
    return { center: { ...mediaPipeCenter }, source: 'mediapipe', detectedWeight: 0 };
  }
  if (confidence >= 0.8) {
    return { center: { ...detectedCenter }, source: 'opencv', detectedWeight: 1 };
  }
  const detectedWeight = 0.7;
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

function detectPupilWithOpenCv(cv, source, landmarks, side, width, height) {
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
  let kernelSmall;
  let kernelLarge;
  let contours;
  let hierarchy;

  try {
    sourceMat = cv.imread(crop.canvas);
    gray = new cv.Mat();
    cv.cvtColor(sourceMat, gray, cv.COLOR_RGBA2GRAY);
    gamma = gammaCorrect(cv, gray, 0.85);
    enhanced = new cv.Mat();
    cv.equalizeHist(gamma, enhanced);
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

    const irisValues = [];
    const grayPixels = blurred.data;
    const xStart = Math.max(0, Math.floor(localIrisCenter.x - localIrisRadius));
    const xEnd = Math.min(blurred.cols - 1, Math.ceil(localIrisCenter.x + localIrisRadius));
    const yStart = Math.max(0, Math.floor(localIrisCenter.y - localIrisRadius));
    const yEnd = Math.min(blurred.rows - 1, Math.ceil(localIrisCenter.y + localIrisRadius));
    for (let y = yStart; y <= yEnd; y += 1) {
      for (let x = xStart; x <= xEnd; x += 1) {
        if (Math.hypot(x - localIrisCenter.x, y - localIrisCenter.y) <= localIrisRadius) {
          irisValues.push(grayPixels[y * blurred.cols + x]);
        }
      }
    }
    const darkThreshold = clamp(percentile(irisValues, 0.24) + 4, 12, 145);

    binary = new cv.Mat();
    cv.threshold(blurred, binary, darkThreshold, 255, cv.THRESH_BINARY_INV);
    maskedBinary = new cv.Mat();
    cv.bitwise_and(binary, mask, maskedBinary);
    kernelSmall = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    kernelLarge = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    opened = new cv.Mat();
    closed = new cv.Mat();
    cv.morphologyEx(maskedBinary, opened, cv.MORPH_OPEN, kernelSmall);
    cv.morphologyEx(opened, closed, cv.MORPH_CLOSE, kernelLarge);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const irisArea = Math.PI * localIrisRadius * localIrisRadius;
    let best = null;

    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      try {
        if (contour.rows < 5) continue;
        const area = cv.contourArea(contour, false);
        const areaRatio = area / irisArea;
        if (areaRatio < 0.018 || areaRatio > 0.60) continue;
        const perimeter = cv.arcLength(contour, true);
        if (!(perimeter > 0)) continue;
        const rotatedRect = cv.fitEllipse(contour);
        const ellipseWidth = Math.max(rotatedRect.size.width, rotatedRect.size.height);
        const ellipseHeight = Math.min(rotatedRect.size.width, rotatedRect.size.height);
        const axisRatio = ellipseHeight / Math.max(1, ellipseWidth);
        const diameterRatio = ellipseWidth / Math.max(1, localIrisDiameter);
        if (axisRatio < 0.50 || diameterRatio < 0.15 || diameterRatio > 0.80) continue;

        const centerDistanceRatio = Math.hypot(
          rotatedRect.center.x - localIrisCenter.x,
          rotatedRect.center.y - localIrisCenter.y,
        ) / localIrisRadius;
        if (centerDistanceRatio > 0.68) continue;

        const circularity = clamp(4 * Math.PI * area / (perimeter * perimeter), 0, 1);
        const ellipse = {
          x: rotatedRect.center.x,
          y: rotatedRect.center.y,
          width: rotatedRect.size.width,
          height: rotatedRect.size.height,
          angle: rotatedRect.angle,
        };
        const intensity = meanEllipseAndRing(
          blurred.data,
          blurred.cols,
          blurred.rows,
          ellipse,
          localIrisCenter,
          localIrisRadius,
        );
        const contrast = clamp((intensity.ringMean - intensity.insideMean) / 70, 0, 1);
        const centerScore = clamp(1 - centerDistanceRatio / 0.68, 0, 1);
        const diameterScore = clamp(1 - Math.abs(diameterRatio - 0.42) / 0.38, 0, 1);
        const confidence = clamp(
          circularity * 0.22
          + axisRatio * 0.20
          + contrast * 0.30
          + centerScore * 0.18
          + diameterScore * 0.10,
          0,
          1,
        );

        if (!best || confidence > best.confidence) {
          best = {
            confidence,
            areaRatio,
            axisRatio,
            circularity,
            contrast,
            diameterRatio,
            centerDistanceRatio,
            localEllipse: ellipse,
            detectedCenter: {
              x: region.x + rotatedRect.center.x / scaleX,
              y: region.y + rotatedRect.center.y / scaleY,
            },
            ellipseGlobal: {
              x: region.x + rotatedRect.center.x / scaleX,
              y: region.y + rotatedRect.center.y / scaleY,
              width: rotatedRect.size.width / scaleX,
              height: rotatedRect.size.height / scaleY,
              angle: rotatedRect.angle,
            },
          };
        }
      } finally {
        contour.delete();
      }
    }

    const selected = selectRefinedPupilCenter(mediaPipeCenter, best?.detectedCenter, best?.confidence ?? 0);
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
    };
  } finally {
    for (const mat of [
      sourceMat, gray, gamma, enhanced, blurred, mask, binary, maskedBinary,
      opened, closed, kernelSmall, kernelLarge, contours, hierarchy,
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

export function refinePupilCenters({ source, landmarks, width, height, cv = window.cv }) {
  if (!cv?.Mat) {
    return {
      right: fallbackPupilResult(landmarks, 'right', width, height),
      left: fallbackPupilResult(landmarks, 'left', width, height),
      available: false,
      meanConfidence: 0,
      minConfidence: 0,
      fallbackCount: 2,
    };
  }

  let right;
  let left;
  try {
    right = detectPupilWithOpenCv(cv, source, landmarks, 'right', width, height);
  } catch (error) {
    console.warn('Right pupil refinement failed.', error);
    right = fallbackPupilResult(landmarks, 'right', width, height, error.message);
  }
  try {
    left = detectPupilWithOpenCv(cv, source, landmarks, 'left', width, height);
  } catch (error) {
    console.warn('Left pupil refinement failed.', error);
    left = fallbackPupilResult(landmarks, 'left', width, height, error.message);
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
