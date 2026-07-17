export const LANDMARKS = Object.freeze({
  rightIrisCenter: 468,
  rightIrisBoundary: [469, 470, 471, 472],
  leftIrisCenter: 473,
  leftIrisBoundary: [474, 475, 476, 477],
});

export function toPixel(landmark, width, height) {
  return {
    x: landmark.x * width,
    y: landmark.y * height,
  };
}

export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function maxPairwiseDistance(points) {
  let maximum = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      maximum = Math.max(maximum, distance(points[i], points[j]));
    }
  }
  return maximum;
}

export function extractPoseDegrees(matrix) {
  if (!matrix?.data || matrix.data.length < 16) {
    return { yaw: Number.NaN, pitch: Number.NaN, roll: Number.NaN };
  }

  // MediaPipe Matrix.data is treated as a flattened 4x4 row-major matrix.
  // Rotation convention: R = Rz(roll) * Ry(yaw) * Rx(pitch).
  const m = matrix.data;
  const r00 = m[0];
  const r10 = m[4];
  const r20 = m[8];
  const r21 = m[9];
  const r22 = m[10];
  const sy = Math.hypot(r00, r10);
  const singular = sy < 1e-6;

  let pitch;
  let yaw;
  let roll;

  if (!singular) {
    pitch = Math.atan2(r21, r22);
    yaw = Math.atan2(-r20, sy);
    roll = Math.atan2(r10, r00);
  } else {
    pitch = Math.atan2(-m[6], m[5]);
    yaw = Math.atan2(-r20, sy);
    roll = 0;
  }

  const degrees = 180 / Math.PI;
  return {
    yaw: yaw * degrees,
    pitch: pitch * degrees,
    roll: roll * degrees,
  };
}

export function measurePd({ landmarks, matrix, width, height, irisReferenceMm = 11.7 }) {
  if (!Array.isArray(landmarks) || landmarks.length < 478) {
    throw new Error('478개 얼굴 랜드마크를 찾지 못했습니다.');
  }
  if (!(width > 0 && height > 0 && irisReferenceMm > 0)) {
    throw new Error('잘못된 이미지 크기 또는 홍채 기준값입니다.');
  }

  const rightCenter = toPixel(landmarks[LANDMARKS.rightIrisCenter], width, height);
  const leftCenter = toPixel(landmarks[LANDMARKS.leftIrisCenter], width, height);
  const rightBoundary = LANDMARKS.rightIrisBoundary.map((index) => toPixel(landmarks[index], width, height));
  const leftBoundary = LANDMARKS.leftIrisBoundary.map((index) => toPixel(landmarks[index], width, height));

  const pdPx = distance(leftCenter, rightCenter);
  const rightIrisPx = maxPairwiseDistance(rightBoundary);
  const leftIrisPx = maxPairwiseDistance(leftBoundary);
  const meanIrisPx = (leftIrisPx + rightIrisPx) / 2;
  const irisDifferenceRatio = Math.abs(leftIrisPx - rightIrisPx) / meanIrisPx;

  if (!(meanIrisPx > 0)) {
    throw new Error('홍채 지름 계산에 실패했습니다.');
  }

  const mmPerPixel = irisReferenceMm / meanIrisPx;
  const pdMm = pdPx * mmPerPixel;

  return {
    pdPx,
    pdMm,
    mmPerPixel,
    leftIrisPx,
    rightIrisPx,
    meanIrisPx,
    irisDifferenceRatio,
    pose: extractPoseDegrees(matrix),
    points: {
      leftCenter,
      rightCenter,
      leftBoundary,
      rightBoundary,
    },
  };
}

export function evaluateQuality(measurement, config) {
  const reasons = [];
  const warnings = [];
  let score = 100;

  const { pose, irisDifferenceRatio, meanIrisPx, pdMm } = measurement;
  const poseAvailable = [pose.yaw, pose.pitch, pose.roll].every(Number.isFinite);

  if (!poseAvailable) {
    warnings.push('얼굴 변환행렬을 읽지 못해 자세 검사가 제한됩니다.');
    score -= 15;
  } else {
    const poseChecks = [
      ['Yaw', Math.abs(pose.yaw), config.maxYaw],
      ['Pitch', Math.abs(pose.pitch), config.maxPitch],
      ['Roll', Math.abs(pose.roll), config.maxRoll],
    ];
    for (const [name, value, limit] of poseChecks) {
      score -= Math.min(18, (value / limit) * 10);
      if (value > limit) reasons.push(`${name} ${value.toFixed(1)}°: 정면 기준 초과`);
    }
  }

  score -= Math.min(25, irisDifferenceRatio * 120);
  if (irisDifferenceRatio > config.maxIrisDifferenceRatio) {
    reasons.push(`좌우 홍채 지름 차이 ${(irisDifferenceRatio * 100).toFixed(1)}%`);
  }

  if (meanIrisPx < config.minIrisPixels) {
    reasons.push(`홍채가 너무 작게 촬영됨 (${meanIrisPx.toFixed(1)}px)`);
    score -= 25;
  }

  if (pdMm < 45 || pdMm > 85) {
    reasons.push(`추정 PD가 비정상 범위 (${pdMm.toFixed(1)}mm)`);
    score -= 25;
  } else if (pdMm < 50 || pdMm > 80) {
    warnings.push('추정 PD가 일반적인 성인 범위의 가장자리에 있습니다.');
    score -= 8;
  }

  score = Math.max(0, Math.round(score));
  const accepted = reasons.length === 0;
  return { accepted, reasons, warnings, score };
}
