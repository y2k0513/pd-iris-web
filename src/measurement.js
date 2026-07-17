export const LANDMARKS = Object.freeze({
  rightIrisCenter: 468,
  rightIrisBoundary: [469, 470, 471, 472],
  leftIrisCenter: 473,
  leftIrisBoundary: [474, 475, 476, 477],
  rightEyeOuter: 33,
  rightEyeInner: 133,
  rightEyeUpper: 159,
  rightEyeLower: 145,
  leftEyeInner: 362,
  leftEyeOuter: 263,
  leftEyeUpper: 386,
  leftEyeLower: 374,
  noseTip: 1,
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

// MediaPipe Face Landmarker의 z는 실제 mm가 아니라 x축과 비슷한 스케일의
// 상대 깊이값이다. x/y/z를 같은 계산 공간으로 옮겨 3D 비율 보정에 사용한다.
export function toRelative3D(landmark, width, height) {
  return {
    x: (landmark.x - 0.5) * width,
    y: (landmark.y - 0.5) * height,
    z: landmark.z * width,
  };
}

export function distance3D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function maxPairwiseDistance3D(points) {
  let maximum = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      maximum = Math.max(maximum, distance3D(points[i], points[j]));
    }
  }
  return maximum;
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

export function projectionFraction(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const denominator = dx * dx + dy * dy;
  if (!(denominator > 0)) return Number.NaN;
  return ((point.x - start.x) * dx + (point.y - start.y) * dy) / denominator;
}

export function extractPoseDegrees(matrix) {
  if (!matrix?.data || matrix.data.length < 16) {
    return { yaw: Number.NaN, pitch: Number.NaN, roll: Number.NaN };
  }

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

export function calculateFraming(landmarks) {
  const facePoints = landmarks.slice(0, 468);
  const xs = facePoints.map((point) => point.x).filter(Number.isFinite);
  const ys = facePoints.map((point) => point.y).filter(Number.isFinite);

  if (!xs.length || !ys.length) {
    return {
      faceWidthRatio: Number.NaN,
      faceHeightRatio: Number.NaN,
      centerOffsetX: Number.NaN,
      centerOffsetY: Number.NaN,
    };
  }

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  return {
    minX,
    maxX,
    minY,
    maxY,
    centerX,
    centerY,
    faceWidthRatio: maxX - minX,
    faceHeightRatio: maxY - minY,
    centerOffsetX: Math.abs(centerX - 0.5),
    centerOffsetY: Math.abs(centerY - 0.5),
  };
}

function calculateEyeAndPerspectiveMetrics(landmarks, width, height, points) {
  const rightOuter = toPixel(landmarks[LANDMARKS.rightEyeOuter], width, height);
  const rightInner = toPixel(landmarks[LANDMARKS.rightEyeInner], width, height);
  const rightUpper = toPixel(landmarks[LANDMARKS.rightEyeUpper], width, height);
  const rightLower = toPixel(landmarks[LANDMARKS.rightEyeLower], width, height);
  const leftInner = toPixel(landmarks[LANDMARKS.leftEyeInner], width, height);
  const leftOuter = toPixel(landmarks[LANDMARKS.leftEyeOuter], width, height);
  const leftUpper = toPixel(landmarks[LANDMARKS.leftEyeUpper], width, height);
  const leftLower = toPixel(landmarks[LANDMARKS.leftEyeLower], width, height);
  const nose = toPixel(landmarks[LANDMARKS.noseTip], width, height);

  const rightEyeWidth = distance(rightOuter, rightInner);
  const leftEyeWidth = distance(leftInner, leftOuter);
  const rightEyeOpeningRatio = distance(rightUpper, rightLower) / rightEyeWidth;
  const leftEyeOpeningRatio = distance(leftUpper, leftLower) / leftEyeWidth;

  const rightGazePosition = projectionFraction(points.rightCenter, rightOuter, rightInner);
  const leftGazePosition = projectionFraction(points.leftCenter, leftInner, leftOuter);
  const gazeOffset = Math.max(
    Math.abs(rightGazePosition - 0.5),
    Math.abs(leftGazePosition - 0.5),
  );

  const noseToRightIris = distance(nose, points.rightCenter);
  const noseToLeftIris = distance(nose, points.leftCenter);
  const noseEyeMean = (noseToRightIris + noseToLeftIris) / 2;
  const perspectiveAsymmetryRatio = Math.abs(noseToRightIris - noseToLeftIris) / noseEyeMean;
  const eyeWidthDifferenceRatio = Math.abs(rightEyeWidth - leftEyeWidth) / ((rightEyeWidth + leftEyeWidth) / 2);
  const irisDepthDifference = Math.abs(
    landmarks[LANDMARKS.leftIrisCenter].z - landmarks[LANDMARKS.rightIrisCenter].z,
  );

  return {
    rightGazePosition,
    leftGazePosition,
    gazeOffset,
    rightEyeOpeningRatio,
    leftEyeOpeningRatio,
    minEyeOpeningRatio: Math.min(rightEyeOpeningRatio, leftEyeOpeningRatio),
    perspectiveAsymmetryRatio,
    eyeWidthDifferenceRatio,
    irisDepthDifference,
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
  const points = { leftCenter, rightCenter, leftBoundary, rightBoundary };

  const pdPx = distance(leftCenter, rightCenter);
  const rightIrisPx = maxPairwiseDistance(rightBoundary);
  const leftIrisPx = maxPairwiseDistance(leftBoundary);
  const meanIrisPx = (leftIrisPx + rightIrisPx) / 2;
  const irisDifferenceRatio = Math.abs(leftIrisPx - rightIrisPx) / meanIrisPx;

  if (!(meanIrisPx > 0)) {
    throw new Error('홍채 지름 계산에 실패했습니다.');
  }

  // 2D 기준값: 기존 픽셀 비율 방식
  const mmPerPixel = irisReferenceMm / meanIrisPx;
  const pdMm2D = pdPx * mmPerPixel;

  // 3D 기준값: MediaPipe의 상대 z를 포함한 얼굴 메시에 같은 11.7mm 스케일을 적용한다.
  // 절대 깊이 센서가 아니라 상대 3D 복원이며, 2D 결과와 교차 검증한다.
  const rightCenter3D = toRelative3D(landmarks[LANDMARKS.rightIrisCenter], width, height);
  const leftCenter3D = toRelative3D(landmarks[LANDMARKS.leftIrisCenter], width, height);
  const rightBoundary3D = LANDMARKS.rightIrisBoundary.map((index) => toRelative3D(landmarks[index], width, height));
  const leftBoundary3D = LANDMARKS.leftIrisBoundary.map((index) => toRelative3D(landmarks[index], width, height));
  const pd3DUnits = distance3D(leftCenter3D, rightCenter3D);
  const rightIris3DUnits = maxPairwiseDistance3D(rightBoundary3D);
  const leftIris3DUnits = maxPairwiseDistance3D(leftBoundary3D);
  const meanIris3DUnits = (leftIris3DUnits + rightIris3DUnits) / 2;

  if (!(meanIris3DUnits > 0)) {
    throw new Error('3D 홍채 지름 계산에 실패했습니다.');
  }

  const pdMm3D = pd3DUnits * irisReferenceMm / meanIris3DUnits;
  const estimateMean = (pdMm2D + pdMm3D) / 2;
  const disagreementRatio = Math.abs(pdMm3D - pdMm2D) / estimateMean;

  // 상대 z는 유용하지만 센서 기반 metric depth는 아니다. 3D를 주값으로 두되
  // 2D 값을 보조로 섞어 홍채 z 노이즈가 최종값을 지배하지 않게 한다.
  const pdMm = pdMm3D * 0.7 + pdMm2D * 0.3;

  return {
    pdPx,
    pdMm,
    pdMm2D,
    pdMm3D,
    mmPerPixel,
    leftIrisPx,
    rightIrisPx,
    meanIrisPx,
    irisDifferenceRatio,
    depthAware: {
      pd3DUnits,
      leftIris3DUnits,
      rightIris3DUnits,
      meanIris3DUnits,
      disagreementRatio,
      fusion3DWeight: 0.7,
    },
    pose: extractPoseDegrees(matrix),
    framing: calculateFraming(landmarks),
    eyeAndPerspective: calculateEyeAndPerspectiveMetrics(landmarks, width, height, points),
    points,
  };
}

export function evaluateQuality(measurement, config) {
  const reasons = [];
  const warnings = [];
  let score = 100;

  const {
    pose,
    irisDifferenceRatio,
    meanIrisPx,
    pdMm,
    framing,
    eyeAndPerspective,
  } = measurement;
  const poseAvailable = [pose.yaw, pose.pitch, pose.roll].every(Number.isFinite);

  if (!poseAvailable) {
    if (config.requirePose) reasons.push('얼굴 자세를 계산하지 못함');
    else warnings.push('얼굴 변환행렬을 읽지 못해 자세 검사가 제한됩니다.');
    score -= 20;
  } else {
    const poseChecks = [
      ['Yaw', Math.abs(pose.yaw), config.maxYaw],
      ['Pitch', Math.abs(pose.pitch), config.maxPitch],
      ['Roll', Math.abs(pose.roll), config.maxRoll],
    ];
    for (const [name, value, limit] of poseChecks) {
      score -= Math.min(20, (value / limit) * 10);
      if (value > limit) reasons.push(`${name} ${value.toFixed(1)}°: 정면 기준 초과`);
    }
  }

  score -= Math.min(25, irisDifferenceRatio * 150);
  if (irisDifferenceRatio > config.maxIrisDifferenceRatio) {
    reasons.push(`좌우 홍채 지름 차이 ${(irisDifferenceRatio * 100).toFixed(1)}%`);
  }

  if (meanIrisPx < config.minIrisPixels) {
    reasons.push(`홍채가 너무 작게 촬영됨 (${meanIrisPx.toFixed(1)}px)`);
    score -= 25;
  }

  if (framing && Number.isFinite(framing.faceHeightRatio)) {
    if (framing.faceHeightRatio < config.minFaceHeightRatio) {
      reasons.push(`얼굴이 너무 작음 (${(framing.faceHeightRatio * 100).toFixed(0)}%)`);
      score -= 20;
    }
    if (framing.faceHeightRatio > config.maxFaceHeightRatio) {
      reasons.push(`얼굴이 너무 가까움 (${(framing.faceHeightRatio * 100).toFixed(0)}%)`);
      score -= 18;
    }
    if (framing.centerOffsetX > config.maxFaceCenterOffsetX) {
      reasons.push('얼굴을 좌우 중앙에 맞추세요');
      score -= 15;
    }
    if (framing.centerOffsetY > config.maxFaceCenterOffsetY) {
      reasons.push('얼굴을 상하 중앙에 맞추세요');
      score -= 15;
    }
  }

  if (eyeAndPerspective) {
    if (eyeAndPerspective.gazeOffset > config.maxGazeOffset) {
      reasons.push('화면이 아니라 카메라 렌즈를 바라보세요');
      score -= 18;
    }
    if (eyeAndPerspective.minEyeOpeningRatio < config.minEyeOpeningRatio) {
      reasons.push('양쪽 눈을 충분히 뜨세요');
      score -= 15;
    }
    if (eyeAndPerspective.perspectiveAsymmetryRatio > config.maxPerspectiveAsymmetryRatio) {
      reasons.push(`얼굴 원근 비대칭 ${(eyeAndPerspective.perspectiveAsymmetryRatio * 100).toFixed(1)}%`);
      score -= 20;
    }
    if (eyeAndPerspective.eyeWidthDifferenceRatio > config.maxEyeWidthDifferenceRatio) {
      reasons.push(`좌우 눈 크기 비대칭 ${(eyeAndPerspective.eyeWidthDifferenceRatio * 100).toFixed(1)}%`);
      score -= 14;
    }
    if (eyeAndPerspective.irisDepthDifference > config.maxIrisDepthDifference) {
      warnings.push('좌우 눈의 상대 깊이 차이가 큽니다.');
      score -= 8;
    }
  }

  if (measurement.depthAware) {
    const disagreement = measurement.depthAware.disagreementRatio;
    score -= Math.min(20, disagreement * 120);
    if (disagreement > config.max2D3DDisagreementRatio) {
      reasons.push(`2D/3D 추정 차이 ${(disagreement * 100).toFixed(1)}%`);
    } else if (disagreement > config.warn2D3DDisagreementRatio) {
      warnings.push(`2D/3D 추정값 차이가 ${(disagreement * 100).toFixed(1)}%입니다.`);
    }
  }

  if (pdMm < 45 || pdMm > 85) {
    reasons.push(`추정 PD가 비정상 범위 (${pdMm.toFixed(1)}mm)`);
    score -= 25;
  } else if (pdMm < 50 || pdMm > 80) {
    warnings.push('추정 PD가 일반적인 성인 범위의 가장자리에 있습니다.');
    score -= 8;
  }

  score = Math.max(0, Math.round(score));
  return { accepted: reasons.length === 0, reasons, warnings, score };
}
