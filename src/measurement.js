
export const UNIVERSAL_PD_PRIOR = Object.freeze({
  label: '공통',
  minMm: 58,
  maxMm: 70,
  centerMm: 64,
  scaleMm: 3,
});

export const SEX_PD_PRIORS = Object.freeze({
  male: UNIVERSAL_PD_PRIOR,
  female: UNIVERSAL_PD_PRIOR,
  universal: UNIVERSAL_PD_PRIOR,
});

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function median(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return Number.NaN;
  const middle = Math.floor(finite.length / 2);
  return finite.length % 2 ? finite[middle] : (finite[middle - 1] + finite[middle]) / 2;
}

// 성별 분포를 hard clipping이 아닌 soft prior로 사용한다.
// priorLoss는 분포 중심에서 반폭(scaleMm)만큼 떨어지면 1이 되는 정규화 제곱손실이다.
// 최종값은 측정값과 prior 중심의 정밀도 가중 평균이며, 촬영 품질이 낮을수록 prior 영향이 커진다.
export function applySexPdPrior({
  rawPdMm,
  sex,
  qualityScore = 100,
  strength = 0.6,
}) {
  const prior =
    UNIVERSAL_PD_PRIOR;
  if (!Number.isFinite(rawPdMm)) {
    throw new Error('유효한 PD 측정값이 필요합니다.');
  }

  const boundedStrength = clamp(Number(strength) || 0, 0, 1);
  const boundedQuality = clamp(Number(qualityScore) || 0, 0, 100) / 100;
  const normalizedDistance = (rawPdMm - prior.centerMm) / prior.scaleMm;
  const priorLoss = normalizedDistance ** 2;

  // 품질 100점에서는 측정 표준오차를 약 0.8mm, 낮은 품질에서는 최대 3.0mm로 본다.
  const measurementSigmaMm = 0.8 + (1 - boundedQuality) * 2.2;
  const measurementPrecision = 1 / (measurementSigmaMm ** 2);
  const distanceMultiplier =
    clamp(
      priorLoss,
      0,
      4,
    );

  const priorPrecision =
    boundedStrength
    * distanceMultiplier
    / (prior.scaleMm ** 2);
  const totalPrecision = measurementPrecision + priorPrecision;
  const priorWeight = totalPrecision > 0 ? priorPrecision / totalPrecision : 0;
  const adjustedPdMm = rawPdMm * (1 - priorWeight) + prior.centerMm * priorWeight;

  return {
    sex: sex || 'universal',
    label: prior.label,
    minMm: prior.minMm,
    maxMm: prior.maxMm,
    centerMm: prior.centerMm,
    scaleMm: prior.scaleMm,
    rawPdMm,
    adjustedPdMm,
    normalizedDistance,
    priorLoss,
    distanceMultiplier,
    priorWeight,
    measurementSigmaMm,
    withinTypicalRange: rawPdMm >= prior.minMm && rawPdMm <= prior.maxMm,
  };
}

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

export function measurePd({
  landmarks,
  matrix,
  width,
  height,
  irisReferenceMm = 11.7,
  centerOverrides = null,
  diameterOverrides = null,
}) {
  if (!Array.isArray(landmarks) || landmarks.length < 478) {
    throw new Error('478개 얼굴 랜드마크를 찾지 못했습니다.');
  }
  if (!(width > 0 && height > 0 && irisReferenceMm > 0)) {
    throw new Error('잘못된 이미지 크기 또는 홍채 기준값입니다.');
  }

  const mediaPipeRightCenter = toPixel(landmarks[LANDMARKS.rightIrisCenter], width, height);
  const mediaPipeLeftCenter = toPixel(landmarks[LANDMARKS.leftIrisCenter], width, height);
  const rightCenter = centerOverrides?.right && Number.isFinite(centerOverrides.right.x) && Number.isFinite(centerOverrides.right.y)
    ? { ...centerOverrides.right }
    : mediaPipeRightCenter;
  const leftCenter = centerOverrides?.left && Number.isFinite(centerOverrides.left.x) && Number.isFinite(centerOverrides.left.y)
    ? { ...centerOverrides.left }
    : mediaPipeLeftCenter;
  const rightBoundary = LANDMARKS.rightIrisBoundary.map((index) => toPixel(landmarks[index], width, height));
  const leftBoundary = LANDMARKS.leftIrisBoundary.map((index) => toPixel(landmarks[index], width, height));
  const points = {
    leftCenter,
    rightCenter,
    leftBoundary,
    rightBoundary,
    mediaPipeLeftCenter,
    mediaPipeRightCenter,
  };

  // 최종 PD 중심 사이의 픽셀 거리
  const pdPx = distance(
    leftCenter,
    rightCenter,
  );

  // MediaPipe 경계 지름은 테스트 및 실시간 프리뷰용 fallback이다.
  const mediaPipeRightIrisPx =
    maxPairwiseDistance(
      rightBoundary,
    );

  const mediaPipeLeftIrisPx =
    maxPairwiseDistance(
      leftBoundary,
    );

  // 최종 촬영 분석에서는 pupil.js의
  // 3.5% 확대된 보라색 원 지름이 전달된다.
  const rightPurpleDiameter = Number(
    diameterOverrides?.right,
  );

  const leftPurpleDiameter = Number(
    diameterOverrides?.left,
  );

  const hasPurpleDiameterOverrides =
    Number.isFinite(
      rightPurpleDiameter,
    )
    && rightPurpleDiameter > 0
    && Number.isFinite(
      leftPurpleDiameter,
    )
    && leftPurpleDiameter > 0;

  const rightIrisPx =
    hasPurpleDiameterOverrides
      ? rightPurpleDiameter
      : mediaPipeRightIrisPx;

  const leftIrisPx =
    hasPurpleDiameterOverrides
      ? leftPurpleDiameter
      : mediaPipeLeftIrisPx;

  const meanIrisPx =
    (leftIrisPx + rightIrisPx) / 2;

  const irisDifferenceRatio =
    Math.abs(
      leftIrisPx - rightIrisPx,
    ) / meanIrisPx;

  if (!(meanIrisPx > 0)) {
    throw new Error('홍채 지름 계산에 실패했습니다.');
  }

  // 2D 기준값: 기존 픽셀 비율 방식
  const mmPerPixel = irisReferenceMm / meanIrisPx;
  const pdMm2D = pdPx * mmPerPixel;

  /*
   * 2D/3D 비교는 같은 중심과 같은 기준 지름을 써야 한다.
   *
   * 최종 촬영:
   * - x/y: 보라색 원 중심
   * - z: MediaPipe 상대 깊이
   * - 기준 지름: 3.5% 확대된 보라색 원 평균 지름
   *
   * 라이브 프리뷰 및 기존 테스트:
   * - 기존 MediaPipe 상대 3D 계산 유지
   */
  let pd3DUnits;
  let rightIris3DUnits;
  let leftIris3DUnits;
  let meanIris3DUnits;
  let pdMm3D;

  if (hasPurpleDiameterOverrides) {
    const rightMediaPipe3D = toRelative3D(
      landmarks[LANDMARKS.rightIrisCenter],
      width,
      height,
    );

    const leftMediaPipe3D = toRelative3D(
      landmarks[LANDMARKS.leftIrisCenter],
      width,
      height,
    );

    /*
     * rightCenter/leftCenter는 이 경로에서
     * main.js가 전달한 보라색 원 중심이다.
     *
     * toRelative3D와 같은 좌표계가 되도록
     * 이미지 중앙을 원점으로 변경한다.
     */
    const rightPurpleCenter3D = {
      x: rightCenter.x - width / 2,
      y: rightCenter.y - height / 2,
      z: rightMediaPipe3D.z,
    };

    const leftPurpleCenter3D = {
      x: leftCenter.x - width / 2,
      y: leftCenter.y - height / 2,
      z: leftMediaPipe3D.z,
    };

    pd3DUnits = distance3D(
      leftPurpleCenter3D,
      rightPurpleCenter3D,
    );

    /*
     * 2D와 동일한 보라색 원 기준 스케일 사용.
     * rightIrisPx와 leftIrisPx는 이 경로에서
     * 3.5% 확대된 보라색 원 지름이다.
     */
    rightIris3DUnits = rightIrisPx;
    leftIris3DUnits = leftIrisPx;

    meanIris3DUnits =
      (
        leftIris3DUnits
        + rightIris3DUnits
      ) / 2;

    pdMm3D =
      pd3DUnits
      * mmPerPixel;
  } else {
    /*
     * 보라색 원이 없는 라이브 프리뷰와 단위 테스트는
     * 기존 MediaPipe 상대 3D 계산을 유지한다.
     */
    const rightCenter3D = toRelative3D(
      landmarks[LANDMARKS.rightIrisCenter],
      width,
      height,
    );

    const leftCenter3D = toRelative3D(
      landmarks[LANDMARKS.leftIrisCenter],
      width,
      height,
    );

    const rightBoundary3D =
      LANDMARKS.rightIrisBoundary.map(
        (index) => toRelative3D(
          landmarks[index],
          width,
          height,
        ),
      );

    const leftBoundary3D =
      LANDMARKS.leftIrisBoundary.map(
        (index) => toRelative3D(
          landmarks[index],
          width,
          height,
        ),
      );

    pd3DUnits = distance3D(
      leftCenter3D,
      rightCenter3D,
    );

    rightIris3DUnits =
      maxPairwiseDistance3D(
        rightBoundary3D,
      );

    leftIris3DUnits =
      maxPairwiseDistance3D(
        leftBoundary3D,
      );

    meanIris3DUnits =
      (
        leftIris3DUnits
        + rightIris3DUnits
      ) / 2;

    if (!(meanIris3DUnits > 0)) {
      throw new Error(
        '3D 홍채 지름 계산에 실패했습니다.',
      );
    }

    pdMm3D =
      pd3DUnits
      * irisReferenceMm
      / meanIris3DUnits;
  }

  const estimateMean =
    (pdMm2D + pdMm3D) / 2;

  const disagreementRatio =
    estimateMean > 0
      ? Math.abs(
        pdMm3D - pdMm2D,
      ) / estimateMean
      : 0;

  // 최종 PD는 반복성이 더 좋은 2D 홍채 비율값을 사용한다.
  // 상대 3D 값은 최종값에 섞지 않고 원근/품질 검증에만 사용한다.
  const pdMm = pdMm2D;

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
      fusion3DWeight: 0,
    },
    pose: extractPoseDegrees(matrix),
    framing: calculateFraming(landmarks),
    eyeAndPerspective: calculateEyeAndPerspectiveMetrics(landmarks, width, height, points),
    points,
    centerSource: centerOverrides
      ? 'refined-pupil'
      : 'mediapipe-iris',

    diameterSource:
      hasPurpleDiameterOverrides
        ? 'opencv-purple-circle'
        : 'mediapipe-iris-boundary',
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

  if (measurement.eyeImageQuality) {
    const eyeQuality = measurement.eyeImageQuality;
    if (eyeQuality.minSharpness < config.minEyeSharpness) {
      reasons.push(`눈 영역 초점이 흐림 (${eyeQuality.minSharpness.toFixed(0)})`);
      score -= 18;
    }
    if (eyeQuality.minBrightness < config.minEyeBrightness) {
      reasons.push(`눈 영역이 너무 어두움 (${eyeQuality.minBrightness.toFixed(0)})`);
      score -= 15;
    }
    if (eyeQuality.maxBrightness > config.maxEyeBrightness) {
      reasons.push(`눈 영역이 너무 밝음 (${eyeQuality.maxBrightness.toFixed(0)})`);
      score -= 12;
    }
    if (eyeQuality.maxUnderexposedRatio > 0.30) {
      warnings.push('눈 주변의 검게 뭉개진 픽셀이 많습니다.');
      score -= 5;
    }
    if (eyeQuality.maxOverexposedRatio > 0.24) {
      warnings.push('눈 주변의 과노출 픽셀이 많습니다.');
      score -= 5;
    }
  }

  if (measurement.pupilRefinement) {
    const pupil = measurement.pupilRefinement;
    if (!pupil.available) {
      warnings.push('OpenCV 동공 보정을 사용할 수 없어 MediaPipe 홍채 중심을 사용했습니다.');
      score -= 3;
    } else if (pupil.fallbackCount > 0) {
      warnings.push(`${pupil.fallbackCount}개 눈은 동공 검출 신뢰도가 낮아 MediaPipe 중심으로 대체했습니다.`);
      score -= pupil.fallbackCount * 3;
    } else if (pupil.minConfidence < config.minPupilConfidence) {
      warnings.push(`동공 검출 신뢰도가 낮습니다 (${(pupil.minConfidence * 100).toFixed(0)}%).`);
      score -= 4;
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
