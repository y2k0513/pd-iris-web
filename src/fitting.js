export const DEFAULT_PUPIL_FIT_THRESHOLDS = Object.freeze({
  minScore: 0.50,
  minIou: 0.46,
  maxCenterDistanceRatio: 0.48,
  minAxisRatio: 0.88,
});

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function median(values) {
  if (!values.length) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function solve3x3(matrix, vector) {
  const augmented = matrix.map(
    (row, index) => [...row, vector[index]],
  );

  for (let column = 0; column < 3; column += 1) {
    let pivotRow = column;

    for (let row = column + 1; row < 3; row += 1) {
      if (
        Math.abs(augmented[row][column])
        > Math.abs(augmented[pivotRow][column])
      ) {
        pivotRow = row;
      }
    }

    if (
      Math.abs(augmented[pivotRow][column]) < 1e-9
    ) {
      return null;
    }

    [
      augmented[column],
      augmented[pivotRow],
    ] = [
      augmented[pivotRow],
      augmented[column],
    ];

    const pivot = augmented[column][column];

    for (let entry = column; entry < 4; entry += 1) {
      augmented[column][entry] /= pivot;
    }

    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;

      const factor = augmented[row][column];

      for (let entry = column; entry < 4; entry += 1) {
        augmented[row][entry] -=
          factor * augmented[column][entry];
      }
    }
  }

  return augmented.map((row) => row[3]);
}

function algebraicCircleFit(points) {
  if (!Array.isArray(points) || points.length < 3) {
    return null;
  }

  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  let sumB = 0;
  let sumXB = 0;
  let sumYB = 0;

  for (const point of points) {
    const x = point.x;
    const y = point.y;
    const b = -(x * x + y * y);

    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumYY += y * y;
    sumXY += x * y;
    sumB += b;
    sumXB += x * b;
    sumYB += y * b;
  }

  const solution = solve3x3(
    [
      [sumXX, sumXY, sumX],
      [sumXY, sumYY, sumY],
      [sumX, sumY, points.length],
    ],
    [sumXB, sumYB, sumB],
  );

  if (!solution) return null;

  const [d, e, f] = solution;

  const x = -d / 2;
  const y = -e / 2;
  const radiusSquared = x * x + y * y - f;

  if (
    !Number.isFinite(radiusSquared)
    || radiusSquared <= 0
  ) {
    return null;
  }

  return {
    x,
    y,
    radius: Math.sqrt(radiusSquared),
  };
}

function refineCircleGeometrically(
  points,
  initial,
  iterations = 8,
) {
  let current = { ...initial };

  for (
    let iteration = 0;
    iteration < iterations;
    iteration += 1
  ) {
    const normal = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];

    const rhs = [0, 0, 0];

    for (const point of points) {
      const dx = current.x - point.x;
      const dy = current.y - point.y;

      const distance = Math.max(
        1e-6,
        Math.hypot(dx, dy),
      );

      const residual = distance - current.radius;

      const jacobian = [
        dx / distance,
        dy / distance,
        -1,
      ];

      for (let row = 0; row < 3; row += 1) {
        rhs[row] += -jacobian[row] * residual;

        for (let column = 0; column < 3; column += 1) {
          normal[row][column] +=
            jacobian[row] * jacobian[column];
        }
      }
    }

    const delta = solve3x3(normal, rhs);

    if (!delta) break;

    current = {
      x: current.x + delta[0],
      y: current.y + delta[1],
      radius: Math.max(
        1,
        current.radius + delta[2],
      ),
    };

    if (
      Math.hypot(
        delta[0],
        delta[1],
        delta[2],
      ) < 1e-4
    ) {
      break;
    }
  }

  return current;
}

function circleResiduals(points, circle) {
  return points.map((point) => Math.abs(
    Math.hypot(
      point.x - circle.x,
      point.y - circle.y,
    ) - circle.radius,
  ));
}

function angularCoverage(points, center) {
  if (points.length < 2) return 0;

  const angles = points
    .map((point) => {
      const angle = Math.atan2(
        point.y - center.y,
        point.x - center.x,
      );

      return angle < 0
        ? angle + Math.PI * 2
        : angle;
    })
    .sort((a, b) => a - b);

  let largestGap = 0;

  for (
    let index = 1;
    index < angles.length;
    index += 1
  ) {
    largestGap = Math.max(
      largestGap,
      angles[index] - angles[index - 1],
    );
  }

  largestGap = Math.max(
    largestGap,
    angles[0]
      + Math.PI * 2
      - angles[angles.length - 1],
  );

  return clamp(
    1 - largestGap / (Math.PI * 2),
    0,
    1,
  );
}

function circleFromThreePoints(a, b, c) {
  const denominator = 2 * (
    a.x * (b.y - c.y)
    + b.x * (c.y - a.y)
    + c.x * (a.y - b.y)
  );

  if (Math.abs(denominator) < 1e-7) {
    return null;
  }

  const aSquared = a.x * a.x + a.y * a.y;
  const bSquared = b.x * b.x + b.y * b.y;
  const cSquared = c.x * c.x + c.y * c.y;

  const x = (
    aSquared * (b.y - c.y)
    + bSquared * (c.y - a.y)
    + cSquared * (a.y - b.y)
  ) / denominator;

  const y = (
    aSquared * (c.x - b.x)
    + bSquared * (a.x - c.x)
    + cSquared * (b.x - a.x)
  ) / denominator;

  const radius = Math.hypot(
    a.x - x,
    a.y - y,
  );

  if (
    ![x, y, radius].every(Number.isFinite)
    || radius <= 0
  ) {
    return null;
  }

  return {
    x,
    y,
    radius,
  };
}

function createDeterministicRandom(seed) {
  let state = Math.max(1, seed >>> 0);

  return () => {
    state = (
      Math.imul(state, 1664525)
      + 1013904223
    ) >>> 0;

    return state / 4294967296;
  };
}

function estimateHorizontalChord(
  points,
  center,
  equivalentRadius,
) {
  if (!points.length || !center) return null;

  const rowHeight = 2;
  const verticalRange = Math.max(
    3,
    equivalentRadius * 0.34,
  );

  const rows = new Map();

  for (const point of points) {
    if (
      Math.abs(point.y - center.y)
      > verticalRange
    ) {
      continue;
    }

    const row =
      Math.round(point.y / rowHeight) * rowHeight;

    const current = rows.get(row) || {
      minX: point.x,
      maxX: point.x,
    };

    current.minX = Math.min(
      current.minX,
      point.x,
    );

    current.maxX = Math.max(
      current.maxX,
      point.x,
    );

    rows.set(row, current);
  }

  const centers = [];
  const radii = [];

  for (const row of rows.values()) {
    const width = row.maxX - row.minX;

    if (
      width < equivalentRadius * 1.1
      || width > equivalentRadius * 2.8
    ) {
      continue;
    }

    centers.push(
      (row.minX + row.maxX) / 2,
    );

    radii.push(width / 2);
  }

  if (radii.length < 2) return null;

  return {
    x: median(centers),
    radius: median(radii),
    sampleCount: radii.length,
  };
}

function summarizeHorizontalCut(
  points,
  indices,
  equivalentRadius,
  minimumSpanRatio,
) {
  const selected = [...indices].map(
    (index) => points[index],
  );

  const span = selected.length
    ? Math.max(
      ...selected.map((point) => point.x),
    ) - Math.min(
      ...selected.map((point) => point.x),
    )
    : 0;

  const spanRatio =
    span / Math.max(1, equivalentRadius * 2);

  return {
    detected:
      selected.length >= 4
      && spanRatio >= minimumSpanRatio,
    indices,
    spanRatio,
    extentRatio: 1,
  };
}

function detectHorizontalCuts(
  points,
  irisCenter,
  equivalentRadius,
) {
  const upperIndices = new Set();
  const lowerIndices = new Set();

  if (points.length < 5) {
    return {
      upper: {
        detected: false,
        indices: upperIndices,
        spanRatio: 0,
        extentRatio: 1,
      },
      lower: {
        detected: false,
        indices: lowerIndices,
        spanRatio: 0,
        extentRatio: 1,
      },
    };
  }

  const upperLimit =
    irisCenter.y - equivalentRadius * 0.10;

  const lowerLimit =
    irisCenter.y + equivalentRadius * 0.10;

  for (
    let index = 0;
    index < points.length;
    index += 1
  ) {
    const point = points[index];
    const dx = point.x - irisCenter.x;

    // 동공 중앙 부근의 절단 경계만 검사한다.
    if (
      Math.abs(dx) > equivalentRadius * 1.05
    ) {
      continue;
    }

    const previous =
      points[
        (index - 2 + points.length)
        % points.length
      ];

    const next =
      points[
        (index + 2)
        % points.length
      ];

    const tangentX = next.x - previous.x;
    const tangentY = next.y - previous.y;

    const upperHorizontal =
      Math.abs(tangentY)
      <= Math.max(
        1.25,
        Math.abs(tangentX) * 0.28,
      );

    // 아래 눈꺼풀은 위쪽보다 곡률이 클 수 있으므로
    // 수평 판정을 약간 더 느슨하게 적용한다.
    const lowerHorizontal =
      Math.abs(tangentY)
      <= Math.max(
        1.5,
        Math.abs(tangentX) * 0.38,
      );

    if (
      point.y < upperLimit
      && upperHorizontal
    ) {
      upperIndices.add(index);
    }

    if (
      point.y > lowerLimit
      && lowerHorizontal
    ) {
      lowerIndices.add(index);
    }
  }

  const upper = summarizeHorizontalCut(
    points,
    upperIndices,
    equivalentRadius,
    0.20,
  );

  const lower = summarizeHorizontalCut(
    points,
    lowerIndices,
    equivalentRadius,
    0.15,
  );

  /*
   * 정상적인 원도 위·아래 끝에서 접선이 수평이다.
   * 따라서 단순한 수평 접선만으로는 눈꺼풀 절단을
   * 확정하지 않고, 실제 세로 반경이 좌우 반경보다
   * 짧게 눌렸는지도 함께 검사한다.
   */
  const minX = Math.min(
    ...points.map((point) => point.x),
  );

  const maxX = Math.max(
    ...points.map((point) => point.x),
  );

  const minY = Math.min(
    ...points.map((point) => point.y),
  );

  const maxY = Math.max(
    ...points.map((point) => point.y),
  );

  const horizontalRadius = Math.max(
    1,
    (maxX - minX) / 2,
  );

  const upperExtent = Math.max(
    0,
    irisCenter.y - minY,
  );

  const lowerExtent = Math.max(
    0,
    maxY - irisCenter.y,
  );

  upper.extentRatio =
    upperExtent / horizontalRadius;

  lower.extentRatio =
    lowerExtent / horizontalRadius;

  upper.detected =
    upper.detected
    && upper.extentRatio <= 0.92;

  lower.detected =
    lower.detected
    && lower.extentRatio <= 0.92;

  return {
    upper,
    lower,
  };
}

function samplePointsByAngle(
  points,
  center,
  binDegrees = 5,
) {
  const bins = new Map();

  for (const point of points) {
    let angle = Math.atan2(
      point.y - center.y,
      point.x - center.x,
    );

    if (angle < 0) {
      angle += Math.PI * 2;
    }

    const degrees =
      angle * 180 / Math.PI;

    const bin =
      Math.floor(degrees / binDegrees);

    const radius = Math.hypot(
      point.x - center.x,
      point.y - center.y,
    );

    const current = bins.get(bin);

    // 각도 구간마다 가장 바깥쪽 경계점만 사용한다.
    if (
      !current
      || radius > current.radius
    ) {
      bins.set(bin, {
        point,
        radius,
      });
    }
  }

  return [...bins.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, value]) => value.point);
}

function calculateSideBalance(
  points,
  center,
) {
  let left = 0;
  let right = 0;

  for (const point of points) {
    if (point.x < center.x) {
      left += 1;
    } else {
      right += 1;
    }
  }

  return Math.min(left, right)
    / Math.max(1, left, right);
}

/**
 * 위쪽 또는 아래쪽 눈꺼풀이 동공을 가렸을 때
 * 절단 경계는 제외하고 실제 원호로 보이는 점만 남긴다.
 */
export function selectVisiblePupilArcPoints(
  points,
  {
    irisCenter,
    irisRadius,
    equivalentRadius,
    angleBinDegrees = 4,
  } = {},
) {
  if (
    !Array.isArray(points)
    || !irisCenter
  ) {
    return [];
  }

  const safeIrisRadius = Math.max(
    1,
    irisRadius
      || equivalentRadius
      || 1,
  );

  const safeEquivalentRadius = Math.max(
    1,
    equivalentRadius
      || safeIrisRadius * 0.55,
  );

  const cuts = detectHorizontalCuts(
    points,
    irisCenter,
    safeEquivalentRadius,
  );

  const sideThreshold = Math.max(
    safeEquivalentRadius * 0.52,
    safeIrisRadius * 0.27,
  );

  let visiblePoints = points.filter(
    (point, index) => {
      const dx = point.x - irisCenter.x;
      const dy = point.y - irisCenter.y;

      const upperCentral =
        dy < -safeEquivalentRadius * 0.08
        && Math.abs(dx)
          < safeEquivalentRadius * 0.90;

      const lowerCentral =
        dy > safeEquivalentRadius * 0.08
        && Math.abs(dx)
          < safeEquivalentRadius * 0.90;

      if (
        cuts.upper.detected
        && (
          cuts.upper.indices.has(index)
          || upperCentral
        )
      ) {
        return false;
      }

      if (
        cuts.lower.detected
        && (
          cuts.lower.indices.has(index)
          || lowerCentral
        )
      ) {
        return false;
      }

      const lateralArc =
        Math.abs(dx) >= sideThreshold
        && Math.abs(dy)
          <= safeIrisRadius * 0.68;

      // 위·아래가 모두 가려졌다면 좌우 원호를 사용한다.
      if (
        cuts.upper.detected
        && cuts.lower.detected
      ) {
        return lateralArc;
      }

      // 위쪽이 가려졌다면 아래쪽과 좌우 원호를 사용한다.
      if (cuts.upper.detected) {
        const lowerOrMiddleArc =
          dy >= -safeIrisRadius * 0.18;

        return lowerOrMiddleArc
          || lateralArc;
      }

      // 아래쪽이 가려졌다면 위쪽과 좌우 원호를 사용한다.
      if (cuts.lower.detected) {
        const upperOrMiddleArc =
          dy <= safeIrisRadius * 0.18;

        return upperOrMiddleArc
          || lateralArc;
      }

      // 가림이 명확하지 않으면 전체 contour를 사용한다.
      return true;
    },
  );

  if (visiblePoints.length < 10) {
    visiblePoints = points.filter(
      (point, index) => (
        !cuts.upper.indices.has(index)
        && !cuts.lower.indices.has(index)
      ),
    );
  }

  const sampled = samplePointsByAngle(
    visiblePoints,
    irisCenter,
    angleBinDegrees,
  );

  const chordEstimate =
    estimateHorizontalChord(
      points,
      irisCenter,
      safeEquivalentRadius,
    );

  Object.defineProperty(
    sampled,
    'meta',
    {
      enumerable: false,
      configurable: true,
      value: {
        topOcclusionDetected:
          cuts.upper.detected,

        bottomOcclusionDetected:
          cuts.lower.detected,

        occlusionDetected:
          cuts.upper.detected
          || cuts.lower.detected,

        upperCutSpanRatio:
          cuts.upper.spanRatio,

        lowerCutSpanRatio:
          cuts.lower.spanRatio,

        upperExtentRatio:
          cuts.upper.extentRatio,

        lowerExtentRatio:
          cuts.lower.extentRatio,

        chordEstimate,

        sideBalance:
          calculateSideBalance(
            sampled,
            irisCenter,
          ),

        angleBinDegrees,
      },
    },
  );

  return sampled;
}

/**
 * 가시 원호에 RANSAC 원 fitting을 수행하고,
 * inlier만 이용해 기하학적으로 다시 보정한다.
 */
export function fitPartialArcCircle(
  points,
  {
    minPoints = 10,
    ransacIterations = 320,
    irisCenter = null,
    irisRadius = null,
    equivalentRadius = null,
  } = {},
) {
  if (
    !Array.isArray(points)
    || points.length < minPoints
  ) {
    return null;
  }

  const pointCenter = irisCenter || {
    x: median(
      points.map((point) => point.x),
    ),
    y: median(
      points.map((point) => point.y),
    ),
  };

  const radialDistances = points.map(
    (point) => Math.hypot(
      point.x - pointCenter.x,
      point.y - pointCenter.y,
    ),
  );

  const safeEquivalentRadius = Math.max(
    1,
    equivalentRadius
      || median(radialDistances),
  );

  const safeIrisRadius = Math.max(
    safeEquivalentRadius * 1.1,
    irisRadius
      || safeEquivalentRadius * 1.7,
  );

  const metadata = points.meta || {};

  const chordEstimate =
    metadata.chordEstimate
    || estimateHorizontalChord(
      points,
      pointCenter,
      safeEquivalentRadius,
    );

  const minRadius =
    safeEquivalentRadius * 0.90;

  const maxRadius = Math.min(
    safeIrisRadius * 1.14,
    safeEquivalentRadius * 1.42,
  );

  const maximumCenterDistance =
    safeIrisRadius * 0.76;

  const random =
    createDeterministicRandom(
      points.length * 2654435761,
    );

  let best = null;

  for (
    let iteration = 0;
    iteration < ransacIterations;
    iteration += 1
  ) {
    const first =
      Math.floor(random() * points.length);

    let second =
      Math.floor(random() * points.length);

    while (second === first) {
      second =
        Math.floor(random() * points.length);
    }

    let third =
      Math.floor(random() * points.length);

    while (
      third === first
      || third === second
    ) {
      third =
        Math.floor(random() * points.length);
    }

    const circle = circleFromThreePoints(
      points[first],
      points[second],
      points[third],
    );

    if (!circle) continue;

    if (
      circle.radius < minRadius
      || circle.radius > maxRadius
    ) {
      continue;
    }

    const centerDistance = Math.hypot(
      circle.x - pointCenter.x,
      circle.y - pointCenter.y,
    );

    if (
      centerDistance
      > maximumCenterDistance
    ) {
      continue;
    }

    const inlierThreshold = Math.max(
      1.5,
      circle.radius * 0.045,
    );

    const residuals =
      circleResiduals(points, circle);

    const inlierIndices = [];

    for (
      let index = 0;
      index < residuals.length;
      index += 1
    ) {
      if (
        residuals[index]
        <= inlierThreshold
      ) {
        inlierIndices.push(index);
      }
    }

    if (
      inlierIndices.length < minPoints
    ) {
      continue;
    }

    const meanResidual =
      inlierIndices.reduce(
        (sum, index) =>
          sum + residuals[index],
        0,
      ) / inlierIndices.length;

    const inlierRatio =
      inlierIndices.length
      / points.length;

    const chordScore = chordEstimate
      ? clamp(
        1 - Math.abs(
          circle.radius
          - chordEstimate.radius,
        ) / Math.max(
          1,
          chordEstimate.radius * 0.45,
        ),
        0,
        1,
      )
      : 0.5;

    const score =
      inlierRatio
      - (
        meanResidual
        / inlierThreshold
      ) * 0.12
      - (
        centerDistance
        / safeIrisRadius
      ) * 0.08
      + chordScore * 0.05;

    if (
      !best
      || score > best.score
    ) {
      best = {
        circle,
        score,
        inlierIndices,
        inlierThreshold,
      };
    }
  }

  let working;

  if (best) {
    working = best.inlierIndices.map(
      (index) => points[index],
    );
  } else {
    // RANSAC 실패 시 전체 가시 원호로 최소제곱 fitting.
    working = points.map((point) => ({
      x: point.x,
      y: point.y,
    }));
  }

  let circle =
    algebraicCircleFit(working)
    || best?.circle
    || null;

  if (!circle) return null;

  circle = refineCircleGeometrically(
    working,
    circle,
    10,
  );

  const finalThreshold = Math.max(
    1.5,
    circle.radius * 0.05,
  );

  const refinedPoints = points.filter(
    (point) => (
      Math.abs(
        Math.hypot(
          point.x - circle.x,
          point.y - circle.y,
        ) - circle.radius
      ) <= finalThreshold
    ),
  );

  if (
    refinedPoints.length >= minPoints
  ) {
    const refit =
      algebraicCircleFit(refinedPoints);

    if (refit) {
      circle = refineCircleGeometrically(
        refinedPoints,
        refit,
        10,
      );

      working = refinedPoints;
    }
  }

  if (
    circle.radius < minRadius
    || circle.radius > maxRadius
  ) {
    return null;
  }

  const centerDistance = Math.hypot(
    circle.x - pointCenter.x,
    circle.y - pointCenter.y,
  );

  if (
    centerDistance
    > maximumCenterDistance
  ) {
    return null;
  }

  const residuals = circleResiduals(
    working,
    circle,
  ).sort((a, b) => a - b);

  if (!residuals.length) {
    return null;
  }

  const p90Index = Math.min(
    residuals.length - 1,
    Math.floor(
      residuals.length * 0.90,
    ),
  );

  const arcCoverage = angularCoverage(
    working,
    circle,
  );

  return {
    ...circle,

    pointCount:
      working.length,

    originalPointCount:
      points.length,

    meanResidual:
      residuals.reduce(
        (sum, value) => sum + value,
        0,
      ) / residuals.length,

    p90Residual:
      residuals[p90Index],

    arcCoverage,

    angularCoverageDegrees:
      arcCoverage * 360,

    inlierRatio:
      working.length / points.length,

    sideBalance:
      calculateSideBalance(
        working,
        circle,
      ),

    topOcclusionDetected:
      Boolean(
        metadata.topOcclusionDetected,
      ),

    bottomOcclusionDetected:
      Boolean(
        metadata.bottomOcclusionDetected,
      ),

    occlusionDetected:
      Boolean(
        metadata.occlusionDetected,
      ),

    upperCutSpanRatio:
      Number(
        metadata.upperCutSpanRatio,
      ) || 0,

    lowerCutSpanRatio:
      Number(
        metadata.lowerCutSpanRatio,
      ) || 0,

    upperExtentRatio:
      Number(
        metadata.upperExtentRatio,
      ) || 1,

    lowerExtentRatio:
      Number(
        metadata.lowerExtentRatio,
      ) || 1,

    chordRadius:
      chordEstimate?.radius ?? null,

    chordCenterX:
      chordEstimate?.x ?? null,

    ransacIterations,
  };
}

/**
 * 일반적인 마스크 기반 원·타원 적합도 점수.
 */
export function scorePupilFitMetrics({
  iou = 0,
  coverage = 0,
  precision = 0,
  centerDistanceRatio = 1,
  axisRatio = 0,
  diameterRatio = 0,
} = {}) {
  const boundedIou =
    clamp(iou, 0, 1);

  const boundedCoverage =
    clamp(coverage, 0, 1);

  const boundedPrecision =
    clamp(precision, 0, 1);

  const centerScore = clamp(
    1 - centerDistanceRatio / 0.80,
    0,
    1,
  );

  const roundnessScore =
    clamp(axisRatio, 0, 1);

  let diameterScore = 0;

  if (diameterRatio > 0) {
    if (diameterRatio <= 1.10) {
      diameterScore = clamp(
        1 - Math.abs(
          diameterRatio - 0.58,
        ) / 0.70,
        0,
        1,
      );
    } else {
      diameterScore = clamp(
        1 - (
          diameterRatio - 1.10
        ) / 0.60,
        0,
        1,
      );
    }
  }

  return clamp(
    boundedIou * 0.42
    + boundedCoverage * 0.18
    + boundedPrecision * 0.12
    + centerScore * 0.16
    + roundnessScore * 0.09
    + diameterScore * 0.03,
    0,
    1,
  );
}

export function chooseBestPupilFit(
  candidates = [],
  thresholds = DEFAULT_PUPIL_FIT_THRESHOLDS,
) {
  const valid = candidates.filter(
    (candidate) => (
      candidate
      && Number.isFinite(candidate.score)
      && Number.isFinite(candidate.iou)
      && Number.isFinite(
        candidate.centerDistanceRatio,
      )
    ),
  );

  if (!valid.length) {
    return {
      best: null,
      accepted: false,
      reason: 'no-candidate',
    };
  }

  valid.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }

    if (b.iou !== a.iou) {
      return b.iou - a.iou;
    }

    return (
      a.centerDistanceRatio
      - b.centerDistanceRatio
    );
  });

  const best = valid[0];

  const accepted =
    best.score >= thresholds.minScore
    && best.iou >= thresholds.minIou
    && best.centerDistanceRatio
      <= thresholds.maxCenterDistanceRatio
    && (
      !Number.isFinite(
        thresholds.minAxisRatio,
      )
      || !Number.isFinite(
        best.axisRatio,
      )
      || best.axisRatio
        >= thresholds.minAxisRatio
    );

  return {
    best,
    accepted,
    reason: accepted
      ? 'accepted'
      : 'below-threshold',
  };
}

/**
 * 마스크 IoU가 가려진 동공의 면적 원을 과도하게
 * 선호하는 문제를 보완한다.
 */
export function chooseOcclusionAwarePupilFit(
  candidates = [],
  thresholds = DEFAULT_PUPIL_FIT_THRESHOLDS,
) {
  const normalSelection =
    chooseBestPupilFit(
      candidates,
      thresholds,
    );

  const partial = candidates.find(
    (candidate) =>
      candidate?.type
      === 'partial-arc-circle',
  );

  const equivalent = candidates.find(
    (candidate) =>
      candidate?.type
      === 'equivalent-circle',
  );

  if (!partial || !equivalent) {
    return normalSelection;
  }

  const expansion = Number.isFinite(
    partial.radiusExpansionRatio,
  )
    ? partial.radiusExpansionRatio
    : (
      partial.width
      / Math.max(1, equivalent.width)
    );

  const radius =
    partial.width / 2;

  const residualLimit = Math.max(
    1.8,
    radius * 0.075,
  );

  const inlierRatio = Number.isFinite(
    partial.arcInlierRatio,
  )
    ? partial.arcInlierRatio
    : 1;

  const sideBalance = Number.isFinite(
    partial.arcSideBalance,
  )
    ? partial.arcSideBalance
    : 1;

  const occlusionDetected = Boolean(
    partial.occlusionDetected
    || partial.topOcclusionDetected
    || partial.bottomOcclusionDetected
  );

  const bothSidesOccluded = Boolean(
    partial.topOcclusionDetected
    && partial.bottomOcclusionDetected
  );

  const requiredArcCoverage =
    bothSidesOccluded
      ? 0.26
      : 0.31;

  const requiredInlierRatio =
    bothSidesOccluded
      ? 0.52
      : 0.56;

  const minimumPartialScore =
    occlusionDetected
      ? 0.46
      : 0.50;

  const partialPlausible =
    partial.score >= Math.max(
      minimumPartialScore,
      thresholds.minScore,
    )
    && partial.centerDistanceRatio
      <= Math.min(
        0.48,
        thresholds.maxCenterDistanceRatio,
      )
    && partial.arcCoverage
      >= requiredArcCoverage
    && partial.arcPointCount >= 10
    && inlierRatio
      >= requiredInlierRatio
    && sideBalance >= 0.32
    && partial.arcResidualP90
      <= residualLimit
    && expansion >= 1.02
    && expansion <= 1.25;

  const occlusionPreferred =
    occlusionDetected
    && partial.arcCoverage
      >= requiredArcCoverage
    && inlierRatio
      >= requiredInlierRatio;

  const competitive =
    !normalSelection.best
    || partial.score
      >= normalSelection.best.score - 0.08;

  if (
    partialPlausible
    && (
      occlusionPreferred
      || competitive
    )
  ) {
    return {
      best: partial,
      accepted: true,
      reason: occlusionPreferred
        ? 'ransac-occlusion-recovery'
        : 'partial-arc-recovery',
      replaced: normalSelection.best,
    };
  }

  return normalSelection;
}
