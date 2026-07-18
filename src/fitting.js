export const DEFAULT_PUPIL_FIT_THRESHOLDS = Object.freeze({
  minScore: 0.38,
  minIou: 0.30,
  maxCenterDistanceRatio: 0.80,
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
  const augmented = matrix.map((row, index) => [...row, vector[index]]);

  for (let column = 0; column < 3; column += 1) {
    let pivotRow = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row;
      }
    }

    if (Math.abs(augmented[pivotRow][column]) < 1e-9) return null;
    [augmented[column], augmented[pivotRow]] = [augmented[pivotRow], augmented[column]];

    const pivot = augmented[column][column];
    for (let entry = column; entry < 4; entry += 1) {
      augmented[column][entry] /= pivot;
    }

    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let entry = column; entry < 4; entry += 1) {
        augmented[row][entry] -= factor * augmented[column][entry];
      }
    }
  }

  return augmented.map((row) => row[3]);
}

function algebraicCircleFit(points) {
  if (!Array.isArray(points) || points.length < 3) return null;

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
  if (!Number.isFinite(radiusSquared) || radiusSquared <= 0) return null;

  return { x, y, radius: Math.sqrt(radiusSquared) };
}

function refineCircleGeometrically(points, initial, iterations = 8) {
  let current = { ...initial };

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const normal = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const rhs = [0, 0, 0];

    for (const point of points) {
      const dx = current.x - point.x;
      const dy = current.y - point.y;
      const distance = Math.max(1e-6, Math.hypot(dx, dy));
      const residual = distance - current.radius;
      const jacobian = [dx / distance, dy / distance, -1];

      for (let row = 0; row < 3; row += 1) {
        rhs[row] += -jacobian[row] * residual;
        for (let column = 0; column < 3; column += 1) {
          normal[row][column] += jacobian[row] * jacobian[column];
        }
      }
    }

    const delta = solve3x3(normal, rhs);
    if (!delta) break;
    current = {
      x: current.x + delta[0],
      y: current.y + delta[1],
      radius: Math.max(1, current.radius + delta[2]),
    };

    if (Math.hypot(delta[0], delta[1], delta[2]) < 1e-4) break;
  }

  return current;
}

function circleResiduals(points, circle) {
  return points.map((point) => Math.abs(
    Math.hypot(point.x - circle.x, point.y - circle.y) - circle.radius,
  ));
}

function angularCoverage(points, center) {
  if (points.length < 2) return 0;
  const angles = points
    .map((point) => {
      const angle = Math.atan2(point.y - center.y, point.x - center.x);
      return angle < 0 ? angle + Math.PI * 2 : angle;
    })
    .sort((a, b) => a - b);

  let largestGap = 0;
  for (let index = 1; index < angles.length; index += 1) {
    largestGap = Math.max(largestGap, angles[index] - angles[index - 1]);
  }
  largestGap = Math.max(
    largestGap,
    angles[0] + Math.PI * 2 - angles[angles.length - 1],
  );
  return clamp(1 - largestGap / (Math.PI * 2), 0, 1);
}

function circleFromThreePoints(a, b, c) {
  const denominator = 2 * (
    a.x * (b.y - c.y)
    + b.x * (c.y - a.y)
    + c.x * (a.y - b.y)
  );

  if (Math.abs(denominator) < 1e-7) return null;

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

  const radius = Math.hypot(a.x - x, a.y - y);

  if (![x, y, radius].every(Number.isFinite) || radius <= 0) {
    return null;
  }

  return { x, y, radius };
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

function estimateHorizontalChord(points, center, equivalentRadius) {
  if (!points.length || !center) return null;

  const rowHeight = 2;
  const verticalRange = Math.max(3, equivalentRadius * 0.34);
  const rows = new Map();

  for (const point of points) {
    if (Math.abs(point.y - center.y) > verticalRange) continue;

    const row = Math.round(point.y / rowHeight) * rowHeight;
    const current = rows.get(row) || {
      minX: point.x,
      maxX: point.x,
    };

    current.minX = Math.min(current.minX, point.x);
    current.maxX = Math.max(current.maxX, point.x);
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

    centers.push((row.minX + row.maxX) / 2);
    radii.push(width / 2);
  }

  if (radii.length < 2) return null;

  return {
    x: median(centers),
    radius: median(radii),
    sampleCount: radii.length,
  };
}

function detectUpperHorizontalCut(
  points,
  irisCenter,
  equivalentRadius,
) {
  const flatIndices = new Set();

  if (points.length < 5) {
    return {
      detected: false,
      flatIndices,
      spanRatio: 0,
    };
  }

  const upperLimit =
    irisCenter.y - equivalentRadius * 0.10;

  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];

    if (
      point.y > upperLimit
      || Math.abs(point.x - irisCenter.x)
        > equivalentRadius * 1.05
    ) {
      continue;
    }

    const previous =
      points[(index - 2 + points.length) % points.length];
    const next =
      points[(index + 2) % points.length];

    const tangentX = next.x - previous.x;
    const tangentY = next.y - previous.y;

    const nearlyHorizontal =
      Math.abs(tangentY)
      <= Math.max(1.25, Math.abs(tangentX) * 0.28);

    if (nearlyHorizontal) {
      flatIndices.add(index);
    }
  }

  const flatPoints = [...flatIndices].map(
    (index) => points[index],
  );

  const span = flatPoints.length
    ? Math.max(...flatPoints.map((point) => point.x))
      - Math.min(...flatPoints.map((point) => point.x))
    : 0;

  const spanRatio = span / Math.max(1, equivalentRadius * 2);

  return {
    detected:
      flatPoints.length >= 4
      && spanRatio >= 0.25,
    flatIndices,
    spanRatio,
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

    if (angle < 0) angle += Math.PI * 2;

    const degrees = angle * 180 / Math.PI;
    const bin = Math.floor(degrees / binDegrees);
    const radius = Math.hypot(
      point.x - center.x,
      point.y - center.y,
    );

    const current = bins.get(bin);

    // 각 방향에서 가장 바깥쪽 경계점 하나만 사용한다.
    if (!current || radius > current.radius) {
      bins.set(bin, { point, radius });
    }
  }

  return [...bins.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, value]) => value.point);
}

function calculateSideBalance(points, center) {
  let left = 0;
  let right = 0;

  for (const point of points) {
    if (point.x < center.x) left += 1;
    else right += 1;
  }

  return Math.min(left, right) / Math.max(1, left, right);
}
/**
 * Remove the central upper contour where an eyelid commonly cuts across the
 * pupil/iris. Lower and lateral arc points are kept so a full circle can be
 * reconstructed from the visible perimeter.
 */
export function selectVisiblePupilArcPoints(points, {
  irisCenter,
  irisRadius,
  equivalentRadius,
  angleBinDegrees = 5,
} = {}) {
  if (!Array.isArray(points) || !irisCenter) return [];

  const safeIrisRadius = Math.max(
    1,
    irisRadius || equivalentRadius || 1,
  );

  const safeEquivalentRadius = Math.max(
    1,
    equivalentRadius || safeIrisRadius * 0.55,
  );

  const horizontalCut = detectUpperHorizontalCut(
    points,
    irisCenter,
    safeEquivalentRadius,
  );

  const sideThreshold = Math.max(
    safeEquivalentRadius * 0.52,
    safeIrisRadius * 0.27,
  );

  let visiblePoints = points.filter((point, index) => {
    const dx = point.x - irisCenter.x;
    const dy = point.y - irisCenter.y;

    const upperCentral =
      dy < -safeEquivalentRadius * 0.08
      && Math.abs(dx) < safeEquivalentRadius * 0.90;

    if (
      horizontalCut.detected
      && (
        horizontalCut.flatIndices.has(index)
        || upperCentral
      )
    ) {
      return false;
    }

    const lowerOrMiddleArc =
      dy >= -safeIrisRadius * 0.18;

    const lateralArc =
      Math.abs(dx) >= sideThreshold
      && dy >= -safeIrisRadius * 0.62;

    return lowerOrMiddleArc || lateralArc;
  });

  if (visiblePoints.length < 12) {
    visiblePoints = points.filter((point) => {
      const dx = point.x - irisCenter.x;
      const dy = point.y - irisCenter.y;

      return (
        dy >= -safeIrisRadius * 0.20
        || (
          Math.abs(dx) >= sideThreshold
          && dy >= -safeIrisRadius * 0.65
        )
      );
    });
  }

  const sampled = samplePointsByAngle(
    visiblePoints,
    irisCenter,
    angleBinDegrees,
  );

  const chordEstimate = estimateHorizontalChord(
    points,
    irisCenter,
    safeEquivalentRadius,
  );

  Object.defineProperty(sampled, 'meta', {
    enumerable: false,
    configurable: true,
    value: {
      topOcclusionDetected: horizontalCut.detected,
      upperCutSpanRatio: horizontalCut.spanRatio,
      chordEstimate,
      sideBalance: calculateSideBalance(
        sampled,
        irisCenter,
      ),
      angleBinDegrees,
    },
  });

  return sampled;
}

/**
 * Robustly fit a circle to a partial visible arc. Algebraic fitting supplies
 * the initial estimate, geometric least squares refines it, and MAD trimming
 * removes eyelid/threshold outliers.
 */
export function fitPartialArcCircle(points, {
  minPoints = 10,
  ransacIterations = 200,
  irisCenter = null,
  irisRadius = null,
  equivalentRadius = null,
} = {}) {
  if (!Array.isArray(points) || points.length < minPoints) {
    return null;
  }

  const pointCenter = irisCenter || {
    x: median(points.map((point) => point.x)),
    y: median(points.map((point) => point.y)),
  };

  const radialDistances = points.map((point) => Math.hypot(
    point.x - pointCenter.x,
    point.y - pointCenter.y,
  ));

  const safeEquivalentRadius = Math.max(
    1,
    equivalentRadius || median(radialDistances),
  );

  const safeIrisRadius = Math.max(
    safeEquivalentRadius * 1.1,
    irisRadius || safeEquivalentRadius * 1.7,
  );

  const metadata = points.meta || {};
  const chordEstimate =
    metadata.chordEstimate
    || estimateHorizontalChord(
      points,
      pointCenter,
      safeEquivalentRadius,
    );

  const minRadius = safeEquivalentRadius * 0.90;
  const maxRadius = Math.min(
    safeIrisRadius * 1.14,
    safeEquivalentRadius * 1.42,
  );

  const maximumCenterDistance = safeIrisRadius * 0.76;
  const random = createDeterministicRandom(
    points.length * 2654435761,
  );

  let best = null;

  for (
    let iteration = 0;
    iteration < ransacIterations;
    iteration += 1
  ) {
    const first = Math.floor(random() * points.length);

    let second = Math.floor(random() * points.length);
    while (second === first) {
      second = Math.floor(random() * points.length);
    }

    let third = Math.floor(random() * points.length);
    while (third === first || third === second) {
      third = Math.floor(random() * points.length);
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

    if (centerDistance > maximumCenterDistance) continue;

    const inlierThreshold = Math.max(
      1.5,
      circle.radius * 0.045,
    );

    const residuals = circleResiduals(points, circle);
    const inlierIndices = [];

    for (
      let index = 0;
      index < residuals.length;
      index += 1
    ) {
      if (residuals[index] <= inlierThreshold) {
        inlierIndices.push(index);
      }
    }

    if (inlierIndices.length < minPoints) continue;

    const meanResidual = inlierIndices.reduce(
      (sum, index) => sum + residuals[index],
      0,
    ) / inlierIndices.length;

    const inlierRatio =
      inlierIndices.length / points.length;

    const chordScore = chordEstimate
      ? clamp(
        1
          - Math.abs(
            circle.radius - chordEstimate.radius,
          ) / Math.max(1, chordEstimate.radius * 0.45),
        0,
        1,
      )
      : 0.5;

    const score =
      inlierRatio
      - meanResidual / inlierThreshold * 0.12
      - centerDistance / safeIrisRadius * 0.08
      + chordScore * 0.05;

    if (!best || score > best.score) {
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
    // RANSAC이 실패하면 기존 최소제곱 방식을 fallback으로 사용한다.
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

  // 최종 원을 기준으로 한 번 더 이상점을 제거한다.
  const finalThreshold = Math.max(
    1.5,
    circle.radius * 0.05,
  );

  const refinedPoints = points.filter((point) => (
    Math.abs(
      Math.hypot(
        point.x - circle.x,
        point.y - circle.y,
      ) - circle.radius
    ) <= finalThreshold
  ));

  if (refinedPoints.length >= minPoints) {
    const refit = algebraicCircleFit(refinedPoints);

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

  if (centerDistance > maximumCenterDistance) {
    return null;
  }

  const residuals = circleResiduals(
    working,
    circle,
  ).sort((a, b) => a - b);

  const p90Index = Math.min(
    residuals.length - 1,
    Math.floor(residuals.length * 0.90),
  );

  const arcCoverage = angularCoverage(
    working,
    circle,
  );

  return {
    ...circle,
    pointCount: working.length,
    originalPointCount: points.length,
    meanResidual:
      residuals.reduce((sum, value) => sum + value, 0)
      / residuals.length,
    p90Residual: residuals[p90Index],
    arcCoverage,
    angularCoverageDegrees: arcCoverage * 360,
    inlierRatio: working.length / points.length,
    sideBalance: calculateSideBalance(
      working,
      circle,
    ),
    topOcclusionDetected:
      Boolean(metadata.topOcclusionDetected),
    upperCutSpanRatio:
      Number(metadata.upperCutSpanRatio) || 0,
    chordRadius: chordEstimate?.radius ?? null,
    chordCenterX: chordEstimate?.x ?? null,
    ransacIterations,
  };
}
/**
 * Combine geometric fit metrics into a single 0..1 confidence score.
 * The mask overlap is intentionally dominant. This lets a clean circular
 * component survive even when its apparent diameter is larger than a typical
 * pupil because the final PD calculation only needs a reliable center.
 */
export function scorePupilFitMetrics({
  iou = 0,
  coverage = 0,
  precision = 0,
  centerDistanceRatio = 1,
  axisRatio = 0,
  diameterRatio = 0,
} = {}) {
  const boundedIou = clamp(iou, 0, 1);
  const boundedCoverage = clamp(coverage, 0, 1);
  const boundedPrecision = clamp(precision, 0, 1);
  const centerScore = clamp(1 - centerDistanceRatio / 0.80, 0, 1);
  const roundnessScore = clamp(axisRatio, 0, 1);

  // Broad plausibility only. A near-iris-sized dark circle is still useful
  // for center estimation, so diameter contributes little to rejection.
  let diameterScore = 0;
  if (diameterRatio > 0) {
    if (diameterRatio <= 1.10) {
      diameterScore = clamp(1 - Math.abs(diameterRatio - 0.58) / 0.70, 0, 1);
    } else {
      diameterScore = clamp(1 - (diameterRatio - 1.10) / 0.60, 0, 1);
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

export function chooseBestPupilFit(candidates = [], thresholds = DEFAULT_PUPIL_FIT_THRESHOLDS) {
  const valid = candidates.filter((candidate) => (
    candidate
    && Number.isFinite(candidate.score)
    && Number.isFinite(candidate.iou)
    && Number.isFinite(candidate.centerDistanceRatio)
  ));

  if (!valid.length) {
    return { best: null, accepted: false, reason: 'no-candidate' };
  }

  valid.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (b.iou !== a.iou) return b.iou - a.iou;
    return a.centerDistanceRatio - b.centerDistanceRatio;
  });

  const best = valid[0];
  const accepted = best.score >= thresholds.minScore
    && best.iou >= thresholds.minIou
    && best.centerDistanceRatio <= thresholds.maxCenterDistanceRatio;

  return {
    best,
    accepted,
    reason: accepted ? 'accepted' : 'below-threshold',
  };
}

/**
 * IoU naturally favors the visible-area equivalent circle, even when an
 * eyelid has removed the upper cap. Prefer a robust partial-arc circle when it
 * expands the radius moderately, follows a sufficiently broad arc, remains
 * centered, and is not substantially worse than the ordinary overlap winner.
 */
export function chooseOcclusionAwarePupilFit(
  candidates = [],
  thresholds = DEFAULT_PUPIL_FIT_THRESHOLDS,
) {
  const normalSelection = chooseBestPupilFit(
    candidates,
    thresholds,
  );

  const partial = candidates.find(
    (candidate) =>
      candidate?.type === 'partial-arc-circle',
  );

  const equivalent = candidates.find(
    (candidate) =>
      candidate?.type === 'equivalent-circle',
  );

  if (!partial || !equivalent) {
    return normalSelection;
  }

  const expansion = Number.isFinite(
    partial.radiusExpansionRatio,
  )
    ? partial.radiusExpansionRatio
    : partial.width / Math.max(1, equivalent.width);

  const radius = partial.width / 2;
  const residualLimit = Math.max(
    2.2,
    radius * 0.09,
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

  const partialPlausible =
    partial.score >= Math.max(
      0.44,
      thresholds.minScore,
    )
    && partial.centerDistanceRatio
      <= Math.min(
        0.66,
        thresholds.maxCenterDistanceRatio,
      )
    && partial.arcCoverage >= 0.38
    && partial.arcPointCount >= 10
    && inlierRatio >= 0.52
    && sideBalance >= 0.28
    && partial.arcResidualP90 <= residualLimit
    && expansion >= 1.015
    && expansion <= 1.35;

  const occlusionPreferred =
    Boolean(partial.topOcclusionDetected)
    && partial.arcCoverage >= 0.38
    && inlierRatio >= 0.52;

  const competitive =
    !normalSelection.best
    || partial.score
      >= normalSelection.best.score - 0.12;

  if (
    partialPlausible
    && (occlusionPreferred || competitive)
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