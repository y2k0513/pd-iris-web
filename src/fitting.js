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

/**
 * Remove the central upper contour where an eyelid commonly cuts across the
 * pupil/iris. Lower and lateral arc points are kept so a full circle can be
 * reconstructed from the visible perimeter.
 */
export function selectVisiblePupilArcPoints(points, {
  irisCenter,
  irisRadius,
  equivalentRadius,
} = {}) {
  if (!Array.isArray(points) || !irisCenter) return [];
  const safeIrisRadius = Math.max(1, irisRadius || equivalentRadius || 1);
  const safeEquivalentRadius = Math.max(1, equivalentRadius || safeIrisRadius * 0.55);
  const sideThreshold = Math.max(
    safeEquivalentRadius * 0.54,
    safeIrisRadius * 0.28,
  );

  return points.filter((point) => {
    const dx = point.x - irisCenter.x;
    const dy = point.y - irisCenter.y;
    const lowerOrMiddleArc = dy >= -safeIrisRadius * 0.17;
    const lateralArc = Math.abs(dx) >= sideThreshold
      && dy >= -safeIrisRadius * 0.58;
    return lowerOrMiddleArc || lateralArc;
  });
}

/**
 * Robustly fit a circle to a partial visible arc. Algebraic fitting supplies
 * the initial estimate, geometric least squares refines it, and MAD trimming
 * removes eyelid/threshold outliers.
 */
export function fitPartialArcCircle(points, {
  maxTrimIterations = 3,
  minPoints = 12,
} = {}) {
  if (!Array.isArray(points) || points.length < minPoints) return null;

  let working = points.map((point) => ({ x: point.x, y: point.y }));
  let circle = algebraicCircleFit(working);
  if (!circle) return null;

  for (let iteration = 0; iteration < maxTrimIterations; iteration += 1) {
    circle = refineCircleGeometrically(working, circle);
    const residuals = circleResiduals(working, circle);
    const residualMedian = median(residuals);
    const mad = median(residuals.map((value) => Math.abs(value - residualMedian)));
    const trimLimit = Math.max(1.25, residualMedian + Math.max(1.0, mad * 3.2));
    const trimmed = working.filter((_, index) => residuals[index] <= trimLimit);
    if (trimmed.length < minPoints || trimmed.length === working.length) break;
    working = trimmed;
    const refit = algebraicCircleFit(working);
    if (!refit) break;
    circle = refit;
  }

  circle = refineCircleGeometrically(working, circle);
  const residuals = circleResiduals(working, circle).sort((a, b) => a - b);
  const p90Index = Math.min(
    residuals.length - 1,
    Math.floor(residuals.length * 0.90),
  );

  return {
    ...circle,
    pointCount: working.length,
    originalPointCount: points.length,
    meanResidual: residuals.reduce((sum, value) => sum + value, 0) / residuals.length,
    p90Residual: residuals[p90Index],
    arcCoverage: angularCoverage(working, circle),
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
  const normalSelection = chooseBestPupilFit(candidates, thresholds);
  const partial = candidates.find((candidate) => candidate?.type === 'partial-arc-circle');
  const equivalent = candidates.find((candidate) => candidate?.type === 'equivalent-circle');

  if (!partial || !equivalent) return normalSelection;

  const expansion = Number.isFinite(partial.radiusExpansionRatio)
    ? partial.radiusExpansionRatio
    : partial.width / Math.max(1, equivalent.width);
  const residualLimit = Math.max(2.5, (partial.width / 2) * 0.14);
  const partialPlausible = partial.score >= Math.max(0.40, thresholds.minScore)
    && partial.iou >= Math.max(0.24, thresholds.minIou - 0.08)
    && partial.centerDistanceRatio <= Math.min(0.58, thresholds.maxCenterDistanceRatio)
    && partial.arcCoverage >= 0.46
    && partial.arcPointCount >= 12
    && partial.arcResidualP90 <= residualLimit
    && expansion >= 1.015
    && expansion <= 1.32;
  const competitive = !normalSelection.best
    || partial.score >= normalSelection.best.score - 0.16;

  if (partialPlausible && competitive) {
    return {
      best: partial,
      accepted: true,
      reason: 'occlusion-recovery',
      replaced: normalSelection.best,
    };
  }

  return normalSelection;
}
