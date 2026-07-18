export const DEFAULT_PUPIL_FIT_THRESHOLDS = Object.freeze({
  minScore: 0.38,
  minIou: 0.30,
  maxCenterDistanceRatio: 0.80,
});

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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
