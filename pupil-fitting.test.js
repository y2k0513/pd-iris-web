import test from 'node:test';
import assert from 'node:assert/strict';

import {
  chooseBestPupilFit,
  scorePupilFitMetrics,
} from './src/fitting.js';

test('fit score rewards mask overlap and centered round shapes', () => {
  const strong = scorePupilFitMetrics({
    iou: 0.82,
    coverage: 0.94,
    precision: 0.86,
    centerDistanceRatio: 0.08,
    axisRatio: 0.96,
    diameterRatio: 0.72,
  });
  const weak = scorePupilFitMetrics({
    iou: 0.28,
    coverage: 0.52,
    precision: 0.40,
    centerDistanceRatio: 0.68,
    axisRatio: 0.58,
    diameterRatio: 1.35,
  });

  assert.ok(strong > 0.75);
  assert.ok(weak < strong);
});

test('best fit selection prefers overlap over candidate type', () => {
  const result = chooseBestPupilFit([
    { type: 'ellipse', score: 0.58, iou: 0.50, centerDistanceRatio: 0.15 },
    { type: 'equivalent-circle', score: 0.77, iou: 0.73, centerDistanceRatio: 0.10 },
  ]);

  assert.equal(result.accepted, true);
  assert.equal(result.best.type, 'equivalent-circle');
});

test('fit selection rejects a badly displaced component', () => {
  const result = chooseBestPupilFit([
    { type: 'circle', score: 0.68, iou: 0.67, centerDistanceRatio: 0.95 },
  ]);

  assert.equal(result.accepted, false);
  assert.equal(result.reason, 'below-threshold');
});

import {
  chooseOcclusionAwarePupilFit,
  fitPartialArcCircle,
  selectVisiblePupilArcPoints,
} from './src/fitting.js';

test('partial arc fitting reconstructs a circle hidden by an upper eyelid cut', () => {
  const center = { x: 80, y: 70 };
  const radius = 32;
  const contour = [];

  for (let degree = 0; degree < 360; degree += 3) {
    const angle = degree * Math.PI / 180;
    const x = center.x + radius * Math.cos(angle);
    const y = center.y + radius * Math.sin(angle);
    const upperCentralOcclusion = y < center.y - 8 && Math.abs(x - center.x) < 24;
    if (!upperCentralOcclusion) {
      const deterministicNoise = ((degree % 9) - 4) * 0.035;
      contour.push({
        x: x + deterministicNoise,
        y: y - deterministicNoise,
      });
    }
  }

  // Add a flat eyelid edge that should be rejected by visible-arc selection.
  for (let x = 60; x <= 100; x += 2) contour.push({ x, y: 48 });

  const visible = selectVisiblePupilArcPoints(contour, {
    irisCenter: center,
    irisRadius: 42,
    equivalentRadius: 29,
  });
  const fitted = fitPartialArcCircle(visible);

  assert.ok(fitted);
  assert.ok(Math.abs(fitted.x - center.x) < 1.5);
  assert.ok(Math.abs(fitted.y - center.y) < 1.5);
  assert.ok(Math.abs(fitted.radius - radius) < 1.5);
  assert.ok(fitted.arcCoverage > 0.5);
});

test('occlusion-aware selection can prefer a larger partial-arc reconstruction', () => {
  const candidates = [
    {
      type: 'equivalent-circle',
      x: 50,
      y: 50,
      width: 40,
      height: 40,
      score: 0.82,
      iou: 0.90,
      centerDistanceRatio: 0.05,
    },
    {
      type: 'partial-arc-circle',
      x: 50.5,
      y: 50.2,
      width: 44,
      height: 44,
      score: 0.72,
      iou: 0.70,
      centerDistanceRatio: 0.12,

      arcCoverage: 0.62,
      arcPointCount: 18,
      arcResidualP90: 1.2,
      arcInlierRatio: 0.78,
      arcSideBalance: 0.83,

      radiusExpansionRatio: 1.10,
      topOcclusionDetected: true,
    },
  ];

  const selection = chooseOcclusionAwarePupilFit(candidates);

  assert.equal(selection.accepted, true);
  assert.equal(selection.best.type, 'partial-arc-circle');
  assert.equal(selection.reason, 'ransac-occlusion-recovery');
});