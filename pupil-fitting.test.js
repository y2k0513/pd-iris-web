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
