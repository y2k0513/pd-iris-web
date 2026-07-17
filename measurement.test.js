import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateFraming,
  extractPoseDegrees,
  maxPairwiseDistance,
  maxPairwiseDistance3D,
  measurePd,
  projectionFraction,
} from './src/measurement.js';

test('maxPairwiseDistance finds circle diameter', () => {
  const points = [{ x: 0, y: -2 }, { x: 2, y: 0 }, { x: 0, y: 2 }, { x: -2, y: 0 }];
  assert.equal(maxPairwiseDistance(points), 4);
});

test('projectionFraction returns midpoint', () => {
  assert.equal(projectionFraction({ x: 5, y: 0 }, { x: 0, y: 0 }, { x: 10, y: 0 }), 0.5);
});

test('identity matrix yields zero pose', () => {
  const pose = extractPoseDegrees({ data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] });
  assert.ok(Math.abs(pose.yaw) < 1e-9);
  assert.ok(Math.abs(pose.pitch) < 1e-9);
  assert.ok(Math.abs(pose.roll) < 1e-9);
});

test('framing calculates face size and center offsets', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  landmarks[0] = { x: 0.3, y: 0.2, z: 0 };
  landmarks[1] = { x: 0.7, y: 0.8, z: 0 };
  const framing = calculateFraming(landmarks);
  assert.ok(Math.abs(framing.faceWidthRatio - 0.4) < 1e-9);
  assert.ok(Math.abs(framing.faceHeightRatio - 0.6) < 1e-9);
  assert.ok(framing.centerOffsetX < 1e-9);
  assert.ok(framing.centerOffsetY < 1e-9);
});

test('iris scale converts pixel PD to millimetres and calculates symmetry', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const width = 1000;
  const height = 1000;

  landmarks[468] = { x: 0.375, y: 0.5, z: 0 };
  landmarks[473] = { x: 0.625, y: 0.5, z: 0 };
  for (const [index, x, y] of [
    [469, 0.352, 0.5], [470, 0.375, 0.477], [471, 0.398, 0.5], [472, 0.375, 0.523],
    [474, 0.602, 0.5], [475, 0.625, 0.477], [476, 0.648, 0.5], [477, 0.625, 0.523],
    [33, 0.32, 0.5], [133, 0.43, 0.5], [159, 0.375, 0.49], [145, 0.375, 0.51],
    [362, 0.57, 0.5], [263, 0.68, 0.5], [386, 0.625, 0.49], [374, 0.625, 0.51],
    [1, 0.5, 0.56],
  ]) landmarks[index] = { x, y, z: 0 };

  const result = measurePd({ landmarks, width, height, irisReferenceMm: 11.7 });
  assert.ok(Math.abs(result.pdPx - 250) < 1e-6);
  assert.ok(Math.abs(result.meanIrisPx - 46) < 1e-6);
  assert.ok(Math.abs(result.pdMm - (250 * 11.7 / 46)) < 1e-6);
  assert.ok(result.eyeAndPerspective.gazeOffset < 0.01);
  assert.ok(result.eyeAndPerspective.perspectiveAsymmetryRatio < 0.01);
});


test('3D iris ratio is retained for quality checking while final PD stays 2D', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const width = 1000;
  const height = 1000;

  landmarks[468] = { x: 0.375, y: 0.5, z: 0.01 };
  landmarks[473] = { x: 0.625, y: 0.5, z: -0.01 };
  for (const [index, x, y, z] of [
    [469, 0.352, 0.5, 0.01], [470, 0.375, 0.477, 0.01], [471, 0.398, 0.5, 0.01], [472, 0.375, 0.523, 0.01],
    [474, 0.602, 0.5, -0.01], [475, 0.625, 0.477, -0.01], [476, 0.648, 0.5, -0.01], [477, 0.625, 0.523, -0.01],
    [33, 0.32, 0.5, 0], [133, 0.43, 0.5, 0], [159, 0.375, 0.49, 0], [145, 0.375, 0.51, 0],
    [362, 0.57, 0.5, 0], [263, 0.68, 0.5, 0], [386, 0.625, 0.49, 0], [374, 0.625, 0.51, 0],
    [1, 0.5, 0.56, -0.03],
  ]) landmarks[index] = { x, y, z };

  const result = measurePd({ landmarks, width, height, irisReferenceMm: 11.7 });
  assert.ok(result.pdMm3D > result.pdMm2D);
  assert.ok(result.depthAware.disagreementRatio > 0);
  assert.ok(Math.abs(result.pdMm - result.pdMm2D) < 1e-9);
  assert.equal(result.depthAware.fusion3DWeight, 0);
});

test('maxPairwiseDistance3D includes z depth', () => {
  const points = [{ x: 0, y: 0, z: -2 }, { x: 0, y: 0, z: 2 }];
  assert.equal(maxPairwiseDistance3D(points), 4);
});
