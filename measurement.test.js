import test from 'node:test';
import assert from 'node:assert/strict';
import { extractPoseDegrees, maxPairwiseDistance, measurePd } from './src/measurement.js';

test('maxPairwiseDistance finds circle diameter', () => {
  const points = [{ x: 0, y: -2 }, { x: 2, y: 0 }, { x: 0, y: 2 }, { x: -2, y: 0 }];
  assert.equal(maxPairwiseDistance(points), 4);
});

test('identity matrix yields zero pose', () => {
  const pose = extractPoseDegrees({ data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] });
  assert.ok(Math.abs(pose.yaw) < 1e-9);
  assert.ok(Math.abs(pose.pitch) < 1e-9);
  assert.ok(Math.abs(pose.roll) < 1e-9);
});

test('iris scale converts pixel PD to millimetres', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const width = 1000;
  const height = 1000;

  landmarks[468] = { x: 0.375, y: 0.5, z: 0 };
  landmarks[473] = { x: 0.625, y: 0.5, z: 0 };
  for (const [index, x, y] of [
    [469, 0.477, 0.5], [470, 0.5, 0.477], [471, 0.523, 0.5], [472, 0.5, 0.523],
    [474, 0.602, 0.5], [475, 0.625, 0.477], [476, 0.648, 0.5], [477, 0.625, 0.523],
  ]) landmarks[index] = { x, y, z: 0 };

  const result = measurePd({ landmarks, width, height, irisReferenceMm: 11.7 });
  assert.ok(Math.abs(result.pdPx - 250) < 1e-6);
  assert.ok(Math.abs(result.meanIrisPx - 46) < 1e-6);
  assert.ok(Math.abs(result.pdMm - (250 * 11.7 / 46)) < 1e-6);
});
