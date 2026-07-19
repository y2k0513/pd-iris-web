import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ellipseResidual,
} from './src/iris.js';

test('ellipseResidual is near zero on an axis-aligned ellipse boundary', () => {
  const ellipse = {
    x: 100,
    y: 80,
    width: 40,
    height: 20,
    angle: 0,
  };

  assert.ok(
    ellipseResidual(
      { x: 120, y: 80 },
      ellipse,
    ) < 1e-9,
  );

  assert.ok(
    ellipseResidual(
      { x: 100, y: 90 },
      ellipse,
    ) < 1e-9,
  );
});

test('ellipseResidual handles rotated ellipses', () => {
  const ellipse = {
    x: 0,
    y: 0,
    width: 20,
    height: 10,
    angle: 90,
  };

  assert.ok(
    ellipseResidual(
      { x: 0, y: 10 },
      ellipse,
    ) < 1e-9,
  );

  assert.ok(
    ellipseResidual(
      { x: 5, y: 0 },
      ellipse,
    ) < 1e-9,
  );
});
