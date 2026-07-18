import test from 'node:test';
import assert from 'node:assert/strict';
import { fillSmallBlackHoles4Connected } from './src/binary.js';

function image(rows) {
  const height = rows.length;
  const width = rows[0].length;
  const data = new Uint8Array(width * height);
  rows.forEach((row, y) => {
    [...row].forEach((value, x) => {
      data[y * width + x] = value === '1' ? 255 : 0;
    });
  });
  return { data, width, height };
}

test('fills an enclosed black component below the pixel threshold', () => {
  const source = image([
    '0000000',
    '0111110',
    '0111110',
    '0110110',
    '0111110',
    '0111110',
    '0000000',
  ]);
  const result = fillSmallBlackHoles4Connected(source.data, source.width, source.height, 2);
  assert.equal(result.data[3 * source.width + 3], 255);
  assert.equal(result.filledComponentCount, 1);
  assert.equal(result.filledPixelCount, 1);
});

test('does not fill the exterior background or an oversized hole', () => {
  const source = image([
    '0000000',
    '0111110',
    '0111110',
    '0110010',
    '0110010',
    '0111110',
    '0000000',
  ]);
  const result = fillSmallBlackHoles4Connected(source.data, source.width, source.height, 3);
  assert.equal(result.data[0], 0);
  assert.equal(result.data[3 * source.width + 3], 0);
  assert.equal(result.filledComponentCount, 0);
});

test('uses 4-neighbour connectivity rather than diagonal connectivity', () => {
  const source = image([
    '00000',
    '01110',
    '01010',
    '01100',
    '00000',
  ]);
  const result = fillSmallBlackHoles4Connected(source.data, source.width, source.height, 1);
  // Center black pixel is only diagonally adjacent to exterior black pixels,
  // so it is an enclosed 4-connected component and gets filled.
  assert.equal(result.data[2 * source.width + 2], 255);
  assert.equal(result.filledComponentCount, 1);
});
