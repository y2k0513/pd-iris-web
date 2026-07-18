/**
 * Fill small black (0) holes enclosed by white (255) foreground.
 *
 * Connectivity is strictly 4-neighbour (up/down/left/right). Any black
 * component touching the image border is treated as the exterior background
 * and is never filled. This is equivalent to inverting the binary image,
 * labelling white components, removing small non-border components, and
 * inverting it back, but avoids two extra full-image inversions.
 */
export function fillSmallBlackHoles4Connected(
  binary,
  width,
  height,
  maxHolePixels,
) {
  if (!binary || !Number.isInteger(width) || !Number.isInteger(height)) {
    throw new TypeError('binary, width, height가 필요합니다.');
  }
  if (width <= 0 || height <= 0 || binary.length !== width * height) {
    throw new RangeError('이진 영상 크기가 올바르지 않습니다.');
  }

  const threshold = Math.max(0, Math.floor(Number(maxHolePixels) || 0));
  const output = new Uint8Array(binary.length);
  output.set(binary);

  if (threshold === 0) {
    return {
      data: output,
      filledComponentCount: 0,
      filledPixelCount: 0,
      maxHolePixels: threshold,
    };
  }

  const visited = new Uint8Array(binary.length);
  const queue = new Int32Array(binary.length);
  const component = new Int32Array(binary.length);
  let filledComponentCount = 0;
  let filledPixelCount = 0;

  for (let seed = 0; seed < binary.length; seed += 1) {
    if (visited[seed] || binary[seed] >= 128) continue;

    let head = 0;
    let tail = 0;
    let componentSize = 0;
    let touchesBorder = false;

    queue[tail++] = seed;
    visited[seed] = 1;

    while (head < tail) {
      const index = queue[head++];
      component[componentSize++] = index;

      const x = index % width;
      const y = Math.floor(index / width);
      if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
        touchesBorder = true;
      }

      // Left
      if (x > 0) {
        const next = index - 1;
        if (!visited[next] && binary[next] < 128) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
      // Right
      if (x + 1 < width) {
        const next = index + 1;
        if (!visited[next] && binary[next] < 128) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
      // Up
      if (y > 0) {
        const next = index - width;
        if (!visited[next] && binary[next] < 128) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
      // Down
      if (y + 1 < height) {
        const next = index + width;
        if (!visited[next] && binary[next] < 128) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
    }

    if (!touchesBorder && componentSize <= threshold) {
      for (let i = 0; i < componentSize; i += 1) {
        output[component[i]] = 255;
      }
      filledComponentCount += 1;
      filledPixelCount += componentSize;
    }
  }

  return {
    data: output,
    filledComponentCount,
    filledPixelCount,
    maxHolePixels: threshold,
  };
}
