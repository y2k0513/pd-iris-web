import { createWriteStream } from 'node:fs';
import { mkdir, stat, unlink } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { finished } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const targetDir = path.join(root, 'public', 'opencv');
const target = path.join(targetDir, 'opencv.js');
const temporary = `${target}.download`;

await mkdir(targetDir, { recursive: true });

try {
  const existing = await stat(target);
  if (existing.size > 5_000_000) {
    console.log(`OpenCV.js already exists: ${target}`);
    process.exit(0);
  }
} catch {
  // Download below.
}

console.log('Downloading OpenCV.js...');
const response = await fetch(OPENCV_URL);
if (!response.ok || !response.body) {
  throw new Error(`OpenCV.js download failed: HTTP ${response.status}`);
}

try {
  await finished(Readable.fromWeb(response.body).pipe(createWriteStream(temporary)));
  const downloaded = await stat(temporary);
  if (downloaded.size < 5_000_000) {
    throw new Error(`Downloaded OpenCV.js is unexpectedly small: ${downloaded.size} bytes`);
  }
  await unlink(target).catch(() => {});
  const { rename } = await import('node:fs/promises');
  await rename(temporary, target);
  console.log(`Saved: ${target}`);
} catch (error) {
  await unlink(temporary).catch(() => {});
  throw error;
}
