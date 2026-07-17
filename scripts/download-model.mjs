import { createWriteStream } from 'node:fs';
import { mkdir, stat } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { finished } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const targetDir = path.join(root, 'public', 'models');
const target = path.join(targetDir, 'face_landmarker.task');

await mkdir(targetDir, { recursive: true });

try {
  const existing = await stat(target);
  if (existing.size > 1_000_000) {
    console.log(`Model already exists: ${target}`);
    process.exit(0);
  }
} catch {
  // Download below.
}

console.log('Downloading Face Landmarker model...');
const response = await fetch(MODEL_URL);
if (!response.ok || !response.body) {
  throw new Error(`Model download failed: HTTP ${response.status}`);
}
await finished(Readable.fromWeb(response.body).pipe(createWriteStream(target)));
console.log(`Saved: ${target}`);
