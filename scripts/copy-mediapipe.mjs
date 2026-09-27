import fs from 'node:fs';
import path from 'node:path';

const source = path.resolve('node_modules/@mediapipe/tasks-vision/wasm');
const target = path.resolve('public/wasm');
if (!fs.existsSync(source)) process.exit(0);
fs.mkdirSync(target, { recursive: true });
for (const file of fs.readdirSync(source)) {
  fs.copyFileSync(path.join(source, file), path.join(target, file));
}
console.log('Copied MediaPipe WASM assets to public/wasm');
