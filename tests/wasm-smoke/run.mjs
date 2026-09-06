import { readFile } from 'node:fs/promises';

const bytes = await readFile(new URL('./target/wasm32-unknown-unknown/release/oximond_wasm_smoke.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(bytes, {});
const result = instance.exports.smoke();
if (result !== 0) throw new Error(`WASM smoke test failed with code ${result}`);
console.log('WASM interpreter smoke test passed for all compatibility profiles.');
