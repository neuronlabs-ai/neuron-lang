/* tslint:disable */
/* eslint-disable */

/**
 * Compile NEURON source code to IR string representation (JSON).
 */
export function compile_to_ir(source: string): string;

/**
 * Evaluate NEURON source code in the VM and return result output (JSON).
 */
export function eval_neuron(source: string): string;

/**
 * Generate reply for a prompt using the loaded WASM LLM model.
 */
export function generate_llm_reply(prompt: string): string;

/**
 * Load a GGUF model from a Uint8Array into WASM memory.
 */
export function init_llm_gguf(model_bytes: Uint8Array): string;

/**
 * Transpile NEURON source code into PyTorch Python script string.
 */
export function transpile_to_python(source: string): string;

/**
 * Type-check NEURON source code and return result as JSON string.
 */
export function type_check(source: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly compile_to_ir: (a: number, b: number) => [number, number];
    readonly eval_neuron: (a: number, b: number) => [number, number];
    readonly generate_llm_reply: (a: number, b: number) => [number, number];
    readonly init_llm_gguf: (a: number, b: number) => [number, number];
    readonly transpile_to_python: (a: number, b: number) => [number, number];
    readonly type_check: (a: number, b: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
