use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct WasmResult {
    pub success: bool,
    pub output: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Type-check NEURON source code and return result as JSON string.
#[wasm_bindgen]
pub fn type_check(source: &str) -> String {
    let result = neuron_compiler::check_with_imports(source, "wasm_input");
    let has_errors = result.has_errors();
    let res = WasmResult {
        success: !has_errors,
        output: if !has_errors { "Type check passed cleanly.".to_string() } else { "Type check failed.".to_string() },
        errors: result.errors.iter().map(|e| e.to_string()).collect(),
        warnings: result.warnings.iter().map(|w| w.to_string()).collect(),
    };
    serde_json::to_string(&res).unwrap_or_default()
}

/// Compile NEURON source code to IR string representation (JSON).
#[wasm_bindgen]
pub fn compile_to_ir(source: &str) -> String {
    match neuron_compiler::compile_with_imports(source, "wasm_input") {
        Ok(output) => {
            let res = WasmResult {
                success: true,
                output: format!("{:#?}", output.ir),
                errors: vec![],
                warnings: output.result.warnings.iter().map(|w| w.to_string()).collect(),
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
        Err(result) => {
            let res = WasmResult {
                success: false,
                output: "".to_string(),
                errors: result.errors.iter().map(|e| e.to_string()).collect(),
                warnings: result.warnings.iter().map(|w| w.to_string()).collect(),
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
    }
}

/// Evaluate NEURON source code in the VM and return result output (JSON).
#[wasm_bindgen]
pub fn eval_neuron(source: &str) -> String {
    match neuron_compiler::compile_with_imports(source, "wasm_input") {
        Ok(output) => {
            let mut vm = neuron_runtime::vm::VM::new();
            vm.load(&output.ir);

            match vm.run_main() {
                Ok(val) => {
                    let display_str = match val {
                        neuron_runtime::vm::Value::Void => "Executed successfully (Void return)".to_string(),
                        v => v.display(),
                    };
                    let res = WasmResult {
                        success: true,
                        output: display_str,
                        errors: vec![],
                        warnings: output.result.warnings.iter().map(|w| w.to_string()).collect(),
                    };
                    serde_json::to_string(&res).unwrap_or_default()
                }
                Err(e) => {
                    let res = WasmResult {
                        success: false,
                        output: "".to_string(),
                        errors: vec![format!("Runtime Error: {}", e)],
                        warnings: output.result.warnings.iter().map(|w| w.to_string()).collect(),
                    };
                    serde_json::to_string(&res).unwrap_or_default()
                }
            }
        }
        Err(result) => {
            let res = WasmResult {
                success: false,
                output: "".to_string(),
                errors: result.errors.iter().map(|e| e.to_string()).collect(),
                warnings: result.warnings.iter().map(|w| w.to_string()).collect(),
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
    }
}

/// Transpile NEURON source code into PyTorch Python script string.
#[wasm_bindgen]
pub fn transpile_to_python(source: &str) -> String {
    match neuron_compiler::compile_with_imports(source, "wasm_input") {
        Ok(output) => {
            let python_script = neuron_compiler::py_transpiler::PyTranspiler::transpile(&output.ir);
            let res = WasmResult {
                success: true,
                output: python_script,
                errors: vec![],
                warnings: output.result.warnings.iter().map(|w| w.to_string()).collect(),
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
        Err(result) => {
            let res = WasmResult {
                success: false,
                output: "".to_string(),
                errors: result.errors.iter().map(|e| e.to_string()).collect(),
                warnings: result.warnings.iter().map(|w| w.to_string()).collect(),
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
    }
}

// ─────────────────────────────────────────────
// WASM LLM Direct Inference
// ─────────────────────────────────────────────

use std::sync::Mutex;
use neuron_runtime::neuron_lm::NeuronLM;

static WASM_LLM: Mutex<Option<NeuronLM>> = Mutex::new(None);

/// Load a GGUF model from a Uint8Array into WASM memory.
#[wasm_bindgen]
pub fn init_llm_gguf(model_bytes: &[u8]) -> String {
    match NeuronLM::new_from_gguf_bytes(model_bytes.to_vec()) {
        Ok(lm) => {
            let mut global = WASM_LLM.lock().unwrap();
            *global = Some(lm);
            let res = WasmResult {
                success: true,
                output: "TinyLlama GGUF model loaded cleanly into WASM memory.".to_string(),
                errors: vec![],
                warnings: vec![],
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
        Err(e) => {
            let res = WasmResult {
                success: false,
                output: "".to_string(),
                errors: vec![format!("Failed to load GGUF model: {}", e)],
                warnings: vec![],
            };
            serde_json::to_string(&res).unwrap_or_default()
        }
    }
}

/// Generate reply for a prompt using the loaded WASM LLM model.
#[wasm_bindgen]
pub fn generate_llm_reply(prompt: &str) -> String {
    let global = WASM_LLM.lock().unwrap();
    if let Some(ref lm) = *global {
        let reply = lm.generate_reply(prompt);
        let res = WasmResult {
            success: true,
            output: reply,
            errors: vec![],
            warnings: vec![],
        };
        serde_json::to_string(&res).unwrap_or_default()
    } else {
        let res = WasmResult {
            success: false,
            output: "".to_string(),
            errors: vec!["LLM model not initialized. Call init_llm_gguf first.".to_string()],
            warnings: vec![],
        };
        serde_json::to_string(&res).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_type_check() {
        let src = "fn main():\n  let x = 42\n  print(x)\n";
        let json_res = type_check(src);
        assert!(json_res.contains("\"success\":true"));
    }

    #[test]
    fn test_wasm_eval_neuron() {
        let src = "fn main() -> Int:\n  return 42\n";
        let json_res = eval_neuron(src);
        assert!(json_res.contains("42"));
        assert!(json_res.contains("\"success\":true"));
    }

    #[test]
    fn test_wasm_transpile_to_python() {
        let src = "fn main():\n  let x = randn(2, 2)\n";
        let json_res = transpile_to_python(src);
        assert!(json_res.contains("import torch"));
        assert!(json_res.contains("\"success\":true"));
    }
}
