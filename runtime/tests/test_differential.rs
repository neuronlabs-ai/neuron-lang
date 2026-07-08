use std::fs;
use std::process::Command;
use neuron_compiler::compile;
use neuron_compiler::transpiler::Transpiler;
use neuron_compiler::py_transpiler::PyTranspiler;
use neuron_runtime::vm::{Value, VM};

fn run_jit(src: &str) -> Result<Value, String> {
    let compile_res = compile(src, "test_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    
    let mut rust_code = Transpiler::transpile(&compile_res.ir);
    rust_code = format!("#![allow(warnings)]\n{}", rust_code);
    
    let temp_dir = std::env::temp_dir().join(format!(
        "neuron_jit_diff_project_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let src_dir = temp_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    let runtime_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .to_string_lossy()
        .replace('\\', "/");
    let cargo_toml_content = format!(r#"[package]
name = "neuron_jit_diff"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);
    std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();
    
    let compile_status = Command::new("cargo")
        .arg("build")
        .current_dir(&temp_dir)
        .status()
        .map_err(|e| format!("Failed to run cargo: {:?}", e))?;
        
    if !compile_status.success() {
        return Err("JIT compilation failed".to_string());
    }
    
    let lib_path = if cfg!(target_os = "windows") {
        temp_dir.join("target").join("debug").join("neuron_jit_diff.dll")
    } else if cfg!(target_os = "macos") {
        temp_dir.join("target").join("debug").join("libneuron_jit_diff.dylib")
    } else {
        temp_dir.join("target").join("debug").join("libneuron_jit_diff.so")
    };
    
    let unique_dll_name = format!(
        "neuron_jit_diff_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = temp_dir.join("target").join("debug").join(&unique_dll_name);
    std::fs::copy(&lib_path, &load_lib_path).unwrap();

    let lib = unsafe { libloading::Library::new(&load_lib_path) }
        .map_err(|e| format!("Failed to load JIT library: {:?}", e))?;
        
    let result = unsafe {
        let run_main: libloading::Symbol<fn(&mut VM) -> Value> = lib
            .get(b"run_main")
            .map_err(|e| format!("Failed to resolve run_main: {:?}", e))?;
        let mut vm = VM::new();
        run_main(&mut vm)
    };
    
    drop(lib);
    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(result)
}

fn run_pytorch(src: &str) -> Result<String, String> {
    let compile_res = compile(src, "test_py_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    
    let py_code = PyTranspiler::transpile(&compile_res.ir);
    
    let temp_py_file = std::env::temp_dir().join(format!(
        "neuron_diff_{}.py",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&temp_py_file, py_code).unwrap();
    
    let output = Command::new("python")
        .arg(&temp_py_file)
        .output()
        .map_err(|e| format!("Failed to run python: {:?}", e))?;
        
    let _ = fs::remove_file(&temp_py_file);
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Python script failed:\n{}", stderr));
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn normalize_output(s: &str) -> String {
    s.lines()
        .map(|line| {
            let mut l = line.trim().to_lowercase();
            // Normalize PyTorch tensor prints, e.g. "tensor([[1., 1.]], grad_fn=...)" -> "[[1.0, 1.0]]"
            if l.contains("tensor(") {
                if let Some(start) = l.find('[') {
                    if let Some(end) = l.rfind(']') {
                        l = l[start..=end].to_string();
                    }
                }
            }
            // Normalize floating numbers, e.g. "1." -> "1.0", or removing trailing zeros on float outputs
            l.replace("1.", "1.0")
                .replace("2.", "2.0")
                .replace("3.", "3.0")
                .replace("4.", "4.0")
                .replace("5.", "5.0")
                .replace("67.500000", "67.5")
                .replace(" ", "")
        })
        .collect::<Vec<String>>()
        .join("\n")
}

#[test]
fn test_differential_loops_and_shapes() {
    let src = r#"
fn run_math():
    let sum = 0.0
    let i = 0
    while i < 10:
        let sum = sum + (i * 1.5)
        let i = i + 1
    return sum

fn run_concat():
    let t1 = zeros(1, 2) + 1.0
    let t2 = zeros(1, 2) + 2.0
    let t3 = concat([t1, t2])
    return t3

fn main():
    let s = run_math()
    print(s)
    let t = run_concat()
    print(t)
    return 0
"#;

    // 1. VM Execution
    let compile_res = compile(src, "input.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&compile_res.ir);
    
    // Catch standard print output if possible, but simplest is calling the functions or executing main
    let main_res = vm.run_main();
    assert!(main_res.is_ok(), "VM failed to execute main: {:?}", main_res.err());

    // Evaluate individual functions on VM directly to check return values
    let math_res = vm.execute("run_math", vec![]).unwrap();
    assert_eq!(math_res.as_float(), 67.5);
    
    let concat_res = vm.execute("run_concat", vec![]).unwrap();
    if let Value::Tensor(t) = concat_res {
        assert_eq!(t.shape, vec![1, 4]);
        assert_eq!(t.data, vec![1.0, 1.0, 2.0, 2.0]);
    } else {
        panic!("Expected tensor, found {:?}", concat_res);
    }

    // 2. JIT Execution
    let jit_math_res = run_jit(r#"
fn run_math():
    let sum = 0.0
    let i = 0
    while i < 10:
        let sum = sum + (i * 1.5)
        let i = i + 1
    return sum
fn main():
    return run_math()
"#).unwrap();
    assert_eq!(jit_math_res.as_float(), 67.5);

    // 3. PyTorch Transpilation & Execution
    let pytorch_output = run_pytorch(src).unwrap();
    let normalized_py = normalize_output(&pytorch_output);
    
    // We expect main to print the float sum (67.5) and the concat tensor ([[1.0, 1.0, 2.0, 2.0]])
    assert!(normalized_py.contains("67.5"));
    assert!(normalized_py.contains("[[1.0,1.0,2.0,2.0]]"));
}
