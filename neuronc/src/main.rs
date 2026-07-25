/// neuronc — the NEURON Language Compiler CLI.
///
/// Usage:
///   neuronc check     <file.nr>   — type-check, print errors/warnings
///   neuronc build     <file.nr>   — compile to NEURON IR
///   neuronc run       <file.nr>   — compile and execute
///   neuronc jit       <file.nr>   — compile and execute using native Rust JIT compilation
///   neuronc aot       <file.nr>   — compile to standalone native executable binary
///   neuronc transpile <file.nr>   — transpile to PyTorch Python script
///   neuronc lsp                   — run Language Server Protocol engine over stdio
///   neuronc pycheck  <file.py>     — scan Python scripts for temporal, causal & uncertainty bugs
///
/// Exit codes: 0 = success, 1 = errors, 2 = warnings only

use std::env;
use std::fs;
use std::process;
use std::io::{Read, Write, BufWriter};

mod repl;
mod pkg;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "check" => {
            if args.len() < 3 {
                eprintln!("error: neuronc check requires a file argument");
                process::exit(1);
            }
            cmd_check(&args[2]);
        }
        "repl" => {
            repl::run_repl();
        }
        "add" => {
            if args.len() < 3 {
                eprintln!("error: neuronc add requires a dependency name");
                process::exit(1);
            }
            let dep_name = &args[2];
            let mut path = None;
            let mut git = None;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--path" && i + 1 < args.len() {
                    path = Some(args[i+1].as_str());
                    i += 2;
                } else if args[i] == "--git" && i + 1 < args.len() {
                    git = Some(args[i+1].as_str());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if path.is_none() && git.is_none() {
                path = Some("../");
            }
            if let Err(e) = pkg::add_dependency(".", dep_name, path, git) {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        }
        "build" => {
            let path = if args.len() >= 3 { &args[2] } else { "." };
            let path_obj = std::path::Path::new(path);
            if path_obj.is_dir() && path_obj.join("neuron.toml").exists() {
                match pkg::build_package(path) {
                    Ok(source) => {
                        match neuron_compiler::compile(&source, "package_build") {
                            Ok(output) => {
                                let nir_path = path_obj.join("target").join("package.nir");
                                std::fs::create_dir_all(path_obj.join("target")).unwrap();
                                std::fs::write(&nir_path, format!("{:?}", output.ir)).unwrap();
                                println!("✓ Package built successfully to {:?}", nir_path);
                            }
                            Err(result) => {
                                eprintln!("error: package compilation failed");
                                for err in result.errors {
                                    eprintln!("  {}", err);
                                }
                                process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                }
            } else {
                if args.len() < 3 {
                    eprintln!("error: neuronc build requires a file argument or a package directory with neuron.toml");
                    process::exit(1);
                }
                cmd_build(&args[2]);
            }
        }
        "run" => {
            if args.len() < 3 {
                eprintln!("error: neuronc run requires a file argument");
                process::exit(1);
            }
            let precision = parse_precision(&args[2..]);
            let file_arg = args[2..].iter().find(|a| !a.starts_with('-') && *a != "f32" && *a != "f64").unwrap_or(&args[2]);
            cmd_run(file_arg, precision);
        }
        "jit" => {
            if args.len() < 3 {
                eprintln!("error: neuronc jit requires a file argument");
                process::exit(1);
            }
            cmd_jit(&args[2]);
        }
        "aot" => {
            if args.len() < 3 {
                eprintln!("error: neuronc aot requires a file argument");
                process::exit(1);
            }
            let file_path = &args[2];
            let mut output_path = None;
            let mut i = 3;
            while i < args.len() {
                if (args[i] == "-o" || args[i] == "--output") && i + 1 < args.len() {
                    output_path = Some(args[i + 1].as_str());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            cmd_aot(file_path, output_path);
        }
        "transpile" => {
            if args.len() < 3 {
                eprintln!("error: neuronc transpile requires a file argument");
                process::exit(1);
            }
            let file_path = &args[2];
            let mut target = "python";
            let mut output_path = None;
            let mut i = 3;
            while i < args.len() {
                if args[i] == "--target" && i + 1 < args.len() {
                    target = &args[i + 1];
                    i += 2;
                } else if (args[i] == "-o" || args[i] == "--output") && i + 1 < args.len() {
                    output_path = Some(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if target != "python" {
                eprintln!("error: unsupported target '{}' (only 'python' is supported)", target);
                process::exit(1);
            }
            cmd_transpile(file_path, output_path.map(|s| s.as_str()));
        }
        "lsp" => {
            cmd_lsp();
        }
        "pycheck" => {
            if args.len() < 3 {
                eprintln!("error: neuronc pycheck requires a Python file argument");
                eprintln!("usage: neuronc pycheck <file.py>");
                process::exit(1);
            }
            cmd_pycheck(&args[2]);
        }
        "version" | "--version" | "-v" => {
            println!("neuronc {} — the NEURON Language Compiler", env!("CARGO_PKG_VERSION"));
            println!("Built for AGI model creation");
        }
        "help" | "--help" | "-h" => {
            print_usage();
        }
        _ => {
            eprintln!("error: unknown command '{}'. Use 'neuronc help' for usage.", command);
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
r#"neuronc — the NEURON Language Compiler

USAGE:
    neuronc <command> [options] <file.nr>

COMMANDS:
    check    Type-check a NEURON source file
    build    Compile to NEURON IR (produces .nir file, or builds package)
    run      Compile and execute a NEURON program
    jit      Compile and execute using native Rust JIT compilation
    aot      Compile to a standalone native binary
    transpile Transpile NEURON code to PyTorch Python script
    lsp      Run Language Server Protocol engine over stdio
    pycheck  Scan Python scripts for temporal, causal & uncertainty bugs
    repl     Start interactive NEURON REPL
    add      Add a local or git dependency to neuron.toml
    version  Print version information

FLAGS:
    -h, --help       Print help
    -v, --version    Print version

EXAMPLES:
    neuronc check  examples/transformer.nr
    neuronc run    examples/simple_shapes.nr
    neuronc aot    examples/colab_mlp.nr -o mlp_bin
"#
    );
}

fn read_source(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path, e);
            process::exit(1);
        }
    }
}

fn cmd_check(path: &str) {
    let source = read_source(path);
    let result = neuron_compiler::check_with_imports(&source, path);

    let mut exit_code = 0;

    if result.has_errors() {
        eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
        for err in &result.errors {
            eprintln!("  {}", err);
            eprintln!();
        }
        exit_code = 1;
    }

    if result.has_warnings() {
        eprintln!("\n{} — {} warning(s):\n", path, result.warnings.len());
        for warn in &result.warnings {
            eprintln!("  {}", warn);
            eprintln!();
        }
        if exit_code == 0 { exit_code = 0; }
    }

    if exit_code == 0 {
        eprintln!("✓ {} — no errors", path);
    }

    process::exit(exit_code);
}

fn cmd_build(path: &str) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            if output.result.has_warnings() {
                for warn in &output.result.warnings {
                    eprintln!("  {}", warn);
                }
            }

            let n_funcs = output.ir.functions.len();
            let n_globals = output.ir.globals.len();
            let total_ops: usize = output.ir.functions.iter().map(|f| f.blocks.iter().map(|b| b.instructions.len()).sum::<usize>()).sum();

            eprintln!("✓ {} — compiled to NEURON IR", path);
            eprintln!("  {} function(s), {} global(s), {} IR node(s)", n_funcs, n_globals, total_ops);

            for func in &output.ir.functions {
                let func_ops: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
                eprintln!("  fn {}({} params) → {} nodes",
                    func.name, func.params.len(), func_ops);
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn parse_precision(args: &[String]) -> neuron_runtime::tensor::DType {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--f32" {
            return neuron_runtime::tensor::DType::F32;
        } else if args[i] == "--f64" {
            return neuron_runtime::tensor::DType::F64;
        } else if args[i] == "--precision" && i + 1 < args.len() {
            if args[i + 1].to_lowercase() == "f32" {
                return neuron_runtime::tensor::DType::F32;
            } else if args[i + 1].to_lowercase() == "f64" {
                return neuron_runtime::tensor::DType::F64;
            }
        }
        i += 1;
    }
    neuron_runtime::tensor::DType::F64
}

fn cmd_run(path: &str, precision: neuron_runtime::tensor::DType) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            for warn in &output.result.warnings {
                eprintln!("  {}", warn);
            }

            let mut vm = neuron_runtime::vm::VM::new().with_precision(precision);
            vm.load(&output.ir);

            match vm.run_main() {
                Ok(result) => {
                    match result {
                        neuron_runtime::vm::Value::Void => {}
                        _ => println!("{}", result.display()),
                    }
                }
                Err(e) => {
                    eprintln!("\nRUNTIME ERROR: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_jit(path: &str) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            for warn in &output.result.warnings {
                eprintln!("  {}", warn);
            }

            let rust_code = neuron_compiler::transpiler::Transpiler::transpile(&output.ir);

            let temp_dir = std::env::temp_dir().join(format!("neuron_jit_{}", std::process::id()));
            let src_dir = temp_dir.join("src");
            std::fs::create_dir_all(&src_dir).unwrap();

            let runtime_path = find_runtime_path();
            let cargo_toml_content = format!(r#"[package]
name = "neuron_jit"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);
            std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
            std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();

            eprintln!("Compiling JIT library with cargo build --release...");
            let compile_start = std::time::Instant::now();
            let compile_status = std::process::Command::new("cargo")
                .arg("build")
                .arg("--release")
                .env("RUSTFLAGS", "-C target-cpu=native")
                .current_dir(&temp_dir)
                .status()
                .expect("Failed to run cargo build");

            if !compile_status.success() {
                eprintln!("error: JIT compilation failed");
                std::process::exit(1);
            }
            let compile_dur = compile_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("✓ JIT compilation completed in {:.2} ms", compile_dur);

            let lib_path = if cfg!(target_os = "windows") {
                temp_dir.join("target").join("release").join("neuron_jit.dll")
            } else if cfg!(target_os = "macos") {
                temp_dir.join("target").join("release").join("libneuron_jit.dylib")
            } else {
                temp_dir.join("target").join("release").join("libneuron_jit.so")
            };

            let lib = unsafe { libloading::Library::new(lib_path) }
                .expect("Failed to load compiled JIT library");

            let run_main: libloading::Symbol<fn(&mut neuron_runtime::vm::VM) -> neuron_runtime::vm::Value> = unsafe {
                lib.get(b"run_main")
            }.expect("Failed to resolve JIT run_main symbol");

            let mut vm = neuron_runtime::vm::VM::new();
            let run_start = std::time::Instant::now();
            let result = run_main(&mut vm);
            let run_dur = run_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("✓ JIT execution completed in {:.2} ms", run_dur);

            match result {
                neuron_runtime::vm::Value::Err(msg) => {
                    eprintln!("\nRUNTIME ERROR: {}", msg);
                    std::process::exit(1);
                }
                neuron_runtime::vm::Value::Void => {}
                _ => println!("{}", result.display()),
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_aot(path: &str, output_path: Option<&str>) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            for warn in &output.result.warnings {
                eprintln!("  {}", warn);
            }

            let rust_code = neuron_compiler::transpiler::Transpiler::transpile(&output.ir);

            let temp_dir = std::env::temp_dir().join(format!("neuron_aot_{}", std::process::id()));
            let src_dir = temp_dir.join("src");
            std::fs::create_dir_all(&src_dir).unwrap();

            let runtime_path = find_runtime_path();
            let cargo_toml_content = format!(r#"[package]
name = "neuron_aot"
version = "0.1.0"
edition = "2021"

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);

            let main_rs_content = format!(r#"
mod user_code {{
{}
}}

fn main() {{
    let mut vm = neuron_runtime::vm::VM::new();
    let res = user_code::main(&mut vm, vec![]);
    match res {{
        neuron_runtime::vm::Value::Void => {{}},
        _ => println!("{{}}", res.display()),
    }}
}}
"#, rust_code);

            std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
            std::fs::write(src_dir.join("main.rs"), main_rs_content).unwrap();

            eprintln!("Compiling Ahead-Of-Time (AOT) native binary with target-cpu=native...");
            let compile_start = std::time::Instant::now();
            let compile_status = std::process::Command::new("cargo")
                .arg("build")
                .arg("--release")
                .env("RUSTFLAGS", "-C target-cpu=native")
                .current_dir(&temp_dir)
                .status()
                .expect("Failed to run cargo build for AOT");

            if !compile_status.success() {
                eprintln!("error: AOT compilation failed");
                std::process::exit(1);
            }
            let compile_dur = compile_start.elapsed().as_secs_f64() * 1000.0;

            let built_bin = if cfg!(target_os = "windows") {
                temp_dir.join("target").join("release").join("neuron_aot.exe")
            } else {
                temp_dir.join("target").join("release").join("neuron_aot")
            };

            let dest_path = match output_path {
                Some(p) => std::path::PathBuf::from(p),
                None => {
                    let file_name = std::path::Path::new(path).file_stem().unwrap_or_default().to_string_lossy();
                    if cfg!(target_os = "windows") {
                        std::path::PathBuf::from(format!("{}.exe", file_name))
                    } else {
                        std::path::PathBuf::from(file_name.to_string())
                    }
                }
            };

            if let Err(e) = std::fs::copy(&built_bin, &dest_path) {
                eprintln!("error: failed to save output binary: {}", e);
                std::process::exit(1);
            }

            eprintln!("✓ AOT compilation completed in {:.2} ms", compile_dur);
            eprintln!("✓ Native binary written to {:?}", dest_path);
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn cmd_transpile(path: &str, output_path: Option<&str>) {
    let source = read_source(path);

    match neuron_compiler::compile_with_imports(&source, path) {
        Ok(output) => {
            if output.result.has_warnings() {
                for warn in &output.result.warnings {
                    eprintln!("  {}", warn);
                }
            }

            let py_code = neuron_compiler::py_transpiler::PyTranspiler::transpile(&output.ir);

            match output_path {
                Some(out) => {
                    if let Err(e) = std::fs::write(out, &py_code) {
                        eprintln!("error: failed to write output file: {}", e);
                        process::exit(1);
                    }
                    eprintln!("✓ Transpiled successfully to {}", out);
                }
                None => {
                    println!("{}", py_code);
                }
            }
        }
        Err(result) => {
            eprintln!("\n{} — {} error(s) found:\n", path, result.errors.len());
            for err in &result.errors {
                eprintln!("  {}", err);
                eprintln!();
            }
            process::exit(1);
        }
    }
}

fn find_runtime_path() -> String {
    if let Ok(p) = std::env::var("NEURON_RUNTIME_PATH") {
        return p.replace("\\", "/");
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if let Some(target) = parent.parent() {
                if let Some(root) = target.parent() {
                    let path = root.join("runtime");
                    if path.exists() {
                        return path.to_string_lossy().replace("\\", "/");
                    }
                }
            }
        }
    }
    let local_path = std::path::Path::new("runtime");
    if local_path.exists() {
        if let Ok(abs) = std::fs::canonicalize(local_path) {
            return abs.to_string_lossy().replace("\\", "/");
        }
    }
    "runtime".to_string()
}

/// Scan a Python file for temporal leaks, causal confusion, and uncertainty bugs.
fn cmd_pycheck(file_path: &str) {
    use std::process::Command;

    // Verify the file exists and is a .py file
    let path = std::path::Path::new(file_path);
    if !path.exists() {
        eprintln!("error: file '{}' not found", file_path);
        process::exit(1);
    }
    if path.extension().map_or(true, |e| e != "py") {
        eprintln!("error: neuronc pycheck only supports Python (.py) files");
        eprintln!("  got: {}", file_path);
        process::exit(1);
    }

    // Find the analyzer script relative to the neuronc binary
    let exe_path = std::env::current_exe().unwrap_or_default();
    let exe_dir = exe_path.parent().unwrap_or(std::path::Path::new("."));

    // Try multiple locations for the analyzer
    let analyzer_candidates = vec![
        std::path::PathBuf::from("pycheck/analyzer.py"),
        exe_dir.join("../../pycheck/analyzer.py"),
        exe_dir.join("../../../pycheck/analyzer.py"),
        exe_dir.join("pycheck/analyzer.py"),
    ];

    let analyzer_path = analyzer_candidates.iter()
        .find(|p| p.exists())
        .cloned();

    let analyzer = match analyzer_path {
        Some(p) => p,
        None => {
            eprintln!("error: could not find pycheck/analyzer.py");
            eprintln!("  searched in:");
            for c in &analyzer_candidates {
                eprintln!("    {}", c.display());
            }
            process::exit(1);
        }
    };

    // Run the analyzer via Python
    let output = Command::new("python")
        .env("PYTHONIOENCODING", "utf-8")
        .arg(analyzer.to_str().unwrap())
        .arg(file_path)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !stdout.is_empty() {
                print!("{}", stdout);
            }
            if !stderr.is_empty() {
                eprint!("{}", stderr);
            }
            // Exit with the same code as the analyzer
            if !out.status.success() {
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("error: failed to run Python: {}", e);
            eprintln!("  Is Python installed and in your PATH?");
            process::exit(1);
        }
    }
}

fn cmd_lsp() {
    use serde_json::{json, Value};

    eprintln!("[NEURON LSP] Language Server starting (stdio)...");

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    loop {
        // ── Read LSP header (Content-Length: N\r\n\r\n) ──
        let mut header = String::new();
        loop {
            let mut buf = [0u8; 1];
            if stdin.lock().read_exact(&mut buf).is_err() {
                return; // stdin closed
            }
            header.push(buf[0] as char);
            if header.ends_with("\r\n\r\n") {
                break;
            }
        }

        // Parse content length
        let content_length: usize = header
            .lines()
            .find_map(|line| {
                if line.starts_with("Content-Length:") {
                    line["Content-Length:".len()..].trim().parse().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);

        if content_length == 0 {
            continue;
        }

        // Read body
        let mut body = vec![0u8; content_length];
        if stdin.lock().read_exact(&mut body).is_err() {
            return;
        }

        let msg: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        match method {
            // ── Initialize ──
            "initialize" => {
                let result = json!({
                    "capabilities": {
                        "textDocumentSync": {
                            "openClose": true,
                            "change": 1,
                            "save": { "includeText": true }
                        },
                        "diagnosticProvider": {
                            "interFileDependencies": false,
                            "workspaceDiagnostics": false
                        },
                        "hoverProvider": true
                    },
                    "serverInfo": {
                        "name": "neuron-lsp",
                        "version": "1.0.0"
                    }
                });
                if let Some(req_id) = id {
                    lsp_send_response(&stdout, req_id, result);
                }
                eprintln!("[NEURON LSP] Initialized.");
            }

            "initialized" => {
                // Client acknowledged — no response needed
                eprintln!("[NEURON LSP] Client connected.");
            }

            // ── Document opened / saved / changed — run diagnostics ──
            "textDocument/didOpen" | "textDocument/didSave" | "textDocument/didChange" => {
                let params = msg.get("params").cloned().unwrap_or(json!({}));

                // Extract URI and text
                let (uri, text) = if method == "textDocument/didChange" {
                    let uri = params.get("textDocument")
                        .and_then(|td| td.get("uri"))
                        .and_then(|u| u.as_str())
                        .unwrap_or("")
                        .to_string();
                    let text = params.get("contentChanges")
                        .and_then(|cc| cc.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|change| change.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    (uri, text)
                } else {
                    let uri = params.get("textDocument")
                        .and_then(|td| td.get("uri"))
                        .and_then(|u| u.as_str())
                        .unwrap_or("")
                        .to_string();
                    let text_direct = params.get("textDocument")
                        .and_then(|td| td.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let text = if text_direct.is_empty() {
                        // didSave with includeText
                        params.get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string()
                    } else {
                        text_direct.to_string()
                    };
                    // If we still don't have text, try reading the file
                    let text = if text.is_empty() {
                        lsp_uri_to_path(&uri)
                            .and_then(|p| std::fs::read_to_string(&p).ok())
                            .unwrap_or_default()
                    } else {
                        text
                    };
                    (uri, text)
                };

                if text.is_empty() {
                    continue;
                }

                let filepath = lsp_uri_to_path(&uri).unwrap_or_else(|| uri.clone());

                // Run the NEURON type checker
                let result = neuron_compiler::check_with_imports(&text, &filepath);

                // Convert errors + warnings to LSP diagnostics
                let mut diagnostics: Vec<Value> = Vec::new();

                for err in &result.errors {
                    diagnostics.push(json!({
                        "range": {
                            "start": { "line": err.span.line.saturating_sub(1), "character": err.span.col.saturating_sub(1) },
                            "end": { "line": err.span.line.saturating_sub(1), "character": err.span.col.saturating_sub(1) + err.span.len }
                        },
                        "severity": 1, // Error
                        "source": "neuronc",
                        "code": format!("{:?}", err.code),
                        "message": format_diagnostic_message(
                            &err.message,
                            err.expected.as_deref(),
                            err.actual.as_deref(),
                            err.fix.as_deref(),
                            &err.notes,
                        )
                    }));
                }

                for warn in &result.warnings {
                    diagnostics.push(json!({
                        "range": {
                            "start": { "line": warn.span.line.saturating_sub(1), "character": warn.span.col.saturating_sub(1) },
                            "end": { "line": warn.span.line.saturating_sub(1), "character": warn.span.col.saturating_sub(1) + warn.span.len }
                        },
                        "severity": 2, // Warning
                        "source": "neuronc",
                        "code": format!("{:?}", warn.code),
                        "message": format_diagnostic_message(
                            &warn.message,
                            None,
                            None,
                            warn.fix.as_deref(),
                            &warn.notes,
                        )
                    }));
                }

                // Send diagnostics notification
                let notification = json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "diagnostics": diagnostics
                    }
                });

                lsp_send_notification(&stdout, notification);
                eprintln!("[NEURON LSP] Published {} diagnostic(s) for {}", diagnostics.len(), filepath);
            }

            // ── Hover — show type info ──
            "textDocument/hover" => {
                // Return basic hover info
                let hover_result = json!({
                    "contents": {
                        "kind": "markdown",
                        "value": "**NEURON** — AI-native programming language\n\nHover details coming soon."
                    }
                });
                if let Some(req_id) = id {
                    lsp_send_response(&stdout, req_id, hover_result);
                }
            }

            // ── Shutdown ──
            "shutdown" => {
                eprintln!("[NEURON LSP] Shutting down...");
                if let Some(req_id) = id {
                    lsp_send_response(&stdout, req_id, json!(null));
                }
            }

            "exit" => {
                eprintln!("[NEURON LSP] Exiting.");
                process::exit(0);
            }

            // Unknown method — ignore notifications, respond to requests
            _ => {
                if let Some(req_id) = id {
                    // It's a request — send method not found
                    let err_response = json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found: {}", method)
                        }
                    });
                    let body = serde_json::to_string(&err_response).unwrap();
                    let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
                    let mut out = BufWriter::new(stdout.lock());
                    let _ = out.write_all(msg.as_bytes());
                    let _ = out.flush();
                }
            }
        }
    }
}

/// Convert an LSP file URI to a local filesystem path.
fn lsp_uri_to_path(uri: &str) -> Option<String> {
    if uri.starts_with("file:///") {
        // Windows: file:///C:/path  →  C:/path
        let path = &uri["file:///".len()..];
        // URL-decode common sequences
        let decoded = path
            .replace("%20", " ")
            .replace("%3A", ":")
            .replace("%5C", "\\");
        Some(decoded)
    } else if uri.starts_with("file://") {
        Some(uri["file://".len()..].to_string())
    } else {
        Some(uri.to_string())
    }
}

/// Format a diagnostic message with expected/actual/fix/notes info.
fn format_diagnostic_message(
    message: &str,
    expected: Option<&str>,
    actual: Option<&str>,
    fix: Option<&str>,
    notes: &[String],
) -> String {
    let mut msg = message.to_string();
    if let Some(exp) = expected {
        msg.push_str(&format!("\n  expected: {}", exp));
    }
    if let Some(act) = actual {
        msg.push_str(&format!("\n  got: {}", act));
    }
    for note in notes {
        msg.push_str(&format!("\n  note: {}", note));
    }
    if let Some(f) = fix {
        msg.push_str(&format!("\n  help: {}", f));
    }
    msg
}

/// Send an LSP JSON-RPC response.
fn lsp_send_response(stdout: &std::io::Stdout, id: serde_json::Value, result: serde_json::Value) {
    use serde_json::json;
    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    });
    let body = serde_json::to_string(&response).unwrap();
    let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
    let mut out = BufWriter::new(stdout.lock());
    let _ = out.write_all(msg.as_bytes());
    let _ = out.flush();
}

/// Send an LSP JSON-RPC notification (no id).
fn lsp_send_notification(stdout: &std::io::Stdout, notification: serde_json::Value) {
    let body = serde_json::to_string(&notification).unwrap();
    let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
    let mut out = BufWriter::new(stdout.lock());
    let _ = out.write_all(msg.as_bytes());
    let _ = out.flush();
}

