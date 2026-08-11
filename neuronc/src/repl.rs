/// NEURON REPL — Interactive Read-Eval-Print Loop.
///
/// A premium interactive terminal for the NEURON language with:
///   - Colored output and branding
///   - Multi-line input (auto-detect incomplete blocks)
///   - History navigation
///   - Special commands (:help, :load, :check, :type, :explain, :clear, :quit)
///   - Persistent VM state between evaluations
///   - Expression evaluation with automatic print

use std::io::{self, Write};
use neuron_compiler::{compile, compile_with_imports, types::TypeChecker, parser::Parser, lexer::Lexer};
use neuron_runtime::vm::VM;

// ── ANSI color codes ──
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";


pub fn run_repl() {
    print_banner();

    let mut accumulated_code = String::new();
    let mut vm = VM::new();
    let mut line_number: usize = 1;
    let mut multi_line_buffer = String::new();
    let mut in_multi_line = false;
    let mut history: Vec<String> = Vec::new();

    loop {
        // Print prompt
        if in_multi_line {
            print!("{CYAN}{BOLD}  ...│{RESET} ");
        } else {
            print!("{CYAN}{BOLD}  nr │{RESET} ");
        }
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                println!();
                print_goodbye();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                println!("{RED}Error reading input: {}{RESET}", e);
                break;
            }
        }

        let line = input.trim_end_matches('\n').trim_end_matches('\r');

        // Handle empty line in multi-line mode: submit the block
        if line.is_empty() && in_multi_line {
            let full_input = multi_line_buffer.trim().to_string();
            multi_line_buffer.clear();
            in_multi_line = false;
            if !full_input.is_empty() {
                history.push(full_input.clone());
                eval_input(&full_input, &mut accumulated_code, &mut vm, line_number);
                line_number += 1;
            }
            continue;
        }

        if line.is_empty() {
            continue;
        }

        // Handle special commands
        if line.starts_with(':') {
            handle_command(line, &mut accumulated_code, &mut vm, &history);
            continue;
        }

        // Check if this line starts a multi-line block (ends with ':')
        let trimmed = line.trim();
        if trimmed.ends_with(':') && !trimmed.contains("print(") {
            in_multi_line = true;
            multi_line_buffer = line.to_string();
            multi_line_buffer.push('\n');
            continue;
        }

        // If in multi-line mode, accumulate
        if in_multi_line {
            multi_line_buffer.push_str(line);
            multi_line_buffer.push('\n');
            continue;
        }

        // Single line evaluation
        history.push(line.to_string());
        eval_input(line, &mut accumulated_code, &mut vm, line_number);
        line_number += 1;
    }
}

fn eval_input(input: &str, accumulated_code: &mut String, _vm: &mut VM, _line_num: usize) {
    let trimmed = input.trim();

    // ── 1. Try as a top-level declaration (model, fn, let at module level) ──
    if trimmed.starts_with("model ") || trimmed.starts_with("fn ") || trimmed.contains("\n") {
        let decl_code = format!("{}\n{}\nfn __repl_run__():\n  return 0\n", accumulated_code, trimmed);
        match compile(&decl_code, "repl") {
            Ok(_output) => {
                accumulated_code.push_str("\n");
                accumulated_code.push_str(trimmed);
                println!("  {GREEN}✓{RESET} {DIM}defined{RESET}");
                return;
            }
            Err(_) => {}
        }
    }

    // ── 2. Try as expression (wrap in fn main and print) ──
    let expr_code = format!("{}\nfn main():\n  let __result = {}\n  print(__result)\n", accumulated_code, trimmed);
    if let Ok(output) = compile(&expr_code, "repl") {
        let mut eval_vm = VM::new();
        eval_vm.load(&output.ir);
        match eval_vm.run_main() {
            Ok(_) => {}
            Err(e) => {
                println!("  {RED}Runtime Error:{RESET} {}", e);
            }
        }
        return;
    }

    // ── 3. Try as a statement inside fn main (let bindings, update, etc.) ──
    let stmt_code = format!("{}\nfn main():\n  {}\n", accumulated_code, trimmed);
    match compile(&stmt_code, "repl") {
        Ok(output) => {
            let mut eval_vm = VM::new();
            eval_vm.load(&output.ir);
            match eval_vm.run_main() {
                Ok(result) => {
                    match result {
                        neuron_runtime::vm::Value::Void => {
                            println!("  {GREEN}✓{RESET}");
                        }
                        _ => {
                            println!("  {GREEN}={RESET} {}", result.display());
                        }
                    }
                }
                Err(e) => {
                    println!("  {RED}Runtime Error:{RESET} {}", e);
                }
            }
            return;
        }
        Err(_) => {}
    }

    // ── 4. Try wrapping bare expression as print ──
    let print_code = format!("{}\nfn main():\n  print({})\n", accumulated_code, trimmed);
    match compile(&print_code, "repl") {
        Ok(output) => {
            let mut eval_vm = VM::new();
            eval_vm.load(&output.ir);
            match eval_vm.run_main() {
                Ok(_) => {}
                Err(e) => {
                    println!("  {RED}Runtime Error:{RESET} {}", e);
                }
            }
            return;
        }
        Err(result) => {
            // Print the first compile error
            if let Some(err) = result.errors.first() {
                println!("  {RED}Error:{RESET} {}", err);
            }
        }
    }
}

fn handle_command(line: &str, accumulated_code: &mut String, vm: &mut VM, history: &[String]) {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = if parts.len() > 1 { parts[1].trim() } else { "" };

    match cmd {
        ":q" | ":quit" | ":exit" => {
            print_goodbye();
            std::process::exit(0);
        }
        ":help" | ":h" | ":?" => {
            print_help();
        }
        ":clear" | ":reset" => {
            accumulated_code.clear();
            *vm = VM::new();
            println!("  {GREEN}✓{RESET} {DIM}State cleared{RESET}");
        }
        ":type" => {
            if arg.is_empty() {
                println!("  {YELLOW}Usage:{RESET} :type <expression>");
            } else {
                handle_type_command(accumulated_code, arg);
            }
        }
        ":explain" => {
            if arg.is_empty() {
                println!("  {YELLOW}Usage:{RESET} :explain <expression>");
            } else {
                handle_explain_command(accumulated_code, arg);
            }
        }
        ":load" => {
            if arg.is_empty() {
                println!("  {YELLOW}Usage:{RESET} :load <file.nr>");
            } else {
                handle_load_command(arg, accumulated_code, vm);
            }
        }
        ":check" => {
            if arg.is_empty() {
                println!("  {YELLOW}Usage:{RESET} :check <file.py>");
            } else {
                handle_check_command(arg);
            }
        }
        ":history" | ":hist" => {
            if history.is_empty() {
                println!("  {DIM}No history yet{RESET}");
            } else {
                for (i, entry) in history.iter().enumerate() {
                    println!("  {DIM}[{}]{RESET} {}", i + 1, entry);
                }
            }
        }
        ":env" => {
            if accumulated_code.is_empty() {
                println!("  {DIM}No declarations in scope{RESET}");
            } else {
                println!("  {BOLD}Current declarations:{RESET}");
                for line in accumulated_code.lines() {
                    let l = line.trim();
                    if !l.is_empty() {
                        println!("  {DIM}│{RESET} {}", l);
                    }
                }
            }
        }
        _ => {
            println!("  {YELLOW}Unknown command:{RESET} {}", cmd);
            println!("  {DIM}Type :help for available commands{RESET}");
        }
    }
}

fn handle_type_command(accumulated_code: &str, expr: &str) {
    let test_code = format!("{}\nfn __repl_temp_expr__():\n  return {}\n", accumulated_code, expr);

    let tokens = match Lexer::new(&test_code).tokenize() {
        Ok(t) => t,
        Err(e) => {
            println!("  {RED}Lex Error:{RESET} {}", e);
            return;
        }
    };
    let program = match Parser::new(tokens, "repl_type").parse() {
        Ok(p) => p,
        Err(e) => {
            println!("  {RED}Parse Error:{RESET} {}", e);
            return;
        }
    };

    let mut checker = TypeChecker::new("repl_type");
    checker.check(&program);

    if checker.result.has_errors() {
        for err in &checker.result.errors {
            println!("  {RED}Type Error:{RESET} {}", err);
        }
    } else if let Some(ty) = checker.lookup("__repl_temp_expr__") {
        if let neuron_compiler::types::NType::Fn_(_, ret, _) = ty {
            println!("  {MAGENTA}Type:{RESET} {:?}", ret);
        } else {
            println!("  {MAGENTA}Type:{RESET} {:?}", ty);
        }
    } else {
        println!("  {DIM}Could not determine type{RESET}");
    }
}

fn handle_explain_command(accumulated_code: &str, expr: &str) {
    let explain_code = format!("{}\nfn main():\n  explain({})\n", accumulated_code, expr);
    match compile(&explain_code, "repl_explain") {
        Ok(output) => {
            let mut vm = VM::new();
            vm.load(&output.ir);
            match vm.run_main() {
                Ok(result) => {
                    println!("  {}", result.display());
                }
                Err(e) => {
                    println!("  {RED}Runtime Error:{RESET} {}", e);
                }
            }
        }
        Err(result) => {
            for err in &result.errors {
                println!("  {RED}Compile Error:{RESET} {}", err);
            }
        }
    }
}

fn handle_load_command(path: &str, accumulated_code: &mut String, vm: &mut VM) {
    match std::fs::read_to_string(path) {
        Ok(source) => {
            match compile_with_imports(&source, path) {
                Ok(output) => {
                    vm.load(&output.ir);
                    match vm.run_main() {
                        Ok(result) => {
                            // Add source to accumulated code (without fn main)
                            for line in source.lines() {
                                let l = line.trim();
                                if !l.starts_with("fn main") && !l.is_empty() {
                                    // Only accumulate top-level declarations
                                    if l.starts_with("model ") || l.starts_with("fn ") {
                                        accumulated_code.push_str("\n");
                                        accumulated_code.push_str(line);
                                    }
                                }
                            }
                            println!("  {GREEN}✓{RESET} Loaded and executed {BOLD}{}{RESET}", path);
                            match result {
                                neuron_runtime::vm::Value::Void => {}
                                _ => println!("  {GREEN}={RESET} {}", result.display()),
                            }
                        }
                        Err(e) => {
                            println!("  {RED}Runtime Error:{RESET} {}", e);
                        }
                    }
                }
                Err(result) => {
                    for err in &result.errors {
                        println!("  {RED}Compile Error:{RESET} {}", err);
                    }
                }
            }
        }
        Err(e) => {
            println!("  {RED}Error:{RESET} Cannot read '{}': {}", path, e);
        }
    }
}

fn handle_check_command(path: &str) {
    let output = std::process::Command::new("pycheck")
        .arg(path)
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
        }
        Err(_) => {
            // Try running as python module
            let output2 = std::process::Command::new("python")
                .args(&["-m", "pycheck", path])
                .output();
            match output2 {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if !stdout.is_empty() {
                        print!("{}", stdout);
                    }
                }
                Err(e) => {
                    println!("  {RED}Error:{RESET} Could not run pycheck: {}", e);
                }
            }
        }
    }
}

fn print_banner() {
    println!();
    println!("  {CYAN}{BOLD}╔══════════════════════════════════════════════════╗{RESET}");
    println!("  {CYAN}{BOLD}║{RESET}                                                  {CYAN}{BOLD}║{RESET}");
    println!("  {CYAN}{BOLD}║{RESET}   {BOLD}◈ NEURON{RESET}  {DIM}Interactive Terminal{RESET}                 {CYAN}{BOLD}║{RESET}");
    println!("  {CYAN}{BOLD}║{RESET}   {DIM}v{} — The AI-Native Programming Language{RESET}    {CYAN}{BOLD}║{RESET}", env!("CARGO_PKG_VERSION"));
    println!("  {CYAN}{BOLD}║{RESET}                                                  {CYAN}{BOLD}║{RESET}");
    println!("  {CYAN}{BOLD}║{RESET}   {DIM}Temporal Safety │ Differentiable │ Causal{RESET}    {CYAN}{BOLD}║{RESET}");
    println!("  {CYAN}{BOLD}║{RESET}                                                  {CYAN}{BOLD}║{RESET}");
    println!("  {CYAN}{BOLD}╚══════════════════════════════════════════════════╝{RESET}");
    println!();
    println!("  {DIM}Type :help for commands, :quit to exit{RESET}");
    println!("  {DIM}Every expression is differentiable. Try:{RESET} {GREEN}randn(3, 3){RESET}");
    println!();
}

fn print_help() {
    println!();
    println!("  {BOLD}NEURON REPL Commands{RESET}");
    println!("  {DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("  {CYAN}:help{RESET}     {DIM}│{RESET} Show this help message");
    println!("  {CYAN}:quit{RESET}     {DIM}│{RESET} Exit the REPL");
    println!("  {CYAN}:clear{RESET}    {DIM}│{RESET} Clear all state and declarations");
    println!("  {CYAN}:type{RESET} e   {DIM}│{RESET} Show the type of an expression");
    println!("  {CYAN}:explain{RESET} e{DIM}│{RESET} Explain a causal/temporal expression");
    println!("  {CYAN}:load{RESET} f   {DIM}│{RESET} Load and execute a .nr file");
    println!("  {CYAN}:check{RESET} f  {DIM}│{RESET} Run PyCheck on a .py file");
    println!("  {CYAN}:history{RESET}  {DIM}│{RESET} Show input history");
    println!("  {CYAN}:env{RESET}      {DIM}│{RESET} Show current declarations in scope");
    println!("  {DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();
    println!("  {BOLD}Quick Examples:{RESET}");
    println!("  {GREEN}randn(3, 3){RESET}                   {DIM}— Create a 3x3 random tensor{RESET}");
    println!("  {GREEN}randn(2, 2) @ randn(2, 2){RESET}     {DIM}— Matrix multiplication{RESET}");
    println!("  {GREEN}softmax(randn(1, 10)){RESET}          {DIM}— Softmax activation{RESET}");
    println!("  {GREEN}:load examples/transformer.nr{RESET}  {DIM}— Load a transformer model{RESET}");
    println!("  {GREEN}:check examples/zillow.py{RESET}      {DIM}— Scan for data leakage{RESET}");
    println!();
}

fn print_goodbye() {
    println!("  {DIM}Goodbye! ◈{RESET}");
}
