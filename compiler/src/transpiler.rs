use crate::ir::{IRProgram, IRFunction, IROp, IRConst, IRType, Terminator};

/// Transpiler converts a NEURON lowered IRProgram into optimized, standard Rust source code.
pub struct Transpiler;

impl Transpiler {
    pub fn transpile(program: &IRProgram) -> String {
        let mut rust_code = String::new();

        // 1. Generate Header and Imports
        rust_code.push_str(
r#"// ═══════════════════════════════════════════════════════════════════
//  NEURON JIT Transpiled Rust Source Code
// ═══════════════════════════════════════════════════════════════════
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::collections::HashMap;
use neuron_runtime::tensor::{Tensor, tensor_neg, tensor_gelu};
use neuron_runtime::vm::{Value, VM};

"#);

        // 2. Generate Global Initializers if any
        rust_code.push_str("// --- Globals Initialization ---\n");
        rust_code.push_str("pub fn initialize_globals(vm: &mut VM) {\n");
        for g in &program.globals {
            let val_str = match &g.value {
                IRConst::Int(v) => format!("Value::Int({})", v),
                IRConst::Float(v) => {
                    let mut s = format!("{:?}", v);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        s.push_str(".0");
                    }
                    format!("Value::Float({})", s)
                }
                IRConst::Bool(v) => format!("Value::Bool({})", v),
                IRConst::String(s) => format!("Value::Str(r#\"{}\"#.to_string())", s),
                IRConst::Tensor(data, shape) => {
                    format!("Value::Tensor(Tensor::new(vec!{:?}, vec!{:?}))", data, shape)
                }
            };
            rust_code.push_str(&format!("    vm.globals.insert({:?}.to_string(), {});\n", g.name, val_str));
        }
        rust_code.push_str("}\n\n");

        // 3. Generate dynamic dispatcher jit_obj_call
        rust_code.push_str("// --- Dynamic Method Dispatcher ---\n");
        rust_code.push_str("fn jit_obj_call(vm: &mut VM, fn_name: &str, args: Vec<Value>) -> Value {\n");
        rust_code.push_str("    let mut resolved_name = fn_name.to_string();\n");
        rust_code.push_str("    if fn_name.starts_with(\"obj_\") {\n");
        rust_code.push_str("        if let Some(Value::Model { name, .. }) = args.first() {\n");
        rust_code.push_str("            let method = &fn_name[4..];\n");
        rust_code.push_str("            resolved_name = format!(\"{}_{}\", name, method);\n");
        rust_code.push_str("        }\n");
        rust_code.push_str("    }\n");
        rust_code.push_str("    match resolved_name.as_str() {\n");
        
        // Populate dispatcher match arms
        for func in &program.functions {
            rust_code.push_str(&format!("        {:?} => {}(vm, args),\n", func.name, func.name));
            if func.name.ends_with("_new") {
                let model_name = &func.name[..func.name.len() - 4];
                rust_code.push_str(&format!("        {:?} => {}(vm, args),\n", model_name, func.name));
            }
        }
        rust_code.push_str("        _ => vm.execute(resolved_name.as_str(), args).unwrap_or_else(|e| panic!(\"Method '{}' failed: {}\", resolved_name, e)),\n");
        rust_code.push_str("    }\n");
        rust_code.push_str("}\n\n");

        let global_names: std::collections::HashSet<String> = program.globals.iter().map(|g| g.name.clone()).collect();
        let func_names: std::collections::HashSet<String> = program.functions.iter().map(|f| f.name.clone()).collect();

        // 4. Generate Functions
        for func in &program.functions {
            rust_code.push_str(&Self::transpile_function(func, &global_names, &func_names));
        }

        // 5. Generate run_main Entry Point
        rust_code.push_str(
r#"// --- Entry Point ---
#[no_mangle]
pub extern "Rust" fn run_main(vm: &mut VM) -> Value {
    initialize_globals(vm);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        main(vm, vec![])
    }));
    match result {
        Ok(val) => val,
        Err(err) => {
            let msg = if let Some(s) = err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic in JIT execution".to_string()
            };
            Value::Err(msg)
        }
    }
}
"#);

        // 6. Generate helper functions for VM-like operations
        rust_code.push_str(
r#"
// --- JIT Helper Functions ---

fn jit_add(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.add(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.add(&ta, tb))
        }
        _ => Value::Float(a_plain.as_float() + b_plain.as_float()),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_sub(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.sub(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.sub(&ta, tb))
        }
        _ => Value::Float(a_plain.as_float() - b_plain.as_float()),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_mul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.mul(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.mul(&ta, tb))
        }
        _ => Value::Float(a_plain.as_float() * b_plain.as_float()),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_div(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => Value::Tensor(vm.tape.div(ta, tb)),
        (Value::Tensor(ta), Value::Int(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y as f64);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(ta, &tb))
        }
        (Value::Tensor(ta), Value::Float(y)) => {
            let mut tb = Tensor::full(&ta.shape, *y);
            tb.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(ta, &tb))
        }
        (Value::Int(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x as f64);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(&ta, tb))
        }
        (Value::Float(x), Value::Tensor(tb)) => {
            let mut ta = Tensor::full(&tb.shape, *x);
            ta.id = vm.tape.alloc_id();
            Value::Tensor(vm.tape.div(&ta, tb))
        }
        _ => Value::Float(a_plain.as_float() / b_plain.as_float()),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_neg(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = match &a_plain {
        Value::Tensor(t) => Value::Tensor(vm.tape.neg(t)),
        _ => Value::Float(-a_plain.as_float()),
    };
    res.apply_wrappers(wraps)
}

fn jit_matmul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.ndim() < 2 || tb.ndim() < 2 {
                panic!("Runtime Error: MatMul requires 2D tensors, got {}D and {}D", ta.ndim(), tb.ndim());
            }
            let lhs_cols = ta.shape[ta.shape.len() - 1];
            let rhs_rows = tb.shape[tb.shape.len() - 2];
            if lhs_cols != rhs_rows {
                panic!("Runtime Error: Shape mismatch in MatMul: cannot multiply shape {:?} by {:?}", ta.shape, tb.shape);
            }
            Value::Tensor(vm.tape.matmul(ta, tb))
        }
        _ => panic!("MatMul requires tensor operands"),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_relu(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.relu(t))
    } else {
        Value::Float(a_plain.as_float().max(0.0))
    };
    res.apply_wrappers(wraps)
}

fn jit_sigmoid(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.sigmoid(t))
    } else {
        Value::Float(1.0 / (1.0 + (-a_plain.as_float()).exp()))
    };
    res.apply_wrappers(wraps)
}

fn jit_tanh(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.tanh(t))
    } else {
        Value::Float(a_plain.as_float().tanh())
    };
    res.apply_wrappers(wraps)
}

fn jit_softmax(vm: &mut VM, a: &Value, dim: i64) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.softmax(t))
    } else {
        a_plain
    };
    res.apply_wrappers(wraps)
}

fn jit_mse(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                panic!("Runtime Error: MSELoss shape mismatch: pred {:?} vs target {:?}", ta.shape, tb.shape);
            }
            Value::Tensor(vm.tape.mse(ta, tb))
        }
        _ => Value::Float(0.0),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_lt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() < b.as_float())
}

fn jit_lte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() <= b.as_float())
}

fn jit_gt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() > b.as_float())
}

fn jit_gte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() >= b.as_float())
}

fn jit_eq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x == y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x == y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x == y),
        _ => Value::Bool(a.as_float() == b.as_float()),
    }
}

fn jit_neq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x != y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x != y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x != y),
        _ => Value::Bool(a.as_float() != b.as_float()),
    }
}

fn jit_list_len(a: &Value) -> Value {
    if let Value::List(ref l) = a {
        Value::Int(l.len() as i64)
    } else { Value::Int(0) }
}

fn jit_index(a: &Value, idx: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let idx_val = idx.as_int();
    if idx_val < 0 {
        panic!("Runtime Error: Negative index {} is not allowed", idx_val);
    }
    let i = idx_val as usize;
    let res = match a_plain {
        Value::List(items) => {
            items.get(i).cloned().unwrap_or(Value::Void)
        }
        Value::Tensor(t) => {
            if t.ndim() == 2 {
                let cols = t.shape[1];
                let start = i * cols;
                let end = start + cols;
                if end <= t.data.len() {
                    let row_data = t.data[start..end].to_vec();
                    Value::Tensor(Tensor::new(row_data, vec![1, cols]))
                } else {
                    Value::Void
                }
            } else if t.ndim() == 1 {
                if i < t.data.len() {
                    Value::Float(t.data[i])
                } else {
                    Value::Void
                }
            } else {
                Value::Void
            }
        }
        _ => Value::Void,
    };
    res.apply_wrappers(wraps)
}

fn jit_stop_grad(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(t) => {
            let mut t_clone = t.clone();
            t_clone.requires_grad = false;
            t_clone.tape_entry = None;
            vm.tape.detach(t_clone.id);
            Value::Tensor(t_clone)
        }
        other => other.clone(),
    }
}

fn jit_concat(vm: &mut VM, a: &Value, dim: i64) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    if let Value::List(items) = &a_plain {
        let stripped_items: Vec<Value> = items.iter().map(|v| v.clone().strip_wrappers().0).collect();
        let tensors: Vec<&Tensor> = stripped_items.iter().filter_map(|v| v.as_tensor()).collect();

        if !tensors.is_empty() {
            let all_2d = tensors.iter().all(|t| t.ndim() == 2);
            let same_batch = all_2d && {
                let b = tensors[0].shape[0];
                tensors.iter().all(|t| t.shape[0] == b)
            };
            if same_batch {
                let b = tensors[0].shape[0];
                let d_total: usize = tensors.iter().map(|t| t.shape[1]).sum();
                let total_elements = b.checked_mul(d_total).expect("Runtime Error: Concat shape size overflow");
                if total_elements > 10_000_000 {
                    panic!("Runtime Error: Tensor size too large in Concat: {} elements", total_elements);
                }
                let mut data = vec![0.0; b * d_total];
                for row in 0..b {
                    let mut col_offset = 0;
                    for t in &tensors {
                        let t_cols = t.shape[1];
                        let start = row * t_cols;
                        let end = start + t_cols;
                        let dest_start = row * d_total + col_offset;
                        data[dest_start..dest_start + t_cols].copy_from_slice(&t.data[start..end]);
                        col_offset += t_cols;
                    }
                }
                return Value::Tensor(Tensor::new(data, vec![b, d_total])).apply_wrappers(wraps);
            } else {
                let total_len: usize = tensors.iter().map(|t| t.numel()).sum();
                if total_len > 10_000_000 {
                    panic!("Runtime Error: Tensor size too large in Concat: {} elements", total_len);
                }
                let mut data = Vec::with_capacity(total_len);
                for t in &tensors { data.extend_from_slice(&t.data); }
                return Value::Tensor(Tensor::new(data, vec![total_len])).apply_wrappers(wraps);
            }
        }
    }
    Value::Tensor(Tensor::zeros(&[0]))
}

fn jit_gelu(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.gelu(t))
    } else {
        a_plain
    };
    res.apply_wrappers(wraps)
}

fn jit_cross_entropy(vm: &mut VM, a: &Value, b: &Value) -> Value {
    let (a_plain, a_wraps) = a.clone().strip_wrappers();
    let (b_plain, b_wraps) = b.clone().strip_wrappers();
    let res = match (&a_plain, &b_plain) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                panic!("Runtime Error: CrossEntropy shape mismatch: pred {:?} vs target {:?}", ta.shape, tb.shape);
            }
            Value::Tensor(vm.tape.cross_entropy(ta, tb))
        }
        _ => Value::Float(0.0),
    };
    res.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps))
}

fn jit_sum(vm: &mut VM, a: &Value, dim: Option<i64>) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        if let Some(d) = dim {
            let ndim = t.ndim() as i64;
            if d < 0 || d >= ndim {
                panic!("Runtime Error: Sum dimension {} out of bounds for tensor of rank {}", d, ndim);
            }
        }
        Value::Tensor(vm.tape.sum(t, dim.map(|d| d as usize)))
    } else {
        a_plain
    };
    res.apply_wrappers(wraps)
}

fn jit_mean(vm: &mut VM, a: &Value, dim: Option<i64>) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        if let Some(d) = dim {
            let ndim = t.ndim() as i64;
            if d < 0 || d >= ndim {
                panic!("Runtime Error: Mean dimension {} out of bounds for tensor of rank {}", d, ndim);
            }
        }
        Value::Tensor(vm.tape.mean(t, dim.map(|d| d as usize)))
    } else {
        a_plain
    };
    res.apply_wrappers(wraps)
}

fn jit_sqrt(vm: &mut VM, a: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let res = if let Value::Tensor(t) = &a_plain {
        Value::Tensor(vm.tape.sqrt(t))
    } else {
        match a_plain {
            Value::Float(f) => Value::Float(f.sqrt()),
            Value::Int(i) => Value::Float((i as f64).sqrt()),
            _ => a_plain,
        }
    };
    res.apply_wrappers(wraps)
}

fn jit_update_row(vm: &mut VM, a: &Value, idx: &Value, row: &Value) -> Value {
    let (a_plain, wraps) = a.clone().strip_wrappers();
    let (idx_plain, _) = idx.clone().strip_wrappers();
    let (row_plain, _) = row.clone().strip_wrappers();
    let res = if let (Value::Tensor(t), Value::Tensor(r)) = (&a_plain, &row_plain) {
        let idx_val = idx_plain.as_int();
        if idx_val < 0 {
            panic!("Runtime Error: Negative index {} is not allowed in UpdateRow", idx_val);
        }
        let i = idx_val as usize;
        let row_len = r.numel();
        if t.shape.len() >= 2 && row_len != t.shape[1] {
            panic!("Runtime Error: Row width mismatch in UpdateRow: expected {}, got {}", t.shape[1], row_len);
        }
        let mut new_data = t.data.clone();
        let start = i * row_len;
        if start + row_len <= new_data.len() {
            new_data[start..start + row_len].copy_from_slice(&r.data[..row_len]);
            Value::Tensor(Tensor::new(new_data, t.shape.clone()))
        } else {
            panic!("Runtime Error: Index {} out of bounds for UpdateRow on tensor of size {}", i, t.shape[0]);
        }
    } else {
        a_plain
    };
    res.apply_wrappers(wraps)
}

fn jit_validate_shape(shape_i64: Vec<i64>) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut product: usize = 1;
    for val in shape_i64 {
        if val < 0 {
            panic!("Runtime Error: Negative dimension size: {}", val);
        }
        let dim = val as usize;
        if dim > 1_000_000 {
            panic!("Runtime Error: Dimension size too large: {}", dim);
        }
        product = product.checked_mul(dim).expect("Runtime Error: Shape size overflow");
        shape.push(dim);
    }
    if product > 10_000_000 {
        panic!("Runtime Error: Tensor size too large: {} elements", product);
    }
    shape
}

fn jit_mod(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            if *y == 0 {
                panic!("Runtime Error: Division by zero in modulo operation");
            }
            Value::Int(x % y)
        }
        _ => {
            let y_val = b.as_float();
            if y_val == 0.0 {
                panic!("Runtime Error: Division by zero in modulo operation");
            }
            Value::Float(a.as_float() % y_val)
        }
    }
}

fn jit_and(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_bool() && b.as_bool())
}

fn jit_or(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_bool() || b.as_bool())
}

fn jit_not(a: &Value) -> Value {
    Value::Bool(!a.as_bool())
}
"#);
        rust_code
    }

    fn transpile_function(func: &IRFunction, global_names: &std::collections::HashSet<String>, func_names: &std::collections::HashSet<String>) -> String {
        let mut f_code = String::new();
        f_code.push_str(&format!("pub fn {}(vm: &mut VM, args: Vec<Value>) -> Value {{\n", func.name));
        f_code.push_str("    let mut locals = HashMap::<String, Value>::new();\n");

        let is_constructor = func.name.ends_with("_new");
        if is_constructor {
            let model_name = &func.name[..func.name.len() - 4];
            f_code.push_str(&format!(
r#"    locals.insert("self".to_string(), Value::Model {{
        name: {:?}.to_string(),
        fields: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
    }});
"#, model_name));
        }

        // Bind arguments
        for (i, param) in func.params.iter().enumerate() {
            f_code.push_str(&format!("    let mut v{} = args[{}].clone();\n", param.id, i));
            f_code.push_str(&format!("    locals.insert({:?}.to_string(), v{}.clone());\n", param.name, param.id));
        }

        // Declare all SSA values mutably at the top
        let mut all_ids = std::collections::BTreeSet::new();
        for block in &func.blocks {
            for node in &block.instructions {
                all_ids.insert(node.id);
            }
        }
        for id in all_ids {
            f_code.push_str(&format!("    let mut v{} = Value::Void;\n", id));
        }

        f_code.push_str(&format!("    let mut current_block = {};\n", func.entry));
        f_code.push_str("    loop {\n");
        f_code.push_str("        match current_block {\n");

        for block in &func.blocks {
            f_code.push_str(&format!("            {} => {{\n", block.id));
            for node in &block.instructions {
                match &node.op {
                    IROp::Store { name } => {
                        if name.starts_with("self.") {
                            let field = &name[5..];
                            f_code.push_str(&format!(
r#"                if let Some(Value::Model {{ fields, .. }}) = locals.get("self") {{
                    fields.borrow_mut().insert({:?}.to_string(), v{}.clone());
                }}
"#, field, node.inputs[0]));
                        } else {
                            if global_names.contains(name) {
                                f_code.push_str(&format!(
r#"                locals.insert({:?}.to_string(), v{}.clone());
                vm.globals.insert({:?}.to_string(), v{}.clone());
"#, name, node.inputs[0], name, node.inputs[0]));
                            } else {
                                f_code.push_str(&format!(
r#"                locals.insert({:?}.to_string(), v{}.clone());
"#, name, node.inputs[0]));
                            }
                        }
                    }
                    _ => {
                        let op_code = match &node.op {
                            IROp::Const(c) => match c {
                                IRConst::Int(v) => format!("Value::Int({})", v),
                                IRConst::Float(v) => {
                                    let mut s = format!("{:?}", v);
                                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                                        s.push_str(".0");
                                    }
                                    format!("Value::Float({})", s)
                                }
                                IRConst::Bool(v) => format!("Value::Bool({})", v),
                                IRConst::String(s) => format!("Value::Str(r#\"{}\"#.to_string())", s),
                                IRConst::Tensor(data, shape) => {
                                    format!("Value::Tensor(Tensor::new(vec!{:?}, vec!{:?}))", data, shape)
                                }
                            },
                            IROp::Zeros(shape) => {
                                let shape_str = if !node.inputs.is_empty() {
                                    let parts: Vec<String> = node.inputs.iter().map(|id| format!("v{}.as_int()", id)).collect();
                                    format!("jit_validate_shape(vec![{}])", parts.join(", "))
                                } else {
                                    format!("jit_validate_shape(vec!{:?})", shape)
                                };
                                format!(
                                    "{{ let mut t = Tensor::zeros(&{}); t.id = vm.tape.alloc_id(); Value::Tensor(t) }}",
                                    shape_str
                                )
                            }
                            IROp::Ones(shape) => {
                                let shape_str = if !node.inputs.is_empty() {
                                    let parts: Vec<String> = node.inputs.iter().map(|id| format!("v{}.as_int()", id)).collect();
                                    format!("jit_validate_shape(vec![{}])", parts.join(", "))
                                } else {
                                    format!("jit_validate_shape(vec!{:?})", shape)
                                };
                                format!(
                                    "{{ let mut t = Tensor::ones(&{}); t.id = vm.tape.alloc_id(); Value::Tensor(t) }}",
                                    shape_str
                                )
                            }
                            IROp::Glorot(shape) => {
                                let shape_str = if !node.inputs.is_empty() {
                                    let parts: Vec<String> = node.inputs.iter().map(|id| format!("v{}.as_int()", id)).collect();
                                    format!("jit_validate_shape(vec![{}])", parts.join(", "))
                                } else {
                                    format!("jit_validate_shape(vec!{:?})", shape)
                                };
                                format!(
                                    "{{ let mut t = Tensor::glorot(&{}); t.id = vm.tape.alloc_id(); Value::Tensor(t) }}",
                                    shape_str
                                )
                            }
                            IROp::Randn(shape) => {
                                let shape_str = if !node.inputs.is_empty() {
                                    let parts: Vec<String> = node.inputs.iter().map(|id| format!("v{}.as_int()", id)).collect();
                                    format!("jit_validate_shape(vec![{}])", parts.join(", "))
                                } else {
                                    format!("jit_validate_shape(vec!{:?})", shape)
                                };
                                format!(
                                    "{{ let mut t = Tensor::randn(&{}); t.id = vm.tape.alloc_id(); Value::Tensor(t) }}",
                                    shape_str
                                )
                            }
                            IROp::Add => format!("jit_add(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Sub => format!("jit_sub(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Mul => format!("jit_mul(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Div => format!("jit_div(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Mod => format!("jit_mod(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Neg => format!("jit_neg(vm, &v{})", node.inputs[0]),
                            IROp::MatMul => format!("jit_matmul(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::ReLU => format!("jit_relu(vm, &v{})", node.inputs[0]),
                            IROp::Sigmoid => format!("jit_sigmoid(vm, &v{})", node.inputs[0]),
                            IROp::Tanh => format!("jit_tanh(vm, &v{})", node.inputs[0]),
                            IROp::Softmax { dim } => format!("jit_softmax(vm, &v{}, {})", node.inputs[0], dim),
                            IROp::MSELoss => format!("jit_mse(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::GeLU => format!("jit_gelu(vm, &v{})", node.inputs[0]),
                            IROp::CrossEntropy => format!("jit_cross_entropy(vm, &v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Lt => format!("jit_lt(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Lte => format!("jit_lte(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Gt => format!("jit_gt(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Gte => format!("jit_gte(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Eq => format!("jit_eq(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Neq => format!("jit_neq(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::And => format!("jit_and(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Or => format!("jit_or(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::Not => format!("jit_not(&v{})", node.inputs[0]),
                            IROp::ListLen => format!("jit_list_len(&v{})", node.inputs[0]),
                            IROp::Index => format!("jit_index(&v{}, &v{})", node.inputs[0], node.inputs[1]),
                            IROp::StopGrad => format!("jit_stop_grad(vm, &v{})", node.inputs[0]),
                            IROp::Sum { dim } => {
                                format!("jit_sum(vm, &v{}, {:?})", node.inputs[0], dim)
                            }
                            IROp::Mean { dim } => {
                                format!("jit_mean(vm, &v{}, {:?})", node.inputs[0], dim)
                            }
                            IROp::Sqrt => {
                                format!("jit_sqrt(vm, &v{})", node.inputs[0])
                            }
                            IROp::Reshape(new_shape) => {
                                let shape_str = format!("vec!{:?}", new_shape.iter().map(|&x| x as usize).collect::<Vec<_>>());
                                let new_shape_debug = format!("{:?}", new_shape);
                                format!(
            r#"if let Value::Tensor(t) = &v{input} {{
                    let target_shape = {shape};
                    let new_n: usize = target_shape.iter().product();
                    if t.numel() != new_n {{
                        panic!("Runtime Error: Cannot reshape tensor of {{:?}} to {debug_shape}", t.shape);
                    }}
                    Value::Tensor(t.reshape(&target_shape))
                }} else {{ v{input}.clone() }}"#,
                                    input = node.inputs[0],
                                    shape = shape_str,
                                    debug_shape = new_shape_debug)
                            }
                            IROp::UpdateRow => {
                                format!("jit_update_row(vm, &v{}, &v{}, &v{})", node.inputs[0], node.inputs[1], node.inputs[2])
                            }
                            
                            IROp::Grad { wrt } => {
                                let wrt_str = match wrt {
                                    Some(ref w) => format!("Some({:?}.to_string())", w),
                                    None => "None".to_string(),
                                };
                                format!(
            r#"{{
                    let (val_plain, _wraps) = v{}.clone().strip_wrappers();
                    let val = &val_plain;
                    if let Value::Tensor(t) = val {{
                        vm.call_stack.push(neuron_runtime::vm::CallFrame {{
                            function_name: String::new(),
                            current_block: 0,
                            instruction_idx: 0,
                            locals: locals.clone(),
                            ssa_values: HashMap::new(),
                            return_addr: None,
                            fused_groups: HashMap::new(),
                            fused_skipped_ids: std::collections::HashSet::new(),
                        }});
                        let parameter_ids = vm.collect_parameter_ids();
                        vm.tape.parameter_ids = parameter_ids;
                        vm.tape.backward(t.id);
                        vm.call_stack.pop();
                        let mut found = None;
                        if let Some(ref param_name) = {} {{
                            if let Some(Value::Tensor(param)) = locals.get(param_name) {{
                                if let Some(grad_data) = vm.tape.get_grad(param.id) {{
                                    found = Some(Value::Tensor(Tensor::new(grad_data.clone(), param.shape.clone())));
                                }}
                            }}
                        }}
                        if found.is_none() {{
                            if let Some(grad_data) = vm.tape.get_grad(t.id) {{
                                found = Some(Value::Tensor(Tensor::new(grad_data.clone(), t.shape.clone())));
                            }}
                        }}
                        found.unwrap_or_else(|| Value::Tensor(Tensor::zeros(&[1])))
                    }} else {{
                        Value::Tensor(Tensor::zeros(&[1]))
                    }}
                }}"#, node.inputs[0], wrt_str)
                            }

                            IROp::Backward => {
                                format!(
            r#"{{
                    let (val_plain, _wraps) = v{}.clone().strip_wrappers();
                    if let Value::Tensor(t) = &val_plain {{
                        vm.call_stack.push(neuron_runtime::vm::CallFrame {{
                            function_name: String::new(),
                            current_block: 0,
                            instruction_idx: 0,
                            locals: locals.clone(),
                            ssa_values: HashMap::new(),
                            return_addr: None,
                            fused_groups: HashMap::new(),
                            fused_skipped_ids: std::collections::HashSet::new(),
                        }});
                        let parameter_ids = vm.collect_parameter_ids();
                        vm.tape.parameter_ids = parameter_ids;
                        vm.tape.backward(t.id);
                        vm.call_stack.pop();
                    }}
                    Value::Void
                }}"#, node.inputs[0])
                            }

                            IROp::Adam { target, lr, .. } => {
                                let root = target.split('.').next().unwrap_or(target);
                                format!(
            r#"{{
                    vm.call_stack.push(neuron_runtime::vm::CallFrame {{
                        function_name: String::new(),
                        current_block: 0,
                        instruction_idx: 0,
                        locals: locals.clone(),
                        ssa_values: HashMap::new(),
                        return_addr: None,
                        fused_groups: HashMap::new(),
                        fused_skipped_ids: std::collections::HashSet::new(),
                    }});
                    let res = vm.apply_optimizer({:?}, {} as f64, "adam").unwrap_or(Value::Void);
                    if let Some(frame) = vm.call_stack.last() {{
                        if let Some(val) = frame.locals.get({:?}) {{
                            locals.insert({:?}.to_string(), val.clone());
                        }}
                    }}
                    vm.call_stack.pop();
                    res
                }}"#, target, lr, root, root)
                            }
                            IROp::SGD { target, lr, .. } => {
                                let root = target.split('.').next().unwrap_or(target);
                                format!(
            r#"{{
                    vm.call_stack.push(neuron_runtime::vm::CallFrame {{
                        function_name: String::new(),
                        current_block: 0,
                        instruction_idx: 0,
                        locals: locals.clone(),
                        ssa_values: HashMap::new(),
                        return_addr: None,
                        fused_groups: HashMap::new(),
                        fused_skipped_ids: std::collections::HashSet::new(),
                    }});
                    let res = vm.apply_optimizer({:?}, {} as f64, "sgd").unwrap_or(Value::Void);
                    if let Some(frame) = vm.call_stack.last() {{
                        if let Some(val) = frame.locals.get({:?}) {{
                            locals.insert({:?}.to_string(), val.clone());
                        }}
                    }}
                    vm.call_stack.pop();
                    res
                }}"#, target, lr, root, root)
                            }
                            IROp::AdamW { target, lr, .. } => {
                                let root = target.split('.').next().unwrap_or(target);
                                format!(
            r#"{{
                    vm.call_stack.push(neuron_runtime::vm::CallFrame {{
                        function_name: String::new(),
                        current_block: 0,
                        instruction_idx: 0,
                        locals: locals.clone(),
                        ssa_values: HashMap::new(),
                        return_addr: None,
                        fused_groups: HashMap::new(),
                        fused_skipped_ids: std::collections::HashSet::new(),
                    }});
                    let res = vm.apply_optimizer({:?}, {} as f64, "adamw").unwrap_or(Value::Void);
                    if let Some(frame) = vm.call_stack.last() {{
                        if let Some(val) = frame.locals.get({:?}) {{
                            locals.insert({:?}.to_string(), val.clone());
                        }}
                    }}
                    vm.call_stack.pop();
                    res
                }}"#, target, lr, root, root)
                            }

                            IROp::Call { function } => {
                                let args: Vec<String> = node.inputs.iter().map(|id| format!("v{}.clone()", id)).collect();
                                let args_str = args.join(", ");
                                if func_names.contains(function) {
                                    format!("{}(vm, vec![{}])", function, args_str)
                                } else if func_names.contains(&format!("{}_new", function)) {
                                    format!("{}_new(vm, vec![{}])", function, args_str)
                                } else {
                                    format!("jit_obj_call(vm, {:?}, vec![{}])", function, args_str)
                                }
                            }

                            IROp::Load { name } => {
                                if !node.inputs.is_empty() {
                                    format!(
            r#"match &v{} {{
                    Value::Model {{ fields, .. }} => fields.borrow().get({:?}).cloned().unwrap_or(Value::None),
                    Value::Uncertain {{ value, std, confidence }} => match {:?} {{
                        "value" => Value::Float(*value),
                        "std" => Value::Float(*std),
                        "confidence" => Value::Float(*confidence),
                        _ => Value::None,
                    }},
                    _ => Value::None,
                }}"#, node.inputs[0], name, name)
                                } else {
                                    if global_names.contains(name) {
                                        format!(
            r#"locals.get({:?}).cloned()
                    .or_else(|| vm.globals.get({:?}).cloned())
                    .unwrap_or(Value::None)"#, name, name)
                                    } else {
                                        format!(
            r#"locals.get({:?}).cloned()
                    .unwrap_or(Value::None)"#, name)
                                    }
                                }
                            }

                            IROp::Transpose(dim0, dim1) => {
                                format!(
            r#"if let Value::Tensor(t) = &v{} {{
                    Value::Tensor(t.transpose({}, {}))
                }} else {{ Value::Void }}"#, node.inputs[0], dim0, dim1)
                            }

                            IROp::Print => {
                                format!(
            r#"{{
                    println!("{{}}", v{}.display());
                    vm.effect_log.push("io".into());
                    Value::Void
                }}"#, node.inputs[0])
                            }

                            IROp::Input => {
                                r#"{
                    use std::io::{self, Write};
                    print!("> ");
                    io::stdout().flush().unwrap();
                    let mut input_str = String::new();
                    io::stdin().read_line(&mut input_str).unwrap();
                    vm.effect_log.push("io".into());
                    Value::Str(input_str.trim().to_string())
                }"#.to_string()
                            }

                            IROp::EmbedString => {
                                format!(
            r#"{{
                    let val = &v{};
                    if let Value::Str(s) = val {{
                        let mut data = vec![0.0; 8];
                        for (i, c) in s.chars().enumerate() {{
                            let idx = i % 8;
                            data[idx] += (c as u32 as f64).sin();
                        }}
                        for v in &mut data {{
                            *v = v.tanh();
                        }}
                        Value::Tensor(Tensor::new(data, vec![1, 8]))
                    }} else {{
                        Value::Tensor(Tensor::zeros(&[1, 8]))
                    }}
                }}"#, node.inputs[0])
                            }

                            IROp::GenerateReply => {
                                format!(
            r#"{{
                    let val = &v{};
                    if let Value::Str(s) = val {{
                        let model = neuron_runtime::neuron_lm::NeuronLM::new();
                        let reply = model.generate_reply(s);
                        Value::Str(reply)
                    }} else {{
                        Value::Str("[AGI Response]: Got it!".to_string())
                    }}
                }}"#, node.inputs[0])
                            }

                            IROp::Nop => {
                                let items: Vec<String> = node.inputs.iter().map(|id| format!("v{}.clone()", id)).collect();
                                match &node.output_type {
                                    IRType::List(_) => format!("Value::List(vec![{}])", items.join(", ")),
                                    IRType::Tuple(_) => format!("Value::Tuple(vec![{}])", items.join(", ")),
                                    _ => "Value::Void".to_string(),
                                }
                            }

                            IROp::ForgetTask { method, strength } => {
                                let task_data = if node.inputs.len() > 1 {
                                    format!("v{}", node.inputs[1])
                                } else {
                                    "Value::List(vec![])".to_string()
                                };
                                format!(
                                    r#"{{
                                        let mut model = v{}.clone();
                                        let cert = neuron_runtime::forget::forget_task(vm, &mut model, &{}, {:?}, {});
                                        v{} = model;
                                        cert.unwrap_or(Value::Void)
                                    }}"#,
                                    node.inputs[0], task_data, method, strength, node.inputs[0]
                                )
                            }

                            IROp::UncertainWrap => {
                                let std_str = if node.inputs.len() > 1 {
                                    format!("v{}.as_float()", node.inputs[1])
                                } else {
                                    "0.1".to_string()
                                };
                                format!(
                                    r#"{{
                                        let val = v{}.as_float();
                                        let std = {};
                                        let confidence = if std > 0.0 {{ (1.0 - (std / (val.abs() + 1e-8)).min(1.0)).max(0.0) }} else {{ 1.0 }};
                                        Value::Uncertain {{ value: val, std, confidence }}
                                    }}"#,
                                    node.inputs[0], std_str
                                )
                            }

                            IROp::UncertainValue => {
                                format!(
                                    r#"match &v{} {{
                                        Value::Uncertain {{ value, .. }} => Value::Float(*value),
                                        other => other.clone(),
                                    }}"#,
                                    node.inputs[0]
                                )
                            }

                            IROp::UncertainConfidence => {
                                format!(
                                    r#"match &v{} {{
                                        Value::Uncertain {{ confidence, .. }} => Value::Float(*confidence),
                                        _ => Value::Float(1.0),
                                    }}"#,
                                    node.inputs[0]
                                )
                            }

                            IROp::TemporalBefore { .. } => {
                                format!(
                                    r#"match &v{} {{
                                        Value::Temporal {{ data, direction }} => Value::Temporal {{ data: data.clone(), direction: direction.clone() }},
                                        other => other.clone(),
                                    }}"#,
                                    node.inputs[0]
                                )
                            }

                            IROp::TemporalSnapshot { .. } => {
                                format!(
                                    r#"match &v{} {{
                                        Value::Temporal {{ data, .. }} => *data.clone(),
                                        other => other.clone(),
                                    }}"#,
                                    node.inputs[0]
                                )
                            }

                            IROp::TemporalAfter { .. } => {
                                format!(
                                    r#"match &v{} {{
                                        Value::Temporal {{ data, direction }} => {{
                                            let new_dir = if direction == "past_to_future" {{ "future_to_past" }} else {{ "past_to_future" }};
                                            Value::Temporal {{ data: data.clone(), direction: new_dir.to_string() }}
                                        }}
                                        other => other.clone(),
                                    }}"#,
                                    node.inputs[0]
                                )
                            }

                            IROp::TemporalCheckDir { expected } => {
                                format!(
                                    r#"{{
                                        if let Value::Temporal {{ direction, .. }} = &v{} {{
                                            if direction != {:?} && vm.strict_temporal {{
                                                panic!("RUNTIME PANIC: temporal direction violation — expected {{}} but got {{}} — lookahead bias detected", {:?}, direction);
                                            }}
                                        }}
                                        Value::Void
                                    }}"#,
                                    node.inputs[0], expected, expected
                                )
                            }

                            IROp::Observe => {
                                format!("Value::Causal {{ data: Box::new(v{}.clone()), mode: \"observed\".into() }}", node.inputs[0])
                            }

                            IROp::Intervene => {
                                format!("Value::Causal {{ data: Box::new(v{}.clone()), mode: \"intervened\".into() }}", node.inputs[0])
                            }

                            IROp::CausalCheckMode { expected } => {
                                format!(
                                    r#"{{
                                        if let Value::Causal {{ mode, .. }} = &v{} {{
                                            if mode != {:?} && vm.strict_causal {{
                                                panic!("RUNTIME PANIC: causal type mismatch — cannot use {{}} data where {{}} is expected", mode, {:?});
                                            }}
                                        }}
                                        Value::Void
                                    }}"#,
                                    node.inputs[0], expected, expected
                                )
                            }

                            IROp::Explain => {
                                format!("Value::Tuple(vec![v{}.clone(), Value::Str(\"explanation: gradient attribution\".into())])", node.inputs[0])
                            }

                            IROp::MergeModels { strategy } => {
                                format!("Value::Str(format!(\"merged with strategy: {}\"))", strategy)
                            }

                            IROp::MemoryStore => {
                                r#"{
                                    vm.effect_log.push("memory_store".into());
                                    Value::Void
                                }"#.to_string()
                            }

                            IROp::MemoryRecall { .. } => {
                                "Value::List(vec![])".to_string()
                            }

                            IROp::Search { strategy, max_iter } => {
                                format!("Value::Str(format!(\"search: strategy={}, max_iter={}\"))", strategy, max_iter)
                            }

                            IROp::Concat { dim } => {
                                format!("jit_concat(vm, &v{}, {})", node.inputs[0], dim)
                            }

                            _ => "Value::Void".to_string(),
                        };
                        f_code.push_str(&format!("                v{} = {};\n", node.id, op_code));
                    }
                }
            }

            match &block.terminator {
                Terminator::Jump(target) => {
                    f_code.push_str(&format!("                current_block = {};\n", target));
                }
                Terminator::Branch { cond, true_block, false_block } => {
                    f_code.push_str(&format!(
                        "                current_block = if v{}.as_bool() {{ {} }} else {{ {} }};\n",
                        cond, true_block, false_block
                    ));
                }
                Terminator::Return(val_id) => {
                    if is_constructor {
                        f_code.push_str("                return locals.get(\"self\").cloned().unwrap_or(Value::Void);\n");
                    } else if let Some(vid) = val_id {
                        f_code.push_str(&format!("                return v{}.clone();\n", vid));
                    } else {
                        f_code.push_str("                return Value::Void;\n");
                    }
                }
            }
            f_code.push_str("            }\n");
        }
        f_code.push_str("            _ => panic!(\"Invalid basic block ID {}\", current_block),\n");
        f_code.push_str("        }\n");
        f_code.push_str("    }\n");
        f_code.push_str("}\n\n");
        f_code
    }
}
