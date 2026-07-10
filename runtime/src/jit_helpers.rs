/// JIT Helper Functions
///
/// These are the runtime support functions used by transpiled (JIT-compiled) NEURON code.
/// By living in the runtime crate, they are compiled once and shared across all JIT invocations,
/// eliminating the need to recompile them for every property test or JIT build.

use crate::tensor::Tensor;
use crate::vm::{Value, VM};
use std::collections::HashMap;

// ── Arithmetic ──────────────────────────────────────────────────────────────

pub fn jit_add(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
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
        (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
        _ => Value::Float(a.as_float() + b.as_float()),
    }
}

pub fn jit_sub(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
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
        (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
        _ => Value::Float(a.as_float() - b.as_float()),
    }
}

pub fn jit_mul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
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
        (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
        _ => Value::Float(a.as_float() * b.as_float()),
    }
}

pub fn jit_div(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
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
        (Value::Int(x), Value::Int(y)) => Value::Int(x / y),
        (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
        _ => Value::Float(a.as_float() / b.as_float()),
    }
}

pub fn jit_neg(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.neg(ta)),
        _ => Value::Float(-a.as_float()),
    }
}

// ── Activations ─────────────────────────────────────────────────────────────

pub fn jit_gelu(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.gelu(ta)),
        _ => a.clone(),
    }
}

pub fn jit_relu(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.relu(ta)),
        _ => Value::Float(a.as_float().max(0.0)),
    }
}

pub fn jit_sigmoid(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.sigmoid(ta)),
        _ => Value::Float(1.0 / (1.0 + (-a.as_float()).exp())),
    }
}

pub fn jit_tanh(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => Value::Tensor(vm.tape.tanh(ta)),
        _ => Value::Float(a.as_float().tanh()),
    }
}

// ── Tensor constructors ─────────────────────────────────────────────────────

pub fn jit_zeros(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::zeros(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_ones(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::ones(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_randn(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::randn(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

pub fn jit_glorot(vm: &mut VM, shape: Vec<usize>) -> Value {
    let mut t = Tensor::glorot(&shape);
    t.id = vm.tape.alloc_id();
    Value::Tensor(t)
}

// ── Linear algebra ──────────────────────────────────────────────────────────

pub fn jit_matmul(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
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
        _ => panic!("JIT matmul requires tensor operands"),
    }
}

// ── Loss functions ──────────────────────────────────────────────────────────

pub fn jit_mse_loss(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                panic!("Runtime Error: MSELoss shape mismatch: pred {:?} vs target {:?}", ta.shape, tb.shape);
            }
            Value::Tensor(vm.tape.mse(ta, tb))
        }
        _ => panic!("JIT mse_loss requires tensor operands"),
    }
}

pub fn jit_cross_entropy(vm: &mut VM, a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                panic!("Runtime Error: CrossEntropy shape mismatch: pred {:?} vs target {:?}", ta.shape, tb.shape);
            }
            Value::Tensor(vm.tape.cross_entropy(ta, tb))
        }
        _ => panic!("JIT cross_entropy requires tensor operands"),
    }
}

// ── Autograd ────────────────────────────────────────────────────────────────

pub fn jit_apply_optimizer(vm: &mut VM, method: &str, target_name: &str, _grad_val: &Value, args: HashMap<String, f64>) {
    let lr = args.get("lr").cloned().unwrap_or(0.01);
    let _ = vm.apply_optimizer(target_name, lr, method);
}

pub fn jit_grad(vm: &mut VM, loss_val: &Value) -> Value {
    let loss_id = match loss_val {
        Value::Tensor(t) => t.id,
        _ => return Value::Void,
    };
    vm.tape.backward(loss_id);
    Value::Void
}

pub fn jit_backward(vm: &mut VM, loss_val: &Value, _param_names: Vec<&str>) -> Value {
    let loss_id = match loss_val {
        Value::Tensor(t) => t.id,
        _ => return Value::Void,
    };
    vm.tape.backward(loss_id);
    Value::Void
}

pub fn jit_stop_grad(vm: &mut VM, a: &Value) -> Value {
    match a {
        Value::Tensor(ta) => {
            let mut t = ta.clone();
            t.id = vm.tape.alloc_id();
            vm.tape.detach(ta.id);
            Value::Tensor(t)
        }
        _ => a.clone(),
    }
}

// ── Comparisons ─────────────────────────────────────────────────────────────

pub fn jit_lt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() < b.as_float())
}

pub fn jit_lte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() <= b.as_float())
}

pub fn jit_gt(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() > b.as_float())
}

pub fn jit_gte(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_float() >= b.as_float())
}

pub fn jit_eq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x == y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x == y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x == y),
        _ => Value::Bool(a.as_float() == b.as_float()),
    }
}

pub fn jit_neq(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Bool(x != y),
        (Value::Bool(x), Value::Bool(y)) => Value::Bool(x != y),
        (Value::Str(x), Value::Str(y)) => Value::Bool(x != y),
        _ => Value::Bool(a.as_float() != b.as_float()),
    }
}

// ── Collections ─────────────────────────────────────────────────────────────

pub fn jit_list_len(a: &Value) -> Value {
    if let Value::List(ref l) = a {
        Value::Int(l.len() as i64)
    } else { Value::Int(0) }
}

pub fn jit_index(a: &Value, idx: &Value) -> Value {
    let idx_val = idx.as_int();
    if idx_val < 0 {
        panic!("Runtime Error: Negative index {} is not allowed", idx_val);
    }
    let i = idx_val as usize;
    match a {
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
                    return Value::Tensor(Tensor::new(row_data, vec![1, cols]));
                }
            } else if t.ndim() == 1 {
                if i < t.data.len() {
                    return Value::Float(t.data[i]);
                }
            }
            Value::Void
        }
        _ => Value::Void,
    }
}

pub fn jit_sum(vm: &mut VM, a: &Value, dim: Option<i64>) -> Value {
    if let Value::Tensor(t) = a {
        if let Some(d) = dim {
            let ndim = t.ndim() as i64;
            if d < 0 || d >= ndim {
                panic!("Runtime Error: Sum dimension {} out of bounds for tensor of rank {}", d, ndim);
            }
        }
        Value::Tensor(vm.tape.sum(t, dim.map(|d| d as usize)))
    } else {
        a.clone()
    }
}

pub fn jit_mean(vm: &mut VM, a: &Value, dim: Option<i64>) -> Value {
    if let Value::Tensor(t) = a {
        if let Some(d) = dim {
            let ndim = t.ndim() as i64;
            if d < 0 || d >= ndim {
                panic!("Runtime Error: Mean dimension {} out of bounds for tensor of rank {}", d, ndim);
            }
        }
        Value::Tensor(vm.tape.mean(t, dim.map(|d| d as usize)))
    } else {
        a.clone()
    }
}

pub fn jit_sqrt(vm: &mut VM, a: &Value) -> Value {
    if let Value::Tensor(t) = a {
        Value::Tensor(vm.tape.sqrt(t))
    } else {
        match a {
            Value::Float(f) => Value::Float(f.sqrt()),
            Value::Int(i) => Value::Float((*i as f64).sqrt()),
            _ => a.clone(),
        }
    }
}

pub fn jit_concat(vm: &mut VM, a: &Value, dim: i64) -> Value {
    if let Value::List(items) = a {
        let tensors: Vec<&Tensor> = items.iter().filter_map(|v| v.as_tensor()).collect();

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
                return Value::Tensor(Tensor::new(data, vec![b, d_total]));
            } else {
                let total_len: usize = tensors.iter().map(|t| t.numel()).sum();
                if total_len > 10_000_000 {
                    panic!("Runtime Error: Tensor size too large in Concat: {} elements", total_len);
                }
                let mut data = Vec::with_capacity(total_len);
                for t in &tensors { data.extend_from_slice(&t.data); }
                return Value::Tensor(Tensor::new(data, vec![total_len]));
            }
        }
    }
    Value::Tensor(Tensor::zeros(&[0]))
}

pub fn jit_update_row(vm: &mut VM, a: &Value, idx: &Value, row: &Value) -> Value {
    if let (Value::Tensor(t), Value::Tensor(r)) = (a, row) {
        let idx_val = idx.as_int();
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
        a.clone()
    }
}

pub fn jit_validate_shape(shape_i64: Vec<i64>) -> Vec<usize> {
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

pub fn jit_mod(vm: &mut VM, a: &Value, b: &Value) -> Value {
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

pub fn jit_and(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_bool() && b.as_bool())
}

pub fn jit_or(a: &Value, b: &Value) -> Value {
    Value::Bool(a.as_bool() || b.as_bool())
}

pub fn jit_not(a: &Value) -> Value {
    Value::Bool(!a.as_bool())
}
