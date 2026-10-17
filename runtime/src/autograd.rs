/// NEURON Autograd — gradient tape and backward pass.
///
/// Every operation registers its backward pass on the tape. `backward()` walks
/// the tape in reverse, computing gradients via chain rule.
/// `grad(loss, wrt=param)` returns the gradient tensor.

use crate::tensor::*;
use crate::buffer::Buffer;

/// An entry in the gradient tape.
#[derive(Clone)]
pub struct TapeEntry {
    /// The operation that produced this value.
    pub op: TapeOp,
    /// IDs of input tensors.
    pub inputs: Vec<usize>,
    /// ID of the output tensor.
    pub output: usize,
    /// Shape of the output.
    pub output_shape: Vec<usize>,
}

/// Tape operations with the information needed for backward pass.
#[derive(Clone)]
pub enum TapeOp {
    Add { a_shape: Vec<usize>, b_shape: Vec<usize> },
    Sub { a_shape: Vec<usize>, b_shape: Vec<usize> },
    Mul { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    Div { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    Pow { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    Exp { output_data: Buffer },
    Log { input_data: Buffer },
    MatMul { a_data: Buffer, a_shape: Vec<usize>, b_data: Buffer, b_shape: Vec<usize> },
    ReLU { input_data: Buffer },
    Sigmoid { output_data: Buffer },
    Tanh { output_data: Buffer },
    GeLU { input_data: Buffer },
    Softmax { output_data: Buffer, dim: usize },
    Neg,
    Sum { input_shape: Vec<usize>, dim: Option<usize> },
    Mean { input_shape: Vec<usize>, dim: Option<usize> },
    Sqrt { output_data: Buffer },
    Reshape { original_shape: Vec<usize> },
    CrossEntropy { pred_softmax: Buffer, pred_shape: Vec<usize>, target: Buffer, batch_size: usize },
    MSE { pred: Buffer, pred_shape: Vec<usize>, target: Buffer, n: usize },
}

/// The gradient tape — records operations for backward pass.
pub struct GradTape {
    entries: Vec<TapeEntry>,
    /// Gradient storage: maps tensor_id → gradient data
    grads: Vec<Option<Buffer>>,
    /// Next tensor id
    next_id: usize,
    /// Parameter IDs that should have their gradients preserved after backward
    pub parameter_ids: std::collections::HashSet<usize>,
}

impl GradTape {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: Vec::new(),
            next_id: 0,
            parameter_ids: std::collections::HashSet::new(),
        }
    }

    /// Allocate a new tensor id.
    pub fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        // Ensure grads vec is large enough
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        id
    }

    /// Record an operation on the tape.
    pub fn record(&mut self, op: TapeOp, inputs: Vec<usize>, output: usize, output_shape: Vec<usize>) {
        self.entries.push(TapeEntry { op, inputs, output, output_shape });
    }

    /// Run tracked add.
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_add(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Add { a_shape: a.shape.clone(), b_shape: b.shape.clone() },
                    vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sub.
    pub fn sub(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_sub(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sub { a_shape: a.shape.clone(), b_shape: b.shape.clone() },
                    vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked mul.
    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_mul(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Mul {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked div.
    pub fn div(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_div(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Div {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked pow.
    pub fn pow(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_pow(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::Pow {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked exp.
    pub fn exp(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_exp(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Exp { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked log.
    pub fn log(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_log(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Log { input_data: a.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked matmul.
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let result = tensor_matmul(a, b);
        let out_id = self.alloc_id();
        self.record(TapeOp::MatMul {
            a_data: a.data.clone(), a_shape: a.shape.clone(),
            b_data: b.data.clone(), b_shape: b.shape.clone(),
        }, vec![a.id, b.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked relu.
    pub fn relu(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_relu(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::ReLU { input_data: a.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sigmoid.
    pub fn sigmoid(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_sigmoid(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sigmoid { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked tanh.
    pub fn tanh(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_tanh(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Tanh { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked softmax.
    pub fn softmax(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_softmax(a);
        let out_id = self.alloc_id();
        let dim = a.ndim().saturating_sub(1);
        self.record(TapeOp::Softmax { output_data: result.data.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked cross-entropy.
    pub fn cross_entropy(&mut self, pred: &Tensor, target: &Tensor) -> Tensor {
        let sm = tensor_softmax(pred);
        let result = tensor_cross_entropy(pred, target);
        let out_id = self.alloc_id();
        let batch = pred.shape[0];
        self.record(TapeOp::CrossEntropy { pred_softmax: sm.data, pred_shape: pred.shape.clone(), target: target.data.clone(), batch_size: batch },
                    vec![pred.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked MSE.
    pub fn mse(&mut self, pred: &Tensor, target: &Tensor) -> Tensor {
        let result = tensor_mse(pred, target);
        let out_id = self.alloc_id();
        self.record(TapeOp::MSE { pred: pred.data.clone(), pred_shape: pred.shape.clone(), target: target.data.clone(), n: pred.numel() },
                    vec![pred.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked negation.
    pub fn neg(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_neg(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::Neg, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked gelu.
    pub fn gelu(&mut self, a: &Tensor) -> Tensor {
        let result = tensor_gelu(a);
        let out_id = self.alloc_id();
        self.record(TapeOp::GeLU { input_data: a.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sum.
    pub fn sum(&mut self, a: &Tensor, dim: Option<usize>) -> Tensor {
        let result = a.sum(dim);
        let out_id = self.alloc_id();
        self.record(TapeOp::Sum { input_shape: a.shape.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked mean.
    pub fn mean(&mut self, a: &Tensor, dim: Option<usize>) -> Tensor {
        let result = a.mean(dim);
        let out_id = self.alloc_id();
        self.record(TapeOp::Mean { input_shape: a.shape.clone(), dim }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Run tracked sqrt.
    pub fn sqrt(&mut self, a: &Tensor) -> Tensor {
        let result = a.map(|x| x.sqrt());
        let out_id = self.alloc_id();
        self.record(TapeOp::Sqrt { output_data: result.data.clone() }, vec![a.id], out_id, result.shape.clone());
        let mut r = result;
        r.id = out_id;
        r.tape_entry = Some(self.entries.len() - 1);
        r
    }

    /// Backward pass — compute gradients for all tensors on the tape.
    /// Starts from `loss_id` with gradient 1.0.
    pub fn backward(&mut self, loss_id: usize) {
        self.backward_opts(loss_id, false);
    }

    /// Backward pass with optional graph retention for multiple backward passes or higher-order derivatives.
    pub fn backward_opts(&mut self, loss_id: usize, retain_graph: bool) {
        // Clear previous gradients
        for g in self.grads.iter_mut() {
            *g = None;
        }
        // Initialize loss gradient to 1.0
        while self.grads.len() <= loss_id {
            self.grads.push(None);
        }

        // Find the tape entry for the loss
        let loss_shape = self.entries.iter().rev()
            .find(|e| e.output == loss_id)
            .map(|e| e.output_shape.clone())
            .unwrap_or(vec![1]);
        let loss_n: usize = loss_shape.iter().product::<usize>().max(1);
        let mut loss_grad = Buffer::new(loss_n);
        for x in loss_grad.iter_mut() { *x = 1.0; }
        self.grads[loss_id] = Some(loss_grad);

        // Walk tape in reverse
        let entries_to_process = if retain_graph {
            self.entries.clone()
        } else {
            std::mem::take(&mut self.entries)
        };

        for entry in entries_to_process.into_iter().rev() {
            let out_grad = match self.grads.get_mut(entry.output) {
                Some(g_opt) => {
                    if retain_graph || self.parameter_ids.contains(&entry.output) {
                        match g_opt.as_ref() {
                            Some(g) => g.clone(),
                            None => continue,
                        }
                    } else {
                        match g_opt.take() {
                            Some(g) => g,
                            None => continue,
                        }
                    }
                }
                None => continue, // No gradient flowing to this output
            };

            // Skip entries whose inputs were cleared by stop_grad/detach.
            // These nodes have been severed from the computation graph.
            if entry.inputs.is_empty() {
                continue;
            }

            match &entry.op {
                TapeOp::Add { a_shape, b_shape } => {
                    let grad_a = reduce_grad(&out_grad, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&out_grad, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Sub { a_shape, b_shape } => {
                    let grad_a = reduce_grad(&out_grad, &entry.output_shape, a_shape);
                    let mut neg = Buffer::new(out_grad.len());
                    if neg.len() > 65536 {
                        use rayon::prelude::*;
                        neg.par_iter_mut().zip(out_grad.par_iter()).for_each(|(n, &g)| *n = -g);
                    } else {
                        for (n, &g) in neg.iter_mut().zip(out_grad.iter()) { *n = -g; }
                    }
                    let grad_b = reduce_grad(&neg, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Mul { a_data, a_shape, b_data, b_shape } => {
                    let a_tensor = Tensor::new(a_data.clone(), a_shape.clone());
                    let b_tensor = Tensor::new(b_data.clone(), b_shape.clone());
                    let a_bc = a_tensor.broadcast_to(&entry.output_shape);
                    let b_bc = b_tensor.broadcast_to(&entry.output_shape);
                    
                    let mut raw_grad_a = Buffer::new(out_grad.len());
                    let mut raw_grad_b = Buffer::new(out_grad.len());
                    if raw_grad_a.len() > 65536 {
                        use rayon::prelude::*;
                        raw_grad_a.par_iter_mut().zip(out_grad.par_iter().zip(b_bc.data.par_iter()))
                            .for_each(|(g_a, (&g, &b))| { *g_a = g * b; });
                        raw_grad_b.par_iter_mut().zip(out_grad.par_iter().zip(a_bc.data.par_iter()))
                            .for_each(|(g_b, (&g, &a))| { *g_b = g * a; });
                    } else {
                        for (g_a, (&g, &b)) in raw_grad_a.iter_mut().zip(out_grad.iter().zip(b_bc.data.iter())) {
                            *g_a = g * b;
                        }
                        for (g_b, (&g, &a)) in raw_grad_b.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter())) {
                            *g_b = g * a;
                        }
                    }
                    
                    let grad_a = reduce_grad(&raw_grad_a, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&raw_grad_b, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::MatMul { a_data, a_shape, b_data, b_shape } => {
                    let a = Tensor::new(a_data.clone(), a_shape.clone());
                    let b = Tensor::new(b_data.clone(), b_shape.clone());
                    let grad_out = Tensor::new(out_grad, entry.output_shape.clone());

                    let b_t = b.transpose(b.ndim() - 2, b.ndim() - 1);
                    let grad_a = tensor_matmul(&grad_out, &b_t);
                    self.accumulate_grad(entry.inputs[0], grad_a.data);

                    let a_t = a.transpose(a.ndim() - 2, a.ndim() - 1);
                    let grad_b = tensor_matmul(&a_t, &grad_out);
                    self.accumulate_grad(entry.inputs[1], grad_b.data);
                }
                TapeOp::ReLU { input_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(input_data.par_iter()))
                            .for_each(|(g_in, (&g, &x))| { *g_in = if x > 0.0 { g } else { 0.0 }; });
                    } else {
                        for (g_in, (&g, &x)) in grad.iter_mut().zip(out_grad.iter().zip(input_data.iter())) {
                            *g_in = if x > 0.0 { g } else { 0.0 };
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sigmoid { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(output_data.par_iter()))
                            .for_each(|(g_in, (&g, &s))| { *g_in = g * s * (1.0 - s); });
                    } else {
                        for (g_in, (&g, &s)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                            *g_in = g * s * (1.0 - s);
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Tanh { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(output_data.par_iter()))
                            .for_each(|(g_in, (&g, &t))| { *g_in = g * (1.0 - t * t); });
                    } else {
                        for (g_in, (&g, &t)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                            *g_in = g * (1.0 - t * t);
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::GeLU { input_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(input_data.par_iter()))
                            .for_each(|(g_in, (&g, &x))| {
                                let cdf = 0.5 * (1.0 + erf_approx(x / 2.0f64.sqrt()));
                                let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
                                *g_in = g * (cdf + x * pdf);
                            });
                    } else {
                        for (g_in, (&g, &x)) in grad.iter_mut().zip(out_grad.iter().zip(input_data.iter())) {
                            let cdf = 0.5 * (1.0 + erf_approx(x / 2.0f64.sqrt()));
                            let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
                            *g_in = g * (cdf + x * pdf);
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Softmax { output_data, dim } => {
                    let last_dim = entry.output_shape[*dim];
                    let outer = output_data.len() / last_dim;
                    let mut grad_input = Buffer::new(output_data.len());
                    for i in 0..outer {
                        let start = i * last_dim;
                        for j in 0..last_dim {
                            for k in 0..last_dim {
                                let s_j = output_data[start + j];
                                let s_k = output_data[start + k];
                                let kronecker = if j == k { 1.0 } else { 0.0 };
                                grad_input[start + j] += out_grad[start + k] * s_k * (kronecker - s_j);
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad_input);
                }
                TapeOp::Neg => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter()).for_each(|(g_in, &g)| { *g_in = -g; });
                    } else {
                        for (g_in, &g) in grad.iter_mut().zip(out_grad.iter()) {
                            *g_in = -g;
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sum { input_shape, dim } => {
                    let mut grad = Buffer::new(input_shape.iter().product::<usize>().max(1));
                    match dim {
                        None => {
                            let val = out_grad[0];
                            for g in grad.iter_mut() {
                                *g = val;
                            }
                        }
                        Some(d) => {
                            let mut out_shape = input_shape.clone();
                            out_shape[*d] = 1;
                            let mut out_strides = vec![1; out_shape.len()];
                            for j in (0..out_shape.len().saturating_sub(1)).rev() {
                                out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                            }
                            
                            let mut in_strides = vec![1; input_shape.len()];
                            for j in (0..input_shape.len().saturating_sub(1)).rev() {
                                in_strides[j] = in_strides[j + 1] * input_shape[j + 1];
                            }

                            for i in 0..grad.len() {
                                let mut coords = vec![0; input_shape.len()];
                                let mut temp = i;
                                for j in 0..input_shape.len() {
                                    coords[j] = temp / in_strides[j];
                                    temp %= in_strides[j];
                                }
                                let mut out_coords = coords.clone();
                                out_coords[*d] = 0;
                                let out_idx: usize = out_coords.iter().zip(out_strides.iter()).map(|(c, s)| c * s).sum();
                                grad[i] = out_grad[out_idx];
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Mean { input_shape, dim } => {
                    let mut grad = Buffer::new(input_shape.iter().product::<usize>().max(1));
                    match dim {
                        None => {
                            let n = input_shape.iter().product::<usize>() as f64;
                            let scale = if n > 0.0 { 1.0 / n } else { 0.0 };
                            let val = out_grad[0] * scale;
                            for g in grad.iter_mut() {
                                *g = val;
                            }
                        }
                        Some(d) => {
                            let count = input_shape[*d] as f64;
                            let scale = if count > 0.0 { 1.0 / count } else { 0.0 };
                            
                            let mut out_shape = input_shape.clone();
                            out_shape[*d] = 1;
                            let mut out_strides = vec![1; out_shape.len()];
                            for j in (0..out_shape.len().saturating_sub(1)).rev() {
                                out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                            }
                            
                            let mut in_strides = vec![1; input_shape.len()];
                            for j in (0..input_shape.len().saturating_sub(1)).rev() {
                                in_strides[j] = in_strides[j + 1] * input_shape[j + 1];
                            }

                            for i in 0..grad.len() {
                                let mut coords = vec![0; input_shape.len()];
                                let mut temp = i;
                                for j in 0..input_shape.len() {
                                    coords[j] = temp / in_strides[j];
                                    temp %= in_strides[j];
                                }
                                let mut out_coords = coords.clone();
                                out_coords[*d] = 0;
                                let out_idx: usize = out_coords.iter().zip(out_strides.iter()).map(|(c, s)| c * s).sum();
                                grad[i] = out_grad[out_idx] * scale;
                            }
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Sqrt { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(output_data.par_iter()))
                            .for_each(|(g_in, (&g, &y))| { *g_in = if y > 1e-10 { g * 0.5 / y } else { 0.0 }; });
                    } else {
                        for (g_in, (&g, &y)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                            *g_in = if y > 1e-10 { g * 0.5 / y } else { 0.0 };
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Div { a_data, a_shape, b_data, b_shape } => {
                    let a_tensor = Tensor::new(a_data.clone(), a_shape.clone());
                    let b_tensor = Tensor::new(b_data.clone(), b_shape.clone());
                    let a_bc = a_tensor.broadcast_to(&entry.output_shape);
                    let b_bc = b_tensor.broadcast_to(&entry.output_shape);
                    
                    let mut raw_grad_a = Buffer::new(out_grad.len());
                    let mut raw_grad_b = Buffer::new(out_grad.len());
                    if raw_grad_a.len() > 65536 {
                        use rayon::prelude::*;
                        raw_grad_a.par_iter_mut().zip(out_grad.par_iter().zip(b_bc.data.par_iter()))
                            .for_each(|(g_a, (&g, &b))| { *g_a = if b.abs() > 1e-15 { g / b } else { 0.0 }; });
                        raw_grad_b.par_iter_mut().zip(out_grad.par_iter().zip(a_bc.data.par_iter()).zip(b_bc.data.par_iter()))
                            .for_each(|(g_b, ((&g, &a), &b))| { *g_b = if b.abs() > 1e-15 { -g * a / (b * b) } else { 0.0 }; });
                    } else {
                        for (g_a, (&g, &b)) in raw_grad_a.iter_mut().zip(out_grad.iter().zip(b_bc.data.iter())) {
                            *g_a = if b.abs() > 1e-15 { g / b } else { 0.0 };
                        }
                        for (g_b, ((&g, &a), &b)) in raw_grad_b.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter()).zip(b_bc.data.iter())) {
                            *g_b = if b.abs() > 1e-15 { -g * a / (b * b) } else { 0.0 };
                        }
                    }
                    
                    let grad_a = reduce_grad(&raw_grad_a, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&raw_grad_b, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Pow { a_data, a_shape, b_data, b_shape } => {
                    let a_tensor = Tensor::new(a_data.clone(), a_shape.clone());
                    let b_tensor = Tensor::new(b_data.clone(), b_shape.clone());
                    let a_bc = a_tensor.broadcast_to(&entry.output_shape);
                    let b_bc = b_tensor.broadcast_to(&entry.output_shape);
                    
                    let mut raw_grad_a = Buffer::new(out_grad.len());
                    let mut raw_grad_b = Buffer::new(out_grad.len());
                    if raw_grad_a.len() > 65536 {
                        use rayon::prelude::*;
                        raw_grad_a.par_iter_mut().zip(out_grad.par_iter().zip(a_bc.data.par_iter()).zip(b_bc.data.par_iter()))
                            .for_each(|(g_a, ((&g, &a), &b))| { *g_a = if a != 0.0 { g * b * a.powf(b - 1.0) } else { 0.0 }; });
                        raw_grad_b.par_iter_mut().zip(out_grad.par_iter().zip(a_bc.data.par_iter()).zip(b_bc.data.par_iter()))
                            .for_each(|(g_b, ((&g, &a), &b))| { *g_b = if a > 0.0 { g * a.powf(b) * a.ln() } else { 0.0 }; });
                    } else {
                        for (g_a, ((&g, &a), &b)) in raw_grad_a.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter()).zip(b_bc.data.iter())) {
                            *g_a = if a != 0.0 { g * b * a.powf(b - 1.0) } else { 0.0 };
                        }
                        for (g_b, ((&g, &a), &b)) in raw_grad_b.iter_mut().zip(out_grad.iter().zip(a_bc.data.iter()).zip(b_bc.data.iter())) {
                            *g_b = if a > 0.0 { g * a.powf(b) * a.ln() } else { 0.0 };
                        }
                    }
                    
                    let grad_a = reduce_grad(&raw_grad_a, &entry.output_shape, a_shape);
                    let grad_b = reduce_grad(&raw_grad_b, &entry.output_shape, b_shape);
                    self.accumulate_grad(entry.inputs[0], grad_a);
                    self.accumulate_grad(entry.inputs[1], grad_b);
                }
                TapeOp::Exp { output_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(output_data.par_iter()))
                            .for_each(|(g_in, (&g, &out))| { *g_in = g * out; });
                    } else {
                        for (g_in, (&g, &out)) in grad.iter_mut().zip(out_grad.iter().zip(output_data.iter())) {
                            *g_in = g * out;
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Log { input_data } => {
                    let mut grad = Buffer::new(out_grad.len());
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(out_grad.par_iter().zip(input_data.par_iter()))
                            .for_each(|(g_in, (&g, &x))| { *g_in = if x.abs() > 1e-15 { g / x } else { 0.0 }; });
                    } else {
                        for (g_in, (&g, &x)) in grad.iter_mut().zip(out_grad.iter().zip(input_data.iter())) {
                            *g_in = if x.abs() > 1e-15 { g / x } else { 0.0 };
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::CrossEntropy { pred_softmax, pred_shape: _, target, batch_size } => {
                    let mut grad = Buffer::new(pred_softmax.len());
                    let scale = *batch_size as f64;
                    let out_g = out_grad[0];
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(pred_softmax.par_iter().zip(target.par_iter()))
                            .for_each(|(g_in, (&s, &t))| { *g_in = out_g * (s - t) / scale; });
                    } else {
                        for (g_in, (&s, &t)) in grad.iter_mut().zip(pred_softmax.iter().zip(target.iter())) {
                            *g_in = out_g * (s - t) / scale;
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::MSE { pred, pred_shape: _, target, n } => {
                    let mut grad = Buffer::new(pred.len());
                    let scale = *n as f64;
                    let out_g = out_grad[0];
                    if grad.len() > 65536 {
                        use rayon::prelude::*;
                        grad.par_iter_mut().zip(pred.par_iter().zip(target.par_iter()))
                            .for_each(|(g_in, (&p, &t))| { *g_in = out_g * 2.0 * (p - t) / scale; });
                    } else {
                        for (g_in, (&p, &t)) in grad.iter_mut().zip(pred.iter().zip(target.iter())) {
                            *g_in = out_g * 2.0 * (p - t) / scale;
                        }
                    }
                    self.accumulate_grad(entry.inputs[0], grad);
                }
                TapeOp::Reshape { .. } => {
                    self.accumulate_grad(entry.inputs[0], out_grad);
                }
            }
        }
    }

    /// Backward pass with graph creation for higher-order derivatives (grad-of-grad).
    /// Returns a map of (tensor_id -> gradient_tensor) where the gradients themselves
    /// are registered on this tape as differentiable operations.
    pub fn backward_differentiable(&mut self, loss_id: usize) -> std::collections::HashMap<usize, Tensor> {
        let loss_shape = self.entries.iter().rev()
            .find(|e| e.output == loss_id)
            .map(|e| e.output_shape.clone())
            .unwrap_or(vec![1]);

        let mut loss_ones = Tensor::ones(&loss_shape);
        loss_ones.id = self.alloc_id();
        let mut tracked_grads: std::collections::HashMap<usize, Tensor> = std::collections::HashMap::new();
        tracked_grads.insert(loss_id, loss_ones);

        let loss_idx = self.entries.iter().rposition(|e| e.output == loss_id).unwrap_or(self.entries.len());
        let entries_to_process = self.entries[..=loss_idx.min(self.entries.len().saturating_sub(1))].to_vec();

        for entry in entries_to_process.into_iter().rev() {
            let out_grad = match tracked_grads.get(&entry.output) {
                Some(g) => g.clone(),
                None => continue,
            };

            if entry.inputs.is_empty() {
                continue;
            }

            match &entry.op {
                TapeOp::Add { .. } => {
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], out_grad.clone());
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], out_grad);
                }
                TapeOp::Sub { .. } => {
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], out_grad.clone());
                    let neg_grad = self.neg(&out_grad);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], neg_grad);
                }
                TapeOp::Mul { a_data, a_shape, b_data, b_shape } => {
                    let mut a_t = Tensor::new(a_data.clone(), a_shape.clone());
                    a_t.id = entry.inputs[0];
                    let mut b_t = Tensor::new(b_data.clone(), b_shape.clone());
                    b_t.id = entry.inputs[1];
                    let grad_a = self.mul(&out_grad, &b_t);
                    let grad_b = self.mul(&out_grad, &a_t);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_a);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], grad_b);
                }
                TapeOp::Div { a_data, a_shape, b_data, b_shape } => {
                    let mut a_t = Tensor::new(a_data.clone(), a_shape.clone());
                    a_t.id = entry.inputs[0];
                    let mut b_t = Tensor::new(b_data.clone(), b_shape.clone());
                    b_t.id = entry.inputs[1];
                    let grad_a = self.div(&out_grad, &b_t);
                    let b_sq = self.mul(&b_t, &b_t);
                    let neg_out = self.neg(&out_grad);
                    let neg_out_a = self.mul(&neg_out, &a_t);
                    let grad_b = self.div(&neg_out_a, &b_sq);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_a);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], grad_b);
                }
                TapeOp::Pow { a_data, a_shape, b_data, b_shape } => {
                    let mut a_t = Tensor::new(a_data.clone(), a_shape.clone());
                    a_t.id = entry.inputs[0];
                    let mut b_t = Tensor::new(b_data.clone(), b_shape.clone());
                    b_t.id = entry.inputs[1];
                    let mut ones = Tensor::ones(b_shape);
                    ones.id = self.alloc_id();
                    let b_minus_1 = self.sub(&b_t, &ones);
                    let a_pow = self.pow(&a_t, &b_minus_1);
                    let b_a_pow = self.mul(&b_t, &a_pow);
                    let grad_a = self.mul(&out_grad, &b_a_pow);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_a);

                    let a_pow_b = self.pow(&a_t, &b_t);
                    let ln_a = self.log(&a_t);
                    let a_pow_ln = self.mul(&a_pow_b, &ln_a);
                    let grad_b = self.mul(&out_grad, &a_pow_ln);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], grad_b);
                }
                TapeOp::Exp { output_data } => {
                    let mut out_t = Tensor::new(output_data.clone(), entry.output_shape.clone());
                    out_t.id = entry.output;
                    let grad_x = self.mul(&out_grad, &out_t);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Log { input_data } => {
                    let mut x_t = Tensor::new(input_data.clone(), entry.output_shape.clone());
                    x_t.id = entry.inputs[0];
                    let grad_x = self.div(&out_grad, &x_t);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Neg => {
                    let grad_x = self.neg(&out_grad);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::ReLU { input_data } => {
                    let x_t = Tensor::new(input_data.clone(), entry.output_shape.clone());
                    let mut mask_data = Buffer::new(x_t.numel());
                    for (m, &v) in mask_data.iter_mut().zip(x_t.data.iter()) {
                        *m = if v > 0.0 { 1.0 } else { 0.0 };
                    }
                    let mut mask = Tensor::new(mask_data, entry.output_shape.clone());
                    mask.id = self.alloc_id();
                    let grad_x = self.mul(&out_grad, &mask);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Sigmoid { output_data } => {
                    let mut s_t = Tensor::new(output_data.clone(), entry.output_shape.clone());
                    s_t.id = entry.output;
                    let mut ones = Tensor::ones(&entry.output_shape);
                    ones.id = self.alloc_id();
                    let one_minus_s = self.sub(&ones, &s_t);
                    let s_one_minus_s = self.mul(&s_t, &one_minus_s);
                    let grad_x = self.mul(&out_grad, &s_one_minus_s);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Tanh { output_data } => {
                    let mut t_t = Tensor::new(output_data.clone(), entry.output_shape.clone());
                    t_t.id = entry.output;
                    let t_sq = self.mul(&t_t, &t_t);
                    let mut ones = Tensor::ones(&entry.output_shape);
                    ones.id = self.alloc_id();
                    let one_minus_t_sq = self.sub(&ones, &t_sq);
                    let grad_x = self.mul(&out_grad, &one_minus_t_sq);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Sqrt { output_data } => {
                    let mut y_t = Tensor::new(output_data.clone(), entry.output_shape.clone());
                    y_t.id = entry.output;
                    let mut twos = Tensor::new(Buffer::from_slice(&vec![2.0; y_t.numel()]), entry.output_shape.clone());
                    twos.id = self.alloc_id();
                    let two_y = self.mul(&twos, &y_t);
                    let grad_x = self.div(&out_grad, &two_y);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::MatMul { a_data, a_shape, b_data, b_shape } => {
                    let mut a_t = Tensor::new(a_data.clone(), a_shape.clone());
                    a_t.id = entry.inputs[0];
                    let mut b_t = Tensor::new(b_data.clone(), b_shape.clone());
                    b_t.id = entry.inputs[1];
                    let b_trans = b_t.transpose(b_t.ndim() - 2, b_t.ndim() - 1);
                    let a_trans = a_t.transpose(a_t.ndim() - 2, a_t.ndim() - 1);
                    let grad_a = self.matmul(&out_grad, &b_trans);
                    let grad_b = self.matmul(&a_trans, &out_grad);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_a);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[1], grad_b);
                }
                TapeOp::Sum { input_shape, .. } => {
                    let mut ones = Tensor::ones(input_shape);
                    ones.id = self.alloc_id();
                    let grad_x = self.mul(&ones, &out_grad);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Mean { input_shape, .. } => {
                    let scale = 1.0 / (input_shape.iter().product::<usize>().max(1) as f64);
                    let mut scales = Tensor::new(Buffer::from_slice(&vec![scale; input_shape.iter().product::<usize>()]), input_shape.clone());
                    scales.id = self.alloc_id();
                    let grad_x = self.mul(&scales, &out_grad);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::MSE { pred, pred_shape, target, n } => {
                    let scale = 2.0 / (*n as f64);
                    let mut pred_t = Tensor::new(pred.clone(), pred_shape.clone());
                    pred_t.id = entry.inputs[0];
                    let mut target_t = Tensor::new(target.clone(), pred_shape.clone());
                    target_t.id = self.alloc_id();
                    let diff = self.sub(&pred_t, &target_t);
                    let mut scales = Tensor::new(Buffer::from_slice(&vec![scale; pred.len()]), pred_shape.clone());
                    scales.id = self.alloc_id();
                    let scaled = self.mul(&diff, &scales);
                    let grad_x = self.mul(&out_grad, &scaled);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::CrossEntropy { pred_softmax, pred_shape, target, batch_size } => {
                    let scale = 1.0 / (*batch_size as f64);
                    let mut sm_t = Tensor::new(pred_softmax.clone(), pred_shape.clone());
                    sm_t.id = self.alloc_id();
                    let mut tgt_t = Tensor::new(target.clone(), pred_shape.clone());
                    tgt_t.id = self.alloc_id();
                    let diff = self.sub(&sm_t, &tgt_t);
                    let mut scales = Tensor::new(Buffer::from_slice(&vec![scale; pred_softmax.len()]), pred_shape.clone());
                    scales.id = self.alloc_id();
                    let scaled = self.mul(&diff, &scales);
                    let grad_x = self.mul(&out_grad, &scaled);
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], grad_x);
                }
                TapeOp::Reshape { original_shape } => {
                    let reshaped = out_grad.reshape(original_shape);
                    let mut reshaped_t = reshaped;
                    reshaped_t.id = self.alloc_id();
                    Self::accumulate_tracked(&mut tracked_grads, self, entry.inputs[0], reshaped_t);
                }
                _ => {}
            }
        }

        // Also populate self.grads for compatibility
        for (&id, tensor) in &tracked_grads {
            while self.grads.len() <= id {
                self.grads.push(None);
            }
            self.grads[id] = Some(tensor.data.clone());
        }

        tracked_grads
    }

    fn accumulate_tracked(
        tracked_grads: &mut std::collections::HashMap<usize, Tensor>,
        tape: &mut GradTape,
        id: usize,
        grad: Tensor,
    ) {
        if let Some(existing) = tracked_grads.remove(&id) {
            let combined = tape.add(&existing, &grad);
            tracked_grads.insert(id, combined);
        } else {
            tracked_grads.insert(id, grad);
        }
    }

    /// Compute the Hessian-Vector Product: H * v = d/dx (<grad(L, x), v>).
    /// Runs in O(N) time without materializing the N x N Hessian matrix.
    pub fn hessian_vector_product(&mut self, loss_id: usize, wrt_id: usize, v: &Tensor) -> Tensor {
        let cp = self.checkpoint();
        let grads = self.backward_differentiable(loss_id);
        let grad_x = grads.get(&wrt_id).cloned().expect("Target tensor has no gradient in loss");

        let mut v_t = v.clone();
        v_t.id = self.alloc_id();
        let prod = self.mul(&grad_x, &v_t);
        let dot = self.sum(&prod, None);

        self.backward_opts(dot.id, true);
        let hv_data = self.get_grad(wrt_id).cloned().expect("HVP calculation produced no gradient");
        self.truncate(cp);
        Tensor::new(hv_data, v.shape.clone())
    }

    /// Compute the full Hessian matrix: H_ij = d^2 L / (dx_i dx_j).
    pub fn hessian(&mut self, loss_id: usize, wrt_id: usize) -> Tensor {
        let cp_base = self.checkpoint();
        let grads = self.backward_differentiable(loss_id);
        let grad_x = grads.get(&wrt_id).cloned().expect("Target tensor has no gradient in loss");
        let n = grad_x.numel();
        let shape = grad_x.shape.clone();
        let cp_diff = self.checkpoint();

        let mut hessian_rows: Vec<f64> = Vec::with_capacity(n * n);

        for i in 0..n {
            let mut basis = vec![0.0; n];
            basis[i] = 1.0;
            let mut v = Tensor::new(Buffer::from_slice(&basis), shape.clone());
            v.id = self.alloc_id();
            let prod = self.mul(&grad_x, &v);
            let dot = self.sum(&prod, None);
            self.backward_opts(dot.id, true);
            let hv_data = self.get_grad(wrt_id).cloned().expect("HVP calculation produced no gradient");
            hessian_rows.extend_from_slice(&hv_data[..]);
            self.truncate(cp_diff);
        }

        self.truncate(cp_base);
        Tensor::new(Buffer::from_slice(&hessian_rows), vec![n, n])
    }

    /// Get gradient for a tensor.
    pub fn get_grad(&self, tensor_id: usize) -> Option<&Buffer> {
        self.grads.get(tensor_id).and_then(|g| g.as_ref())
    }

    /// Detach a tensor from the tape to sever its gradients.
    pub fn detach(&mut self, tensor_id: usize) {
        if tensor_id < self.grads.len() {
            self.grads[tensor_id] = None;
        }
        for entry in self.entries.iter_mut() {
            if entry.output == tensor_id {
                entry.inputs.clear();
            }
        }
    }

    /// Get a checkpoint (the current number of tape entries).
    pub fn checkpoint(&self) -> usize {
        self.entries.len()
    }

    /// Truncate the tape to a previous checkpoint, releasing cached buffers.
    pub fn truncate(&mut self, checkpoint: usize) {
        if checkpoint < self.entries.len() {
            self.entries.truncate(checkpoint);
        }
    }

    /// Reset the tape.
    pub fn reset(&mut self) {
        self.entries.clear();
        for g in self.grads.iter_mut() { *g = None; }
        self.parameter_ids.clear();
    }

    /// Return the number of entries in the tape.
    pub fn tape_len(&self) -> usize {
        self.entries.len()
    }

    /// Return the ID of the last output tensor registered on the tape.
    pub fn last_output_id(&self) -> Option<usize> {
        self.entries.last().map(|e| e.output)
    }

    /// Zero all gradients without clearing the tape.
    pub fn zero_grad(&mut self) {
        for g in self.grads.iter_mut() { *g = None; }
    }

    fn accumulate_grad(&mut self, id: usize, grad: Buffer) {
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        if let Some(ref mut existing) = self.grads[id] {
            if existing.len() == grad.len() {
                if existing.len() > 65536 {
                    use rayon::prelude::*;
                    existing.par_iter_mut().zip(grad.par_iter()).for_each(|(a, &b)| *a += b);
                } else {
                    for (a, &b) in existing.iter_mut().zip(grad.iter()) { *a += b; }
                }
            } else {
                *existing = grad;
            }
        } else {
            self.grads[id] = Some(grad);
        }
    }

    #[allow(dead_code)]
    fn add_grad(&mut self, id: usize, grad: &[f64]) {
        while self.grads.len() <= id {
            self.grads.push(None);
        }
        if let Some(ref mut existing) = self.grads[id] {
            // Accumulate: handle size mismatch by extending
            if existing.len() == grad.len() {
                if existing.len() > 65536 {
                    use rayon::prelude::*;
                    existing.par_iter_mut().zip(grad.par_iter()).for_each(|(a, b)| *a += b);
                } else {
                    for (a, b) in existing.iter_mut().zip(grad.iter()) { *a += b; }
                }
            } else {
                // Broadcast case — just set
                *existing = Buffer::from_slice(grad);
            }
        } else {
            self.grads[id] = Some(Buffer::from_slice(grad));
        }
    }
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

// ═══════════════════════════════════════════
//  Optimizers
// ═══════════════════════════════════════════

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    /// Apply one step of Adam to a list of (param, grad) pairs.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Vec<f64>]) {
        self.t += 1;
        while self.m.len() < params.len() {
            self.m.push(vec![]);
            self.v.push(vec![]);
        }
        let correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
            }
            let m_slice = &mut self.m[i][..n];
            let v_slice = &mut self.v[i][..n];
            let grad_slice = &grad[..n];
            let param_slice = &mut param.data[..n];

            for j in 0..n {
                let g = grad_slice[j];
                m_slice[j] = self.beta1 * m_slice[j] + (1.0 - self.beta1) * g;
                v_slice[j] = self.beta2 * v_slice[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m_slice[j] / correction1;
                let v_hat = v_slice[j] / correction2;
                param_slice[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<Vec<f64>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self { lr, momentum, velocity: Vec::new() }
    }

    pub fn step(&mut self, params: &mut [Tensor], grads: &[Vec<f64>]) {
        while self.velocity.len() < params.len() {
            self.velocity.push(vec![]);
        }
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.numel();
            if self.velocity[i].is_empty() {
                self.velocity[i] = vec![0.0; n];
            }
            let vel_slice = &mut self.velocity[i][..n];
            let grad_slice = &grad[..n];
            let param_slice = &mut param.data[..n];

            for j in 0..n {
                vel_slice[j] = self.momentum * vel_slice[j] + grad_slice[j];
                param_slice[j] -= self.lr * vel_slice[j];
            }
        }
    }
}

fn reduce_grad(grad: &Buffer, current_shape: &[usize], target_shape: &[usize]) -> Buffer {
    if current_shape == target_shape {
        return grad.clone();
    }
    let mut current_t = Tensor::new(grad.clone(), current_shape.to_vec());
    let ndim = current_shape.len();
    let target_ndim = target_shape.len();
    let offset = ndim.saturating_sub(target_ndim);
    
    let mut padded_target_shape = vec![1; ndim];
    for i in 0..target_ndim {
        padded_target_shape[i + offset] = target_shape[i];
    }
    
    for i in (0..ndim).rev() {
        if padded_target_shape[i] == 1 && current_t.shape[i] > 1 {
            current_t = current_t.sum(Some(i));
        }
    }
    
    let mut out_data = current_t.data;
    let expected_len = target_shape.iter().product::<usize>();
    if out_data.len() != expected_len {
        let mut new_buf = Buffer::new(expected_len);
        let copy_len = out_data.len().min(expected_len);
        new_buf[..copy_len].copy_from_slice(&out_data[..copy_len]);
        out_data = new_buf;
    }
    out_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_add() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        a.id = tape.alloc_id();
        let mut b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        b.id = tape.alloc_id();

        let c = tape.add(&a, &b);
        tape.backward(c.id);

        let grad_a = tape.get_grad(a.id).unwrap();
        let grad_b = tape.get_grad(b.id).unwrap();
        assert_eq!(grad_a, &vec![1.0, 1.0, 1.0]);
        assert_eq!(grad_b, &vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_grad_matmul() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        a.id = tape.alloc_id();
        let mut b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        b.id = tape.alloc_id();

        let c = tape.matmul(&a, &b);
        // To create a scalar loss, just backward from the matmul result directly
        // (treating it as if dL/dC = ones)
        tape.backward(c.id);

        let grad_a = tape.get_grad(a.id).unwrap();
        assert_eq!(grad_a.len(), 4);
        // dL/dA = dL/dC @ B^T (with dL/dC = ones since output_shape matches)
    }

    #[test]
    fn test_grad_relu() {
        let mut tape = GradTape::new();
        let mut a = Tensor::new(vec![-1.0, 0.5, -0.3, 2.0], vec![4]);
        a.id = tape.alloc_id();

        let b = tape.relu(&a);
        tape.backward(b.id);

        let grad = tape.get_grad(a.id).unwrap();
        assert_eq!(grad[0], 0.0);  // -1.0 → dead
        assert_eq!(grad[1], 1.0);  // 0.5 → alive
        assert_eq!(grad[2], 0.0);  // -0.3 → dead
        assert_eq!(grad[3], 1.0);  // 2.0 → alive
    }

    #[test]
    fn test_adam() {
        let param = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = vec![0.1, 0.2, 0.3];
        let mut adam = Adam::new(0.001);
        let _original = param.data.clone();
        adam.step(&mut [param.clone()], &[grad]);
        // Just verify it doesn't panic; values move
    }

    #[test]
    fn test_higher_order_polynomial() {
        // Test f(x) = x^3 at x = 3.0:
        // f'(x)  = 3 * x^2 = 27.0
        // f''(x) = 6 * x   = 18.0
        let mut tape = GradTape::new();
        let mut x = Tensor::new(vec![3.0], vec![1]);
        x.id = tape.alloc_id();

        // Compute x^2 = x * x
        let x2 = tape.mul(&x, &x);
        // Compute x^3 = x^2 * x
        let y = tape.mul(&x2, &x);

        // First derivative (differentiable)
        let grads = tape.backward_differentiable(y.id);
        let grad_x = grads.get(&x.id).expect("Should compute gradient for x");
        assert!((grad_x.data[0] - 27.0).abs() < 1e-6, "f'(3) = 27, got {}", grad_x.data[0]);

        // Second derivative: differentiate grad_x with respect to x!
        tape.backward(grad_x.id);
        let grad2_x = tape.get_grad(x.id).expect("Should compute second derivative for x");
        assert!((grad2_x[0] - 18.0).abs() < 1e-6, "f''(3) = 18, got {}", grad2_x[0]);
    }

    #[test]
    fn test_hessian_vector_product_and_matrix() {
        // f(x1, x2) = x1^2 + 3*x1*x2 + 2*x2^2
        // at (x1=2.0, x2=1.0)
        // grad f = [ 2*x1 + 3*x2, 3*x1 + 4*x2 ] = [ 7.0, 10.0 ]
        // Hessian H = [ [2.0, 3.0], [3.0, 4.0] ]
        let mut tape = GradTape::new();
        let mut x1 = Tensor::new(vec![2.0], vec![1]);
        x1.id = tape.alloc_id();
        let mut x2 = Tensor::new(vec![1.0], vec![1]);
        x2.id = tape.alloc_id();

        let mut three = Tensor::new(vec![3.0], vec![1]);
        three.id = tape.alloc_id();
        let mut two = Tensor::new(vec![2.0], vec![1]);
        two.id = tape.alloc_id();

        let x1_sq = tape.mul(&x1, &x1);
        let x1_x2 = tape.mul(&x1, &x2);
        let three_x1_x2 = tape.mul(&three, &x1_x2);
        let x2_sq = tape.mul(&x2, &x2);
        let two_x2_sq = tape.mul(&two, &x2_sq);

        let term1 = tape.add(&x1_sq, &three_x1_x2);
        let loss = tape.add(&term1, &two_x2_sq);

        // First derivatives
        let grads = tape.backward_differentiable(loss.id);
        let g1 = grads.get(&x1.id).unwrap();
        let g2 = grads.get(&x2.id).unwrap();
        assert!((g1.data[0] - 7.0).abs() < 1e-6, "df/dx1 = 7, got {}", g1.data[0]);
        assert!((g2.data[0] - 10.0).abs() < 1e-6, "df/dx2 = 10, got {}", g2.data[0]);

        // Second derivatives: d^2 f / dx1^2 should be 2.0
        tape.backward_opts(g1.id, true);
        let h11 = tape.get_grad(x1.id).unwrap();
        assert!((h11[0] - 2.0).abs() < 1e-6, "d^2f/dx1^2 = 2, got {}", h11[0]);

        // d^2 f / (dx1 dx2) should be 3.0
        let h12 = tape.get_grad(x2.id).unwrap();
        assert!((h12[0] - 3.0).abs() < 1e-6, "d^2f/dx1dx2 = 3, got {}", h12[0]);

        // HVP on x1: H_11 * 1.0 = 2.0
        let v1 = Tensor::new(vec![1.0], vec![1]);
        let hv1 = tape.hessian_vector_product(loss.id, x1.id, &v1);
        assert!((hv1.data[0] - 2.0).abs() < 1e-6, "HVP x1 = 2, got {}", hv1.data[0]);

        // Hessian 1x1 on x1: [[2.0]]
        let h_mat1 = tape.hessian(loss.id, x1.id);
        assert!((h_mat1.data[0] - 2.0).abs() < 1e-6, "Hessian x1 = 2, got {}", h_mat1.data[0]);

        // Hessian 1x1 on x2: [[4.0]]
        let h_mat2 = tape.hessian(loss.id, x2.id);
        assert!((h_mat2.data[0] - 4.0).abs() < 1e-6, "Hessian x2 = 4, got {}", h_mat2.data[0]);
    }

    #[test]
    fn test_vector_hessian_and_hvp() {
        // f(x) = sum(x * (x * A)) where x is [1, 2], A is [2, 2]
        // f(x) = 2*x0^2 + 6*x0*x1 + 4*x1^2
        // f'(x) = [4*x0 + 6*x1, 6*x0 + 8*x1] = [14.0, 20.0]
        // Hessian H = [[4.0, 6.0], [6.0, 8.0]]
        // H * v for v = [1.0, 2.0] = [16.0, 22.0]
        let mut tape = GradTape::new();
        let mut x = Tensor::new(vec![2.0, 1.0], vec![1, 2]);
        x.id = tape.alloc_id();

        let mut a = Tensor::new(vec![2.0, 3.0, 3.0, 4.0], vec![2, 2]);
        a.id = tape.alloc_id();

        let x_a = tape.matmul(&x, &a); // [1, 2]
        let prod = tape.mul(&x, &x_a);  // [1, 2]
        let loss = tape.sum(&prod, None); // scalar

        // Gradient
        let grads = tape.backward_differentiable(loss.id);
        let g = grads.get(&x.id).expect("Grad for x");
        assert!((g.data[0] - 14.0).abs() < 1e-5, "df/dx0 = 14, got {}", g.data[0]);
        assert!((g.data[1] - 20.0).abs() < 1e-5, "df/dx1 = 20, got {}", g.data[1]);

        // HVP for v = [1.0, 2.0]
        let v = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let hv = tape.hessian_vector_product(loss.id, x.id, &v);
        assert!((hv.data[0] - 16.0).abs() < 1e-5, "HVP[0] = 16, got {}", hv.data[0]);
        assert!((hv.data[1] - 22.0).abs() < 1e-5, "HVP[1] = 22, got {}", hv.data[1]);

        // Full Hessian: 2x2
        let h = tape.hessian(loss.id, x.id);
        assert_eq!(h.shape, vec![2, 2]);
        assert!((h.data[0] - 4.0).abs() < 1e-5, "H[0,0] = 4, got {}", h.data[0]);
        assert!((h.data[1] - 6.0).abs() < 1e-5, "H[0,1] = 6, got {}", h.data[1]);
        assert!((h.data[2] - 6.0).abs() < 1e-5, "H[1,0] = 6, got {}", h.data[2]);
        assert!((h.data[3] - 8.0).abs() < 1e-5, "H[1,1] = 8, got {}", h.data[3]);
    }

    #[test]
    fn test_mse_hessian() {
        // L(w) = MSE(w * x, y) for scalar w, x = 3.0, y = 1.0
        // pred = 3 * w. L = (3*w - 1)^2
        // dL/dw = 2 * (3*w - 1) * 3 = 18*w - 6
        // d^2L/dw^2 = 18.0
        let mut tape = GradTape::new();
        let mut w = Tensor::new(vec![2.0], vec![1]);
        w.id = tape.alloc_id();
        let mut x = Tensor::new(vec![3.0], vec![1]);
        x.id = tape.alloc_id();
        let mut y = Tensor::new(vec![1.0], vec![1]);
        y.id = tape.alloc_id();

        let pred = tape.mul(&w, &x);
        let loss = tape.mse(&pred, &y);

        let h = tape.hessian(loss.id, w.id);
        assert_eq!(h.shape, vec![1, 1]);
        assert!((h.data[0] - 18.0).abs() < 1e-5, "Hessian of MSE = 18, got {}", h.data[0]);
    }
}

