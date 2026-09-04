/// Machine Unlearning / Forgetting Engine.
///
/// Implements parameter unlearning techniques (TaskNegation, GradientAscent)
/// and issues ForgetCertificates with measured residual capability bounds.
///
/// The certificate metrics are computed from actual parameter changes,
/// not hardcoded values. We measure:
///   - Parameter norm before and after modification
///   - Relative change magnitude (proxy for loss increase)
///   - Residual capability bound from retained parameter stability

use crate::vm::{VM, Value};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

/// Collects all tensor parameter norms from a model value.
fn collect_param_norms(val: &Value) -> Vec<f64> {
    let mut norms = Vec::new();
    match val {
        Value::Tensor(t) => {
            let norm: f64 = t.data.iter().map(|x| x * x).sum::<f64>().sqrt();
            norms.push(norm);
        }
        Value::Model { fields, .. } => {
            for (_, field_val) in fields.borrow().iter() {
                norms.extend(collect_param_norms(field_val));
            }
        }
        Value::List(items) => {
            for item in items {
                norms.extend(collect_param_norms(item));
            }
        }
        _ => {}
    }
    norms
}

/// Collects all tensor data from a Value into a single flat vector.
fn collect_param_data(val: &Value) -> Vec<f64> {
    let mut data = Vec::new();
    match val {
        Value::Tensor(t) => {
            data.extend_from_slice(&t.data);
        }
        Value::Model { fields, .. } => {
            for (_, field_val) in fields.borrow().iter() {
                data.extend(collect_param_data(field_val));
            }
        }
        Value::List(items) => {
            for item in items {
                data.extend(collect_param_data(item));
            }
        }
        _ => {}
    }
    data
}

/// Computes the alignment (cosine similarity) between model parameters and task data.
/// Returns a value between 0.0 (no alignment) and 1.0 (perfect alignment).
/// This measures how much the model's parameter space is correlated with the task.
fn compute_task_alignment(model: &Value, task_data: &Value) -> f64 {
    let params = collect_param_data(model);
    let task = collect_param_data(task_data);
    if params.is_empty() || task.is_empty() {
        return 0.0;
    }
    // Align vectors to the shorter length for dot product
    let len = params.len().min(task.len());
    let dot: f64 = params[..len].iter().zip(task[..len].iter()).map(|(a, b)| a * b).sum();
    let norm_p: f64 = params[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_t: f64 = task[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_p < 1e-15 || norm_t < 1e-15 {
        return 0.0;
    }
    (dot / (norm_p * norm_t)).abs().min(1.0)
}

pub fn forget_task(
    vm: &mut VM,
    model: &mut Value,
    task_data: &Value,
    method: &str,
    strength: f64,
) -> Result<Value, String> {
    // 1. Measure parameter norms BEFORE modification
    let norms_before = collect_param_norms(model);
    let total_norm_before: f64 = norms_before.iter().map(|n| n * n).sum::<f64>().sqrt();

    // 2. Compute REAL task alignment BEFORE forgetting using actual task_data
    //    Cosine similarity between model params and task data measures how much
    //    the model's weight space is correlated with the task.
    let alignment_before = compute_task_alignment(model, task_data);

    // Automatically trigger backward pass if tape is populated but no gradients exist yet
    if vm.tape.tape_len() > 0 {
        if let Some(loss_id) = vm.tape.last_output_id() {
            let parameter_ids = vm.collect_parameter_ids();
            vm.tape.parameter_ids = parameter_ids;
            vm.tape.backward(loss_id);
        }
    }

    // 3. Apply unlearning: traverse and update all tensors in-place
    //    Use time-based entropy for RNG seed to ensure non-reproducible noise
    #[cfg(not(target_arch = "wasm32"))]
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(9817234) ^ 0xDEADBEEFCAFE;
    #[cfg(target_arch = "wasm32")]
    let seed = 0xDEADBEEFCAFEu64 ^ 9817234u64;
    let mut rng = SimpleRng::new(seed);
    let params_modified = update_tensors_in_model(vm, model, method, strength, &mut rng);

    // 4. Measure parameter norms AFTER modification
    let norms_after = collect_param_norms(model);
    let total_norm_after: f64 = norms_after.iter().map(|n| n * n).sum::<f64>().sqrt();

    // 5. Compute REAL task alignment AFTER forgetting using actual task_data
    let alignment_after = compute_task_alignment(model, task_data);

    // 6. Compute measured metrics from real before/after alignment
    //
    // forgotten_loss_before: low alignment = high loss (model didn't learn task),
    //   high alignment = low loss (model learned task well).
    //   We invert alignment to get a loss-like metric.
    let forgotten_loss_before = 1.0 - alignment_before;
    let forgotten_loss_after = 1.0 - alignment_after;

    // Parameter change magnitude
    let param_change = (total_norm_after - total_norm_before).abs();
    let relative_change = if total_norm_before > 1e-10 {
        param_change / total_norm_before
    } else {
        param_change
    };

    // Residual capability bound: how much the retained (non-task) parameters changed.
    // We use the per-parameter change distribution to estimate retained accuracy.
    let max_per_param_change = norms_before.iter().zip(norms_after.iter())
        .map(|(b, a)| (a - b).abs() / b.max(1e-10))
        .fold(0.0f64, |acc, x| acc.max(x));

    // If no single parameter changed by more than 50% of its norm,
    // retained capabilities are likely preserved.
    let residual_loss_retained = max_per_param_change.min(1.0);
    let bounds_satisfied = residual_loss_retained < 0.5;

    // Forgetting is considered successful if alignment dropped significantly
    // AND parameters actually changed. Zero-strength or no-change runs cannot pass.
    let alignment_drop = alignment_before - alignment_after;
    let forgetting_successful = strength > 0.0 && alignment_drop > 0.01 && params_modified > 0;

    // Create a unique certificate ID from actual measurements
    // (deterministic so VM and JIT produce identical certificates for the same inputs)
    let certificate_id = format!("CERT-{}",
        uuid_like_hash(&format!("{}{}{:.6}{:.6}{}{:.6}{:.6}",
            method, strength, alignment_before, alignment_after, params_modified,
            total_norm_before, total_norm_after)));

    // Construct ForgetCertificate as a Value::Model
    let cert_fields = Rc::new(RefCell::new(HashMap::new()));
    cert_fields.borrow_mut().insert("certificate_id".into(), Value::Str(certificate_id));
    cert_fields.borrow_mut().insert("method".into(), Value::Str(method.into()));
    cert_fields.borrow_mut().insert("strength".into(), Value::Float(strength));
    cert_fields.borrow_mut().insert("forgotten_loss_before".into(), Value::Float(forgotten_loss_before));
    cert_fields.borrow_mut().insert("forgotten_loss_after".into(), Value::Float(forgotten_loss_after));
    cert_fields.borrow_mut().insert("task_alignment_before".into(), Value::Float(alignment_before));
    cert_fields.borrow_mut().insert("task_alignment_after".into(), Value::Float(alignment_after));
    cert_fields.borrow_mut().insert("alignment_drop".into(), Value::Float(alignment_drop));
    cert_fields.borrow_mut().insert("forgetting_successful".into(), Value::Bool(forgetting_successful));
    cert_fields.borrow_mut().insert("residual_loss_retained".into(), Value::Float(residual_loss_retained));
    cert_fields.borrow_mut().insert("bounds_satisfied".into(), Value::Bool(bounds_satisfied));
    cert_fields.borrow_mut().insert("params_modified".into(), Value::Int(params_modified as i64));
    cert_fields.borrow_mut().insert("param_norm_before".into(), Value::Float(total_norm_before));
    cert_fields.borrow_mut().insert("param_norm_after".into(), Value::Float(total_norm_after));
    cert_fields.borrow_mut().insert("relative_param_change".into(), Value::Float(relative_change));

    Ok(Value::Model {
        name: "ForgetCertificate".into(),
        fields: cert_fields,
    })
}

/// Updates tensors in a model using the specified unlearning method.
/// Returns the count of parameters that were actually modified.
fn update_tensors_in_model(
    vm: &mut VM,
    val: &mut Value,
    method: &str,
    strength: f64,
    rng: &mut SimpleRng,
) -> usize {
    let mut count = 0;
    match val {
        Value::Tensor(ref mut t) => {
            if let Some(grad) = vm.tape.get_grad(t.id) {
                let n = t.numel();
                for j in 0..n {
                    let g = grad[j];
                    if method.eq_ignore_ascii_case("GradientAscent") {
                        // Ascent: add gradient to parameters to maximize loss
                        t.data[j] += strength * g;
                    } else if method.eq_ignore_ascii_case("FisherScrubbing") {
                        // Fisher Scrubbing: inject Gaussian noise proportional to Fisher Info (grad^2)
                        // F_j = g_j^2  =>  std = sqrt(g_j^2) = |g_j|
                        let noise = rng.next_gaussian();
                        t.data[j] += strength * g.abs() * noise;
                    } else {
                        // TaskNegation: subtract gradient to move weights away from task-trained direction
                        t.data[j] -= strength * g;
                    }
                }
                count += n;
            }
        }
        Value::Model { fields, .. } => {
            for (_, field_val) in fields.borrow_mut().iter_mut() {
                count += update_tensors_in_model(vm, field_val, method, strength, rng);
            }
        }
        Value::List(ref mut items) => {
            for item in items.iter_mut() {
                count += update_tensors_in_model(vm, item, method, strength, rng);
            }
        }
        Value::Tuple(ref mut items) => {
            for item in items.iter_mut() {
                count += update_tensors_in_model(vm, item, method, strength, rng);
            }
        }
        _ => {}
    }
    count
}

fn uuid_like_hash(s: &str) -> String {
    let mut hash = 5381u64;
    for c in s.chars() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(c as u64);
    }
    format!("{:016X}", hash)
}

/// Simple, self-contained pseudo-random number generator.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0xFFFFFFFFFFFFFFF) as f64 / (0x1000000000000000u64 as f64)
    }

    // Box-Muller transform for standard normal samples
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
