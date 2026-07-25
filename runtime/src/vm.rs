/// NEURON Virtual Machine — executes NEURON IR against the runtime.
///
/// Maintains a value stack, call stack, and gradient tape.
/// This is v1 execution: compile to IR, interpret IR.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use neuron_compiler::ir::*;
use crate::tensor::*;
use crate::autograd::*;
use crate::buffer::Buffer;

pub struct CudaModuleFunction {
    pub module: *mut std::ffi::c_void,
    pub function: *mut std::ffi::c_void,
}

unsafe impl Send for CudaModuleFunction {}
unsafe impl Sync for CudaModuleFunction {}

static GLOBAL_KERNEL_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, CudaModuleFunction>>> = std::sync::OnceLock::new();

fn validate_shape(inputs: &[ValueId], get: impl Fn(ValueId) -> Value) -> Result<Vec<usize>, String> {
    let mut shape = Vec::new();
    let mut product: usize = 1;
    for &id in inputs {
        let val = get(id).as_int();
        if val < 0 {
            return Err(format!("Runtime Error: Negative dimension size: {}", val));
        }
        let dim = val as usize;
        if dim > 1_000_000 {
            return Err(format!("Runtime Error: Dimension size too large: {}", dim));
        }
        product = product.checked_mul(dim)
            .ok_or_else(|| "Runtime Error: Shape size overflow".to_string())?;
        shape.push(dim);
    }
    if product > 10_000_000 {
        return Err(format!("Runtime Error: Tensor size too large: {} elements", product));
    }
    Ok(shape)
}

fn validate_static_shape(shape: &[i64]) -> Result<Vec<usize>, String> {
    let mut s = Vec::new();
    let mut product: usize = 1;
    for &val in shape {
        if val < 0 {
            return Err(format!("Runtime Error: Negative dimension size: {}", val));
        }
        let dim = val as usize;
        if dim > 1_000_000 {
            return Err(format!("Runtime Error: Dimension size too large: {}", dim));
        }
        product = product.checked_mul(dim)
            .ok_or_else(|| "Runtime Error: Shape size overflow".to_string())?;
        s.push(dim);
    }
    if product > 10_000_000 {
        return Err(format!("Runtime Error: Tensor size too large: {} elements", product));
    }
    Ok(s)
}

/// Runtime value — everything the VM can hold.
#[derive(Clone, Debug)]
pub enum Value {
    Tensor(Tensor),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Uncertain { value: f64, std: f64, confidence: f64 },
    Random { mean: f64, variance: f64 },
    Temporal { data: Box<Value>, direction: String },
    Causal { data: Box<Value>, mode: String },
    Model { name: String, fields: std::rc::Rc<std::cell::RefCell<HashMap<String, Value>>> },
    CausalModel { name: String, variables: Vec<String> },
    Void,
    None,
    Err(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValueWrapper {
    Temporal(String),
    Causal(String),
}

impl Value {
    pub fn unwrap_ref(&self) -> &Value {
        let mut curr = self;
        loop {
            match curr {
                Value::Temporal { data, .. } => curr = data,
                Value::Causal { data, .. } => curr = data,
                _ => break,
            }
        }
        curr
    }

    pub fn strip_wrappers(self) -> (Value, Vec<ValueWrapper>) {
        let mut val = self;
        let mut wrappers = Vec::new();
        loop {
            match val {
                Value::Temporal { data, direction } => {
                    wrappers.push(ValueWrapper::Temporal(direction));
                    val = *data;
                }
                Value::Causal { data, mode } => {
                    wrappers.push(ValueWrapper::Causal(mode));
                    val = *data;
                }
                _ => break,
            }
        }
        (val, wrappers)
    }

    pub fn apply_wrappers(mut self, wrappers: Vec<ValueWrapper>) -> Self {
        for w in wrappers.into_iter().rev() {
            match w {
                ValueWrapper::Temporal(dir) => self = Value::Temporal { data: Box::new(self), direction: dir },
                ValueWrapper::Causal(mode) => self = Value::Causal { data: Box::new(self), mode },
            }
        }
        self
    }

    pub fn combine_wrappers(mut a_wraps: Vec<ValueWrapper>, b_wraps: Vec<ValueWrapper>) -> Vec<ValueWrapper> {
        for w in b_wraps {
            if !a_wraps.iter().any(|cw| match (cw, &w) {
                (ValueWrapper::Temporal(d1), ValueWrapper::Temporal(d2)) => d1 == d2,
                (ValueWrapper::Causal(m1), ValueWrapper::Causal(m2)) => m1 == m2,
                _ => false,
            }) {
                a_wraps.push(w);
            }
        }
        a_wraps
    }

    pub fn as_tensor(&self) -> Option<&Tensor> {
        if let Value::Tensor(t) = self.unwrap_ref() { Some(t) } else { None }
    }
    pub fn as_float(&self) -> f64 {
        match self.unwrap_ref() {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            Value::Tensor(t) if t.numel() == 1 => t.data[0],
            _ => panic!("Runtime TypeError: Expected Float, found {:?}", self),
        }
    }
    pub fn as_int(&self) -> i64 {
        match self.unwrap_ref() {
            Value::Int(i) => *i,
            Value::Float(f) => *f as i64,
            _ => panic!("Runtime TypeError: Expected Int, found {:?}", self),
        }
    }
    pub fn as_bool(&self) -> bool {
        match self.unwrap_ref() {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            _ => panic!("Runtime TypeError: Expected Bool, found {:?}", self),
        }
    }
    pub fn display(&self) -> String {
        match self {
            Value::Tensor(t) => format!("{}", t),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => format!("{:.6}", f),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => s.clone(),
            Value::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", inner.join(", "))
            }
            Value::Tuple(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("({})", inner.join(", "))
            }
            Value::Uncertain { value, std, confidence } =>
                format!("Uncertain(value={:.4}, std={:.4}, confidence={:.4})", value, std, confidence),
            Value::Random { mean, variance } =>
                format!("Random(mean={:.4}, variance={:.4})", mean, variance),
            Value::Temporal { direction, .. } =>
                format!("Temporal(direction={})", direction),
            Value::Causal { mode, .. } =>
                format!("Causal(mode={})", mode),
            Value::Model { name, fields } => {
                let fields_ref = fields.borrow();
                if fields_ref.is_empty() {
                    format!("<Model {}>", name)
                } else {
                    let mut parts: Vec<String> = fields_ref.iter()
                        .map(|(k, v)| format!("  {}: {}", k, v.display()))
                        .collect();
                    parts.sort();
                    format!("<{}>\n{}", name, parts.join("\n"))
                }
            },
            Value::CausalModel { name, .. } => format!("<CausalModel {}>", name),
            Value::Void => "void".into(),
            Value::None => "None".into(),
            Value::Err(msg) => format!("Runtime Error: {}", msg),
        }
    }
}

/// The NEURON VM — interprets IR programs.
pub struct VM {
    /// Global variable storage.
    pub globals: HashMap<String, Value>,
    /// Function table (name → IR function).
    pub functions: HashMap<String, IRFunction>,
    /// Gradient tape.
    pub tape: GradTape,
    /// Call stack for nested function calls.
    pub call_stack: Vec<CallFrame>,
    /// Maximum recursion depth.
    pub max_depth: usize,
    /// Effect tracker.
    pub effect_log: Vec<String>,
    /// Captured print output.
    pub stdout_log: Vec<String>,
    /// Temporal direction violations — causes panic.
    strict_temporal: bool,
    /// Strict causal mode checking.
    strict_causal: bool,
    /// Optimizer states.
    adam_m: HashMap<TensorId, Vec<f64>>,
    adam_v: HashMap<TensorId, Vec<f64>>,
    optimizer_step: usize,
    /// Loaded CUDA modules and functions.
    pub cuda_kernels: HashMap<String, CudaModuleFunction>,
    /// Functions that have had their kernels compiled.
    pub compiled_functions: std::collections::HashSet<String>,
    /// RNG counter for sample_categorical.
    rng_counter: u64,
    /// Execution floating-point precision (F64 or F32).
    pub precision: crate::tensor::DType,
    /// Distributed cluster training manager.
    pub dist_manager: crate::distributed::DistributedManager,
}

pub struct CallFrame {
    pub function_name: String,
    pub current_block: BlockId,
    pub instruction_idx: usize,
    pub locals: HashMap<String, Value>,
    pub ssa_values: HashMap<ValueId, Value>,
    pub return_addr: Option<(String, BlockId, ValueId)>, // (caller_function_name, caller_block, output_val_id)
    pub fused_groups: HashMap<ValueId, (usize, neuron_compiler::cuda_codegen::FusedGroup)>,
    pub fused_skipped_ids: std::collections::HashSet<ValueId>,
}

impl VM {
    pub fn new() -> Self {
        Self {
            globals: HashMap::new(),
            functions: HashMap::new(),
            tape: GradTape::new(),
            call_stack: Vec::new(),
            max_depth: 256,
            effect_log: Vec::new(),
            stdout_log: Vec::new(),
            strict_temporal: true,
            strict_causal: true,
            adam_m: HashMap::new(),
            adam_v: HashMap::new(),
            optimizer_step: 0,
            cuda_kernels: HashMap::new(),
            compiled_functions: std::collections::HashSet::new(),
            rng_counter: 0,
            precision: crate::tensor::DType::F64,
            dist_manager: crate::distributed::DistributedManager::new(
                crate::distributed::DistributedConfig::single_node()
            ),
        }
    }

    pub fn with_precision(mut self, precision: crate::tensor::DType) -> Self {
        self.precision = precision;
        self
    }

    /// Load an IR program into the VM.
    pub fn load(&mut self, program: &IRProgram) {
        for func in &program.functions {
            self.functions.insert(func.name.clone(), func.clone());
        }
        for global in &program.globals {
            let val = match &global.value {
                IRConst::Int(v) => Value::Int(*v),
                IRConst::Float(v) => Value::Float(*v),
                IRConst::Bool(v) => Value::Bool(*v),
                IRConst::String(s) => Value::Str(s.clone()),
                IRConst::Tensor(data, shape) => {
                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                    Value::Tensor(Tensor::new(data.clone(), shape_usize))
                }
            };
            self.globals.insert(global.name.clone(), val);
        }
    }

    /// Execute a function by name.
    pub fn execute(&mut self, fn_name: &str, args: Vec<Value>) -> Result<Value, String> {
        let mut resolved_name = fn_name.to_string();
        let mut keyword_args = Vec::new();
        if fn_name.contains(':') {
            let parts: Vec<&str> = fn_name.split(':').collect();
            resolved_name = parts[0].to_string();
            if parts.len() > 1 {
                keyword_args = parts[1].split(',').map(|s| s.to_string()).collect();
            }
        }
        
        // Check if this is a causal model constructor call
        if let Some(Value::Str(ref s)) = self.globals.get(&resolved_name) {
            if s.starts_with("causal_model:") {
                let parts: Vec<&str> = s.split(':').collect();
                if parts.len() >= 3 {
                    let name = parts[1].to_string();
                    let mut variables = Vec::new();
                    for part in &parts[2..] {
                        if part.starts_with("variables=") {
                            variables = part["variables=".len()..].split(',').map(|v| v.to_string()).filter(|v| !v.is_empty()).collect();
                        }
                    }
                    return Ok(Value::CausalModel { name, variables });
                }
            }
        }
        
        if resolved_name.starts_with("obj_") {
            if let Some(first_arg) = args.first() {
                if let Value::CausalModel { name: model_name, variables } = first_arg {
                    if resolved_name == "obj_observe" || resolved_name == "obj_intervene" {
                        let mut weights = vec![vec![0.0; variables.len()]; variables.len()];
                        let noise_vars = vec![1.0; variables.len()];
                        let noise_means = vec![0.0; variables.len()];

                        if let Some(Value::Str(ref global_val)) = self.globals.get(model_name) {
                            if global_val.starts_with("causal_model:") {
                                let parts: Vec<&str> = global_val.split(':').collect();
                                for part in parts {
                                    if part.starts_with("edges=") {
                                        let edges_str = &part["edges=".len()..];
                                        for edge in edges_str.split(',') {
                                            let nodes: Vec<&str> = edge.split("->").collect();
                                            if nodes.len() == 2 {
                                                if let (Some(&u), Some(&v)) = (
                                                    variables.iter().position(|x| x == nodes[0]).as_ref(),
                                                    variables.iter().position(|x| x == nodes[1]).as_ref(),
                                                ) {
                                                    weights[u][v] = 1.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let mut map = HashMap::new();
                        for (i, kw) in keyword_args.iter().enumerate() {
                            if let Some(val) = args.get(i + 1) {
                                map.insert(kw.clone(), val.as_float());
                            }
                        }

                        let model = crate::causal::CausalModel::new(variables.clone(), weights, noise_vars, noise_means);
                        if resolved_name == "obj_observe" {
                            let obs_res = model.observe(&map)
                                .ok_or_else(|| "Failed to compute observational inference".to_string())?;
                            return Ok(convert_causal_results(obs_res));
                        } else {
                            let int_res = model.intervene(&map)
                                .ok_or_else(|| "Failed to compute interventional inference".to_string())?;
                            return Ok(convert_causal_results(int_res));
                        }
                    }
                } else if let Value::Model { name, .. } = first_arg {
                    let method = &resolved_name[4..];
                    resolved_name = format!("{}_{}", name, method);
                }
            }
        }

        // Intercept native/built-in functions
        if resolved_name == "causal_discover" || resolved_name == "causal::discover" {
            if args.len() < 3 {
                return Err("causal_discover requires 3 arguments: (data, names, alpha)".into());
            }
            let data_list = match &args[0] {
                Value::List(l) => l,
                _ => return Err("causal_discover argument 1 must be a list of lists".into()),
            };
            let mut data = Vec::new();
            for item in data_list {
                if let Value::List(row) = item {
                    let mut r = Vec::new();
                    for val in row {
                        r.push(val.as_float());
                    }
                    data.push(r);
                } else {
                    return Err("causal_discover argument 1 must be a list of lists".into());
                }
            }
            let names_list = match &args[1] {
                Value::List(l) => l,
                _ => return Err("causal_discover argument 2 must be a list of strings".into()),
            };
            let mut names = Vec::new();
            for val in names_list {
                names.push(val.display());
            }
            let alpha = args[2].as_float();

            let pc_res = crate::causal::discover(&data, names, alpha);

            let fields = Rc::new(RefCell::new(HashMap::new()));
            
            let mut adj_rows = Vec::new();
            for row in pc_res.adjacency {
                let r_vals = row.into_iter().map(Value::Float).collect();
                adj_rows.push(Value::List(r_vals));
            }
            fields.borrow_mut().insert("adjacency".into(), Value::List(adj_rows));

            let mut conf_rows = Vec::new();
            for row in pc_res.confidences {
                let r_vals = row.into_iter().map(Value::Float).collect();
                conf_rows.push(Value::List(r_vals));
            }
            fields.borrow_mut().insert("confidences".into(), Value::List(conf_rows));

            let name_vals = pc_res.names.into_iter().map(Value::Str).collect();
            fields.borrow_mut().insert("names".into(), Value::List(name_vals));

            return Ok(Value::Model {
                name: "PCResult".into(),
                fields,
            });
        }

        if resolved_name == "causal_observe" || resolved_name == "causal::observe" {
            if args.len() < 5 {
                return Err("causal_observe requires 5 arguments: (names, weights, noise_vars, noise_means, evidence)".into());
            }
            let names = parse_string_list(&args[0])?;
            let weights = parse_float_matrix(&args[1])?;
            let noise_vars = parse_float_list(&args[2])?;
            let noise_means = parse_float_list(&args[3])?;
            let evidence = parse_evidence_map(&args[4])?;

            let model = crate::causal::CausalModel::new(names, weights, noise_vars, noise_means);
            let obs_res = model.observe(&evidence)
                .ok_or_else(|| "Failed to compute observational inference".to_string())?;

            return Ok(convert_causal_results(obs_res));
        }

        if resolved_name == "causal_intervene" || resolved_name == "causal::intervene" {
            if args.len() < 5 {
                return Err("causal_intervene requires 5 arguments: (names, weights, noise_vars, noise_means, interventions)".into());
            }
            let names = parse_string_list(&args[0])?;
            let weights = parse_float_matrix(&args[1])?;
            let noise_vars = parse_float_list(&args[2])?;
            let noise_means = parse_float_list(&args[3])?;
            let interventions = parse_evidence_map(&args[4])?;

            let model = crate::causal::CausalModel::new(names, weights, noise_vars, noise_means);
            let int_res = model.intervene(&interventions)
                .ok_or_else(|| "Failed to compute interventional inference".to_string())?;

            return Ok(convert_causal_results(int_res));
        }

        if resolved_name == "causal_counterfactual" || resolved_name == "causal::counterfactual" {
            if args.len() < 7 {
                return Err("causal_counterfactual requires 7 arguments: (names, weights, noise_vars, noise_means, evidence, interventions, queries)".into());
            }
            let names = parse_string_list(&args[0])?;
            let weights = parse_float_matrix(&args[1])?;
            let noise_vars = parse_float_list(&args[2])?;
            let noise_means = parse_float_list(&args[3])?;
            let evidence = parse_evidence_map(&args[4])?;
            let interventions = parse_evidence_map(&args[5])?;
            let queries = parse_string_list(&args[6])?;

            let model = crate::causal::CausalModel::new(names, weights, noise_vars, noise_means);
            let cf_res = model.counterfactual(&evidence, &interventions, &queries)
                .ok_or_else(|| "Failed to compute counterfactual inference".to_string())?;

            let fields = Rc::new(RefCell::new(HashMap::new()));
            for (name, val) in cf_res {
                fields.borrow_mut().insert(name, Value::Float(val));
            }
            return Ok(Value::Model {
                name: "CounterfactualResult".into(),
                fields,
            });
        }

        if resolved_name == "range" {
            if args.is_empty() {
                return Err("range requires 1 argument: (n)".into());
            }
            let n = args[0].as_int();
            let mut list = Vec::new();
            for i in 0..n {
                list.push(Value::Int(i));
            }
            return Ok(Value::List(list));
        }

        if resolved_name == "load" {
            let mut t = Tensor::zeros(&[1, 20]);
            t.id = self.tape.alloc_id();
            return Ok(Value::Temporal {
                data: Box::new(Value::Tensor(t)),
                direction: "past_to_future".into(),
            });
        }

        if resolved_name == "sin" {
            if args.is_empty() {
                return Err("sin requires 1 argument: (x)".into());
            }
            let val = args[0].as_float();
            return Ok(Value::Float(val.sin()));
        }

        if resolved_name == "cos" {
            if args.is_empty() {
                return Err("cos requires 1 argument: (x)".into());
            }
            let val = args[0].as_float();
            return Ok(Value::Float(val.cos()));
        }

        if resolved_name == "load_ohlcv" {
            if args.is_empty() {
                return Err("load_ohlcv requires a file path argument".into());
            }
            let path = args[0].display();
            if path.contains("..") || path.starts_with('/') || path.starts_with('\\') || (path.len() >= 2 && path.chars().next().unwrap().is_alphabetic() && path.chars().nth(1).unwrap() == ':') {
                return Err(format!("Security Error: Access to path '{}' is restricted", path));
            }
            let mut data = Vec::new();
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 5 {
                        let row: Vec<Value> = parts.iter().map(|p| {
                            Value::Float(p.trim().parse::<f64>().unwrap_or(0.0))
                        }).collect();
                        data.push(Value::List(row));
                    }
                }
            }
            if data.is_empty() {
                for i in 0..10 {
                    let base = 100.0 + i as f64 * 0.5;
                    data.push(Value::List(vec![
                        Value::Float(base),
                        Value::Float(base + 1.2),
                        Value::Float(base - 0.8),
                        Value::Float(base + 0.4),
                        Value::Float(1000.0 + i as f64 * 100.0),
                    ]));
                }
            }
            let mut flat_data = Vec::new();
            let rows = data.len();
            let cols = 5;
            for row_val in data {
                if let Value::List(row_items) = row_val {
                    for item in row_items.iter().take(5) {
                        flat_data.push(item.as_float());
                    }
                }
            }
            let mut t = Tensor::new(flat_data, vec![rows, cols]);
            t.id = self.tape.alloc_id();
            return Ok(Value::Temporal {
                data: Box::new(Value::Tensor(t)),
                direction: "past_to_future".into(),
            });
        }

        if resolved_name == "load_ohlcv_list" {
            if args.is_empty() {
                return Err("load_ohlcv_list requires a file path argument".into());
            }
            let path = args[0].display();
            if path.contains("..") || path.starts_with('/') || path.starts_with('\\') || (path.len() >= 2 && path.chars().next().unwrap().is_alphabetic() && path.chars().nth(1).unwrap() == ':') {
                return Err(format!("Security Error: Access to path '{}' is restricted", path));
            }
            let mut data = Vec::new();
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 5 {
                        let row: Vec<Value> = parts.iter().map(|p| {
                            Value::Float(p.trim().parse::<f64>().unwrap_or(0.0))
                        }).collect();
                        data.push(Value::List(row));
                    }
                }
            }
            return Ok(Value::List(data));
        }

        if resolved_name == "load_tensor" {
            if args.len() < 3 {
                return Err("load_tensor requires 3 arguments: (path, rows, cols)".into());
            }
            let path = args[0].display();
            let rows = args[1].as_int() as usize;
            let cols = args[2].as_int() as usize;
            let expected_len = rows * cols;

            let data = match std::fs::read(&path) {
                Ok(bytes) => {
                    let num_f64 = bytes.len() / 8;
                    if num_f64 < expected_len {
                        return Err(format!(
                            "load_tensor: file '{}' has {} f64 values but expected {} ({}x{})",
                            path, num_f64, expected_len, rows, cols
                        ));
                    }
                    let mut vals = Vec::with_capacity(expected_len);
                    for i in 0..expected_len {
                        let offset = i * 8;
                        let bytes_arr: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
                        vals.push(f64::from_le_bytes(bytes_arr));
                    }
                    vals
                }
                Err(e) => {
                    return Err(format!("load_tensor: cannot read file '{}': {}", path, e));
                }
            };

            let mut t = Tensor::new(data, vec![rows, cols]);
            t.id = self.tape.alloc_id();
            return Ok(Value::Tensor(t));
        }

        if resolved_name == "argmax" {
            if args.is_empty() {
                return Err("argmax requires 1 argument: (tensor)".into());
            }
            let t = match &args[0] {
                Value::Tensor(t) => t,
                other => return Err(format!("argmax expects a Tensor, got {:?}", other)),
            };
            if t.data.is_empty() {
                return Err("argmax on empty tensor".into());
            }
            // For a 2D tensor, return argmax of the last row (last token position)
            // For a 1D tensor, return argmax of the whole vector
            let slice = if t.ndim() == 2 {
                let cols = t.shape[1];
                let last_row_start = (t.shape[0] - 1) * cols;
                &t.data[last_row_start..last_row_start + cols]
            } else {
                &t.data[..]
            };
            let mut max_idx = 0usize;
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..slice.len() {
                if slice[i] > max_val {
                    max_val = slice[i];
                    max_idx = i;
                }
            }
            return Ok(Value::Int(max_idx as i64));
        }

        if resolved_name == "sample_categorical" {
            if args.len() < 2 {
                return Err("sample_categorical requires 2 arguments: (logits_tensor, temperature)".into());
            }
            let t = match &args[0] {
                Value::Tensor(t) => t,
                other => return Err(format!("sample_categorical expects a Tensor as first arg, got {:?}", other)),
            };
            let temperature = match &args[1] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                other => return Err(format!("sample_categorical expects a Float temperature, got {:?}", other)),
            };
            if t.data.is_empty() {
                return Err("sample_categorical on empty tensor".into());
            }
            if temperature <= 0.0 {
                return Err("sample_categorical: temperature must be > 0".into());
            }

            // Get the logits slice (last row for 2D, whole vector for 1D)
            let slice = if t.ndim() == 2 {
                let cols = t.shape[1];
                let last_row_start = (t.shape[0] - 1) * cols;
                &t.data[last_row_start..last_row_start + cols]
            } else {
                &t.data[..]
            };

            // Apply temperature scaling
            let scaled: Vec<f64> = slice.iter().map(|&x| x / temperature).collect();

            // Numerically stable softmax
            let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

            // Sample from the categorical distribution
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            // Use a simple PRNG seeded from time + a counter for reproducibility within a session
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            self.rng_counter += 1;
            self.rng_counter.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert hash to uniform [0, 1)
            let r: f64 = (hash as f64) / (u64::MAX as f64);

            let mut cumulative = 0.0;
            let mut sampled_idx = probs.len() - 1; // fallback to last
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    sampled_idx = i;
                    break;
                }
            }
            return Ok(Value::Int(sampled_idx as i64));
        }

        if resolved_name == "save_tensor" {
            if args.len() < 2 {
                return Err("save_tensor requires 2 arguments: (tensor, path)".into());
            }
            let t = match &args[0] {
                Value::Tensor(t) => t,
                other => return Err(format!("save_tensor expects a Tensor, got {:?}", other)),
            };
            let path = args[1].display();
            // Create parent directories if needed
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let slice = &t.data[..];
            let mut bytes = Vec::with_capacity(slice.len() * 8);
            for i in 0..slice.len() {
                bytes.extend_from_slice(&slice[i].to_le_bytes());
            }
            std::fs::write(&path, &bytes)
                .map_err(|e| format!("save_tensor: cannot write '{}': {}", path, e))?;
            return Ok(Value::Void);
        }

        if resolved_name == "char_from_int" {
            if args.is_empty() {
                return Err("char_from_int requires 1 argument: (int)".into());
            }
            let code = args[0].as_int() as u32;
            let ch = std::char::from_u32(code).unwrap_or('?');
            return Ok(Value::Str(ch.to_string()));
        }

        if resolved_name == "onehot" {
            if args.len() < 2 {
                return Err("onehot requires 2 arguments: (index, size)".into());
            }
            let idx = args[0].as_int() as usize;
            let size = args[1].as_int() as usize;
            let mut data = vec![0.0; size];
            if idx < size {
                data[idx] = 1.0;
            }
            let mut t = Tensor::new(data, vec![1, size]);
            t.id = self.tape.alloc_id();
            return Ok(Value::Tensor(t));
        }

        if resolved_name == "forget" {
            if args.len() < 4 {
                return Err("forget requires 4 arguments: (model, task_data, method, strength)".into());
            }
            let mut model = args[0].clone();
            let task_data = &args[1];
            let method = args[2].display();
            let strength = args[3].as_float();
            let cert = crate::forget::forget_task(self, &mut model, task_data, &method, strength)?;
            return Ok(cert);
        }

        let func = if let Some(f) = self.functions.get(&resolved_name) {
            f.clone()
        } else if self.functions.contains_key(&format!("{}_new", resolved_name)) {
            resolved_name = format!("{}_new", resolved_name);
            self.functions.get(&resolved_name)
                .ok_or_else(|| format!("Function '{}' not found", resolved_name))?.clone()
        } else {
            return Err(format!("Function '{}' not found", resolved_name));
        };

        if crate::device::Device::cuda_available() && !self.compiled_functions.contains(&resolved_name) {
            self.compile_function_kernels(&func);
        }

        if self.call_stack.len() >= self.max_depth {
            return Err("Stack overflow: maximum recursion depth exceeded".into());
        }

        // Initialize the first CallFrame
        let mut frame = CallFrame {
            function_name: resolved_name.clone(),
            current_block: func.entry,
            instruction_idx: 0,
            locals: HashMap::new(),
            ssa_values: HashMap::new(),
            return_addr: None,
            fused_groups: build_fused_groups_cache(&func, &self.functions.values().cloned().collect::<Vec<_>>()),
            fused_skipped_ids: std::collections::HashSet::new(),
        };

        // If it's a constructor, pre-initialize "self"
        let is_constructor = resolved_name.ends_with("_new");
        if is_constructor {
            let model_name = &resolved_name[..resolved_name.len() - 4];
            frame.locals.insert("self".to_string(), Value::Model {
                name: model_name.to_string(),
                fields: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
            });
        }

        // Bind arguments to parameters
        for (param, arg) in func.params.iter().zip(args.into_iter()) {
            frame.locals.insert(param.name.clone(), arg.clone());
            frame.ssa_values.insert(param.id, arg);
        }

        self.call_stack.push(frame);

        let mut final_result = Value::Void;

        while !self.call_stack.is_empty() {
            let frame_idx = self.call_stack.len() - 1;
            let current_func_name = self.call_stack[frame_idx].function_name.clone();
            
            let func = self.functions.get(&current_func_name)
                .ok_or_else(|| format!("Function '{}' not found during execution", current_func_name))?.clone();
            let current_block_id = self.call_stack[frame_idx].current_block;
            let inst_idx = self.call_stack[frame_idx].instruction_idx;
            
            let block = func.blocks.iter().find(|b| b.id == current_block_id)
                .ok_or_else(|| format!("Block {} not found in function {}", current_block_id, current_func_name))?;
            
            if inst_idx < block.instructions.len() {
                let node = block.instructions[inst_idx].clone();
                
                if self.call_stack[frame_idx].fused_skipped_ids.contains(&node.id) {
                    self.call_stack[frame_idx].instruction_idx += 1;
                    continue;
                }
                
                if crate::device::Device::cuda_available() {
                    if let Some((g_idx, group)) = self.call_stack[frame_idx].fused_groups.get(&node.id).cloned() {
                        self.execute_fused_group(frame_idx, &func, g_idx, &group)?;
                        self.call_stack[frame_idx].instruction_idx += 1;
                        continue;
                    }
                }
                
                if let IROp::Call { ref function } = node.op {
                    if self.call_stack.len() >= self.max_depth {
                        return Err("Stack overflow: maximum recursion depth exceeded".into());
                    }
                    
                    // Retrieve arguments
                    let mut call_args = Vec::new();
                    for &input_id in &node.inputs {
                        let val = self.call_stack[frame_idx].ssa_values.get(&input_id).cloned()
                            .or_else(|| self.globals.get(&input_id.to_string()).cloned())
                            .unwrap_or(Value::Void);
                        call_args.push(val);
                    }
                    
                    // Resolve target function
                    let mut resolved_callee = function.to_string();
                    if function.starts_with("obj_") {
                        if let Some(first_arg) = call_args.first() {
                            if let Value::Model { name, .. } = first_arg {
                                let method = &function[4..];
                                resolved_callee = format!("{}_{}", name, method);
                            }
                        }
                    }
                    
                    let callee = if let Some(f) = self.functions.get(&resolved_callee) {
                        Some(f.clone())
                    } else if self.functions.contains_key(&format!("{}_new", resolved_callee)) {
                        resolved_callee = format!("{}_new", resolved_callee);
                        Some(self.functions.get(&resolved_callee)
                            .ok_or_else(|| format!("Function '{}' not found", resolved_callee))?.clone())
                    } else {
                        None
                    };
                    
                    if let Some(callee) = callee {
                        if crate::device::Device::cuda_available() && !self.compiled_functions.contains(&resolved_callee) {
                            self.compile_function_kernels(&callee);
                        }

                        // Update caller frame instruction index to point to next instruction
                        self.call_stack[frame_idx].instruction_idx += 1;
                        
                        // Create callee frame
                        let mut callee_frame = CallFrame {
                            function_name: resolved_callee.clone(),
                            current_block: callee.entry,
                            instruction_idx: 0,
                            locals: HashMap::new(),
                            ssa_values: HashMap::new(),
                            return_addr: Some((current_func_name, current_block_id, node.id)),
                            fused_groups: build_fused_groups_cache(&callee, &self.functions.values().cloned().collect::<Vec<_>>()),
                            fused_skipped_ids: std::collections::HashSet::new(),
                        };
                        
                        let is_constructor = resolved_callee.ends_with("_new");
                        if is_constructor {
                            let model_name = &resolved_callee[..resolved_callee.len() - 4];
                            callee_frame.locals.insert("self".to_string(), Value::Model {
                                name: model_name.to_string(),
                                fields: std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
                            });
                        }
                        
                        // Bind callee params
                        for (param, arg) in callee.params.iter().zip(call_args.into_iter()) {
                            callee_frame.locals.insert(param.name.clone(), arg.clone());
                            callee_frame.ssa_values.insert(param.id, arg);
                        }
                        
                        self.call_stack.push(callee_frame);
                    } else {
                        // Execute as built-in
                        match self.execute(&resolved_callee, call_args) {
                            Ok(res) => {
                                self.call_stack[frame_idx].ssa_values.insert(node.id, res);
                                self.call_stack[frame_idx].instruction_idx += 1;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                } else {
                    let ssa_values = self.call_stack[frame_idx].ssa_values.clone();
                    let result = self.exec_node(&node, &ssa_values)?;
                    self.call_stack[frame_idx].ssa_values.insert(node.id, result);
                    self.call_stack[frame_idx].instruction_idx += 1;
                }
            } else {
                let next_state = match &block.terminator {
                    Terminator::Jump(target) => {
                        self.call_stack[frame_idx].current_block = *target;
                        self.call_stack[frame_idx].instruction_idx = 0;
                        self.call_stack[frame_idx].fused_skipped_ids.clear();
                        true
                    }
                    Terminator::Branch { cond, true_block, false_block } => {
                        let cond_val = self.call_stack[frame_idx].ssa_values.get(cond)
                            .or_else(|| self.globals.get(&cond.to_string()))
                            .cloned()
                            .unwrap_or(Value::Bool(false));
                        
                        let target = if cond_val.as_bool() { *true_block } else { *false_block };
                        self.call_stack[frame_idx].current_block = target;
                        self.call_stack[frame_idx].instruction_idx = 0;
                        self.call_stack[frame_idx].fused_skipped_ids.clear();
                        true
                    }
                    Terminator::Return(val_id) => {
                        let is_constructor = current_func_name.ends_with("_new");
                        let ret_val = if is_constructor {
                            self.call_stack[frame_idx].locals.get("self").cloned().unwrap_or(Value::Void)
                        } else if let Some(vid) = val_id {
                            self.call_stack[frame_idx].ssa_values.get(vid).cloned().unwrap_or(Value::Void)
                        } else {
                            Value::Void
                        };
                        
                        let finished_frame = self.call_stack.pop()
                            .ok_or_else(|| "Internal error: call stack unexpectedly empty".to_string())?;
                        
                        if let Some((_caller_name, _caller_block, dest_val_id)) = finished_frame.return_addr {
                            let caller_idx = self.call_stack.len() - 1;
                            self.call_stack[caller_idx].ssa_values.insert(dest_val_id, ret_val);
                            true
                        } else {
                            final_result = ret_val;
                            false
                        }
                    }
                };
                if !next_state {
                    break;
                }
            }
        }
        
        Ok(final_result)
    }

    fn exec_node(&mut self, node: &IRNode, values: &HashMap<ValueId, Value>) -> Result<Value, String> {
        let get = |id: &ValueId| -> Value {
            values.get(id).cloned()
                .or_else(|| self.globals.get(&id.to_string()).cloned())
                .unwrap_or(Value::Void)
        };
        let get_stripped = |id: &ValueId| -> (Value, Vec<ValueWrapper>) {
            get(id).strip_wrappers()
        };

        // Device Check & Fallback Tracking
        let target_device = match node.device {
            DeviceTarget::CPU => crate::device::Device::CPU,
            DeviceTarget::CUDA(id) => crate::device::Device::CUDA(id),
            DeviceTarget::Auto => crate::device::Device::auto(),
        };
        if let crate::device::Device::CUDA(id) = target_device {
            if !crate::device::Device::cuda_available() {
                self.effect_log.push(format!("fallback_to_cpu_cuda_{}", id));
            } else {
                self.effect_log.push(format!("cuda_exec_{}", id));
            }
        }

        match &node.op {
            IROp::Const(c) => Ok(match c {
                IRConst::Int(v) => Value::Int(*v),
                IRConst::Float(v) => Value::Float(*v),
                IRConst::Bool(v) => Value::Bool(*v),
                IRConst::String(s) => Value::Str(s.clone()),
                IRConst::Tensor(data, shape) => {
                    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                    Value::Tensor(Tensor::new(data.clone(), shape_usize))
                }
            }),

            IROp::Zeros(shape) => {
                let s = if !node.inputs.is_empty() {
                    validate_shape(&node.inputs, |id| get(&id))?
                } else {
                    validate_static_shape(shape)?
                };
                let mut t = Tensor::zeros(&s);
                t.id = self.tape.alloc_id();
                Ok(Value::Tensor(t))
            }
            IROp::Ones(shape) => {
                let s = if !node.inputs.is_empty() {
                    validate_shape(&node.inputs, |id| get(&id))?
                } else {
                    validate_static_shape(shape)?
                };
                let mut t = Tensor::ones(&s);
                t.id = self.tape.alloc_id();
                Ok(Value::Tensor(t))
            }
            IROp::Glorot(shape) => {
                let s = if !node.inputs.is_empty() {
                    validate_shape(&node.inputs, |id| get(&id))?
                } else {
                    validate_static_shape(shape)?
                };
                let mut t = Tensor::glorot(&s).with_grad();
                t.id = self.tape.alloc_id();
                Ok(Value::Tensor(t))
            }
            IROp::Randn(shape) => {
                let s = if !node.inputs.is_empty() {
                    validate_shape(&node.inputs, |id| get(&id))?
                } else {
                    validate_static_shape(shape)?
                };
                let mut t = Tensor::randn(&s);
                t.id = self.tape.alloc_id();
                Ok(Value::Tensor(t))
            }

            IROp::Add => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        let r = self.tape.add(ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Int(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y as f64);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.add(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Float(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.add(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Int(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x as f64);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.add(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Float(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.add(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
                    (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
                    (Value::List(x), Value::List(y)) => {
                        let mut res = x.clone();
                        res.extend(y.clone());
                        Ok(Value::List(res))
                    }
                    (Value::Str(x), Value::Str(y)) => {
                        Ok(Value::Str(format!("{}{}", x, y)))
                    }
                    _ => Ok(Value::Float(a.as_float() + b.as_float())),
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }
            IROp::Sub => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        let r = self.tape.sub(ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Int(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y as f64);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.sub(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Float(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.sub(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Int(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x as f64);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.sub(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Float(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.sub(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    _ => Ok(Value::Float(a.as_float() - b.as_float())),
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }
            IROp::Mul => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        let r = self.tape.mul(ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Int(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y as f64);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.mul(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Float(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.mul(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Int(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x as f64);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.mul(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Float(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.mul(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    _ => Ok(Value::Float(a.as_float() * b.as_float())),
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }
            IROp::Div => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        let r = self.tape.div(ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Int(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y as f64);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.div(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Tensor(ta), Value::Float(y)) => {
                        let mut tb = Tensor::full(&ta.shape, *y);
                        tb.id = self.tape.alloc_id();
                        let r = self.tape.div(ta, &tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Int(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x as f64);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.div(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    (Value::Float(x), Value::Tensor(tb)) => {
                        let mut ta = Tensor::full(&tb.shape, *x);
                        ta.id = self.tape.alloc_id();
                        let r = self.tape.div(&ta, tb);
                        Ok(Value::Tensor(r))
                    }
                    _ => Ok(Value::Float(a.as_float() / b.as_float())),
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }
            IROp::Mod => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => {
                        if *y == 0 {
                            return Err("Runtime Error: Division by zero in modulo operation".to_string());
                        }
                        Ok(Value::Int(x % y))
                    }
                    _ => {
                        let y_val = b.as_float();
                        if y_val == 0.0 {
                            return Err("Runtime Error: Division by zero in modulo operation".to_string());
                        }
                        Ok(Value::Float(a.as_float() % y_val))
                    }
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }
            IROp::Neg => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = match &a {
                    Value::Tensor(t) => Ok(Value::Tensor(self.tape.neg(t))),
                    _ => Ok(Value::Float(-a.as_float())),
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::MatMul => {
                let (a, a_wraps) = get_stripped(&node.inputs[0]);
                let (b, b_wraps) = get_stripped(&node.inputs[1]);
                let res = match (&a, &b) {
                    (Value::Tensor(ta), Value::Tensor(tb)) => {
                        if ta.ndim() < 2 || tb.ndim() < 2 {
                            return Err(format!("Runtime Error: MatMul requires 2D tensors, got {}D and {}D", ta.ndim(), tb.ndim()));
                        }
                        let lhs_cols = ta.shape[ta.shape.len() - 1];
                        let rhs_rows = tb.shape[tb.shape.len() - 2];
                        if lhs_cols != rhs_rows {
                            return Err(format!("Runtime Error: Shape mismatch in MatMul: cannot multiply shape {:?} by {:?}", ta.shape, tb.shape));
                        }
                        let target_device = match node.device {
                            DeviceTarget::CPU => crate::device::Device::CPU,
                            DeviceTarget::CUDA(id) => crate::device::Device::CUDA(id),
                            DeviceTarget::Auto => crate::device::Device::auto(),
                        };
                        let is_cuda = match target_device {
                            crate::device::Device::CUDA(_) => crate::device::Device::cuda_available(),
                            _ => false,
                        };
                        if is_cuda {
                            let is_differentiable = if let Some(frame) = self.call_stack.last() {
                                if let Some(func) = self.functions.get(&frame.function_name) {
                                    func.is_differentiable
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            if !is_differentiable {
                                self.execute_cuda_matmul(node, ta, tb)
                            } else {
                                let mut r = self.tape.matmul(ta, tb);
                                let gpu_val = self.execute_cuda_matmul(node, ta, tb)?;
                                if let Value::Tensor(gpu_tensor) = gpu_val {
                                    r.data = gpu_tensor.data;
                                }
                                Ok(Value::Tensor(r))
                            }
                        } else {
                            let r = self.tape.matmul(ta, tb);
                            Ok(Value::Tensor(r))
                        }
                    }
                    _ => Err("MatMul requires tensor operands".into()),
                };
                res.map(|v| v.apply_wrappers(Value::combine_wrappers(a_wraps, b_wraps)))
            }

            IROp::ReLU => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.relu(t)))
                } else { Ok(Value::Float(a.as_float().max(0.0))) };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::GeLU => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.gelu(t)))
                } else { Ok(a) };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Sigmoid => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.sigmoid(t)))
                } else { Ok(Value::Float(1.0 / (1.0 + (-a.as_float()).exp()))) };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Tanh => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.tanh(t)))
                } else { Ok(Value::Float(a.as_float().tanh())) };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Softmax { dim: _ } => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.softmax(t)))
                } else { Ok(a) };
                res.map(|v| v.apply_wrappers(wraps))
            }

            IROp::CrossEntropy => {
                let (pred, pred_wraps) = get_stripped(&node.inputs[0]);
                let (target, target_wraps) = get_stripped(&node.inputs[1]);
                let wraps = Value::combine_wrappers(pred_wraps, target_wraps);
                let res = if let (Value::Tensor(p), Value::Tensor(t)) = (&pred, &target) {
                    if p.shape != t.shape {
                        return Err(format!("Runtime Error: CrossEntropy shape mismatch: pred {:?} vs target {:?}", p.shape, t.shape));
                    }
                    Ok(Value::Tensor(self.tape.cross_entropy(p, t)))
                } else { Ok(Value::Float(0.0)) };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::MSELoss => {
                let (pred, pred_wraps) = get_stripped(&node.inputs[0]);
                let (target, target_wraps) = get_stripped(&node.inputs[1]);
                let wraps = Value::combine_wrappers(pred_wraps, target_wraps);
                let res = if let (Value::Tensor(p), Value::Tensor(t)) = (&pred, &target) {
                    if p.shape != t.shape {
                        return Err(format!("Runtime Error: MSELoss shape mismatch: pred {:?} vs target {:?}", p.shape, t.shape));
                    }
                    Ok(Value::Tensor(self.tape.mse(p, t)))
                } else { Ok(Value::Float(0.0)) };
                res.map(|v| v.apply_wrappers(wraps))
            }

            IROp::Grad { wrt } => {
                // Run backward on the expression, return gradient
                if let Some(input_id) = node.inputs.first() {
                    let (val, _wraps) = get_stripped(input_id);
                    if let Value::Tensor(t) = &val {
                        let parameter_ids = self.collect_parameter_ids();
                        self.tape.parameter_ids = parameter_ids;
                        self.tape.backward(t.id);
                        // Find the gradient for the requested parameter
                        if let Some(ref param_name) = wrt {
                            if let Some(frame) = self.call_stack.last() {
                                if let Some(Value::Tensor(param)) = frame.locals.get(param_name) {
                                    if let Some(grad_data) = self.tape.get_grad(param.id) {
                                        return Ok(Value::Tensor(Tensor::new(grad_data.clone(), param.shape.clone())));
                                    }
                                }
                            }
                        }
                        // Return gradient of loss itself
                        if let Some(grad_data) = self.tape.get_grad(t.id) {
                            return Ok(Value::Tensor(Tensor::new(grad_data.clone(), t.shape.clone())));
                        }
                    }
                }
                Ok(Value::Tensor(Tensor::zeros(&[1])))
            }
            IROp::Backward => {
                if let Some(input_id) = node.inputs.first() {
                    let (val, _wraps) = get_stripped(input_id);
                    if let Value::Tensor(t) = &val {
                        let parameter_ids = self.collect_parameter_ids();
                        self.tape.parameter_ids = parameter_ids;
                        self.tape.backward(t.id);
                    }
                }
                Ok(Value::Void)
            }

            IROp::Adam { target, lr, beta1: _, beta2: _ } => {
                self.apply_optimizer(target, *lr, "adam")
            }
            IROp::SGD { target, lr, momentum: _ } => {
                self.apply_optimizer(target, *lr, "sgd")
            }
            IROp::AdamW { target, lr, weight_decay: _ } => {
                self.apply_optimizer(target, *lr, "adamw")
            }

            IROp::Call { function } => {
                let args: Vec<Value> = node.inputs.iter().map(|id| get(id)).collect();
                self.execute(function, args)
            }
            IROp::Return => {
                if let Some(id) = node.inputs.first() {
                    Ok(get(id))
                } else { Ok(Value::Void) }
            }

            IROp::Load { name } => {
                if !node.inputs.is_empty() {
                    let obj = get(&node.inputs[0]);
                    match obj {
                        Value::Model { fields, .. } => {
                            Ok(fields.borrow().get(name).cloned().unwrap_or(Value::None))
                        }
                        Value::Uncertain { value, std, confidence } => {
                            match name.as_str() {
                                "value" => Ok(Value::Float(value)),
                                "std" => Ok(Value::Float(std)),
                                "confidence" => Ok(Value::Float(confidence)),
                                _ => Err(format!("Uncertain value has no field '{}'", name)),
                            }
                        }
                        _ => Err(format!("Cannot access field '{}' on {:?}", name, obj)),
                    }
                } else {
                    if let Some(frame) = self.call_stack.last() {
                        if let Some(val) = frame.locals.get(name) {
                            return Ok(val.clone());
                        }
                    }
                    Ok(self.globals.get(name).cloned().unwrap_or(Value::None))
                }
            }
            IROp::Store { name } => {
                if let Some(id) = node.inputs.first() {
                    let val = get(id);
                    if name.starts_with("self.") {
                        let field_name = &name[5..];
                        if let Some(frame) = self.call_stack.last_mut() {
                            if let Some(Value::Model { fields, .. }) = frame.locals.get_mut("self") {
                                fields.borrow_mut().insert(field_name.to_string(), val.clone());
                            }
                        }
                    } else {
                        if let Some(frame) = self.call_stack.last_mut() {
                            frame.locals.insert(name.clone(), val.clone());
                        }
                        self.globals.insert(name.clone(), val);
                    }
                }
                Ok(Value::Void)
            }

            IROp::UncertainWrap => {
                let value = get(&node.inputs[0]).as_float();
                let std = if node.inputs.len() > 1 { get(&node.inputs[1]).as_float() } else { 0.1 };
                let confidence = if std > 0.0 { (1.0 - (std / (value.abs() + 1e-8)).min(1.0)).max(0.0) } else { 1.0 };
                Ok(Value::Uncertain { value, std, confidence })
            }
            IROp::UncertainValue => {
                if let Value::Uncertain { value, .. } = get(&node.inputs[0]) {
                    Ok(Value::Float(value))
                } else { Ok(get(&node.inputs[0])) }
            }
            IROp::UncertainConfidence => {
                if let Value::Uncertain { confidence, .. } = get(&node.inputs[0]) {
                    Ok(Value::Float(confidence))
                } else { Ok(Value::Float(1.0)) }
            }

            IROp::TemporalBefore { .. } => {
                let val = get(&node.inputs[0]);
                if let Value::Temporal { data, direction } = &val {
                    Ok(Value::Temporal { data: data.clone(), direction: direction.clone() })
                } else { Ok(val) }
            }
            IROp::TemporalSnapshot { .. } => {
                let val = get(&node.inputs[0]);
                if let Value::Temporal { data, .. } = &val {
                    Ok(*data.clone())
                } else { Ok(val) }
            }
            IROp::TemporalAfter { .. } => {
                let val = get(&node.inputs[0]);
                if let Value::Temporal { data, direction } = &val {
                    let new_dir = if direction == "past_to_future" { "future_to_past" } else { "past_to_future" };
                    Ok(Value::Temporal { data: data.clone(), direction: new_dir.into() })
                } else { Ok(val) }
            }
            IROp::TemporalCheckDir { expected } => {
                let val = get(&node.inputs[0]);
                if let Value::Temporal { direction, .. } = &val {
                    if direction != expected && self.strict_temporal {
                        return Err(format!(
                            "RUNTIME PANIC: temporal direction violation — expected {} but got {} — lookahead bias detected",
                            expected, direction
                        ));
                    }
                }
                Ok(Value::Void)
            }

            IROp::Observe => {
                let val = get(&node.inputs[0]);
                Ok(Value::Causal { data: Box::new(val), mode: "observed".into() })
            }
            IROp::Intervene => {
                let val = get(&node.inputs[0]);
                Ok(Value::Causal { data: Box::new(val), mode: "intervened".into() })
            }
            IROp::CausalCheckMode { expected } => {
                let val = get(&node.inputs[0]);
                if let Value::Causal { mode, .. } = &val {
                    if mode != expected && self.strict_causal {
                        return Err(format!(
                            "RUNTIME PANIC: causal type mismatch — cannot use {} data where {} is expected",
                            mode, expected
                        ));
                    }
                }
                Ok(Value::Void)
            }

            IROp::Explain => {
                let val = get(&node.inputs[0]);
                Ok(Value::Tuple(vec![
                    val,
                    Value::Str("explanation: gradient attribution".into()),
                ]))
            }

            IROp::MergeModels { strategy } => {
                Ok(Value::Str(format!("merged with strategy: {}", strategy)))
            }
            IROp::ForgetTask { method, strength } => {
                let mut model = get(&node.inputs[0]);
                let task_data = if node.inputs.len() > 1 {
                    get(&node.inputs[1])
                } else {
                    Value::List(vec![])
                };
                let cert = crate::forget::forget_task(self, &mut model, &task_data, method, *strength)?;
                Ok(cert)
            }

            IROp::MemoryStore => {
                self.effect_log.push("memory_store".into());
                Ok(Value::Void)
            }
            IROp::MemoryRecall { k: _ } => {
                Ok(Value::List(vec![]))
            }
            IROp::Search { strategy, max_iter } => {
                Ok(Value::Str(format!("search: strategy={}, max_iter={}", strategy, max_iter)))
            }

            IROp::Print => {
                let val = get(&node.inputs[0]);
                let disp = val.display();
                println!("{}", disp);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                self.stdout_log.push(disp);
                self.effect_log.push("io".into());
                Ok(Value::Void)
            }

            IROp::PrintInline => {
                let val = get(&node.inputs[0]);
                let disp = val.display();
                print!("{}", disp);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                self.stdout_log.push(disp);
                self.effect_log.push("io".into());
                Ok(Value::Void)
            }

            IROp::Input => {
                use std::io::{self, Write};
                print!("> ");
                io::stdout().flush().unwrap();
                let mut input_str = String::new();
                io::stdin().read_line(&mut input_str).unwrap();
                self.effect_log.push("io".into());
                Ok(Value::Str(input_str.trim().to_string()))
            }

            IROp::EmbedString => {
                if let Some(first) = node.inputs.first() {
                    let val = get(first);
                    if let Value::Str(s) = val {
                        let mut data = vec![0.0; 8];
                        for (i, c) in s.chars().enumerate() {
                            let idx = i % 8;
                            data[idx] += (c as u32 as f64).sin();
                        }
                        for v in &mut data {
                            *v = v.tanh();
                        }
                        return Ok(Value::Tensor(Tensor::new(data, vec![1, 8])));
                    }
                }
                Ok(Value::Tensor(Tensor::zeros(&[1, 8])))
            }

            IROp::GenerateReply => {
                if let Some(first) = node.inputs.first() {
                    let val = get(first);
                    if let Value::Str(s) = val {
                        let model = crate::neuron_lm::NeuronLM::new();
                        let reply = model.generate_reply(&s);
                        return Ok(Value::Str(reply));
                    }
                }
                Ok(Value::Str("[AGI Response]: I'm listening! What's on your mind?".to_string()))
            }

            IROp::EffectCheck { expected: _ } => Ok(Value::Void),

            IROp::Concat { dim: _ } => {
                // Concatenate tensors — strip wrappers from the list and its items
                if let Some(first) = node.inputs.first() {
                    let (val, wraps) = get_stripped(first);
                    if let Value::List(items) = &val {
                        // Strip wrappers from individual list items too
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
                                let total_elements = b.checked_mul(d_total)
                                    .ok_or_else(|| "Runtime Error: Concat shape size overflow".to_string())?;
                                if total_elements > 10_000_000 {
                                    return Err(format!("Runtime Error: Tensor size too large in Concat: {} elements", total_elements));
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
                                return Ok(Value::Tensor(Tensor::new(data, vec![b, d_total])).apply_wrappers(wraps));
                            } else {
                                let total_len: usize = tensors.iter().map(|t| t.numel()).sum();
                                if total_len > 10_000_000 {
                                    return Err(format!("Runtime Error: Tensor size too large in Concat: {} elements", total_len));
                                }
                                let mut data = Vec::with_capacity(total_len);
                                for t in &tensors { data.extend_from_slice(&t.data); }
                                return Ok(Value::Tensor(Tensor::new(data, vec![total_len])).apply_wrappers(wraps));
                            }
                        }
                    }
                }
                Ok(Value::Tensor(Tensor::zeros(&[0])))
            }

            IROp::Transpose(dim0, dim1) => {
                if let Some(first) = node.inputs.first() {
                    let (val, wraps) = get_stripped(first);
                    if let Value::Tensor(t) = &val {
                        let ndim = t.ndim();
                        if *dim0 >= ndim || *dim1 >= ndim {
                            return Err(format!("Runtime Error: Transpose dimensions ({}, {}) out of bounds for tensor of rank {}", dim0, dim1, ndim));
                        }
                        return Ok(Value::Tensor(t.transpose(*dim0, *dim1)).apply_wrappers(wraps));
                    }
                }
                Ok(Value::Void)
            }

            IROp::Nop => {
                let items: Vec<Value> = node.inputs.iter().map(|id| get(id)).collect();
                match &node.output_type {
                    IRType::List(_) => Ok(Value::List(items)),
                    IRType::Tuple(_) => Ok(Value::Tuple(items)),
                    _ => {
                        if items.len() == 1 {
                            Ok(items.into_iter().next().unwrap())
                        } else {
                            Ok(Value::List(items))
                        }
                    }
                }
            }

            IROp::Lt => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_float() < b.as_float()))
            }
            IROp::Lte => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_float() <= b.as_float()))
            }
            IROp::Gt => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_float() > b.as_float()))
            }
            IROp::Gte => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_float() >= b.as_float()))
            }
            IROp::Eq => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => x == y,
                    (Value::Bool(x), Value::Bool(y)) => x == y,
                    (Value::Str(x), Value::Str(y)) => x == y,
                    _ => a.as_float() == b.as_float(),
                }))
            }
            IROp::Neq => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => x != y,
                    (Value::Bool(x), Value::Bool(y)) => x != y,
                    (Value::Str(x), Value::Str(y)) => x != y,
                    _ => a.as_float() != b.as_float(),
                }))
            }
            IROp::And => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_bool() && b.as_bool()))
            }
            IROp::Or => {
                let a = get(&node.inputs[0]);
                let b = get(&node.inputs[1]);
                Ok(Value::Bool(a.as_bool() || b.as_bool()))
            }
            IROp::Not => {
                let a = get(&node.inputs[0]);
                Ok(Value::Bool(!a.as_bool()))
            }
            IROp::ListLen => {
                let a = get(&node.inputs[0]);
                Ok(Value::Int(if let Value::List(items) = a {
                    items.len() as i64
                } else {
                    0
                }))
            }
            IROp::Index => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let idx = get(&node.inputs[1]);
                let idx_val = idx.as_int();
                if idx_val < 0 {
                    return Err(format!("Runtime Error: Negative index {} is not allowed", idx_val));
                }
                let i = idx_val as usize;
                let res = match a {
                    Value::List(items) => {
                        if i < items.len() {
                            Ok(items[i].clone())
                        } else {
                            Err(format!("Index {} out of bounds for list of length {}", i, items.len()))
                        }
                    }
                    Value::Tensor(t) => {
                        if t.ndim() == 2 {
                            let cols = t.shape[1];
                            let start = i * cols;
                            let end = start + cols;
                            if end <= t.data.len() {
                                let row_data = t.data[start..end].to_vec();
                                Ok(Value::Tensor(Tensor::new(row_data, vec![1, cols])))
                            } else {
                                Err(format!("Index {} out of bounds for tensor with {} rows", i, t.shape[0]))
                            }
                        } else if t.ndim() == 1 {
                            if i < t.data.len() {
                                Ok(Value::Float(t.data[i]))
                            } else {
                                Err(format!("Index {} out of bounds for tensor of length {}", i, t.data.len()))
                            }
                        } else {
                            Err(format!("Indexing not supported for {}-dimensional tensors", t.ndim()))
                        }
                    }
                    other => Err(format!("Cannot index into {:?}", other)),
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::StopGrad => {
                let a = get(&node.inputs[0]);
                Ok(match a {
                    Value::Tensor(mut t) => {
                        t.requires_grad = false;
                        t.tape_entry = None;
                        self.tape.detach(t.id);
                        Value::Tensor(t)
                    }
                    other => other,
                })
            }
            IROp::Sum { dim } => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    if let Some(d) = dim {
                        if *d < 0 || (*d as usize) >= t.ndim() {
                            return Err(format!("Runtime Error: Sum dimension {} out of bounds for tensor of rank {}", d, t.ndim()));
                        }
                    }
                    let dim_usize = dim.map(|d| d as usize);
                    Ok(Value::Tensor(self.tape.sum(t, dim_usize)))
                } else {
                    Ok(a)
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Mean { dim } => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    if let Some(d) = dim {
                        if *d < 0 || (*d as usize) >= t.ndim() {
                            return Err(format!("Runtime Error: Mean dimension {} out of bounds for tensor of rank {}", d, t.ndim()));
                        }
                    }
                    let dim_usize = dim.map(|d| d as usize);
                    Ok(Value::Tensor(self.tape.mean(t, dim_usize)))
                } else {
                    Ok(a)
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Sqrt => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    Ok(Value::Tensor(self.tape.sqrt(t)))
                } else {
                    match a {
                        Value::Float(f) => Ok(Value::Float(f.sqrt())),
                        _ => Ok(a),
                    }
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::Reshape(new_shape) => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let res = if let Value::Tensor(t) = &a {
                    // Validate new shape doesn't have negative numbers or zero-element overflow
                    let mut new_n: usize = 1;
                    let mut shape_usize = Vec::new();
                    for &x in new_shape {
                        if x <= 0 {
                            return Err(format!("Runtime Error: Reshape dimension must be positive, got {}", x));
                        }
                        let x_usize = x as usize;
                        if let Some(prod) = new_n.checked_mul(x_usize) {
                            new_n = prod;
                        } else {
                            return Err("Runtime Error: Reshape dimension product overflows usize".to_string());
                        }
                        shape_usize.push(x_usize);
                    }
                    if t.numel() != new_n {
                        return Err(format!("Runtime Error: Cannot reshape tensor of size {} to shape {:?}", t.numel(), shape_usize));
                    }
                    Ok(Value::Tensor(t.reshape(&shape_usize)))
                } else {
                    Ok(a)
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::UpdateRow => {
                let (a, wraps) = get_stripped(&node.inputs[0]);
                let (idx, _) = get_stripped(&node.inputs[1]);
                let (row, _) = get_stripped(&node.inputs[2]);
                let res = if let (Value::Tensor(t), Value::Tensor(r)) = (&a, &row) {
                    let idx_val = idx.as_int();
                    if idx_val < 0 {
                        return Err(format!("Runtime Error: Negative index {} is not allowed in UpdateRow", idx_val));
                    }
                    let i = idx_val as usize;
                    if t.shape.len() >= 2 && r.numel() != t.shape[1] {
                        return Err(format!("Runtime Error: Row width mismatch in UpdateRow: expected {}, got {}", t.shape[1], r.numel()));
                    }
                    let mut new_data = t.data.clone();
                    let row_len = r.numel();
                    let start = i * row_len;
                    if start + row_len <= new_data.len() {
                        new_data[start..start + row_len].copy_from_slice(&r.data[..row_len]);
                        Ok(Value::Tensor(Tensor::new(new_data, t.shape.clone())))
                    } else {
                        Err(format!("Runtime Error: Index {} out of bounds for UpdateRow on tensor of size {}", i, t.shape[0]))
                    }
                } else {
                    Ok(a)
                };
                res.map(|v| v.apply_wrappers(wraps))
            }
            IROp::PythonCall { ref module, ref function } => {
                // ── Python Interop Bridge ──
                // Shells out to the system Python interpreter, passing input tensors
                // as JSON and parsing the result back into NEURON values.
                use std::process::Command;

                // Collect input values as JSON arrays
                let mut py_inputs = Vec::new();
                for input_id in &node.inputs {
                    let (val, _) = get_stripped(input_id);
                    match &val {
                        Value::Tensor(t) => {
                            let data_str: Vec<String> = t.data.iter().map(|v| format!("{}", v)).collect();
                            py_inputs.push(format!(
                                "{{\"type\":\"tensor\",\"data\":[{}],\"shape\":[{}]}}",
                                data_str.join(","),
                                t.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
                            ));
                        }
                        Value::Int(i) => py_inputs.push(format!("{{\"type\":\"int\",\"value\":{}}}", i)),
                        Value::Float(f) => py_inputs.push(format!("{{\"type\":\"float\",\"value\":{}}}", f)),
                        Value::Bool(b) => py_inputs.push(format!("{{\"type\":\"bool\",\"value\":{}}}", b)),
                        Value::Str(s) => py_inputs.push(format!("{{\"type\":\"str\",\"value\":\"{}\"}}", s)),
                        _ => py_inputs.push(format!("{{\"type\":\"none\"}}")),
                    }
                }

                // Generate Python script
                let py_script = format!(
r#"import json, sys
import numpy as np
try:
    import {module}
except ImportError:
    print(json.dumps({{"error": "Module '{module}' not found. Install with: pip install {module}"}}))
    sys.exit(0)

# Parse inputs
inputs_json = '{inputs_json}'
inputs = json.loads(inputs_json)

args = []
for inp in inputs:
    if inp["type"] == "tensor":
        args.append(np.array(inp["data"], dtype=np.float64).reshape(inp["shape"]))
    elif inp["type"] == "int":
        args.append(inp["value"])
    elif inp["type"] == "float":
        args.append(inp["value"])
    elif inp["type"] == "bool":
        args.append(inp["value"])
    elif inp["type"] == "str":
        args.append(inp["value"])
    else:
        args.append(None)

# Call the function
fn = {module}.{function}
try:
    result = fn(*args)
except TypeError:
    # Some numpy functions expect a single tuple arg (e.g., np.zeros((3,3)) not np.zeros(3,3))
    # Try packing integer args as a tuple
    int_args = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
    if len(int_args) == len(args) and len(args) > 1:
        result = fn(tuple(int(a) for a in args))
    else:
        raise

# Serialize result
if isinstance(result, np.ndarray):
    print(json.dumps({{"type":"tensor","data":result.flatten().tolist(),"shape":list(result.shape)}}))
elif isinstance(result, (int, bool)):
    print(json.dumps({{"type":"int","value":int(result)}}))
elif isinstance(result, float):
    print(json.dumps({{"type":"float","value":result}}))
elif isinstance(result, str):
    print(json.dumps({{"type":"str","value":result}}))
elif hasattr(result, 'numpy'):
    # PyTorch tensor
    arr = result.detach().cpu().numpy()
    print(json.dumps({{"type":"tensor","data":arr.flatten().tolist(),"shape":list(arr.shape)}}))
else:
    print(json.dumps({{"type":"str","value":str(result)}}))
"#,
                    module = module,
                    function = function,
                    inputs_json = format!("[{}]", py_inputs.join(",")),
                );

                // Execute Python
                let output = Command::new("python")
                    .arg("-c")
                    .arg(&py_script)
                    .output();

                match output {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);

                        if !stderr.is_empty() {
                            eprintln!("[NEURON Python Bridge] stderr: {}", stderr.trim());
                        }

                        // Parse JSON result
                        let trimmed = stdout.trim();
                        if trimmed.is_empty() {
                            Ok(Value::None)
                        } else if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                            if let Some(err) = parsed.get("error") {
                                Err(format!("Python Error: {}", err.as_str().unwrap_or("unknown")))
                            } else {
                                let typ = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("none");
                                match typ {
                                    "tensor" => {
                                        let data: Vec<f64> = parsed.get("data")
                                            .and_then(|d| d.as_array())
                                            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                                            .unwrap_or_default();
                                        let shape: Vec<usize> = parsed.get("shape")
                                            .and_then(|s| s.as_array())
                                            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                                            .unwrap_or_default();
                                        Ok(Value::Tensor(Tensor::new(data, shape)))
                                    }
                                    "int" => {
                                        let v = parsed.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                                        Ok(Value::Int(v))
                                    }
                                    "float" => {
                                        let v = parsed.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        Ok(Value::Float(v))
                                    }
                                    "str" => {
                                        let v = parsed.get("value").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        Ok(Value::Str(v))
                                    }
                                    "bool" => {
                                        let v = parsed.get("value").and_then(|v| v.as_bool()).unwrap_or(false);
                                        Ok(Value::Bool(v))
                                    }
                                    _ => Ok(Value::None),
                                }
                            }
                        } else {
                            // Raw string output
                            Ok(Value::Str(trimmed.to_string()))
                        }
                    }
                    Err(e) => Err(format!("Failed to execute Python: {}. Is Python installed and in PATH?", e)),
                }
            }
            other => Err(format!("Unhandled IR operation: {:?}", other)),
        }
    }

    /// Run the main/entry function if it exists.
    pub fn run_main(&mut self) -> Result<Value, String> {
        // Look for main, __main__, or train function
        let entry = if self.functions.contains_key("main") { "main" }
            else if self.functions.contains_key("__main__") { "__main__" }
            else if self.functions.contains_key("train") { "train" }
            else if self.functions.contains_key("__global_init__") { "__global_init__" }
            else { return Err("No entry point found (main, train, or __global_init__)".into()); };
        self.execute(entry, vec![])
    }

    pub fn collect_parameter_ids(&self) -> std::collections::HashSet<usize> {
        let mut ids = std::collections::HashSet::new();
        // Traverse globals
        for val in self.globals.values() {
            self.collect_tensor_ids_from_value(val, &mut ids);
        }
        // Traverse call stack locals
        for frame in &self.call_stack {
            for val in frame.locals.values() {
                self.collect_tensor_ids_from_value(val, &mut ids);
            }
        }
        ids
    }

    fn collect_tensor_ids_from_value(&self, val: &Value, ids: &mut std::collections::HashSet<usize>) {
        match val {
            Value::Tensor(t) => {
                ids.insert(t.id);
            }
            Value::Model { fields, .. } => {
                if let Ok(map) = fields.try_borrow() {
                    for field_val in map.values() {
                        self.collect_tensor_ids_from_value(field_val, ids);
                    }
                }
            }
            Value::List(items) | Value::Tuple(items) => {
                for item in items {
                    self.collect_tensor_ids_from_value(item, ids);
                }
            }
            _ => {}
        }
    }

    /// Get a global variable.
    pub fn get_global(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Set a global variable.
    pub fn set_global(&mut self, name: String, val: Value) {
        self.globals.insert(name, val);
    }

    fn update_nested_target(&mut self, val: &mut Value, parts: &[&str], lr: f64, method: &str) {
        if parts.is_empty() {
            self.update_value_tensors(val, lr, method);
            return;
        }
        let part = parts[0];
        match val {
            Value::Model { fields, .. } => {
                let mut map = fields.borrow_mut();
                if let Some(sub_val) = map.get_mut(part) {
                    self.update_nested_target(sub_val, &parts[1..], lr, method);
                }
            }
            _ => {}
        }
    }

    pub fn apply_optimizer(&mut self, target: &str, lr: f64, method: &str) -> Result<Value, String> {
        self.optimizer_step += 1;
        let parts: Vec<&str> = target.split('.').collect();
        if parts.is_empty() { return Ok(Value::Void); }
        
        let root = parts[0];
        
        // Retrieve the root value out of locals or globals mutably
        let mut root_val = None;
        let mut is_local = false;
        if let Some(frame) = self.call_stack.last_mut() {
            if let Some(val) = frame.locals.get(root) {
                root_val = Some(val.clone());
                is_local = true;
            }
        }
        if root_val.is_none() {
            if let Some(val) = self.globals.get(root) {
                root_val = Some(val.clone());
            }
        }
        
        if let Some(mut val) = root_val {
            self.update_nested_target(&mut val, &parts[1..], lr, method);
            
            // Put it back
            if is_local {
                if let Some(frame) = self.call_stack.last_mut() {
                    frame.locals.insert(root.to_string(), val);
                }
            } else {
                self.globals.insert(root.to_string(), val);
            }
        }
        
        Ok(Value::Void)
    }

    fn update_value_tensors(&mut self, val: &mut Value, lr: f64, method: &str) {
        match val {
            Value::Tensor(ref mut t) => {
                self.update_tensor_in_place(t, lr, method);
            }
            Value::Model { fields, .. } => {
                for (_, field_val) in fields.borrow_mut().iter_mut() {
                    self.update_value_tensors(field_val, lr, method);
                }
            }
            Value::List(ref mut items) => {
                for item in items.iter_mut() {
                    self.update_value_tensors(item, lr, method);
                }
            }
            Value::Tuple(ref mut items) => {
                for item in items.iter_mut() {
                    self.update_value_tensors(item, lr, method);
                }
            }
            _ => {}
        }
    }

    fn update_tensor_in_place(&mut self, t: &mut Tensor, lr: f64, method: &str) {
        if let Some(grad) = self.tape.get_grad(t.id) {
            let n = t.numel();
            match method {
                "adam" | "adamw" => {
                    let step = self.optimizer_step;
                    let beta1: f64 = 0.9;
                    let beta2: f64 = 0.999;
                    let eps = 1e-8;
                    
                    let m = self.adam_m.entry(t.id).or_insert_with(|| vec![0.0; n]);
                    let v = self.adam_v.entry(t.id).or_insert_with(|| vec![0.0; n]);
                    
                    // Precompute powi outside the loop
                    let beta1_pow = beta1.powi(step as i32);
                    let beta2_pow = beta2.powi(step as i32);
                    let correction1 = 1.0 - beta1_pow;
                    let correction2 = 1.0 - beta2_pow;

                    // Slice to eliminate bound checks
                    let m_slice = &mut m[..n];
                    let v_slice = &mut v[..n];
                    let grad_slice = &grad[..n];
                    let t_slice = &mut t.data[..n];
                    
                    for i in 0..n {
                        let g = grad_slice[i];
                        m_slice[i] = beta1 * m_slice[i] + (1.0 - beta1) * g;
                        v_slice[i] = beta2 * v_slice[i] + (1.0 - beta2) * g * g;
                        let m_hat = m_slice[i] / correction1;
                        let v_hat = v_slice[i] / correction2;
                        let update_val = lr * m_hat / (v_hat.sqrt() + eps);
                        
                        if method == "adamw" {
                            t_slice[i] -= lr * 0.01 * t_slice[i] + update_val;
                        } else {
                            t_slice[i] -= update_val;
                        }
                    }
                }
                "sgd" => {
                    let grad_slice = &grad[..n];
                    let t_slice = &mut t.data[..n];
                    for i in 0..n {
                        t_slice[i] -= lr * grad_slice[i];
                    }
                }
                _ => {}
            }
        }
    }

    // Helper functions for causal native methods
    

    fn compile_function_kernels(&mut self, func: &IRFunction) {
        if crate::device::is_simulate_cuda() {
            self.compiled_functions.insert(func.name.clone());
            return;
        }
        
        let all_funcs: Vec<IRFunction> = self.functions.values().cloned().collect();
        let kernels = neuron_compiler::cuda_codegen::generate_cuda_kernels(func, &all_funcs);
        
        if let Some(ctx) = crate::device::get_cuda_context() {
            if !kernels.is_empty() && (std::env::var("NEURON_VERBOSE").is_ok() || std::env::var("NEURON_LOG").is_ok()) {
                println!("[NEURON JIT] Compiling {} GPU kernels for function '{}'...", kernels.len(), func.name);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            let cache_mutex = GLOBAL_KERNEL_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
            for kernel in kernels {
                let mut cache = cache_mutex.lock().unwrap();
                if let Some(k_func) = cache.get(&kernel.code) {
                    self.cuda_kernels.insert(kernel.name.clone(), CudaModuleFunction {
                        module: k_func.module,
                        function: k_func.function,
                    });
                    continue;
                }
                
                let start_time = std::time::Instant::now();
                match ctx.compile_to_ptx(&kernel.name, &kernel.code) {
                    Ok(ptx) => {
                        match ctx.load_module_and_get_function(&ptx, &kernel.name) {
                            Ok((module, function)) => {
                                let k_func = CudaModuleFunction { module, function };
                                cache.insert(kernel.code.clone(), CudaModuleFunction { module, function });
                                self.cuda_kernels.insert(kernel.name.clone(), k_func);
                                if std::env::var("NEURON_VERBOSE").is_ok() || std::env::var("NEURON_LOG").is_ok() {
                                    println!("  ✓ Kernel '{}' compiled in {}ms", kernel.name, start_time.elapsed().as_millis());
                                    let _ = std::io::Write::flush(&mut std::io::stdout());
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to load CUDA kernel {}: {}", kernel.name, e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to compile CUDA kernel {}: {}", kernel.name, e);
                    }
                }
            }
        }
        self.compiled_functions.insert(func.name.clone());
    }

    fn execute_fused_group(
        &mut self,
        frame_idx: usize,
        func: &IRFunction,
        g_idx: usize,
        group: &neuron_compiler::cuda_codegen::FusedGroup,
    ) -> Result<(), String> {
        let kernel_name = format!("fused_{}_{}", func.name, g_idx);
        
        for node in &group.instructions {
            self.call_stack[frame_idx].fused_skipped_ids.insert(node.id);
        }
        
        let all_funcs: Vec<IRFunction> = self.functions.values().cloned().collect();
        let kernels = neuron_compiler::cuda_codegen::generate_cuda_kernels(func, &all_funcs);
        let kernel = kernels.iter().find(|k| k.name == kernel_name)
            .ok_or_else(|| format!("Kernel {} metadata not found", kernel_name))?;
            
        let terminal_node = group.instructions.last().ok_or_else(|| "Empty fused group in CUDA execution".to_string())?;
        let output_id = terminal_node.id;
        
        if crate::device::is_simulate_cuda() {
            let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
            
            for node in &group.instructions {
                let res = self.exec_node(node, &local_ssa)?;
                local_ssa.insert(node.id, res.clone());
                self.call_stack[frame_idx].ssa_values.insert(node.id, res);
            }
            
            self.effect_log.push(format!("cuda_exec_{}", g_idx));
            return Ok(());
        }
        
        // Pre-pass for non-simulated mode to ensure constants are populated on CPU
        let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
        for node in &group.instructions {
            if matches!(node.op, IROp::Const(_)) {
                let res = self.exec_node(node, &local_ssa)?;
                local_ssa.insert(node.id, res.clone());
                self.call_stack[frame_idx].ssa_values.insert(node.id, res);
            }
        }
        
        // If the entire group is only Const nodes, all values are already
        // materialized by the pre-pass above. Skip kernel launch to avoid
        // overwriting scalar values with a Tensor output.
        let all_const = group.instructions.iter().all(|n| matches!(n.op, IROp::Const(_)));
        if all_const {
            return Ok(());
        }
        
        if let Some(ctx) = crate::device::get_cuda_context() {
            let k_func = self.cuda_kernels.get(&kernel_name)
                .ok_or_else(|| format!("CUDA kernel {} not loaded", kernel_name))?;
                
            let get_val = |id: usize| -> Value {
                self.call_stack[frame_idx].ssa_values.get(&id).cloned()
                    .or_else(|| self.globals.get(&id.to_string()).cloned())
                    .unwrap_or(Value::Void)
            };
                
            let mut output_shape = vec![1];
            for &input_id in &kernel.inputs {
                if let Value::Tensor(t) = get_val(input_id) {
                    output_shape = t.shape.clone();
                    break;
                }
            }
            let numel = output_shape.iter().product::<usize>();
            
            // Guard: grid=0 is an invalid CUDA launch. Fall back to CPU.
            if numel == 0 {
                // Execute all nodes on CPU instead
                let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
                for node in &group.instructions {
                    let res = self.exec_node(node, &local_ssa)?;
                    local_ssa.insert(node.id, res.clone());
                    self.call_stack[frame_idx].ssa_values.insert(node.id, res);
                }
                self.effect_log.push(format!("fallback_to_cpu_cuda_{}", g_idx));
                return Ok(());
            }
            
            // Allocate output in dedicated VRAM (persistent — stays on GPU)
            let output_tensor = Tensor::new(Buffer::new_vram(numel), output_shape);
            let out_ptr = output_tensor.device_ptr();
            
            // If VRAM allocation failed (ptr=0), fall back to CPU
            if out_ptr == 0 {
                let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
                for node in &group.instructions {
                    let res = self.exec_node(node, &local_ssa)?;
                    local_ssa.insert(node.id, res.clone());
                    self.call_stack[frame_idx].ssa_values.insert(node.id, res);
                }
                self.effect_log.push(format!("fallback_to_cpu_cuda_{}", g_idx));
                return Ok(());
            }
            
            enum CudaArg {
                Ptr(u64),
                Float(f64),
                Int(i32),
            }
            
            let mut arg_values = Vec::new();
            arg_values.push(CudaArg::Ptr(out_ptr));
            
            // IMPORTANT: Collect all cloned input Values FIRST so their VRAM buffers
            // stay alive until after the kernel launch. Previously, each `get_val()` 
            // clone was dropped at the end of the loop iteration, returning the VRAM
            // pointer to the pool. The next iteration's clone could reuse (and zero) 
            // the same pointer, causing all inputs to alias the last cloned buffer.
            let input_vals: Vec<Value> = kernel.inputs.iter().map(|&id| get_val(id)).collect();
            
            let mut any_input_missing = false;
            for (idx, val) in input_vals.iter().enumerate() {
                if kernel.input_is_tensor[idx] {
                    let tensor = match val.as_tensor() {
                        Some(t) => t,
                        None => {
                            any_input_missing = true;
                            break;
                        }
                    };
                    // ensure_device() handles both VRAM (no-op if already there) and UVM (prefetch)
                    tensor.data.ensure_device();
                    let ptr = tensor.device_ptr();
                    if ptr == 0 {
                        any_input_missing = true;
                        break;
                    }
                    arg_values.push(CudaArg::Ptr(ptr));
                } else {
                    arg_values.push(CudaArg::Float(val.as_float()));
                }
            }
            
            if any_input_missing {
                let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
                for node in &group.instructions {
                    let res = self.exec_node(node, &local_ssa)?;
                    local_ssa.insert(node.id, res.clone());
                    self.call_stack[frame_idx].ssa_values.insert(node.id, res);
                }
                self.effect_log.push(format!("fallback_to_cpu_cuda_{}", g_idx));
                return Ok(());
            }
            
            arg_values.push(CudaArg::Int(numel as i32));
            
            let mut kernel_params: Vec<*mut std::ffi::c_void> = Vec::new();
            for arg in &mut arg_values {
                let ptr = match arg {
                    CudaArg::Ptr(p) => p as *mut u64 as *mut std::ffi::c_void,
                    CudaArg::Float(f) => f as *mut f64 as *mut std::ffi::c_void,
                    CudaArg::Int(i) => i as *mut i32 as *mut std::ffi::c_void,
                };
                kernel_params.push(ptr);
            }

            let block_size = 256;
            let grid_size = (numel + block_size - 1) / block_size;
            
            let res = unsafe {
                (ctx.cuda.cuLaunchKernel)(
                    k_func.function,
                    grid_size as u32, 1, 1,
                    block_size as u32, 1, 1,
                    0,
                    std::ptr::null_mut(),
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if res != 0 {
                let err_str = cuda_error_string(ctx, res);
                return Err(format!("cuLaunchKernel failed (code {}): {} [grid={}, block={}, args={}, numel={}]",
                    res, err_str, grid_size, block_size, kernel_params.len(), numel));
            }
            
            // Mark the output tensor's VRAM as current — NO download to host!
            // The data stays in VRAM until the CPU actually needs to read it
            // (via Deref, which transparently calls ensure_host()).
            output_tensor.data.mark_device_dirty();
            
            // If the function is not differentiable, skip the CPU re-execution entirely
            // to avoid all CPU compute and Host buffer allocation overhead.
            if !func.is_differentiable {
                self.call_stack[frame_idx].ssa_values.insert(output_id, Value::Tensor(output_tensor));
                self.effect_log.push(format!("cuda_exec_{}", g_idx));
                return Ok(());
            }

            // Execute nodes on CPU to populate tape and intermediate values for autograd
            let old_force = crate::device::is_force_cpu();
            crate::device::set_force_cpu(true);
            let mut local_ssa = self.call_stack[frame_idx].ssa_values.clone();
            let mut cpu_res = Ok(());
            for node in &group.instructions {
                match self.exec_node(node, &local_ssa) {
                    Ok(res) => {
                        local_ssa.insert(node.id, res.clone());
                        self.call_stack[frame_idx].ssa_values.insert(node.id, res);
                    }
                    Err(e) => {
                        cpu_res = Err(e);
                        break;
                    }
                }
            }
            crate::device::set_force_cpu(old_force);
            cpu_res?;
            
            // Swap the CPU-computed tensor's buffer with the GPU-computed one, preserving the tape ID
            if let Some(Value::Tensor(ref mut cpu_tensor)) = self.call_stack[frame_idx].ssa_values.get_mut(&output_id) {
                cpu_tensor.data = output_tensor.data;
            }
            
            self.effect_log.push(format!("cuda_exec_{}", g_idx));
            
            Ok(())
        } else {
            Err("CUDA context not available".to_string())
        }
    }

    fn execute_cuda_matmul(&mut self, _node: &IRNode, ta: &Tensor, tb: &Tensor) -> Result<Value, String> {
        let (lhs_rows, lhs_cols) = (ta.shape[0], ta.shape[1]);
        let (rhs_rows, rhs_cols) = (tb.shape[0], tb.shape[1]);
        if lhs_cols != rhs_rows {
            return Err(format!("Shape mismatch in MatMul: cannot multiply {}x{} by {}x{}", lhs_rows, lhs_cols, rhs_rows, rhs_cols));
        }

        // Ensure operands are on the device
        ta.data.ensure_device();
        tb.data.ensure_device();

        let numel = lhs_rows * rhs_cols;
        let output_shape = vec![lhs_rows, rhs_cols];

        let output_tensor = Tensor::new(Buffer::new_vram(numel), output_shape);
        let out_ptr = output_tensor.device_ptr();

        if out_ptr == 0 {
            // Fall back to CPU matmul
            let r = self.tape.matmul(ta, tb);
            return Ok(Value::Tensor(r));
        }

        let ctx = crate::device::get_cuda_context()
            .ok_or_else(|| "CUDA context not available".to_string())?;

        let cublas = ctx.cublas.as_ref()
            .ok_or_else(|| "cuBLAS library not loaded. Ensure libcublas is installed.".to_string())?;

        let handle = ctx.cublas_handle;
        if handle.is_null() {
            return Err("cuBLAS handle is null".to_string());
        }

        // zero-overhead transpose trick: row-major C = A x B via column-major B x A
        let m = rhs_cols as i32;
        let n = lhs_rows as i32;
        let k = lhs_cols as i32;

        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;

        let a_ptr = ta.device_ptr() as *const f64;
        let b_ptr = tb.device_ptr() as *const f64;
        let c_ptr = out_ptr as *mut f64;

        let res = unsafe {
            (cublas.cublasDgemm_v2)(
                handle,
                0, // CUBLAS_OP_N
                0, // CUBLAS_OP_N
                m, n, k,
                &alpha,
                b_ptr, m, // lda
                a_ptr, k, // ldb
                &beta,
                c_ptr, m, // ldc
            )
        };

        if res != 0 {
            return Err(format!("cublasDgemm_v2 failed: error code {}", res));
        }

        // No cuCtxSynchronize needed here — cuBLAS calls on the default stream
        // are serialized automatically. The Buffer will sync lazily when host
        // data is actually needed (e.g., in ensure_host / as_slice).

        output_tensor.data.mark_device_dirty();
        Ok(Value::Tensor(output_tensor))
    }
}

impl Drop for VM {
    fn drop(&mut self) {
        // CUDA modules are cached globally in GLOBAL_KERNEL_CACHE and remain loaded
        // for the process duration to avoid reload/recompilation overhead.
    }
}

/// Convert a CUDA driver error code into a human-readable string.
fn cuda_error_string(ctx: &crate::device::CudaContext, code: u32) -> String {
    let mut ptr: *const std::os::raw::c_char = std::ptr::null();
    let res = unsafe { (ctx.cuda.cuGetErrorString)(code, &mut ptr) };
    if res == 0 && !ptr.is_null() {
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        cstr.to_string_lossy().into_owned()
    } else {
        format!("unknown error (code {})", code)
    }
}

fn build_fused_groups_cache(func: &IRFunction, all_funcs: &[IRFunction]) -> HashMap<ValueId, (usize, neuron_compiler::cuda_codegen::FusedGroup)> {
    let mut map = HashMap::new();
    let tensor_ids = neuron_compiler::cuda_codegen::get_tensor_ids(func, all_funcs);
    let groups = neuron_compiler::cuda_codegen::find_fused_groups(func, all_funcs);
    for (g_idx, group) in groups.into_iter().enumerate() {
        if !group.is_empty() {
            let has_tensors = group.instructions.iter().any(|node| tensor_ids.contains(&node.id));
            if has_tensors {
                let first_id = group.instructions[0].id;
                map.insert(first_id, (g_idx, group));
            }
        }
    }
    map
}


fn parse_string_list(val: &Value) -> Result<Vec<String>, String> {
    if let Value::List(l) = val {
        Ok(l.iter().map(|item| item.display()).collect())
    } else {
        Err("Expected a list of strings".into())
    }
}

fn parse_float_list(val: &Value) -> Result<Vec<f64>, String> {
    if let Value::List(l) = val {
        Ok(l.iter().map(|item| item.as_float()).collect())
    } else {
        Err("Expected a list of floats".into())
    }
}

fn parse_float_matrix(val: &Value) -> Result<Vec<Vec<f64>>, String> {
    if let Value::List(l) = val {
        let mut mat = Vec::new();
        for item in l {
            mat.push(parse_float_list(item)?);
        }
        Ok(mat)
    } else {
        Err("Expected a list of list of floats".into())
    }
}

fn parse_evidence_map(val: &Value) -> Result<HashMap<String, f64>, String> {
    let mut map = HashMap::new();
    match val {
        Value::Model { fields, .. } => {
            for (name, f_val) in fields.borrow().iter() {
                map.insert(name.clone(), f_val.as_float());
            }
        }
        Value::List(pairs) => {
            for pair in pairs {
                match pair {
                    Value::Tuple(p) | Value::List(p) if p.len() >= 2 => {
                        let name = p[0].display();
                        let v = p[1].as_float();
                        map.insert(name, v);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    Ok(map)
}

fn convert_causal_results(results: HashMap<String, (f64, f64)>) -> Value {
    let fields = Rc::new(RefCell::new(HashMap::new()));
    for (name, (mean, std)) in results {
        fields.borrow_mut().insert(name, Value::Uncertain {
            value: mean,
            std,
            confidence: 1.0,
        });
    }
    Value::Model {
        name: "CausalInferenceResult".into(),
        fields,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_const() {
        let mut prog = IRProgram::new();
        let mut func = IRFunction::new("main");
        let block = BasicBlock {
            id: 0,
            instructions: vec![
                IRNode {
                    id: 0, op: IROp::Const(IRConst::Int(42)),
                    inputs: vec![], output_type: IRType::I64, output_shape: vec![],
                    grad_fn: None, device: DeviceTarget::Auto, temporal_dir: None, effects: vec![],
                },
                IRNode {
                    id: 1, op: IROp::Print, inputs: vec![0],
                    output_type: IRType::Void, output_shape: vec![],
                    grad_fn: None, device: DeviceTarget::Auto, temporal_dir: None, effects: vec![],
                },
            ],
            terminator: Terminator::Return(Some(0)),
        };
        func.blocks.push(block);
        func.entry = 0;
        prog.functions.push(func);

        let mut vm = VM::new();
        vm.load(&prog);
        let result = vm.run_main().unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_vm_tensor_ops() {
        let mut prog = IRProgram::new();
        let mut func = IRFunction::new("main");
        // zeros(2,3)
        let block = BasicBlock {
            id: 0,
            instructions: vec![
                IRNode {
                    id: 0, op: IROp::Zeros(vec![2, 3]),
                    inputs: vec![], output_type: IRType::Tensor(vec![2, 3]), output_shape: vec![2, 3],
                    grad_fn: None, device: DeviceTarget::Auto, temporal_dir: None, effects: vec![],
                },
            ],
            terminator: Terminator::Return(Some(0)),
        };
        func.blocks.push(block);
        func.entry = 0;
        prog.functions.push(func);

        let mut vm = VM::new();
        vm.load(&prog);
        let result = vm.run_main().unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.shape, vec![2, 3]);
        } else { panic!("Expected tensor"); }
    }
}
