/// NEURON internal type representations and type checker.
///
/// Types are resolved from AST TypeExpr nodes and used for all type checking.
/// The type checker walks the AST, builds scoped symbol tables, and enforces
/// all NEURON type rules: tensor shapes, uncertainty, temporal, causal, effects.

use crate::ast::*;
use crate::token::Span;
use crate::errors::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════
//  Internal type representations
// ═══════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum NType {
    Base(String),                          // Int, Float, Bool, String, Timestamp, Loss, Dataset
    Tensor(Vec<Dim>),                      // Tensor[dims]
    Uncertain(Box<NType>),                 // Uncertain[T]
    Random(Box<NType>),                    // Random[T]
    Prob(Box<NType>),                      // Prob[T]
    Temporal(Box<NType>, TemporalSpec),    // Temporal[T, direction/offset]
    Causal(Box<NType>, String),            // Causal[T, mode]
    Learnable(String, Option<Box<NType>>), // Learnable[FnType]
    Effect(Vec<EffectEntry>),              // Effect[IO, Rand, Mut[x]]
    List(Box<NType>),                      // List[T]
    Option_(Box<NType>),                   // Option[T]
    Tuple(Vec<NType>),                     // (T1, T2, ...)
    Fn_(Vec<NType>, Box<NType>, Option<Box<NType>>), // fn(params) -> ret [effects]
    Model(String, HashMap<String, NType>, HashMap<String, NType>), // Model { fields, methods }
    CausalModel(String, Vec<String>),      // CausalModel { variables }
    // AGI types
    Memory(Box<NType>),
    EpisodicMemory(Box<NType>),
    SemanticMemory(Box<NType>),
    WorkingMemory(Box<NType>, Option<i64>),
    Reward(Box<NType>),
    Agent(String),
    Void,
    Any,
    Explanation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Static(i64),
    Symbolic(String),
    Named(String, String),
    Dynamic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectEntry {
    pub kind: String,
    pub target: Option<String>,
}

impl NType {
    pub fn display(&self) -> String {
        match self {
            NType::Base(n) => n.clone(),
            NType::Tensor(dims) => {
                if dims.is_empty() { "Tensor".into() }
                else {
                    let ds: Vec<String> = dims.iter().map(|d| match d {
                        Dim::Static(v) => v.to_string(),
                        Dim::Symbolic(s) => s.clone(),
                        Dim::Named(a, n) => format!("{}:{}", a, n),
                        Dim::Dynamic => "?".into(),
                    }).collect();
                    format!("Tensor[{}]", ds.join(", "))
                }
            }
            NType::Uncertain(inner) => format!("Uncertain[{}]", inner.display()),
            NType::Random(inner) => format!("Random[{}]", inner.display()),
            NType::Temporal(inner, spec) => format!("Temporal[{}, {}]", inner.display(), spec),
            NType::Causal(inner, mode) => format!("Causal[{}, {}]", inner.display(), mode),
            NType::List(inner) => format!("List[{}]", inner.display()),
            NType::Model(name, _, _) => format!("Model[{}]", name),
            NType::Memory(inner) => format!("Memory[{}]", inner.display()),
            NType::EpisodicMemory(inner) => format!("EpisodicMemory[{}]", inner.display()),
            NType::Reward(inner) => format!("Reward[{}]", inner.display()),
            NType::Agent(name) => format!("Agent[{}]", name),
            NType::Void => "Void".into(),
            NType::Any => "Any".into(),
            _ => format!("{:?}", self),
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, NType::Base(n) if n == "Int" || n == "Float" || n == "Loss")
    }
    pub fn is_tensor(&self) -> bool { matches!(self, NType::Tensor(_)) }
    pub fn is_temporal(&self) -> bool { matches!(self, NType::Temporal(_, _)) }
    pub fn is_causal(&self) -> bool { matches!(self, NType::Causal(_, _)) }
    pub fn is_uncertain(&self) -> bool { matches!(self, NType::Uncertain(_)) }
}

#[derive(Debug, Clone, PartialEq)]
enum TypeWrapper {
    Temporal(TemporalSpec),
    Causal(String),
    Uncertain,
    Random,
}

fn strip_wrappers(mut ty: NType) -> (NType, Vec<TypeWrapper>) {
    let mut wrappers = Vec::new();
    loop {
        match ty {
            NType::Temporal(inner, spec) => {
                wrappers.push(TypeWrapper::Temporal(spec));
                ty = *inner;
            }
            NType::Causal(inner, mode) => {
                wrappers.push(TypeWrapper::Causal(mode));
                ty = *inner;
            }
            NType::Uncertain(inner) => {
                wrappers.push(TypeWrapper::Uncertain);
                ty = *inner;
            }
            NType::Random(inner) => {
                wrappers.push(TypeWrapper::Random);
                ty = *inner;
            }
            _ => break,
        }
    }
    (ty, wrappers)
}

fn apply_wrappers(mut ty: NType, wrappers: Vec<TypeWrapper>) -> NType {
    for w in wrappers.into_iter().rev() {
        match w {
            TypeWrapper::Temporal(spec) => ty = NType::Temporal(Box::new(ty), spec),
            TypeWrapper::Causal(mode) => ty = NType::Causal(Box::new(ty), mode),
            TypeWrapper::Uncertain => ty = NType::Uncertain(Box::new(ty)),
            TypeWrapper::Random => ty = NType::Random(Box::new(ty)),
        }
    }
    ty
}

pub fn types_compatible(a: &NType, b: &NType) -> bool {
    if matches!(a, NType::Any) || matches!(b, NType::Any) { return true; }
    if matches!(a, NType::Base(ref n) if n == "Any") || matches!(b, NType::Base(ref n) if n == "Any") { return true; }
    match (a, b) {
        (NType::Base(x), NType::Base(y)) => x == y || (x == "Float" && y == "Loss") || (x == "Loss" && y == "Float"),
        (NType::Tensor(_), NType::Base(y)) if y == "Loss" || y == "Tensor" => true,
        (NType::Base(x), NType::Tensor(_)) if x == "Loss" || x == "Tensor" => true,
        (NType::Tensor(_), NType::Tensor(_)) => true, // Shape checked separately
        (NType::Uncertain(x), NType::Uncertain(y)) => types_compatible(x, y),
        (NType::Random(x), NType::Random(y)) => types_compatible(x, y),
        (NType::Temporal(x, s1), NType::Temporal(y, s2)) => {
            // Semantic Invariant: An expression whose information dependency is no later
            // than the maximum allowed dependency is safe (actual <= expected).
            let spec_compat = match (s1, s2) {
                (TemporalSpec::Direction(d1), TemporalSpec::Direction(d2)) => d1 == d2,
                (TemporalSpec::Offset(o_exp), TemporalSpec::Offset(o_act)) => {
                    // Subtyping: actual data offset must be <= expected offset bound.
                    // Past data (e.g. -5) safely satisfies present (0) or future (+5) expectations.
                    o_act <= o_exp
                }
                (TemporalSpec::Direction(d_exp), TemporalSpec::Offset(o_act)) => {
                    if d_exp == "past_to_future" { *o_act <= 0 }
                    else if d_exp == "future_to_past" { *o_act > 0 }
                    else { false }
                }
                (TemporalSpec::Offset(o_exp), TemporalSpec::Direction(d_act)) => {
                    if d_act == "past_to_future" { *o_exp >= 0 }
                    else { false }
                }
            };
            spec_compat && types_compatible(x, y)
        }
        (NType::Causal(x, m1), NType::Causal(y, m2)) => m1 == m2 && types_compatible(x, y),
        (NType::List(x), NType::List(y)) => types_compatible(x, y),
        (NType::Tuple(ts1), NType::Tuple(ts2)) => {
            ts1.len() == ts2.len() && ts1.iter().zip(ts2.iter()).all(|(t1, t2)| types_compatible(t1, t2))
        }
        (NType::Option_(x), NType::Option_(y)) => types_compatible(x, y),
        (NType::Model(a, _, _), NType::Model(b, _, _)) => a == b,
        (NType::Model(a, _, _), NType::Base(b)) => a == b,
        (NType::Base(a), NType::Model(b, _, _)) => a == b,
        (NType::Void, NType::Void) => true,
        // Safe ingestion: raw data can be introduced into Temporal or Causal ("observed").
        // Raw types can NEVER accept Temporal or Causal, preventing egress/erasure bypasses.
        (NType::Temporal(x, _), actual) if !actual.is_temporal() => types_compatible(x, actual),
        (NType::Causal(x, ref mode), actual) if mode == "observed" && !actual.is_causal() => types_compatible(x, actual),
        _ => false,
    }
}


fn type_from_ast(te: &TypeExpr) -> NType {
    match te {
        TypeExpr::Base(name, _) => {
            if name == "Any" { NType::Any }
            else if name == "Tensor" { NType::Tensor(vec![]) }
            else { NType::Base(name.clone()) }
        }
        TypeExpr::Tensor(dims, _) => NType::Tensor(dims.iter().map(|d| match d {
            DimExpr::Static(v) => Dim::Static(*v),
            DimExpr::Symbolic(s) => Dim::Symbolic(s.clone()),
            DimExpr::Named(a, n) => Dim::Named(a.clone(), n.clone()),
            DimExpr::Dynamic => Dim::Dynamic,
        }).collect()),
        TypeExpr::Uncertain(inner, _) => NType::Uncertain(Box::new(type_from_ast(inner))),
        TypeExpr::Random(inner, _) => NType::Random(Box::new(type_from_ast(inner))),
        TypeExpr::Prob(inner, _) => NType::Prob(Box::new(type_from_ast(inner))),
        TypeExpr::Temporal(inner, spec, _) => NType::Temporal(Box::new(type_from_ast(inner)), spec.clone()),
        TypeExpr::Causal(inner, mode, _) => NType::Causal(Box::new(type_from_ast(inner)), mode.clone()),
        TypeExpr::Learnable(fn_type, _, _) => NType::Learnable(fn_type.clone(), None),
        TypeExpr::ListType(inner, _) => NType::List(Box::new(type_from_ast(inner))),
        TypeExpr::OptionType(inner, _) => NType::Option_(Box::new(type_from_ast(inner))),
        TypeExpr::Memory(inner, _) => NType::Memory(Box::new(type_from_ast(inner))),
        TypeExpr::EpisodicMemory(inner, _) => NType::EpisodicMemory(Box::new(type_from_ast(inner))),
        TypeExpr::SemanticMemory(inner, _) => NType::SemanticMemory(Box::new(type_from_ast(inner))),
        TypeExpr::WorkingMemory(inner, cap, _) => {
            let c = cap.as_ref().and_then(|e| if let Expr::IntLit(v, _) = e.as_ref() { Some(*v) } else { None });
            NType::WorkingMemory(Box::new(type_from_ast(inner)), c)
        }
        TypeExpr::RewardType(inner, _) => NType::Reward(Box::new(type_from_ast(inner))),
        TypeExpr::Fn(params, ret, _) => {
            let ps: Vec<NType> = params.iter().map(|p| type_from_ast(p)).collect();
            NType::Fn_(ps, Box::new(type_from_ast(ret)), None)
        }
        TypeExpr::UserDefined(name, _) => {
            if name == "Any" { NType::Any }
            else if name == "Tensor" { NType::Tensor(vec![]) }
            else { NType::Base(name.clone()) }
        }
    }
}

// ═══════════════════════════════════════════
//  Unification environment for symbolic dims
// ═══════════════════════════════════════════

#[derive(Debug, Default)]
struct UnificationEnv {
    bindings: HashMap<String, Dim>,
}

impl UnificationEnv {
    fn resolve(&self, d: &Dim) -> Dim {
        match d {
            Dim::Symbolic(name) => {
                if let Some(bound) = self.bindings.get(name) {
                    self.resolve(bound)
                } else { d.clone() }
            }
            Dim::Named(alias, _) => {
                if let Some(bound) = self.bindings.get(alias) {
                    self.resolve(bound)
                } else { d.clone() }
            }
            _ => d.clone(),
        }
    }

    fn unify(&mut self, a: &Dim, b: &Dim) -> bool {
        let ra = self.resolve(a);
        let rb = self.resolve(b);
        match (&ra, &rb) {
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            (Dim::Static(x), Dim::Static(y)) => x == y,
            (Dim::Symbolic(sa), Dim::Symbolic(sb)) if sa == sb => true,
            (Dim::Symbolic(s), other) | (other, Dim::Symbolic(s)) => {
                self.bindings.insert(s.clone(), other.clone());
                true
            }
            (Dim::Named(a, _), Dim::Named(b, _)) if a == b => true,
            (Dim::Named(a, _), other) | (other, Dim::Named(a, _)) => {
                self.bindings.insert(a.clone(), other.clone());
                true
            }
        }
    }
}

fn extract_symbolic_dims(te: &TypeExpr, out: &mut Vec<String>) {
    match te {
        TypeExpr::Tensor(dims, _) => {
            for d in dims {
                if let DimExpr::Symbolic(ref s) = d {
                    if !out.contains(s) {
                        out.push(s.clone());
                    }
                }
            }
        }
        TypeExpr::Temporal(inner, _, _)
        | TypeExpr::Causal(inner, _, _)
        | TypeExpr::Uncertain(inner, _)
        | TypeExpr::Random(inner, _)
        | TypeExpr::Prob(inner, _)
        | TypeExpr::ListType(inner, _)
        | TypeExpr::OptionType(inner, _)
        | TypeExpr::Memory(inner, _)
        | TypeExpr::RewardType(inner, _) => {
            extract_symbolic_dims(inner, out);
        }
        _ => {}
    }
}

// ═══════════════════════════════════════════
//  Scope / Symbol Table
// ═══════════════════════════════════════════

#[derive(Debug)]
struct Scope {
    symbols: HashMap<String, NType>,
    mutations: Vec<String>,
    uncertain_accessed: Vec<(String, Span)>,
    uncertain_confidence_checked: Vec<String>,
    const_ints: HashMap<String, i64>,
}

impl Scope {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            mutations: Vec::new(),
            uncertain_accessed: Vec::new(),
            uncertain_confidence_checked: Vec::new(),
            const_ints: HashMap::new(),
        }
    }
    
    fn record_uncertain_access(&mut self, name: &str, span: Span) {
        self.uncertain_accessed.push((name.to_string(), span));
    }

    fn record_uncertain_confidence_checked(&mut self, name: &str) {
        self.uncertain_confidence_checked.push(name.to_string());
    }

    fn define(&mut self, name: &str, ty: NType) {
        self.symbols.insert(name.to_string(), ty);
    }

    fn lookup(&self, name: &str) -> Option<&NType> {
        self.symbols.get(name)
    }
}

struct SymbolTable {
    scopes: Vec<Scope>,
}

impl SymbolTable {
    fn new() -> Self {
        let mut global = Scope::new();
        // Built-in functions
        let tensor = NType::Tensor(vec![]);
        let float = NType::Base("Float".into());
        let int = NType::Base("Int".into());
        let loss = NType::Base("Loss".into());
        let dataset = NType::Base("Dataset".into());
        let any = NType::Any;

        global.define("zeros", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("ones", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("randn", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("glorot", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("relu", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("gelu", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("sqrt", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("sin", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("cos", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("len", NType::Fn_(vec![any.clone()], Box::new(int.clone()), None));
        global.define("transpose", NType::Fn_(vec![tensor.clone(), int.clone(), int.clone()], Box::new(tensor.clone()), None));
        global.define("update_row", NType::Fn_(vec![tensor.clone(), int.clone(), any.clone()], Box::new(tensor.clone()), None));
        global.define("softmax", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("sigmoid", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("tanh", NType::Fn_(vec![tensor.clone()], Box::new(tensor.clone()), None));
        global.define("cross_entropy", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("mse", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("negative_log_likelihood", NType::Fn_(vec![tensor.clone(), tensor.clone()], Box::new(loss.clone()), None));
        global.define("kl_divergence", NType::Fn_(vec![any.clone(), any.clone()], Box::new(loss.clone()), None));
        global.define("concat", NType::Fn_(vec![NType::List(Box::new(tensor.clone()))], Box::new(tensor.clone()), None));
        global.define("range", NType::Fn_(vec![int.clone()], Box::new(NType::List(Box::new(int.clone()))), None));
        global.define("min", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("max", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("abs", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("Normal", NType::Fn_(vec![float.clone(), float.clone()], Box::new(NType::Uncertain(Box::new(float.clone()))), None));
        global.define("Beta", NType::Fn_(vec![float.clone(), float.clone()], Box::new(NType::Uncertain(Box::new(float.clone()))), None));
        global.define("GaussianNoise", NType::Fn_(vec![float.clone()], Box::new(NType::Random(Box::new(float.clone()))), None));
        global.define("load", NType::Fn_(vec![NType::Base("String".into())], Box::new(any.clone()), None));
        global.define("load_dataset", NType::Fn_(vec![NType::Base("String".into())], Box::new(dataset.clone()), None));
        global.define("load_ohlcv", NType::Fn_(vec![NType::Base("String".into())], Box::new(any.clone()), None));
        global.define("load_ohlcv_list", NType::Fn_(vec![NType::Base("String".into())], Box::new(any.clone()), None));
        global.define("load_tensor", NType::Fn_(vec![NType::Base("String".into()), int.clone(), int.clone()], Box::new(tensor.clone()), None));
        global.define("save_tensor", NType::Fn_(vec![tensor.clone(), NType::Base("String".into())], Box::new(NType::Void), None));
        global.define("argmax", NType::Fn_(vec![tensor.clone()], Box::new(int.clone()), None));
        global.define("sample_categorical", NType::Fn_(vec![tensor.clone(), float.clone()], Box::new(int.clone()), None));
        global.define("char_from_int", NType::Fn_(vec![int.clone()], Box::new(NType::Base("String".into())), None));
        global.define("onehot", NType::Fn_(vec![int.clone(), int.clone()], Box::new(tensor.clone()), None));
        global.define("aggregate", NType::Fn_(vec![NType::List(Box::new(any.clone()))], Box::new(any.clone()), None));
        global.define("estimate_epistemic_std", NType::Fn_(vec![tensor.clone()], Box::new(float.clone()), None));
        global.define("fractional_kelly", NType::Fn_(vec![NType::Uncertain(Box::new(float.clone())), float.clone()], Box::new(float.clone()), None));
        global.define("sample", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("condition", NType::Fn_(vec![any.clone(), any.clone()], Box::new(any.clone()), None));
        global.define("print", NType::Fn_(vec![any.clone()], Box::new(NType::Void), None));
        global.define("print_inline", NType::Fn_(vec![any.clone()], Box::new(NType::Void), None));
        global.define("input", NType::Fn_(vec![], Box::new(NType::Base("String".into())), None));
        global.define("embed_string", NType::Fn_(vec![NType::Base("String".into())], Box::new(tensor.clone()), None));
        global.define("generate_reply", NType::Fn_(vec![NType::Base("String".into())], Box::new(NType::Base("String".into())), None));
        global.define("forget", NType::Fn_(vec![any.clone(), any.clone(), any.clone(), any.clone()], Box::new(NType::Base("ForgetCertificate".into())), None));
        global.define("sgd", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("adam", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("grad", NType::Fn_(vec![any.clone()], Box::new(tensor.clone()), None));
        global.define("stop_grad", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("mean", NType::Fn_(vec![any.clone()], Box::new(float.clone()), None));
        global.define("sum", NType::Fn_(vec![any.clone()], Box::new(float.clone()), None));

        // High-performance Mersenne and bitwise built-ins
        let bool_type = NType::Base("Bool".into());
        global.define("pow2_sub1", NType::Fn_(vec![int.clone()], Box::new(int.clone()), None));
        global.define("mersenne_mod", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("mersenne_lucas_lehmer", NType::Fn_(vec![int.clone()], Box::new(bool_type.clone()), None));
        global.define("shl", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("shr", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("band", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("parallel_mersenne_hunt", NType::Fn_(vec![int.clone(), int.clone()], Box::new(NType::List(Box::new(int.clone()))), None));
        global.define("mersenne_factor_sift", NType::Fn_(vec![int.clone(), int.clone()], Box::new(bool_type.clone()), None));
        global.define("mersenne_find_factor", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("mersenne_find_factor_gpu", NType::Fn_(vec![int.clone(), int.clone()], Box::new(int.clone()), None));
        global.define("mersenne_hunt_53rd", NType::Fn_(vec![int.clone(), int.clone(), NType::Base("String".into())], Box::new(bool_type.clone()), None));
        global.define("cuda_available", NType::Fn_(vec![], Box::new(bool_type.clone()), None));
        global.define("is_cuda_available", NType::Fn_(vec![], Box::new(bool_type.clone()), None));
        global.define("cuda_device_name", NType::Fn_(vec![], Box::new(NType::Base("String".into())), None));

        // Built-in type names
        global.define("Int", NType::Base("Int".into()));
        global.define("Float", NType::Base("Float".into()));
        global.define("Bool", NType::Base("Bool".into()));
        global.define("String", NType::Base("String".into()));
        global.define("Timestamp", NType::Base("Timestamp".into()));
        global.define("Loss", NType::Base("Loss".into()));
        global.define("Dataset", NType::Base("Dataset".into()));
        global.define("Any", NType::Any);
        global.define("Uncertain", NType::Fn_(vec![any.clone(), any.clone()], Box::new(NType::Uncertain(Box::new(any.clone()))), None));
        global.define("UNCERTAIN", NType::Fn_(vec![any.clone(), any.clone()], Box::new(NType::Uncertain(Box::new(any.clone()))), None));
        global.define("load", NType::Fn_(vec![any.clone()], Box::new(any.clone()), None));
        global.define("println", NType::Fn_(vec![any.clone()], Box::new(NType::Void), None));

        Self { scopes: vec![global] }
    }

    fn push(&mut self) { self.scopes.push(Scope::new()); }
    fn pop(&mut self) -> Scope { self.scopes.pop().unwrap_or_else(|| Scope::new()) }

    fn define(&mut self, name: &str, ty: NType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.define(name, ty);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<NType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.lookup(name) { return Some(ty.clone()); }
        }
        None
    }

    fn record_mutation(&mut self, target: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.mutations.push(target.to_string());
        }
    }

    fn record_uncertain_access(&mut self, name: &str, span: Span) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.record_uncertain_access(name, span);
        }
    }

    fn record_uncertain_confidence_checked(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.record_uncertain_confidence_checked(name);
        }
    }

    fn propagate_child_scope(&mut self, child: Scope) {
        if let Some(parent) = self.scopes.last_mut() {
            parent.mutations.extend(child.mutations);
            parent.uncertain_accessed.extend(child.uncertain_accessed);
            parent.uncertain_confidence_checked.extend(child.uncertain_confidence_checked);
        }
    }

    fn get_const_int(&self, name: &str) -> Option<i64> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.const_ints.get(name) {
                return Some(*val);
            }
        }
        None
    }

    fn set_const_int(&mut self, name: &str, val: i64) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.symbols.contains_key(name) {
                scope.const_ints.insert(name.to_string(), val);
                return;
            }
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.const_ints.insert(name.to_string(), val);
        }
    }
}

// ═══════════════════════════════════════════
//  Type Checker
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct FnEffectInfo {
    pub param_names: Vec<String>,
    pub effects: Vec<EffectEntry>,
}

pub struct TypeChecker {
    pub result: CompileResult,
    symbols: SymbolTable,
    unifier: UnificationEnv,
    model_types: HashMap<String, NType>,
    current_return_type: Option<NType>,
    effects_map: HashMap<String, FnEffectInfo>,
}

impl TypeChecker {
    pub fn new(filename: &str) -> Self {
        Self {
            result: CompileResult::new(filename),
            symbols: SymbolTable::new(),
            unifier: UnificationEnv::default(),
            model_types: HashMap::new(),
            current_return_type: None,
            effects_map: HashMap::new(),
        }
    }

    pub fn lookup(&self, name: &str) -> Option<NType> {
        self.symbols.lookup(name)
    }

    pub fn check(&mut self, program: &Program) {
        // Phase 1: register all top-level declarations
        for tl in &program.top_levels {
            self.register_top_level(tl);
        }
        // Phase 2: type-check all bodies
        for tl in &program.top_levels {
            self.check_top_level(tl);
        }
    }

    // ── Phase 1: Registration ──

    fn register_top_level(&mut self, tl: &TopLevel) {
        match tl {
            TopLevel::Fn(f) => {
                let fn_ty = self.fn_to_type(f);
                self.symbols.define(&f.name, fn_ty);
                let effects = f.effect_clause.as_ref().map(|ec| {
                    ec.effects.iter().map(|e| EffectEntry {
                        kind: e.kind.clone(),
                        target: e.target.clone(),
                    }).collect()
                }).unwrap_or_default();
                self.effects_map.insert(f.name.clone(), FnEffectInfo {
                    param_names: f.params.iter().map(|p| p.name.clone()).collect(),
                    effects,
                });
            }
            TopLevel::Model(m) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &m.fields {
                    fields.insert(f.name.clone(), type_from_ast(&f.type_ann));
                }
                for p in &m.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &m.methods {
                    methods.insert(met.name.clone(), self.fn_to_type(met));
                    let effects = met.effect_clause.as_ref().map(|ec| {
                        ec.effects.iter().map(|e| EffectEntry {
                            kind: e.kind.clone(),
                            target: e.target.clone(),
                        }).collect()
                    }).unwrap_or_default();
                    self.effects_map.insert(format!("{}.{}", m.name, met.name), FnEffectInfo {
                        param_names: met.params.iter().map(|p| p.name.clone()).collect(),
                        effects,
                    });
                }
                let model_ty = NType::Model(m.name.clone(), fields, methods);
                self.model_types.insert(m.name.clone(), model_ty.clone());
                // Constructor
                let params: Vec<NType> = m.params.iter().map(|p| {
                    p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
                }).collect();
                self.symbols.define(&m.name, NType::Fn_(params, Box::new(model_ty), None));
            }
            TopLevel::Layer(l) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &l.fields { fields.insert(f.name.clone(), type_from_ast(&f.type_ann)); }
                for p in &l.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &l.methods {
                    methods.insert(met.name.clone(), self.fn_to_type(met));
                    let effects = met.effect_clause.as_ref().map(|ec| {
                        ec.effects.iter().map(|e| EffectEntry {
                            kind: e.kind.clone(),
                            target: e.target.clone(),
                        }).collect()
                    }).unwrap_or_default();
                    self.effects_map.insert(format!("{}.{}", l.name, met.name), FnEffectInfo {
                        param_names: met.params.iter().map(|p| p.name.clone()).collect(),
                        effects,
                    });
                }
                let model_ty = NType::Model(l.name.clone(), fields, methods);
                self.model_types.insert(l.name.clone(), model_ty.clone());
                let params: Vec<NType> = l.params.iter().map(|p| {
                    p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
                }).collect();
                self.symbols.define(&l.name, NType::Fn_(params, Box::new(model_ty), None));
            }
            TopLevel::Causal(c) => {
                let mut vars = Vec::new();
                for edge in &c.edges {
                    for s in &edge.sources { if !vars.contains(s) { vars.push(s.clone()); } }
                    if let Some(ref t) = edge.target { if !vars.contains(t) { vars.push(t.clone()); } }
                }
                let cm_ty = NType::CausalModel(c.name.clone(), vars);
                self.symbols.define(&c.name, cm_ty);
            }
            TopLevel::Agent(a) => {
                let mut fields = HashMap::new();
                let mut methods = HashMap::new();
                for f in &a.fields { fields.insert(f.name.clone(), type_from_ast(&f.type_ann)); }
                for p in &a.params {
                    if let Some(ref ta) = p.type_ann {
                        fields.insert(p.name.clone(), type_from_ast(ta));
                    }
                }
                for met in &a.methods {
                    methods.insert(met.name.clone(), self.fn_to_type(met));
                    let effects = met.effect_clause.as_ref().map(|ec| {
                        ec.effects.iter().map(|e| EffectEntry {
                            kind: e.kind.clone(),
                            target: e.target.clone(),
                        }).collect()
                    }).unwrap_or_default();
                    self.effects_map.insert(format!("{}.{}", a.name, met.name), FnEffectInfo {
                        param_names: met.params.iter().map(|p| p.name.clone()).collect(),
                        effects,
                    });
                }
                let agent_ty = NType::Model(a.name.clone(), fields, methods);
                self.model_types.insert(a.name.clone(), agent_ty.clone());
                self.symbols.define(&a.name, agent_ty);
            }
            TopLevel::Let(l) => {
                let ty = l.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                self.symbols.define(&l.name, ty);
            }
            TopLevel::Import(imp) => {
                for name in &imp.names {
                    if self.symbols.lookup(name).is_none() {
                        self.symbols.define(name, NType::Any);
                    }
                }
                if let Some(ref alias) = imp.alias {
                    if self.symbols.lookup(alias).is_none() {
                        self.symbols.define(alias, NType::Any);
                    }
                }
            }
            _ => {}
        }
    }

    fn fn_to_type(&self, f: &FnDecl) -> NType {
        let params: Vec<NType> = f.params.iter().map(|p| {
            p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
        }).collect();
        let ret = f.return_type.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Void);
        let effect = f.effect_clause.as_ref().map(|ec| {
            let entries = ec.effects.iter().map(|e| EffectEntry {
                kind: e.kind.clone(),
                target: e.target.clone(),
            }).collect();
            Box::new(NType::Effect(entries))
        });
        NType::Fn_(params, Box::new(ret), effect)
    }

    // ── Phase 2: Checking ──

    fn check_top_level(&mut self, tl: &TopLevel) {
        match tl {
            TopLevel::Fn(f) => self.check_fn(f, None),
            TopLevel::Model(m) => {
                let self_ty = self.model_types.get(&m.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for p in &m.params {
                    let ty = p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                    self.symbols.define(&p.name, ty);
                    if let Some(ref ta) = p.type_ann {
                        let mut dims = Vec::new();
                        extract_symbolic_dims(ta, &mut dims);
                        for dim in dims {
                            self.symbols.define(&dim, NType::Base("Int".into()));
                        }
                    }
                }
                for f in &m.fields {
                    self.symbols.define(&f.name, type_from_ast(&f.type_ann));
                }
                for method in &m.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Layer(l) => {
                let self_ty = self.model_types.get(&l.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for p in &l.params {
                    let ty = p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any);
                    self.symbols.define(&p.name, ty);
                    if let Some(ref ta) = p.type_ann {
                        let mut dims = Vec::new();
                        extract_symbolic_dims(ta, &mut dims);
                        for dim in dims {
                            self.symbols.define(&dim, NType::Base("Int".into()));
                        }
                    }
                }
                for method in &l.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Agent(a) => {
                let self_ty = self.model_types.get(&a.name).cloned();
                self.symbols.push();
                if let Some(ref st) = self_ty { self.symbols.define("self", st.clone()); }
                for method in &a.methods { self.check_fn(method, self_ty.as_ref()); }
                self.symbols.pop();
            }
            TopLevel::Let(l) => {
                let inferred = self.infer_expr(&l.value);
                if let Some(val) = self.eval_const_int(&l.value) {
                    self.symbols.set_const_int(&l.name, val);
                }
                if let Some(ref ta) = l.type_ann {
                    let declared = type_from_ast(ta);
                    if !types_compatible(&declared, &inferred) && !matches!(inferred, NType::Any) {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Variable '{}' declared as {} but initialized with {}", l.name, declared.display(), inferred.display()),
                            l.span.clone(),
                        ).with_expected(&declared.display()).with_actual(&inferred.display()));
                    }
                    self.symbols.define(&l.name, declared);
                } else {
                    self.symbols.define(&l.name, inferred);
                }
            }
            TopLevel::Meta(m) => self.check_fn(&m.func, None),
            TopLevel::Expr(e) => { self.infer_expr(&e.expr); }
            TopLevel::Update(u) => {
                self.symbols.record_mutation(&u.target);
                self.infer_expr(&u.expr);
            }
            _ => {}
        }
    }

    fn check_fn(&mut self, f: &FnDecl, self_ty: Option<&NType>) {
        self.symbols.push();
        if let Some(st) = self_ty { self.symbols.define("self", st.clone()); }
        for p in &f.params {
            let ty = if p.name == "self" && self_ty.is_some() {
                self_ty.unwrap().clone()
            } else {
                p.type_ann.as_ref().map(|t| type_from_ast(t)).unwrap_or(NType::Any)
            };
            self.symbols.define(&p.name, ty);
            if let Some(ref ta) = p.type_ann {
                let mut dims = Vec::new();
                extract_symbolic_dims(ta, &mut dims);
                for dim in dims {
                    self.symbols.define(&dim, NType::Base("Int".into()));
                }
            }
        }
        // R5 fix: track expected return type so Stmt::Return can validate against it
        let prev_return_type = self.current_return_type.take();
        self.current_return_type = f.return_type.as_ref().map(|t| type_from_ast(t));
        for stmt in &f.body { self.check_stmt(stmt); }
        self.current_return_type = prev_return_type;

        // Effect checking
        let scope = self.symbols.pop();
        
        // Uncertainty confidence check warnings
        for (name, span) in &scope.uncertain_accessed {
            if !scope.uncertain_confidence_checked.contains(name) {
                self.result.add_warning(uncertainty_ignored_warning(span.clone(), name));
            }
        }

        // Effects govern caller-visible state (mutations to parameters or self).
        // Pure local variable mutations within the function body do not require effects.
        let param_names: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        let external_mutations: Vec<&String> = scope.mutations.iter().filter(|m| {
            let root = m.split('.').next().unwrap_or(m.as_str());
            param_names.contains(&root) || root == "self" || (self_ty.is_some() && m.starts_with("self."))
        }).collect();

        if !external_mutations.is_empty() {
            if let Some(ref eff) = f.effect_clause {
                let declared_mut_targets: Vec<String> = eff.effects.iter()
                    .filter(|e| e.kind == "Mut")
                    .filter_map(|e| e.target.clone())
                    .collect();
                let has_generic_mut = eff.effects.iter().any(|e| e.kind == "Mut" && e.target.is_none());

                if !has_generic_mut {
                    let mut unpermitted: Vec<String> = Vec::new();
                    for m in &external_mutations {
                        let covered = declared_mut_targets.iter().any(|dt| {
                            dt == *m || m.starts_with(&format!("{}.", dt)) || (dt == "self" && m.starts_with("self."))
                        });
                        if !covered {
                            unpermitted.push(format!("Mut[{}]", m));
                        }
                    }
                    if !unpermitted.is_empty() {
                        self.result.add_error(effect_undeclared_error(f.span.clone(), &f.name, &unpermitted));
                    }
                }
            } else {
                let missing: Vec<String> = external_mutations.iter().map(|m| format!("Mut[{}]", m)).collect();
                self.result.add_error(effect_undeclared_error(f.span.clone(), &f.name, &missing));
            }
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => {
                let inferred = self.infer_expr(&l.value);
                if let Some(val) = self.eval_const_int(&l.value) {
                    self.symbols.set_const_int(&l.name, val);
                }
                if let Some(ref ta) = l.type_ann {
                    let declared = type_from_ast(ta);
                    if !types_compatible(&declared, &inferred) && !matches!(inferred, NType::Any) {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Variable '{}' declared as {} but initialized with {}", l.name, declared.display(), inferred.display()),
                            l.span.clone(),
                        ).with_expected(&declared.display()).with_actual(&inferred.display()));
                    }
                    self.symbols.define(&l.name, declared);
                } else {
                    self.symbols.define(&l.name, inferred);
                }
            }
            Stmt::For(f) => {
                let iter_ty = self.infer_expr(&f.iter_expr);
                let elem_ty = match &iter_ty {
                    NType::List(inner) => *inner.clone(),
                    _ => NType::Any,
                };
                self.symbols.push();
                self.symbols.define(&f.var, elem_ty);
                for s in &f.body { self.check_stmt(s); }
                let child = self.symbols.pop();
                self.symbols.propagate_child_scope(child);
            }
            Stmt::If(i) => {
                let cond_ty = self.infer_expr(&i.cond);
                if !matches!(cond_ty, NType::Base(ref n) if n == "Bool") && !matches!(cond_ty, NType::Any) {
                    self.result.add_error(NeuronError::new(
                        ErrorCode::TypeMismatch, "If condition must be Bool", i.span.clone(),
                    ).with_actual(&cond_ty.display()));
                }
                self.symbols.push();
                for s in &i.then_body { self.check_stmt(s); }
                let then_child = self.symbols.pop();
                self.symbols.propagate_child_scope(then_child);

                self.symbols.push();
                for s in &i.else_body { self.check_stmt(s); }
                let else_child = self.symbols.pop();
                self.symbols.propagate_child_scope(else_child);
            }
            Stmt::While(w) => {
                let cond_ty = self.infer_expr(&w.cond);
                if !matches!(cond_ty, NType::Base(ref n) if n == "Bool") && !matches!(cond_ty, NType::Any) {
                    self.result.add_error(NeuronError::new(
                        ErrorCode::TypeMismatch, "While condition must be Bool", w.span.clone(),
                    ).with_actual(&cond_ty.display()));
                }
                self.symbols.push();
                for s in &w.body { self.check_stmt(s); }
                let child = self.symbols.pop();
                self.symbols.propagate_child_scope(child);
            }
            Stmt::Return(r) => {
                let inferred = self.infer_expr(&r.value);
                if let Some(ref declared) = self.current_return_type {
                    if !types_compatible(declared, &inferred) && !matches!(inferred, NType::Any) {
                        self.result.add_error(
                            NeuronError::new(
                                ErrorCode::TypeMismatch,
                                format!(
                                    "Function declared to return {} but returns {}",
                                    declared.display(),
                                    inferred.display()
                                ),
                                r.span.clone(),
                            )
                            .with_expected(&declared.display())
                            .with_actual(&inferred.display()),
                        );
                    }
                }
            }
            Stmt::Update(u) => {
                self.symbols.record_mutation(&u.target);
                self.infer_expr(&u.expr);
            }
            Stmt::Expr(e) => { self.infer_expr(&e.expr); }
            Stmt::Constraint(c) => { self.infer_expr(&c.expr); }
        }
    }

    // ── Expression type inference ──

    fn infer_expr(&mut self, expr: &Expr) -> NType {
        match expr {
            Expr::IntLit(_, _) => NType::Base("Int".into()),
            Expr::FloatLit(_, _) => NType::Base("Float".into()),
            Expr::BoolLit(_, _) => NType::Base("Bool".into()),
            Expr::StringLit(_, _) => NType::Base("String".into()),
            Expr::Ident(name, span) => {
                match self.symbols.lookup(name) {
                    Some(ty) => {
                        if let NType::Uncertain(_) = ty {
                            self.symbols.record_uncertain_access(name, span.clone());
                        }
                        ty
                    }
                    None => {
                        // R6 fix: report undefined identifiers instead of silently becoming Any.
                        // We still return Any for error recovery so the checker can continue
                        // finding additional errors in the same compilation.
                        self.result.add_error(NeuronError::new(
                            ErrorCode::UndefinedVariable,
                            format!("Undefined variable '{}'", name),
                            span.clone(),
                        ));
                        NType::Any
                    }
                }
            }
            Expr::Self_(_) => self.symbols.lookup("self").unwrap_or(NType::Any),

            Expr::BinOp(b) => self.infer_binop(b),
            Expr::UnaryOp(u) => {
                let inner = self.infer_expr(&u.operand);
                match u.op {
                    UnaryOp::Neg => {
                        if inner != NType::Any && !inner.is_numeric() && !matches!(inner, NType::Tensor(_)) && !matches!(inner, NType::Uncertain(_)) {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                format!("Numeric negation (-) requires numeric or tensor operand, got {}", inner.display()),
                                u.span.clone(),
                            ));
                        }
                        inner
                    }
                    UnaryOp::Not => {
                        if inner != NType::Any && inner != NType::Base("Bool".into()) {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                format!("Logical negation (!) requires a boolean operand, got {}", inner.display()),
                                u.span.clone(),
                            ));
                        }
                        if inner == NType::Any { NType::Any } else { NType::Base("Bool".into()) }
                    }
                }
            }
            Expr::FnCall(c) => self.infer_fn_call(c),
            Expr::Dot(d) => {
                if d.field == "confidence" {
                    if let Expr::Ident(ref name, _) = d.obj {
                        self.symbols.record_uncertain_confidence_checked(name);
                    }
                }
                self.infer_dot(d)
            }
            Expr::Index(idx) => {
                let obj_ty = self.infer_expr(&idx.obj);
                let (stripped_obj, wrappers) = strip_wrappers(obj_ty);
                let elem_ty = match stripped_obj {
                    NType::Tensor(ref dims) => {
                        if dims.len() <= 1 {
                            NType::Base("Float".into())
                        } else {
                            NType::Tensor(dims[1..].to_vec())
                        }
                    }
                    NType::List(inner) => *inner.clone(),
                    NType::Tuple(ref types) => {
                        if idx.indices.len() == 1 {
                            if let IndexItem::Expr(Expr::IntLit(val, _)) = &idx.indices[0] {
                                let i = *val as usize;
                                if i < types.len() {
                                    types[i].clone()
                                } else {
                                    NType::Any
                                }
                            } else {
                                NType::Any
                            }
                        } else {
                            NType::Any
                        }
                    }
                    _ => NType::Any,
                };
                if elem_ty == NType::Any {
                    NType::Any
                } else {
                    apply_wrappers(elem_ty, wrappers)
                }
            }
            Expr::Grad(g) => {
                let inner = self.infer_expr(&g.expr);
                // grad(loss) returns a tensor-like type
                if inner.is_tensor() { inner } else { NType::Tensor(vec![]) }
            }
            Expr::StopGrad(expr, _) => {
                self.infer_expr(expr)
            }
            Expr::Do(_d) => {
                NType::Causal(Box::new(NType::Any), "intervened".into())
            }
            Expr::Observe(_o) => {
                NType::Causal(Box::new(NType::Any), "observed".into())
            }
            Expr::Explain(e) => {
                let inner = self.infer_expr(&e.expr);
                NType::Tuple(vec![inner, NType::Explanation])
            }
            Expr::Merge(m) => {
                let left = self.infer_expr(&m.left);
                left // Merge returns same type as left operand
            }
            Expr::Forget(f) => {
                self.infer_expr(&f.obj) // Forget returns same type
            }
            Expr::List(elems, _) => {
                if elems.is_empty() {
                    NType::List(Box::new(NType::Any))
                } else {
                    let inner = self.infer_expr(&elems[0]);
                    NType::List(Box::new(inner))
                }
            }
            Expr::ListComp(lc) => {
                let inner = self.infer_expr(&lc.expr);
                NType::List(Box::new(inner))
            }
            Expr::Tuple(elems, _) => {
                let types: Vec<NType> = elems.iter().map(|e| self.infer_expr(e)).collect();
                NType::Tuple(types)
            }
            Expr::SearchExpr(s) => {
                self.infer_expr(&s.space);
                self.infer_expr(&s.evaluate);
                NType::Any // SearchResult
            }
            Expr::RecallExpr(r) => {
                let mem_ty = self.infer_expr(&r.memory);
                match mem_ty {
                    NType::EpisodicMemory(inner) | NType::SemanticMemory(inner) | NType::Memory(inner) => {
                        NType::List(inner)
                    }
                    _ => NType::List(Box::new(NType::Any)),
                }
            }
            Expr::StoreExpr(s) => {
                self.infer_expr(&s.memory);
                self.infer_expr(&s.item);
                NType::Void
            }
        }
    }

    fn infer_binop(&mut self, b: &BinOpExpr) -> NType {
        let left = self.infer_expr(&b.left);
        let right = self.infer_expr(&b.right);

        // ── Uncertainty mismatch ──
        if matches!((&left, &right), (NType::Uncertain(_), NType::Random(_)) | (NType::Random(_), NType::Uncertain(_))) {
            self.result.add_error(uncertainty_mismatch_error(
                b.span.clone(),
                if matches!(left, NType::Uncertain(_)) { "Uncertain" } else { "Random" },
                if matches!(right, NType::Random(_)) { "Random" } else { "Uncertain" },
            ));
            return NType::Any;
        }

        // ── Causal type mismatch ──
        if let (NType::Causal(_, ref m1), NType::Causal(_, ref m2)) = (&left, &right) {
            if m1 != m2 {
                self.result.add_error(causal_type_mismatch_error(b.span.clone(), m1, m2));
                return NType::Any;
            }
        }

        // Strip wrappers for calculation
        let (stripped_left, left_wrappers) = strip_wrappers(left.clone());
        let (stripped_right, right_wrappers) = strip_wrappers(right.clone());

        // Perform normal inference on stripped types
        let mut stripped_result = NType::Any;

        // ── Tensor operations ──
        if b.op == BinOp::MatMul {
            if let (NType::Tensor(ref da), NType::Tensor(ref db)) = (&stripped_left, &stripped_right) {
                stripped_result = self.check_matmul(da, db, &b.span);
            } else if stripped_left == NType::Any || stripped_right == NType::Any {
                stripped_result = NType::Any;
            } else {
                self.result.add_error(NeuronError::new(
                    ErrorCode::TypeMismatch,
                    format!("Unsupported binary operation: {} @ {}", left.display(), right.display()),
                    b.span.clone(),
                ));
            }
        } else if matches!(b.op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) {
            if let (NType::List(ref la), NType::List(ref lb)) = (&stripped_left, &stripped_right) {
                if b.op == BinOp::Add {
                    if types_compatible(la, lb) {
                        stripped_result = NType::List(la.clone());
                    } else if types_compatible(lb, la) {
                        stripped_result = NType::List(lb.clone());
                    } else {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("List concatenation requires compatible element types, got {} and {}", la.display(), lb.display()),
                            b.span.clone(),
                        ));
                    }
                } else {
                    self.result.add_error(NeuronError::new(
                        ErrorCode::TypeMismatch,
                        format!("Unsupported binary operation: {} {} {}", left.display(), b.op.as_str(), right.display()),
                        b.span.clone(),
                    ));
                }
            } else if stripped_left == NType::Base("String".into()) && stripped_right == NType::Base("String".into()) && b.op == BinOp::Add {
                stripped_result = NType::Base("String".into());
            } else if let (NType::Tensor(ref da), NType::Tensor(ref db)) = (&stripped_left, &stripped_right) {
                self.check_elementwise(da, db, &b.span);
                stripped_result = stripped_left.clone();
            } else if (matches!(stripped_left, NType::Tensor(_)) && stripped_right.is_numeric()) || (stripped_left.is_numeric() && matches!(stripped_right, NType::Tensor(_))) {
                stripped_result = if matches!(stripped_left, NType::Tensor(_)) { stripped_left.clone() } else { stripped_right.clone() };
            } else if stripped_left == NType::Any || stripped_right == NType::Any {
                stripped_result = NType::Any;
            } else if stripped_left.is_numeric() && stripped_right.is_numeric() {
                if matches!(stripped_left, NType::Base(ref n) if n == "Float") || matches!(stripped_right, NType::Base(ref n) if n == "Float") {
                    stripped_result = NType::Base("Float".into());
                } else {
                    stripped_result = NType::Base("Int".into());
                }
            } else {
                self.result.add_error(NeuronError::new(
                    ErrorCode::TypeMismatch,
                    format!("Unsupported binary operation: {} {} {}", left.display(), b.op.as_str(), right.display()),
                    b.span.clone(),
                ));
            }
        } else if matches!(b.op, BinOp::Eq | BinOp::Neq) {
            if stripped_left != stripped_right && !(stripped_left.is_numeric() && stripped_right.is_numeric()) {
                self.result.add_error(NeuronError::new(
                    ErrorCode::TypeMismatch,
                    format!("Comparison operator {} requires compatible types, got {} and {}", b.op.as_str(), left.display(), right.display()),
                    b.span.clone(),
                ));
            }
            stripped_result = NType::Base("Bool".into());
        } else if matches!(b.op, BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte) {
            if !stripped_left.is_numeric() || !stripped_right.is_numeric() {
                self.result.add_error(NeuronError::new(
                    ErrorCode::TypeMismatch,
                    format!("Comparison operator {} requires numeric operands, got {} and {}", b.op.as_str(), left.display(), right.display()),
                    b.span.clone(),
                ));
            }
            stripped_result = NType::Base("Bool".into());
        } else if matches!(b.op, BinOp::And | BinOp::Or) {
            if stripped_left != NType::Base("Bool".into()) || stripped_right != NType::Base("Bool".into()) {
                self.result.add_error(NeuronError::new(
                    ErrorCode::TypeMismatch,
                    format!("Logical operator {} requires boolean operands, got {} and {}", b.op.as_str(), left.display(), right.display()),
                    b.span.clone(),
                ));
            }
            stripped_result = NType::Base("Bool".into());
        } else if stripped_left == NType::Any || stripped_right == NType::Any {
            stripped_result = NType::Any;
        } else {
            self.result.add_error(NeuronError::new(
                ErrorCode::TypeMismatch,
                format!("Unsupported binary operation: {} {} {}", left.display(), b.op.as_str(), right.display()),
                b.span.clone(),
            ));
        }

        // Combine wrappers and re-apply
        let mut combined_wrappers = left_wrappers.clone();
        let mut has_temporal_conflict = false;
        for rw in &right_wrappers {
            match rw {
                TypeWrapper::Temporal(ref rspec) => {
                    if let Some(lw) = combined_wrappers.iter_mut().find(|w| matches!(w, TypeWrapper::Temporal(_))) {
                        if let TypeWrapper::Temporal(ref mut lspec) = lw {
                            match (&lspec, &rspec) {
                                (TemporalSpec::Offset(n1), TemporalSpec::Offset(n2)) => {
                                    // R3 fix: The temporal horizon of an expression is governed by its
                                    // latest dependency (max offset). If either operand touches the future,
                                    // the combined result depends on future data and cannot be laundered.
                                    *lspec = TemporalSpec::Offset(std::cmp::max(*n1, *n2));
                                }
                                (TemporalSpec::Direction(d1), TemporalSpec::Direction(d2)) => {
                                    if d1 != d2 {
                                        has_temporal_conflict = true;
                                    }
                                }
                                (TemporalSpec::Direction(d), TemporalSpec::Offset(o)) | (TemporalSpec::Offset(o), TemporalSpec::Direction(d)) => {
                                    if (d == "past_to_future" && *o > 0) || (d == "future_to_past" && *o <= 0) {
                                        has_temporal_conflict = true;
                                    }
                                }
                            }
                        }
                    } else {
                        combined_wrappers.push(rw.clone());
                    }
                }
                TypeWrapper::Causal(ref rmode) => {
                    if let Some(lw) = left_wrappers.iter().find(|w| matches!(w, TypeWrapper::Causal(_))) {
                        if let TypeWrapper::Causal(ref lmode) = lw {
                            if lmode != rmode {
                                // Handled by causal check above
                            }
                        }
                    } else {
                        combined_wrappers.push(rw.clone());
                    }
                }
                TypeWrapper::Uncertain => {
                    if !left_wrappers.iter().any(|w| matches!(w, TypeWrapper::Uncertain)) {
                        combined_wrappers.push(rw.clone());
                    }
                }
                TypeWrapper::Random => {
                    if !left_wrappers.iter().any(|w| matches!(w, TypeWrapper::Random)) {
                        combined_wrappers.push(rw.clone());
                    }
                }
            }
        }

        if has_temporal_conflict {
            self.result.add_error(NeuronError::new(
                ErrorCode::TypeMismatch,
                format!("Temporal direction mismatch in binary operation"),
                b.span.clone(),
            ));
        }

        apply_wrappers(stripped_result, combined_wrappers)
    }

    fn check_matmul(&mut self, a_dims: &[Dim], b_dims: &[Dim], span: &Span) -> NType {
        if a_dims.iter().any(|d| matches!(d, Dim::Dynamic)) || b_dims.iter().any(|d| matches!(d, Dim::Dynamic)) {
            self.result.add_warning(dynamic_dim_warning(span.clone()));
        }
        if a_dims.len() < 2 || b_dims.len() < 2 {
            // Allow for compatibility — return tensor
            return NType::Tensor(vec![]);
        }
        let a_inner = &a_dims[a_dims.len() - 1];
        let b_outer = &b_dims[b_dims.len() - 2];
        if !self.unifier.unify(a_inner, b_outer) {
            let a_str = match a_inner { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
            let b_str = match b_outer { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
            self.result.add_error(shape_mismatch_error(
                span.clone(),
                &format!("inner dim {}", b_str),
                &format!("inner dim {}", a_str),
                "matrix multiply (@)",
            ));
        }
        // Result dims: a[:-1] + b[-1:]
        let mut result_dims: Vec<Dim> = a_dims[..a_dims.len()-1].to_vec();
        result_dims.push(b_dims[b_dims.len()-1].clone());
        NType::Tensor(result_dims)
    }

    fn check_elementwise(&mut self, a: &[Dim], b: &[Dim], span: &Span) {
        if a.iter().any(|d| matches!(d, Dim::Dynamic)) || b.iter().any(|d| matches!(d, Dim::Dynamic)) {
            self.result.add_warning(dynamic_dim_warning(span.clone()));
        }
        if a.is_empty() || b.is_empty() {
            return;
        }
        
        let max_len = std::cmp::max(a.len(), b.len());
        let mut a_padded = vec![Dim::Static(1); max_len - a.len()];
        a_padded.extend_from_slice(a);
        let mut b_padded = vec![Dim::Static(1); max_len - b.len()];
        b_padded.extend_from_slice(b);

        for (da, db) in a_padded.iter().zip(b_padded.iter()) {
            if matches!(da, Dim::Static(1)) || matches!(db, Dim::Static(1)) {
                continue;
            }
            if !self.unifier.unify(da, db) {
                let sa = match da { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
                let sb = match db { Dim::Static(v) => v.to_string(), Dim::Symbolic(s) => s.clone(), _ => "?".into() };
                self.result.add_error(shape_mismatch_error(span.clone(), &sa, &sb, "element-wise operation"));
            }
        }
    }

    fn expr_to_lvalue_string(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Self_(_) => Some("self".to_string()),
            Expr::Ident(name, _) => Some(name.clone()),
            Expr::Dot(d) => {
                let base = self.expr_to_lvalue_string(&d.obj)?;
                Some(format!("{}.{}", base, d.field))
            }
            _ => None,
        }
    }

    fn eval_const_int(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::IntLit(v, _) => Some(*v),
            Expr::Ident(name, _) => self.symbols.get_const_int(name),
            Expr::UnaryOp(u) => {
                let val = self.eval_const_int(&u.operand)?;
                match u.op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Not => Some(if val == 0 { 1 } else { 0 }),
                }
            }
            Expr::BinOp(b) => {
                let left = self.eval_const_int(&b.left)?;
                let right = self.eval_const_int(&b.right)?;
                match b.op {
                    BinOp::Add => left.checked_add(right),
                    BinOp::Sub => left.checked_sub(right),
                    BinOp::Mul => left.checked_mul(right),
                    BinOp::Div => if right != 0 { left.checked_div(right) } else { None },
                    BinOp::Mod => if right != 0 { left.checked_rem(right) } else { None },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn infer_fn_call(&mut self, c: &FnCallExpr) -> NType {
        // Propagate mutations from callee effect clauses to caller scope
        if let Expr::Dot(ref d) = c.callee {
            let obj_ty = self.infer_expr(&d.obj);
            if let NType::Model(ref model_name, _, _) = obj_ty {
                let key = format!("{}.{}", model_name, d.field);
                if let Some(info) = self.effects_map.get(&key).cloned() {
                    for eff in &info.effects {
                        if eff.kind == "Mut" {
                            if eff.target.as_deref() == Some("self") || eff.target.is_none() {
                                if let Some(obj_str) = self.expr_to_lvalue_string(&d.obj) {
                                    self.symbols.record_mutation(&obj_str);
                                }
                            } else if let Some(ref target) = eff.target {
                                if let Some(idx) = info.param_names.iter().position(|p| p == target) {
                                    if idx > 0 && idx - 1 < c.args.len() {
                                        if let Some(arg_str) = self.expr_to_lvalue_string(&c.args[idx - 1].value) {
                                            self.symbols.record_mutation(&arg_str);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if let Expr::Ident(ref name, _) = c.callee {
            if let Some(info) = self.effects_map.get(name).cloned() {
                for eff in &info.effects {
                    if eff.kind == "Mut" {
                        if let Some(ref target) = eff.target {
                            if let Some(idx) = info.param_names.iter().position(|p| p == target) {
                                if idx < c.args.len() {
                                    if let Some(arg_str) = self.expr_to_lvalue_string(&c.args[idx].value) {
                                        self.symbols.record_mutation(&arg_str);
                                    }
                                }
                            }
                        } else {
                            for arg in &c.args {
                                if let Some(arg_str) = self.expr_to_lvalue_string(&arg.value) {
                                    self.symbols.record_mutation(&arg_str);
                                }
                            }
                        }
                    }
                }
            }
        }

        let callee_ty = self.infer_expr(&c.callee);
        if let Expr::Dot(ref d) = c.callee {
            let obj_ty = self.infer_expr(&d.obj);
            if let NType::Temporal(ref inner, ref spec) = obj_ty {
                let current_offset = match spec {
                    TemporalSpec::Offset(n) => *n,
                    TemporalSpec::Direction(dir) => if dir == "past_to_future" { 0 } else { 1 },
                };
                if d.field == "before" {
                    let k = if let Some(arg) = c.args.first() {
                        if let Some(val) = self.eval_const_int(&arg.value) {
                            val
                        } else {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                "Temporal method '.before()' requires a compile-time constant integer offset, but found dynamic expression",
                                arg.value.span().clone(),
                            ));
                            -1_000_000
                        }
                    } else {
                        1
                    };
                    // R2 fix: compose with receiver's offset, never reset
                    return NType::Temporal(inner.clone(), TemporalSpec::Offset(current_offset - k.abs()));
                } else if d.field == "after" {
                    let k = if let Some(arg) = c.args.first() {
                        if let Some(val) = self.eval_const_int(&arg.value) {
                            val
                        } else {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                "Temporal method '.after()' requires a compile-time constant integer offset, but found dynamic expression",
                                arg.value.span().clone(),
                            ));
                            1_000_000
                        }
                    } else {
                        1
                    };
                    return NType::Temporal(inner.clone(), TemporalSpec::Offset(current_offset + k.abs().max(1)));
                } else if d.field == "shift" || d.field == "lag" || d.field == "lead" {
                    if c.args.is_empty() {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Temporal method '.{}()' requires an integer offset argument", d.field),
                            c.span.clone(),
                        ));
                        return NType::Temporal(inner.clone(), TemporalSpec::Offset(current_offset));
                    }
                    let arg = &c.args[0];
                    if let Some(k) = self.eval_const_int(&arg.value) {
                        let new_offset = match d.field.as_str() {
                            "shift" => current_offset + k,
                            "lag" => current_offset - k,
                            "lead" => current_offset + k,
                            _ => current_offset,
                        };
                        return NType::Temporal(inner.clone(), TemporalSpec::Offset(new_offset));
                    } else {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Temporal method '.{}()' requires a compile-time constant integer offset, but found dynamic expression", d.field),
                            arg.value.span().clone(),
                        ));
                        return NType::Temporal(inner.clone(), TemporalSpec::Offset(1_000_000));
                    }
                } else if d.field == "snapshot" {
                    // Formal semantics rule T-Snapshot-Safe: snapshot is only
                    // permitted when the temporal offset k ≤ 0 (past/present data).
                    // Allowing snapshot on k > 0 would declassify future-provenanced
                    // data into a raw type, breaking temporal non-interference.
                    let is_future = match spec {
                        TemporalSpec::Offset(n) => *n > 0,
                        TemporalSpec::Direction(dir) => dir == "future_to_past",
                    };
                    if is_future {
                        let leak_desc = match spec {
                            TemporalSpec::Offset(n) => format!(
                                "snapshot() on Temporal with future offset +{} would declassify future data — use only on past/present offsets (≤ 0)",
                                n
                            ),
                            TemporalSpec::Direction(dir) => format!(
                                "snapshot() on Temporal with direction '{}' would declassify future data — use only on past_to_future data",
                                dir
                            ),
                        };
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TemporalLeak,
                            leak_desc,
                            c.span.clone(),
                        ));
                    }
                    return *inner.clone();
                }
            }
            if let NType::Causal(ref inner, _) = obj_ty {
                if d.field == "extract" || d.field == "value" || d.field == "snapshot" {
                    return *inner.clone();
                }
            }
            if let NType::Uncertain(ref inner) = obj_ty {
                if d.field == "extract" || d.field == "value" || d.field == "snapshot" {
                    return *inner.clone();
                }
            }
            if d.field == "forget" {
                return NType::Base("ForgetCertificate".into());
            }
            if d.field == "observe" {
                return NType::Causal(Box::new(NType::Any), "observed".into());
            }
            if d.field == "intervene" {
                return NType::Causal(Box::new(NType::Any), "intervened".into());
            }
        }
        let callee_name = match &c.callee {
            Expr::Ident(ref name, _) => Some(name.as_str()),
            _ => None,
        };
        if callee_name == Some("Uncertain") || callee_name == Some("UNCERTAIN") {
            let inner = c.args.first().map(|a| self.infer_expr(&a.value)).unwrap_or(NType::Any);
            return NType::Uncertain(Box::new(inner));
        }
        let is_variadic_shape_creator = match callee_name {
            Some("zeros") | Some("glorot") | Some("ones") | Some("randn") => true,
            _ => false,
        };
        let is_variadic_optimizer_or_builtin = match callee_name {
            Some("sgd") | Some("adam") | Some("forget") => true,
            _ => false,
        };

        let mut is_method_call = false;
        if let Expr::Dot(ref d) = c.callee {
            let obj_ty = self.infer_expr(&d.obj);
            if let NType::Model(_, _, ref methods) = obj_ty {
                if methods.contains_key(&d.field) {
                    is_method_call = true;
                }
            }
        }

        match callee_ty {
            NType::Fn_(ref params, ref ret, _) => {
                if is_variadic_shape_creator {
                    // Variadic shape creator: all arguments should be integers, lists, or tuples
                    for (i, arg) in c.args.iter().enumerate() {
                        let arg_ty = self.infer_expr(&arg.value);
                        if !types_compatible(&NType::Base("Int".into()), &arg_ty) && !matches!(arg_ty, NType::List(_) | NType::Tuple(_) | NType::Any) {
                            self.result.add_error(NeuronError::new(
                                ErrorCode::TypeMismatch,
                                format!("Argument {} of shape creator must be an integer, got {}", i + 1, arg_ty.display()),
                                c.span.clone(),
                            ));
                        }
                    }
                } else if is_variadic_optimizer_or_builtin {
                    for arg in &c.args {
                        self.infer_expr(&arg.value);
                    }
                } else {
                    let expected_args_len = if is_method_call { params.len().saturating_sub(1) } else { params.len() };
                    if c.args.len() != expected_args_len {
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TypeMismatch,
                            format!("Function call expected {} arguments but got {}", expected_args_len, c.args.len()),
                            c.span.clone(),
                        ));
                    } else {
                        let params_to_check = if is_method_call { &params[1..] } else { &params[..] };
                        for (i, (param_ty, arg)) in params_to_check.iter().zip(c.args.iter()).enumerate() {
                            let arg_ty = self.infer_expr(&arg.value);
                            if !types_compatible(param_ty, &arg_ty) {
                                self.result.add_error(NeuronError::new(
                                    ErrorCode::TypeMismatch,
                                    format!("Argument {} type mismatch: expected {} but got {}", i + 1, param_ty.display(), arg_ty.display()),
                                    c.span.clone(),
                                ).with_expected(&param_ty.display()).with_actual(&arg_ty.display()));
                            }
                            
                            // Temporal direction and offset validation on calls
                            if let (NType::Temporal(_, ref expected_spec), NType::Temporal(_, ref found_spec)) = (param_ty, &arg_ty) {
                                match (expected_spec, found_spec) {
                                    (TemporalSpec::Direction(exp_dir), TemporalSpec::Direction(found_dir)) => {
                                        if exp_dir == "past_to_future" && found_dir == "future_to_past" {
                                            self.result.add_error(temporal_leak_error(
                                                c.span.clone(),
                                                found_dir,
                                                exp_dir,
                                            ));
                                        }
                                    }
                                    (TemporalSpec::Direction(exp_dir), TemporalSpec::Offset(found_offset)) => {
                                        if exp_dir == "past_to_future" && *found_offset > 0 {
                                            self.result.add_error(temporal_offset_leak_error(
                                                c.span.clone(),
                                                *found_offset,
                                                0,
                                            ));
                                        }
                                    }
                                    (TemporalSpec::Offset(expected_max), TemporalSpec::Offset(found_offset)) => {
                                        if *expected_max <= 0 && *found_offset > *expected_max {
                                            self.result.add_error(temporal_offset_leak_error(
                                                c.span.clone(),
                                                *found_offset,
                                                *expected_max,
                                            ));
                                        } else if *expected_max > 0 && *found_offset != *expected_max {
                                            self.result.add_error(NeuronError::new(
                                                ErrorCode::TypeMismatch,
                                                format!("Temporal horizon mismatch: expected Temporal[..., +{}] but got Temporal[..., {}]", expected_max, found_offset),
                                                c.span.clone(),
                                            ).with_expected(&format!("Temporal[..., +{}]", expected_max)).with_actual(&format!("Temporal[..., {}]", found_offset)));
                                        }
                                    }
                                    (TemporalSpec::Offset(expected_max), TemporalSpec::Direction(found_dir)) => {
                                        if *expected_max <= 0 && found_dir == "future_to_past" {
                                            self.result.add_error(temporal_leak_error(
                                                c.span.clone(),
                                                found_dir,
                                                &format!("offset <= {}", expected_max),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                *ret.clone()
            }
            NType::Model(_, _, _) => callee_ty, // Constructor returns the model type
            _ => NType::Any,
        }
    }

    fn infer_dot(&mut self, d: &DotExpr) -> NType {
        let obj_ty = self.infer_expr(&d.obj);

        // Temporal direction and offset checking
        if let NType::Temporal(ref inner, ref spec) = obj_ty {
            match d.field.as_str() {
                "before" => {
                    let new_spec = match spec {
                        TemporalSpec::Offset(n) => TemporalSpec::Offset(n - 1),
                        TemporalSpec::Direction(_) => TemporalSpec::Direction("past_to_future".into()),
                    };
                    return NType::Temporal(inner.clone(), new_spec);
                }
                "after" => {
                    let new_spec = match spec {
                        TemporalSpec::Offset(n) => TemporalSpec::Offset(n + 1),
                        TemporalSpec::Direction(dir) => {
                            let nd = if dir == "past_to_future" { "future_to_past" } else { "past_to_future" };
                            TemporalSpec::Direction(nd.into())
                        }
                    };
                    return NType::Temporal(inner.clone(), new_spec);
                }
                "shift" | "lag" | "lead" => {
                    return NType::Fn_(vec![NType::Base("Int".into())], Box::new(obj_ty.clone()), None);
                }
                "snapshot" => {
                    // T-Snapshot-Safe: reject snapshot on future-provenanced data
                    let is_future = match spec {
                        TemporalSpec::Offset(n) => *n > 0,
                        TemporalSpec::Direction(dir) => dir == "future_to_past",
                    };
                    if is_future {
                        let leak_desc = match spec {
                            TemporalSpec::Offset(n) => format!(
                                "snapshot() on Temporal with future offset +{} would declassify future data — use only on past/present offsets (≤ 0)",
                                n
                            ),
                            TemporalSpec::Direction(dir) => format!(
                                "snapshot() on Temporal with direction '{}' would declassify future data — use only on past_to_future data",
                                dir
                            ),
                        };
                        // Use a zero-span since DotExpr doesn't carry its own span easily;
                        // the error message is self-descriptive.
                        self.result.add_error(NeuronError::new(
                            ErrorCode::TemporalLeak,
                            leak_desc,
                            d.obj.span().clone(),
                        ));
                    }
                    return *inner.clone();
                }
                _ => {}
            }
        }

        // Causal field/method access
        if let NType::Causal(ref inner, _) = obj_ty {
            if d.field == "extract" || d.field == "value" || d.field == "snapshot" {
                return *inner.clone();
            }
        }

        // Uncertain field access
        if let NType::Uncertain(ref inner) = obj_ty {
            match d.field.as_str() {
                "value" | "extract" | "snapshot" => return *inner.clone(),
                "confidence" => return NType::Base("Float".into()),
                "std" => return NType::Base("Float".into()),
                "bounds" => return NType::Tuple(vec![NType::Base("Float".into()), NType::Base("Float".into())]),
                _ => {}
            }
        }

        // CausalModel methods
        if let NType::CausalModel(_, _) = obj_ty {
            match d.field.as_str() {
                "observe" => return NType::Causal(Box::new(NType::Any), "observed".into()),
                "intervene" => return NType::Causal(Box::new(NType::Any), "intervened".into()),
                _ => {}
            }
        }

        // Model field/method lookup
        if let NType::Model(_, ref fields, ref methods) = obj_ty {
            if let Some(ty) = fields.get(&d.field) { return ty.clone(); }
            if let Some(ty) = methods.get(&d.field) { return ty.clone(); }
        }

        NType::Any
    }
}
