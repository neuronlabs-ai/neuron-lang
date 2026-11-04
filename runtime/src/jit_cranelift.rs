/// Native In-Memory Cranelift JIT Engine for NEURON
///
/// Compiles NEURON SSA BasicBlocks directly into executable machine code pages
/// in memory in single-digit milliseconds, eliminating external Cargo subprocesses.

use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, InstBuilder, types, condcodes::IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use neuron_compiler::ir::{IRFunction, Terminator, IROp, IRConst, BlockId, ValueId};
use std::collections::HashMap;


extern "C" fn neuron_jit_print_i64(val: i64) {
    println!("{}", val);
}

extern "C" fn neuron_jit_print_inline_i64(val: i64) {
    print!("{}", val);
}

pub struct CraneliftEngine {
    builder_context: FunctionBuilderContext,
    ctx: Context,
    module: JITModule,
}

impl CraneliftEngine {
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;
        flag_builder.set("is_pic", "false").map_err(|e| e.to_string())?;
        flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
        let isa_flags = settings::Flags::new(flag_builder);

        let isa = cranelift_native::builder()
            .map_err(|msg| format!("Host machine is not supported: {}", msg))?
            .finish(isa_flags)
            .map_err(|e| e.to_string())?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        builder.symbol("neuron_jit_print_i64", neuron_jit_print_i64 as *const u8);
        builder.symbol("neuron_jit_print_inline_i64", neuron_jit_print_inline_i64 as *const u8);
        let module = JITModule::new(builder);

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
        })
    }

    /// Compiles a simple scalar function to test Cranelift execution:
    /// fn add(a: i64, b: i64) -> i64
    pub fn compile_test_add(&mut self) -> Result<fn(i64, i64) -> i64, String> {
        self.ctx.func.signature.params.push(AbiParam::new(types::I64));
        self.ctx.func.signature.params.push(AbiParam::new(types::I64));
        self.ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let a = builder.block_params(entry_block)[0];
        let b = builder.block_params(entry_block)[1];
        let sum = builder.ins().iadd(a, b);
        builder.ins().return_(&[sum]);
        let target_config = self.module.target_config();
        builder.finalize(target_config);

        let id = self.module.declare_function("test_add", Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;
        self.module.define_function(id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = self.module.get_finalized_function(id);
        let func: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        Ok(func)
    }

    /// Compiles a loop function (Fibonacci) to verify control flow and basic block jumps:
    /// fn fib(n: i64) -> i64
    pub fn compile_test_fib(&mut self) -> Result<fn(i64) -> i64, String> {
        self.ctx.func.signature.params.push(AbiParam::new(types::I64));
        self.ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        let entry = builder.create_block();
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let exit = builder.create_block();

        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let n = builder.block_params(entry)[0];

        let var_a = builder.declare_var(types::I64);
        let var_b = builder.declare_var(types::I64);
        let var_i = builder.declare_var(types::I64);

        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        builder.def_var(var_a, zero);
        builder.def_var(var_b, one);
        builder.def_var(var_i, zero);

        builder.ins().jump(loop_header, &[]);
        builder.seal_block(entry);

        builder.switch_to_block(loop_header);
        let i = builder.use_var(var_i);
        let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, n);
        builder.ins().brif(cmp, loop_body, &[], exit, &[]);

        builder.switch_to_block(loop_body);
        let cur_a = builder.use_var(var_a);
        let cur_b = builder.use_var(var_b);
        let next_b = builder.ins().iadd(cur_a, cur_b);
        let next_i = builder.ins().iadd(i, one);
        builder.def_var(var_a, cur_b);
        builder.def_var(var_b, next_b);
        builder.def_var(var_i, next_i);
        builder.ins().jump(loop_header, &[]);
        builder.seal_block(loop_body);

        builder.switch_to_block(exit);
        let final_a = builder.use_var(var_a);
        builder.ins().return_(&[final_a]);
        builder.seal_block(exit);
        builder.seal_block(loop_header);

        let target_config = self.module.target_config();
        builder.finalize(target_config);

        let id = self.module.declare_function("test_fib", Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;
        self.module.define_function(id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = self.module.get_finalized_function(id);
        let func: fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        Ok(func)
    }

    /// Compiles a NEURON IRFunction consisting of integer/scalar operations directly to machine code
    pub fn compile_scalar_function(&mut self, func: &IRFunction) -> Result<fn() -> i64, String> {
        self.ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut print_sig = self.module.make_signature();
        print_sig.params.push(AbiParam::new(types::I64));
        let print_fn_id = self.module.declare_function("neuron_jit_print_i64", Linkage::Import, &print_sig)
            .map_err(|e| e.to_string())?;

        let mut print_inline_sig = self.module.make_signature();
        print_inline_sig.params.push(AbiParam::new(types::I64));
        let print_inline_fn_id = self.module.declare_function("neuron_jit_print_inline_i64", Linkage::Import, &print_inline_sig)
            .map_err(|e| e.to_string())?;

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Scan all instructions across all blocks to collect all variable names
        let mut var_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for block in &func.blocks {
            for node in &block.instructions {
                match &node.op {
                    IROp::Store { name } | IROp::Load { name } => {
                        var_names.insert(name.clone());
                    }
                    _ => {}
                }
            }
        }

        // Map NEURON BlockId -> Cranelift Block
        let mut cl_blocks: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for block in &func.blocks {
            cl_blocks.insert(block.id, builder.create_block());
        }

        // Create a dedicated prologue block to declare and initialize variables
        let prologue_block = builder.create_block();
        builder.append_block_params_for_function_params(prologue_block);
        builder.switch_to_block(prologue_block);

        let mut var_map: HashMap<String, Variable> = HashMap::new();
        let zero = builder.ins().iconst(types::I64, 0);
        for name in &var_names {
            let var = builder.declare_var(types::I64);
            builder.def_var(var, zero);
            var_map.insert(name.clone(), var);
        }

        let entry_cl = cl_blocks.get(&func.entry)
            .copied()
            .ok_or_else(|| format!("Entry block {} not found in IRFunction", func.entry))?;
        builder.ins().jump(entry_cl, &[]);
        builder.seal_block(prologue_block);

        // Map ValueId -> Cranelift Value
        let mut val_map: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();

        for block in &func.blocks {
            let cl_block = cl_blocks[&block.id];
            builder.switch_to_block(cl_block);

            for node in &block.instructions {
                let cl_val = match &node.op {
                    IROp::Const(IRConst::Int(v)) => {
                        builder.ins().iconst(types::I64, *v)
                    }
                    IROp::Const(IRConst::Float(f)) => {
                        builder.ins().iconst(types::I64, *f as i64)
                    }
                    IROp::Const(IRConst::Bool(b)) => {
                        builder.ins().iconst(types::I64, if *b { 1 } else { 0 })
                    }
                    IROp::Add => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().iadd(a, b)
                    }
                    IROp::Sub => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().isub(a, b)
                    }
                    IROp::Mul => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().imul(a, b)
                    }
                    IROp::Div => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().sdiv(a, b)
                    }
                    IROp::Mod => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().srem(a, b)
                    }
                    IROp::Neg => {
                        let a = val_map[&node.inputs[0]];
                        builder.ins().ineg(a)
                    }
                    IROp::Lt => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Lte => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Gt => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Gte => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Eq => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::Equal, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Neq => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        let cmp = builder.ins().icmp(IntCC::NotEqual, a, b);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::And => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().band(a, b)
                    }
                    IROp::Or => {
                        let a = val_map[&node.inputs[0]];
                        let b = val_map[&node.inputs[1]];
                        builder.ins().bor(a, b)
                    }
                    IROp::Not => {
                        let a = val_map[&node.inputs[0]];
                        let zero = builder.ins().iconst(types::I64, 0);
                        let cmp = builder.ins().icmp(IntCC::Equal, a, zero);
                        builder.ins().uextend(types::I64, cmp)
                    }
                    IROp::Store { name } => {
                        let val = val_map[&node.inputs[0]];
                        let var = var_map[name];
                        builder.def_var(var, val);
                        val
                    }
                    IROp::Load { name } => {
                        let var = match var_map.get(name) {
                            Some(v) => *v,
                            None => return Err(format!("Load of undeclared variable '{}'", name)),
                        };
                        builder.use_var(var)
                    }
                    IROp::Print => {
                        let val = val_map[&node.inputs[0]];
                        let local_callee = self.module.declare_func_in_func(print_fn_id, builder.func);
                        builder.ins().call(local_callee, &[val]);
                        builder.ins().iconst(types::I64, 0)
                    }
                    IROp::PrintInline => {
                        let val = val_map[&node.inputs[0]];
                        let local_callee = self.module.declare_func_in_func(print_inline_fn_id, builder.func);
                        builder.ins().call(local_callee, &[val]);
                        builder.ins().iconst(types::I64, 0)
                    }
                    IROp::Nop => {
                        builder.ins().iconst(types::I64, 0)
                    }
                    _ => {
                        return Err(format!("Unsupported IROp in scalar Cranelift JIT: {:?}", node.op));
                    }
                };
                val_map.insert(node.id, cl_val);
            }

            match &block.terminator {
                Terminator::Jump(target) => {
                    builder.ins().jump(cl_blocks[target], &[]);
                }
                Terminator::Branch { cond, true_block, false_block } => {
                    let c = val_map[cond];
                    let zero = builder.ins().iconst(types::I64, 0);
                    let cond_bool = builder.ins().icmp(IntCC::NotEqual, c, zero);
                    builder.ins().brif(cond_bool, cl_blocks[true_block], &[], cl_blocks[false_block], &[]);
                }
                Terminator::Return(Some(vid)) => {
                    let ret = val_map[vid];
                    builder.ins().return_(&[ret]);
                }
                Terminator::Return(None) => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().return_(&[zero]);
                }
            }
        }

        // Seal all basic blocks
        for block in &func.blocks {
            builder.seal_block(cl_blocks[&block.id]);
        }

        let target_config = self.module.target_config();
        builder.finalize(target_config);

        let func_name = format!("neuron_{}", func.name);
        let id = self.module.declare_function(&func_name, Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| e.to_string())?;
        self.module.define_function(id, &mut self.ctx)
            .map_err(|e| e.to_string())?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = self.module.get_finalized_function(id);
        let native_fn: fn() -> i64 = unsafe { std::mem::transmute(code) };
        Ok(native_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuron_compiler::ir::{BasicBlock, IRNode};

    #[test]
    fn test_cranelift_jit_in_memory() {
        let mut engine = CraneliftEngine::new().expect("Failed to initialize Cranelift engine");
        let add = engine.compile_test_add().expect("Failed to compile Cranelift test_add function");
        let result = add(40, 2);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_cranelift_fibonacci_loop() {
        let mut engine = CraneliftEngine::new().expect("Failed to initialize Cranelift engine");
        let fib = engine.compile_test_fib().expect("Failed to compile Cranelift fib function");
        // fib(0) = 0, fib(1) = 1, fib(2) = 1, fib(3) = 2, fib(4) = 3, fib(5) = 5, fib(10) = 55
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(5), 5);
        assert_eq!(fib(10), 55);
    }

    #[test]
    fn test_cranelift_compile_ir_scalar() {
        let mut engine = CraneliftEngine::new().expect("Engine");
        // Construct an IRFunction that computes:
        // x = 10
        // y = 20
        // z = x * y + 5  (10 * 20 + 5 = 205)
        // return z
        let func = IRFunction {
            name: "calc".into(),
            params: vec![],
            return_type: neuron_compiler::ir::IRType::I64,
            blocks: vec![
                BasicBlock {
                    id: 0,
                    instructions: vec![
                        IRNode {
                            id: 0,
                            op: IROp::Const(IRConst::Int(10)),
                            inputs: vec![],
                            output_type: neuron_compiler::ir::IRType::I64,
                            output_shape: vec![],
                            grad_fn: None,
                            device: neuron_compiler::ir::DeviceTarget::CPU,
                            temporal_dir: None,
                            effects: vec![],
                        },
                        IRNode {
                            id: 1,
                            op: IROp::Const(IRConst::Int(20)),
                            inputs: vec![],
                            output_type: neuron_compiler::ir::IRType::I64,
                            output_shape: vec![],
                            grad_fn: None,
                            device: neuron_compiler::ir::DeviceTarget::CPU,
                            temporal_dir: None,
                            effects: vec![],
                        },
                        IRNode {
                            id: 2,
                            op: IROp::Mul,
                            inputs: vec![0, 1],
                            output_type: neuron_compiler::ir::IRType::I64,
                            output_shape: vec![],
                            grad_fn: None,
                            device: neuron_compiler::ir::DeviceTarget::CPU,
                            temporal_dir: None,
                            effects: vec![],
                        },
                        IRNode {
                            id: 3,
                            op: IROp::Const(IRConst::Int(5)),
                            inputs: vec![],
                            output_type: neuron_compiler::ir::IRType::I64,
                            output_shape: vec![],
                            grad_fn: None,
                            device: neuron_compiler::ir::DeviceTarget::CPU,
                            temporal_dir: None,
                            effects: vec![],
                        },
                        IRNode {
                            id: 4,
                            op: IROp::Add,
                            inputs: vec![2, 3],
                            output_type: neuron_compiler::ir::IRType::I64,
                            output_shape: vec![],
                            grad_fn: None,
                            device: neuron_compiler::ir::DeviceTarget::CPU,
                            temporal_dir: None,
                            effects: vec![],
                        },
                    ],
                    terminator: Terminator::Return(Some(4)),
                }
            ],
            entry: 0,
            is_differentiable: false,
            effects: vec![],
        };

        let calc_fn = engine.compile_scalar_function(&func).expect("Compile failed");
        let result = calc_fn();
        assert_eq!(result, 205);
    }

    #[test]
    fn test_cranelift_while_loop_with_variables() {
        let source = r#"
fn main():
  let sum = 0
  let i = 0
  while i < 100:
    let sum = sum + i
    let i = i + 1
  return sum
"#;
        let output = neuron_compiler::compile(source, "test_loop.nr").expect("compile");
        let main_fn = output.ir.functions.iter().find(|f| f.name == "main").expect("main fn");
        let mut engine = CraneliftEngine::new().expect("engine");
        let native_fn = engine.compile_scalar_function(main_fn).expect("compile_scalar_function");
        let result = native_fn();
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_cranelift_bench_loop_full() {
        let source = std::fs::read_to_string("../examples/bench_loop.nr").expect("read bench_loop.nr");
        let output = neuron_compiler::compile(&source, "bench_loop.nr").expect("compile");
        let main_fn = output.ir.functions.iter().find(|f| f.name == "main").expect("main fn");
        let mut engine = CraneliftEngine::new().expect("engine");
        let native_fn = engine.compile_scalar_function(main_fn).expect("compile_scalar_function");
        let start = std::time::Instant::now();
        let _ = native_fn();
        let elapsed = start.elapsed();
        println!("100k iterations native machine code executed in: {:?}", elapsed);
        assert!(elapsed.as_millis() < 50, "Expected execution to be < 50ms, took {:?}", elapsed);
    }
}
