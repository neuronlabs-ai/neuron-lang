/// Native In-Memory Cranelift JIT Engine for NEURON
///
/// Compiles NEURON SSA BasicBlocks directly into executable machine code pages
/// in memory in single-digit milliseconds, eliminating external Cargo subprocesses.

use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, InstBuilder, types, condcodes::IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use neuron_compiler::ir::{IRFunction, Terminator, IROp, IRConst, BlockId, ValueId};
use std::collections::HashMap;

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

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
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

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Map NEURON BlockId -> Cranelift Block
        let mut cl_blocks: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for block in &func.blocks {
            cl_blocks.insert(block.id, builder.create_block());
        }

        let entry_cl = cl_blocks[&func.entry];
        builder.switch_to_block(entry_cl);

        // Map ValueId -> Cranelift Value
        let mut val_map: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();

        for block in &func.blocks {
            let cl_block = cl_blocks[&block.id];
            if builder.current_block() != Some(cl_block) {
                builder.switch_to_block(cl_block);
            }

            for node in &block.instructions {
                let cl_val = match &node.op {
                    IROp::Const(IRConst::Int(v)) => {
                        builder.ins().iconst(types::I64, *v)
                    }
                    IROp::Const(IRConst::Float(f)) => {
                        builder.ins().f64const(*f);
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
                    _ => {
                        // Default to 0 for unhandled ops in pure scalar prototype
                        builder.ins().iconst(types::I64, 0)
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
                    builder.ins().brif(c, cl_blocks[true_block], &[], cl_blocks[false_block], &[]);
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

        // Seal all blocks
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
}
