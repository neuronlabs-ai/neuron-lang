use neuron_compiler::compile;
use neuron_compiler::cuda_codegen::{find_fused_groups, generate_cuda_kernels};
use neuron_compiler::ir::{IRProgram, IRFunction, IRNode, IROp, IRType, DeviceTarget, BasicBlock, Terminator, IRConst};
use neuron_runtime::vm::VM;
use neuron_runtime::device::set_simulate_cuda;

#[test]
fn test_cuda_operator_fusion_codegen() {
    let src = r#"
fn test_fused_math(x: Tensor[2, 2]) -> Tensor[2, 2]:
  let y = -x
  let z = gelu(y)
  let w = z + 1.5
  return w
"#;

    let compile_res = compile(src, "test_gpu_input.nr").unwrap();
    let func = compile_res.ir.functions.iter().find(|f| f.name == "test_fused_math").unwrap();

    // 1. Verify fused groups are detected
    let groups = find_fused_groups(func, &[]);
    assert!(!groups.is_empty(), "Expected to find at least one fused operator group");
    
    // 2. Generate CUDA kernels
    let kernels = generate_cuda_kernels(func, &[]);
    assert_eq!(kernels.len(), 1, "Expected exactly 1 fused kernel");
    
    let kernel = &kernels[0];
    println!("Generated CUDA Kernel Code:\n{}", kernel.code);
    
    // 3. Verify kernel structural syntax and function arguments
    assert!(kernel.name.contains("test_fused_math"));
    assert!(kernel.code.contains("extern \"C\" __global__ void"));
    assert!(kernel.code.contains("double* v")); // Output pointer
    assert!(kernel.code.contains("const double* v")); // Input pointer
    assert!(kernel.code.contains("int n")); // Size bounds check
    
    // 4. Verify element-wise operations mapped to C++ math built-ins
    assert!(kernel.code.contains("-v")); // Negation
    assert!(kernel.code.contains("erf")); // GeLU erf call
    assert!(kernel.code.contains("1.500000")); // Float constant addition
}

#[test]
fn test_gpu_device_fallback_routing() {
    // 1. Create a dummy function with an Auto device node
    let mut prog = IRProgram::new();
    let mut func = IRFunction::new("main");
    
    let block = BasicBlock {
        id: 0,
        instructions: vec![
            IRNode {
                id: 0,
                op: IROp::Const(IRConst::Float(3.14)),
                inputs: vec![],
                output_type: IRType::F64,
                output_shape: vec![],
                grad_fn: None,
                device: DeviceTarget::CUDA(0), // Set targeting to GPU CUDA 0
                temporal_dir: None,
                effects: vec![],
            }
        ],
        terminator: Terminator::Return(Some(0)),
    };
    
    func.blocks.push(block);
    func.entry = 0;
    prog.functions.push(func);
    
    // 2. Run with CUDA unavailable (Standard CPU Fallback or real GPU execution)
    std::env::remove_var("NEURON_SIMULATE_CUDA");
    set_simulate_cuda(false);
    let mut vm = VM::new();
    vm.load(&prog);
    let res = vm.run_main().unwrap();
    assert_eq!(res.as_float(), 3.14);
    if neuron_runtime::device::get_cuda_context().is_some() {
        assert!(vm.effect_log.contains(&"cuda_exec_0".to_string()));
        assert!(!vm.effect_log.contains(&"fallback_to_cpu_cuda_0".to_string()));
    } else {
        assert!(vm.effect_log.contains(&"fallback_to_cpu_cuda_0".to_string()));
        assert!(!vm.effect_log.contains(&"cuda_exec_0".to_string()));
    }
    
    // 3. Run with CUDA simulated (Device Routing)
    std::env::set_var("NEURON_SIMULATE_CUDA", "1");
    set_simulate_cuda(true);
    let mut vm_sim = VM::new();
    vm_sim.load(&prog);
    let res_sim = vm_sim.run_main().unwrap();
    assert_eq!(res_sim.as_float(), 3.14);
    assert!(vm_sim.effect_log.contains(&"cuda_exec_0".to_string()));
    assert!(!vm_sim.effect_log.contains(&"fallback_to_cpu_cuda_0".to_string()));
    
    // Cleanup
    std::env::remove_var("NEURON_SIMULATE_CUDA");
    set_simulate_cuda(false);
}

#[test]
fn test_cuda_operator_fusion_execution() {
    let src = r#"
fn test_fused_math(x: Tensor[2, 2]) -> Tensor[2, 2]:
  let y = -x
  let z = gelu(y)
  let w = z + 1.5
  return w

fn main() -> Tensor[2, 2]:
  let x = zeros(2, 2) + 1.0
  let out = test_fused_math(x)
  return out
"#;

    let compile_res = compile(src, "test_gpu_input.nr").unwrap();

    // 1. Run in simulated CUDA mode
    std::env::set_var("NEURON_SIMULATE_CUDA", "1");
    set_simulate_cuda(true);
    let mut vm = VM::new();
    let mut prog = compile_res.ir.clone();
    for func in &mut prog.functions {
        if func.name == "test_fused_math" {
            for block in &mut func.blocks {
                for node in &mut block.instructions {
                    node.device = DeviceTarget::CUDA(0);
                }
            }
        }
    }
    
    vm.load(&prog);
    let res = vm.run_main().unwrap();
    
    let t = res.as_tensor().unwrap();
    assert_eq!(t.shape, vec![2, 2]);
    // x = 1.0 -> y = -1.0 -> z = gelu(-1.0) = -0.158655 -> w = 1.341345
    assert!((t.data[0] - 1.341345).abs() < 1e-4);
    
    // Verify that the fused group was executed on the simulated GPU
    assert!(vm.effect_log.iter().any(|e| e.contains("cuda_exec_0")));
    assert!(!vm.effect_log.iter().any(|e| e.contains("fallback_to_cpu_cuda_0")));

    // 2. Run in standard fallback mode (CPU fallback or real GPU execution)
    std::env::remove_var("NEURON_SIMULATE_CUDA");
    set_simulate_cuda(false);
    let mut vm_cpu = VM::new();
    vm_cpu.load(&prog);
    let res_cpu = vm_cpu.run_main().unwrap();
    let t_cpu = res_cpu.as_tensor().unwrap();
    assert!((t_cpu.data[0] - 1.341345).abs() < 1e-4);
    
    // Verify that fallback was logged on CPU-only machines, or real execution on GPU machines
    if neuron_runtime::device::get_cuda_context().is_some() {
        assert!(vm_cpu.effect_log.iter().any(|e| e.contains("cuda_exec_0")));
        assert!(!vm_cpu.effect_log.iter().any(|e| e.contains("fallback_to_cpu_cuda_0")));
    } else {
        assert!(vm_cpu.effect_log.iter().any(|e| e.contains("fallback_to_cpu_cuda_0")));
        assert!(!vm_cpu.effect_log.iter().any(|e| e.contains("cuda_exec_0")));
    }
    
    // Cleanup
    std::env::remove_var("NEURON_SIMULATE_CUDA");
    set_simulate_cuda(false);
}

#[test]
fn test_cuda_transcendental_fusion_codegen() {
    let src = r#"
fn test_transcendental(x: Tensor[4, 4]) -> Tensor[4, 4]:
  let a = exp(x)
  let b = log(a + 1.0)
  let c = abs(b - 2.0)
  let d = sqrt(c)
  return d
"#;

    let compile_res = compile(src, "test_gpu_trans.nr").unwrap();
    let func = compile_res.ir.functions.iter().find(|f| f.name == "test_transcendental").unwrap();

    let groups = find_fused_groups(func, &[]);
    assert!(!groups.is_empty(), "Expected fused groups for transcendental operations");

    let kernels = generate_cuda_kernels(func, &[]);
    assert_eq!(kernels.len(), 1, "Expected 1 fused transcendental CUDA kernel");

    let k = &kernels[0];
    assert!(k.code.contains("exp("), "Expected exp() in CUDA kernel");
    assert!(k.code.contains("log("), "Expected log() in CUDA kernel");
    assert!(k.code.contains("fabs("), "Expected fabs() in CUDA kernel");
    assert!(k.code.contains("sqrt("), "Expected sqrt() in CUDA kernel");
}

#[test]
fn test_matmul_epilogue_fusion_optimization() {
    use neuron_compiler::optimize::optimize_program;

    let src = r#"
fn linear_relu(x: Tensor[2, 3], w: Tensor[3, 2], b: Tensor[2, 2]) -> Tensor[2, 2]:
  let mm = x @ w
  let bias_added = mm + b
  let activated = relu(bias_added)
  return activated

fn main() -> Tensor[2, 2]:
  let x = ones(2, 3)
  let w = ones(3, 2)
  let b = zeros(2, 2) - 1.0
  let out = linear_relu(x, w, b)
  return out
"#;

    let mut compile_res = compile(src, "test_matmul_fused.nr").unwrap();
    optimize_program(&mut compile_res.ir);

    // Verify that linear_relu has MatMulAddReLU fused op
    let lr_func = compile_res.ir.functions.iter().find(|f| f.name == "linear_relu").unwrap();
    let has_fused_matmul = lr_func.blocks.iter().any(|b| {
        b.instructions.iter().any(|n| matches!(n.op, IROp::MatMulAddReLU))
    });
    assert!(has_fused_matmul, "Expected MatMulAddReLU fused operation in optimized IR");

    // Execute in VM to verify mathematical precision
    // x = [1, 1, 1], w = [1, 1; 1, 1; 1, 1] => mm = [3, 3; 3, 3]
    // b = [-1, -1; -1, -1] => bias_added = [2, 2; 2, 2]
    // relu([2, 2; 2, 2]) = [2, 2; 2, 2]
    let mut vm = VM::new();
    vm.load(&compile_res.ir);
    let res = vm.run_main().unwrap();
    let t = res.as_tensor().unwrap();
    assert_eq!(t.shape, vec![2, 2]);
    for i in 0..t.numel() {
        let val = t.data[i];
        assert!((val - 2.0).abs() < 1e-6, "Expected 2.0, got {}", val);
    }
}

#[test]
fn test_triton_kernel_fusion_codegen() {
    use neuron_compiler::triton_codegen::{
        generate_triton_kernels,
        generate_triton_fused_softmax,
        generate_triton_fused_layernorm,
        generate_triton_fused_attention,
    };

    let src = r#"
fn triton_target(x: Tensor[8, 8]) -> Tensor[8, 8]:
  let y = -x
  let z = relu(y)
  let w = exp(z) + 1.0
  return w
"#;

    let compile_res = compile(src, "test_triton.nr").unwrap();
    let func = compile_res.ir.functions.iter().find(|f| f.name == "triton_target").unwrap();

    let triton_kernels = generate_triton_kernels(func, &[]);
    assert_eq!(triton_kernels.len(), 1, "Expected 1 fused Triton kernel");

    let tk = &triton_kernels[0];
    println!("Generated Triton Kernel Code:\n{}", tk.code);

    // Verify structural Triton elements
    assert!(tk.code.contains("@triton.jit"), "Expected @triton.jit decorator");
    assert!(tk.code.contains("import triton.language as tl"), "Expected tl import");
    assert!(tk.code.contains("tl.program_id(axis=0)"), "Expected program_id");
    assert!(tk.code.contains("tl.arange(0, BLOCK_SIZE)"), "Expected block arange");
    assert!(tk.code.contains("tl.load"), "Expected tl.load");
    assert!(tk.code.contains("tl.store"), "Expected tl.store");
    assert!(tk.code.contains("tl.maximum"), "Expected tl.maximum for ReLU");
    assert!(tk.code.contains("tl.exp"), "Expected tl.exp for Exp");
    assert!(tk.code.contains("def launch_"), "Expected Python launcher function");

    // Test specialized templates
    let softmax_code = generate_triton_fused_softmax("fused_softmax_v1");
    assert!(softmax_code.contains("@triton.jit"));
    assert!(softmax_code.contains("tl.max(row, axis=0)"));
    assert!(softmax_code.contains("tl.sum(numerator, axis=0)"));

    let layernorm_code = generate_triton_fused_layernorm("fused_layernorm_v1");
    assert!(layernorm_code.contains("@triton.jit"));
    assert!(layernorm_code.contains("rstd = 1.0 / tl.sqrt(var + eps)"));

    let flash_attn_code = generate_triton_fused_attention("flash_attention_v2");
    assert!(flash_attn_code.contains("@triton.jit"));
    assert!(flash_attn_code.contains("tl.dot(q, k) * sm_scale"));
    assert!(flash_attn_code.contains("tl.maximum(m_i, tl.max(qk, 1))"));
    assert!(flash_attn_code.contains("acc += tl.dot(p.to(tl.float16), v)"));
}


