//! OpenAI Triton Code Generation Engine for NEURON
//!
//! Generates high-performance Triton `@triton.jit` kernels from fused IR blocks,
//! avoiding intermediate global memory roundtrips and maximizing SRAM reuse.
//! Includes specialized templates for Fused Softmax, Fused LayerNorm, and FlashAttention.

use std::collections::HashSet;
use crate::ir::{IRFunction, IRConst, IROp};
use crate::cuda_codegen::{get_tensor_ids, find_fused_groups};

/// A generated OpenAI Triton kernel representation
#[derive(Debug, Clone)]
pub struct TritonKernel {
    pub name: String,
    pub code: String,
    pub inputs: Vec<usize>,
    pub input_is_tensor: Vec<bool>,
    pub output: usize,
    pub output_shape: Vec<i64>,
}

/// Generates OpenAI Triton Python kernels from fused groups in an IR function
pub fn generate_triton_kernels(func: &IRFunction, all_funcs: &[IRFunction]) -> Vec<TritonKernel> {
    let tensor_ids = get_tensor_ids(func, all_funcs);
    let groups = find_fused_groups(func, all_funcs);
    let mut kernels = Vec::new();

    for (g_idx, group) in groups.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        // Only generate kernels for groups that contain tensor operations
        let has_tensors = group.instructions.iter().any(|node| tensor_ids.contains(&node.id));
        if !has_tensors {
            continue;
        }

        // 1. Identify inputs & outputs of this group
        let mut produced_vals = HashSet::new();
        for node in &group.instructions {
            produced_vals.insert(node.id);
        }

        let mut input_set = HashSet::new();
        for node in &group.instructions {
            for &input in &node.inputs {
                if !produced_vals.contains(&input) {
                    input_set.insert(input);
                }
            }
        }

        let mut inputs: Vec<usize> = input_set.into_iter().collect();
        inputs.sort();

        let terminal_node = group.instructions.last().unwrap();
        let output = terminal_node.id;
        let output_shape = terminal_node.output_shape.clone();

        // 2. Classify inputs as Tensor or Scalar
        let mut input_is_tensor = Vec::new();
        let mut triton_args = Vec::new();
        let mut launch_args = Vec::new();

        triton_args.push(format!("out_ptr_{}", output));
        launch_args.push(format!("out_tensor_{}", output));

        for &val_id in &inputs {
            let is_tensor = tensor_ids.contains(&val_id);
            input_is_tensor.push(is_tensor);

            if is_tensor {
                triton_args.push(format!("in_ptr_{}", val_id));
                launch_args.push(format!("in_tensor_{}", val_id));
            } else {
                triton_args.push(format!("scalar_{}", val_id));
                launch_args.push(format!("val_{}", val_id));
            }
        }

        triton_args.push("n_elements".to_string());
        triton_args.push("BLOCK_SIZE: tl.constexpr".to_string());

        // 3. Generate kernel body
        let kernel_name = format!("triton_fused_{}_{}", func.name, g_idx);
        let mut code = String::new();

        code.push_str("import triton\nimport triton.language as tl\nimport torch\n\n");
        code.push_str("@triton.jit\n");
        code.push_str(&format!("def {}(\n    {},\n):\n", kernel_name, triton_args.join(",\n    ")));
        code.push_str("    pid = tl.program_id(axis=0)\n");
        code.push_str("    block_start = pid * BLOCK_SIZE\n");
        code.push_str("    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n");
        code.push_str("    mask = offsets < n_elements\n\n");

        // Load tensor inputs into SRAM registers
        for &val_id in &inputs {
            if tensor_ids.contains(&val_id) {
                code.push_str(&format!(
                    "    v{} = tl.load(in_ptr_{} + offsets, mask=mask)\n",
                    val_id, val_id
                ));
            }
        }

        code.push_str("\n    # Fused element-wise operations\n");
        for node in &group.instructions {
            let expr = match &node.op {
                IROp::Const(c) => match c {
                    IRConst::Int(v) => format!("{}.0", v),
                    IRConst::Float(v) => format!("{:.6}", v),
                    IRConst::Bool(v) => format!("{}", if *v { "1.0" } else { "0.0" }),
                    _ => "0.0".to_string(),
                },
                IROp::Add => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_triton_operand(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} + {}", in0, in1)
                }
                IROp::Sub => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_triton_operand(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} - {}", in0, in1)
                }
                IROp::Mul => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_triton_operand(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} * {}", in0, in1)
                }
                IROp::Div => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_triton_operand(node.inputs[1], &inputs, &tensor_ids);
                    format!("{} / {}", in0, in1)
                }
                IROp::Pow => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    let in1 = format_triton_operand(node.inputs[1], &inputs, &tensor_ids);
                    format!("tl.exp({} * tl.log({}))", in1, in0)
                }
                IROp::Neg => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("-{}", in0)
                }
                IROp::ReLU => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("tl.maximum({}, 0.0)", in0)
                }
                IROp::GeLU => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("{} * 0.5 * (1.0 + tl.erf({} / 1.41421356))", in0, in0)
                }
                IROp::Sigmoid => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("1.0 / (1.0 + tl.exp(-{}))", in0)
                }
                IROp::Tanh => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("(tl.exp(2.0 * {}) - 1.0) / (tl.exp(2.0 * {}) + 1.0)", in0, in0)
                }
                IROp::Exp => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("tl.exp({})", in0)
                }
                IROp::Log => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("tl.log({})", in0)
                }
                IROp::Abs => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("tl.abs({})", in0)
                }
                IROp::Sqrt => {
                    let in0 = format_triton_operand(node.inputs[0], &inputs, &tensor_ids);
                    format!("tl.sqrt({})", in0)
                }
                _ => "0.0".to_string(),
            };

            code.push_str(&format!("    v{} = {}\n", node.id, expr));
        }

        // Store terminal result to output tensor in HBM
        code.push_str(&format!(
            "\n    tl.store(out_ptr_{} + offsets, v{}, mask=mask)\n\n",
            output, output
        ));

        // 4. Generate Python launcher function
        code.push_str(&format!("def launch_{}(\n    {},\n    block_size: int = 1024,\n):\n", kernel_name, launch_args.join(",\n    ")));
        code.push_str(&format!("    n_elements = out_tensor_{}.numel()\n", output));
        code.push_str("    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),\n    )\n");
        code.push_str(&format!("    {}[grid](\n", kernel_name));
        code.push_str(&format!("        out_tensor_{},\n", output));
        for (idx, &val_id) in inputs.iter().enumerate() {
            if input_is_tensor[idx] {
                code.push_str(&format!("        in_tensor_{},\n", val_id));
            } else {
                code.push_str(&format!("        val_{},\n", val_id));
            }
        }
        code.push_str("        n_elements,\n");
        code.push_str("        BLOCK_SIZE=block_size,\n");
        code.push_str("    )\n");
        code.push_str(&format!("    return out_tensor_{}\n", output));

        kernels.push(TritonKernel {
            name: kernel_name,
            code,
            inputs,
            input_is_tensor,
            output,
            output_shape,
        });
    }

    kernels
}

fn format_triton_operand(id: usize, group_inputs: &[usize], tensor_ids: &HashSet<usize>) -> String {
    if group_inputs.contains(&id) && !tensor_ids.contains(&id) {
        format!("scalar_{}", id)
    } else {
        format!("v{}", id)
    }
}

/// Generates a dedicated high-performance Fused Online Softmax Triton kernel
pub fn generate_triton_fused_softmax(kernel_name: &str) -> String {
    format!(r#"import triton
import triton.language as tl
import torch

@triton.jit
def {kernel_name}(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # Load row from HBM into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Numerically stable online softmax reduction in SRAM
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store back to HBM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def launch_{kernel_name}(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    {kernel_name}[(n_rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
"#)
}

/// Generates a dedicated Fused LayerNorm Triton kernel
pub fn generate_triton_fused_layernorm(kernel_name: &str) -> String {
    format!(r#"import triton
import triton.language as tl
import torch

@triton.jit
def {kernel_name}(
    out_ptr,
    in_ptr,
    weight_ptr,
    bias_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = in_ptr + row_idx * stride_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load inputs, weight, bias
    x = tl.load(row_start + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    # Compute mean and variance in single SRAM pass
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    y = diff * rstd * w + b

    # Store output
    out_row_start = out_ptr + row_idx * stride_row
    tl.store(out_row_start + cols, y, mask=mask)
"#)
}

/// Generates a high-performance Fused FlashAttention Triton kernel (Dao et al. algorithm)
pub fn generate_triton_fused_attention(kernel_name: &str) -> String {
    format!(r#"import triton
import triton.language as tl
import torch

@triton.jit
def {kernel_name}(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Block pointers
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on

    # Initialize SRAM running statistics for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q block into SRAM
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Loop over K and V blocks (eliminating HBM materialization of N_CTX x N_CTX attention matrix)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * sm_scale

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_n * stride_vk, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)
"#)
}
