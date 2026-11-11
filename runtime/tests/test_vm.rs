use std::fs;
use neuron_compiler::compile;
use neuron_runtime::vm::VM;

#[test]
fn test_integration_vm_simple_shapes() {
    let src = fs::read_to_string("../examples/simple_shapes.nr")
        .or_else(|_| fs::read_to_string("examples/simple_shapes.nr"))
        .expect("Cannot read examples/simple_shapes.nr");
        
    let out = compile(&src, "simple_shapes.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);
    
    let result = vm.run_main();
    assert!(result.is_ok(), "Running simple_shapes.nr failed: {:?}", result.err());
}

#[test]
fn test_integration_vm_transformer_compiles() {
    let src = fs::read_to_string("../examples/transformer.nr")
        .or_else(|_| fs::read_to_string("examples/transformer.nr"))
        .expect("Cannot read examples/transformer.nr");
        
    let out = compile(&src, "transformer.nr");
    assert!(out.is_ok(), "Compiling transformer.nr failed: {:?}", out.err());
}

#[test]
fn test_integration_vm_agi_agent_compiles() {
    let src = fs::read_to_string("../examples/agi_agent.nr")
        .or_else(|_| fs::read_to_string("examples/agi_agent.nr"))
        .expect("Cannot read examples/agi_agent.nr");
        
    let out = compile(&src, "agi_agent.nr");
    assert!(out.is_ok(), "Compiling agi_agent.nr failed: {:?}", out.err());
}

#[test]
fn test_integration_vm_disconnected_gradient() {
    let src = r#"
fn main():
    let a: Tensor[1] = zeros(1) + 2.0
    let b: Tensor[1] = zeros(1) + 5.0
    let c = a * 3.0
    let g = grad(c, wrt=b)
    print(g)
"#;
    let out = compile(src, "test_grad.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);
    let result = vm.run_main();
    assert!(result.is_ok(), "Running disconnected gradient test failed: {:?}", result.err());
}

#[test]
fn test_integration_vm_tensor_indexing_and_slicing() {
    let src = r#"
fn main():
    let t = ones(3, 4)
    let row0 = t[0]
    let elem = t[0, 1]
    let s_rows = t[0:2]
    let s_cols = t[:, 1:3]
    let s_both = t[0:2, 1:3]
    let s_neg = t[0:-1]
    let v = ones(6)
    let v_slice = v[1:4]
    let l = [10, 20, 30, 40]
    let l_sub = l[1:3]
"#;
    let out = compile(src, "test_slice.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&out.ir);
    let result = vm.run_main();
    assert!(result.is_ok(), "Running indexing and slicing test failed: {:?}", result.err());
}
