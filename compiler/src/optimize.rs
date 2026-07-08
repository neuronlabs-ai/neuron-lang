use std::collections::HashMap;
use crate::ir::{IRProgram, IRConst, IROp, ValueId};

pub fn optimize_program(program: &mut IRProgram) {
    for func in &mut program.functions {
        let mut constants: HashMap<ValueId, IRConst> = HashMap::new();
        
        // Loop multiple times to propagate constant folding across multiple instructions
        for _ in 0..3 {
            for block in &mut func.blocks {
                for node in &mut block.instructions {
                    // 1. If it's already a constant, record it
                    if let IROp::Const(c) = &node.op {
                        constants.insert(node.id, c.clone());
                        continue;
                    }
                    
                    // 2. Try to fold arithmetic and comparison ops if all inputs are constant
                    if node.inputs.is_empty() {
                        continue;
                    }
                    
                    let all_const = node.inputs.iter().all(|id| constants.contains_key(id));
                    if !all_const {
                        continue;
                    }
                    
                    // Retrieve inputs
                    let input_consts: Vec<&IRConst> = node.inputs.iter().map(|id| constants.get(id).unwrap()).collect();
                    
                    let folded = match &node.op {
                        IROp::Add => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Int(a + b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Float(a + b)),
                                _ => None,
                            }
                        }
                        IROp::Sub => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Int(a - b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Float(a - b)),
                                _ => None,
                            }
                        }
                        IROp::Mul => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Int(a * b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Float(a * b)),
                                _ => None,
                            }
                        }
                        IROp::Div => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) if *b != 0 => Some(IRConst::Int(a / b)),
                                (IRConst::Float(a), IRConst::Float(b)) if *b != 0.0 => Some(IRConst::Float(a / b)),
                                _ => None,
                            }
                        }
                        IROp::Neg => {
                            match input_consts[0] {
                                IRConst::Int(a) => Some(IRConst::Int(-a)),
                                IRConst::Float(a) => Some(IRConst::Float(-a)),
                                _ => None,
                            }
                        }
                        IROp::Lt => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a < b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a < b)),
                                _ => None,
                            }
                        }
                        IROp::Lte => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a <= b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a <= b)),
                                _ => None,
                            }
                        }
                        IROp::Gt => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a > b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a > b)),
                                _ => None,
                            }
                        }
                        IROp::Gte => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a >= b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a >= b)),
                                _ => None,
                            }
                        }
                        IROp::Eq => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a == b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a == b)),
                                (IRConst::Bool(a), IRConst::Bool(b)) => Some(IRConst::Bool(a == b)),
                                _ => None,
                            }
                        }
                        IROp::Neq => {
                            match (input_consts[0], input_consts[1]) {
                                (IRConst::Int(a), IRConst::Int(b)) => Some(IRConst::Bool(a != b)),
                                (IRConst::Float(a), IRConst::Float(b)) => Some(IRConst::Bool(a != b)),
                                (IRConst::Bool(a), IRConst::Bool(b)) => Some(IRConst::Bool(a != b)),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    
                    if let Some(c) = folded {
                        node.op = IROp::Const(c.clone());
                        node.inputs.clear();
                        constants.insert(node.id, c);
                    }
                }
            }
        }
    }
}
