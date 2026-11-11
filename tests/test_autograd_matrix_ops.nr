fn main():
    print("=== TEST AUTOGRAD MATRIX OPS ===")

    // 1. Matmul and gradient
    let A = ones(2, 3)
    let B = ones(3, 2)
    let C = matmul(A, B)
    let loss_c = sum(C)
    let grad_A = grad(loss_c, wrt=A)
    print("grad_A shape:")
    print(shape(grad_A))
    print(grad_A)

    // 2. Reshape and gradient with negative dimension inference (-1)
    let x = ones(2, 3)
    let x_flat = x.reshape(-1)
    print("x_flat shape:")
    print(shape(x_flat))
    let loss_x = sum(x_flat * 2.0)
    let grad_x = grad(loss_x, wrt=x)
    print("grad_x shape:")
    print(shape(grad_x))
    print(grad_x)

    // 3. Transpose and gradient
    let M = ones(2, 3)
    let M_T = M.transpose()
    print("M_T shape:")
    print(shape(M_T))
    let loss_m = sum(M_T * 3.0)
    let grad_M = grad(loss_m, wrt=M)
    print("grad_M shape:")
    print(shape(grad_M))
    print(grad_M)

    print("ALL AUTOGRAD MATRIX OPS PASSED")
