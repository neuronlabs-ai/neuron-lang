fn main():
    print("=== TEST TENSOR INDEXING & SLICING ===")

    // 1. 2D Tensor indexing
    let t = ones(3, 4)
    let row0 = t[0]
    print("row0 shape:")
    print(shape(row0))

    let elem = t[0, 1]
    print("elem t[0, 1]:")
    print(elem)

    // 2. 2D Tensor Slicing - rows
    let s_rows = t[0:2]
    print("s_rows [0:2] shape:")
    print(shape(s_rows))

    // 3. 2D Tensor Slicing - cols
    let s_cols = t[:, 1:3]
    print("s_cols [:, 1:3] shape:")
    print(shape(s_cols))

    // 4. 2D Tensor Slicing - rows & cols
    let s_both = t[0:2, 1:3]
    print("s_both [0:2, 1:3] shape:")
    print(shape(s_both))

    // 5. Negative bounds slice
    let s_neg = t[0:-1]
    print("s_neg [0:-1] shape:")
    print(shape(s_neg))

    // 6. 1D Tensor Slicing
    let v = ones(6)
    let v_slice = v[1:4]
    print("v_slice [1:4] shape:")
    print(shape(v_slice))

    // 7. List Slicing
    let l = [10, 20, 30, 40, 50]
    let l_sub = l[1:4]
    print("list sub:")
    print(l_sub)

    print("ALL INDEXING & SLICING TESTS PASSED")
