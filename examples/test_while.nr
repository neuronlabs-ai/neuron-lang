fn test_while():
    let count = 0
    let sum = 0.0
    while count < 5:
        let sum = sum + 1.0
        let count = count + 1
        print(sum)
    print(count)
    print(sum)
    return sum

fn main():
    print("=== While Loop Test ===")
    let result = test_while()
    print("Final result:")
    print(result)

    print("=== Factorial of 6 ===")
    let n = 6
    let fact = 1
    while n > 1:
        let fact = fact * n
        let n = n - 1
    print(fact)

    print("=== Tensor Accumulation ===")
    let t = zeros(1, 4)
    let i = 0
    while i < 3:
        let t = t + 1.0
        let i = i + 1
    print(t)

    print("=== For Loop Mutation ===")
    let sum_for = 0.0
    let my_list = [1.0, 2.0, 3.0, 4.0]
    for x in my_list:
        let sum_for = sum_for + x
    print(sum_for)

    return 0
