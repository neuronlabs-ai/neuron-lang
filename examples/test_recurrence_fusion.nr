fn is_mersenne_textbook(p: Int) -> Bool:
    let mp = pow2_sub1(p)
    let s = 4
    let i = 0
    while i < p - 2:
        let s = (s * s - 2) % mp
        let i = i + 1
    return s == 0

fn test_p(p: Int):
    print("Testing exponent p:")
    print(p)
    let is_p = is_mersenne_textbook(p)
    if is_p:
        print("  >>> [PRIME HIT] M_p is a verified Mersenne Prime! <<<")
    else:
        print("  -> Composite.")

fn main():
    print("==================================================")
    print("Testing Pure Textbook Lucas-Lehmer Recurrence")
    print("NEURON Compiler Recurrence Fusion & Modulo Engine")
    print("==================================================")
    test_p(3)
    test_p(5)
    test_p(7)
    test_p(11)
    test_p(13)
    test_p(17)
    test_p(19)
    test_p(23)
    test_p(29)
    test_p(31)