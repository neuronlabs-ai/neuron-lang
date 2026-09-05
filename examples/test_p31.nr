fn test_p31():
    let p = 31
    let mp = 2147483647
    let s = 4
    let i = 0
    let steps = p - 2
    while i < steps:
        let s2 = s * s - 2
        let s = s2 % mp
        let i = i + 1
    print("Final S for p=31:")
    print(s)

fn main():
    test_p31()
