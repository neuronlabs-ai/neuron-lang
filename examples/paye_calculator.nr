// PAYE (Pay As You Earn) Tax Calculator in NEURON Language
// Equivalent to the C implementation

fn compute_tax(gross: Float) -> Float:
    if gross > 40000.0:
        return gross * 0.30
    else:
        if gross >= 30000.0:
            return gross * 0.25
        else:
            if gross >= 20000.0:
                return gross * 0.15
            else:
                if gross >= 10000.0:
                    return gross * 0.10
                else:
                    return 0.0

fn main():
    print_inline("Enter gross pay: ")
    let gross_str = input()
    let gross = parse_float(gross_str)

    let tax = compute_tax(gross)
    let net = gross - tax

    print("Gross pay   = Ksh", gross)
    print("Tax amount  = Ksh", tax)
    print("Net pay     = Ksh", net)
