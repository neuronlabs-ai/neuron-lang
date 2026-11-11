// Speed Fine Calculator in NEURON Language
// Equivalent to the C implementation

fn compute_fine(excess: Int) -> Int:
    if excess <= 0:
        return 0
    else:
        if excess < 30:
            return 2500
        else:
            return 4000

fn normalize_excess(diff: Int) -> Int:
    if diff <= 0:
        return 0
    else:
        return diff

fn main():
    print_inline("Enter vehicle speed: ")
    let speed_str = input()
    let speed = parse_int(speed_str)

    print_inline("Enter speed limit: ")
    let limit_str = input()
    let limit = parse_int(limit_str)

    let raw_excess = speed - limit
    let excess = normalize_excess(raw_excess)
    let fine = compute_fine(raw_excess)

    print()
    print("Vehicle speed: ", speed, "km/h")
    print("Speed limit:   ", limit, "km/h")
    print("Excess speed:  ", excess, "km/h")
    print("Fine levied:   Kshs.", fine)
