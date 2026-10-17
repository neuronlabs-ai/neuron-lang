fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y: Temporal[Tensor[5000], 3] = build_labels(x)
    
    // Attempting to assign future target to causal feature space:
    let leaked_feature: Temporal[Tensor[5000], 0] = y
    print(leaked_feature)
