fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn enrich_features(features: Temporal[Tensor[5000], 0], labels: Temporal[Tensor[5000], 3]) -> Temporal[Tensor[5000], 0]:
    return labels

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = build_labels(x)
    let f = enrich_features(x, y)
    print(f)
