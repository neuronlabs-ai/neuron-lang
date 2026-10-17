fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn train_model(features: Temporal[Tensor[5000], 0]):
    print(features)

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y: Temporal[Tensor[5000], 3] = build_labels(x)
    train_model(y)
