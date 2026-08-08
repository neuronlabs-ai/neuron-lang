"""Simple PyTorch transformer — test case for py2nr transpiler."""
import torch
import torch.nn.functional as F
import numpy as np


class TransformerBlock:
    def __init__(self, d_model, n_heads):
        self.wq = torch.randn(d_model, d_model)
        self.wk = torch.randn(d_model, d_model)
        self.wv = torch.randn(d_model, d_model)
        self.wo = torch.randn(d_model, d_model)
        self.scale = 1.0 / np.sqrt(d_model)

    def forward(self, x):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)
        scores = torch.matmul(q, k.T) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return torch.matmul(out, self.wo)


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = torch.randn(input_dim, hidden_dim)
        self.w2 = torch.randn(hidden_dim, output_dim)
        self.b1 = torch.zeros(1, hidden_dim)
        self.b2 = torch.zeros(1, output_dim)

    def forward(self, x):
        h = F.relu(torch.matmul(x, self.w1) + self.b1)
        return torch.matmul(h, self.w2) + self.b2


def train_step(model, x, y, lr):
    pred = model.forward(x)
    loss = F.mse_loss(pred, y)
    return loss


if __name__ == '__main__':
    model = MLP(784, 256, 10)
    x = torch.randn(32, 784)
    y = torch.zeros(32, 10)
    loss = train_step(model, x, y, 0.001)
    print(loss)
