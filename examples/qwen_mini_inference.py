# ═══════════════════════════════════════════════════════════════════════
#  Minimal Transformer Inference in Python (numpy)
#  This is a real, working transformer forward pass that py2nr will
#  transpile to .nr source code
# ═══════════════════════════════════════════════════════════════════════

import numpy as np


class RMSNorm:
    def __init__(self, dim: int):
        self.weight = np.ones(dim)
        self.eps = 1e-6

    def forward(self, x):
        norm = np.sqrt(np.mean(x * x) + self.eps)
        return x / norm * self.weight


class AttentionHead:
    def __init__(self, d_model: int, head_dim: int):
        self.wq = np.random.randn(d_model, head_dim)
        self.wk = np.random.randn(d_model, head_dim)
        self.wv = np.random.randn(d_model, head_dim)
        self.wo = np.random.randn(head_dim, d_model)

    def forward(self, x):
        q = np.matmul(x, self.wq)
        k = np.matmul(x, self.wk)
        v = np.matmul(x, self.wv)
        scores = np.matmul(q, np.transpose(k))
        scale = 1.0 / np.sqrt(16.0)
        scaled = scores * scale
        attn = torch.softmax(scaled)
        context = np.matmul(attn, v)
        return np.matmul(context, self.wo)


class SwiGLU_FFN:
    def __init__(self, d_model: int, d_ff: int):
        self.w_gate = np.random.randn(d_model, d_ff)
        self.w_up = np.random.randn(d_model, d_ff)
        self.w_down = np.random.randn(d_ff, d_model)

    def forward(self, x):
        gate = torch.gelu(np.matmul(x, self.w_gate))
        up = np.matmul(x, self.w_up)
        hidden = gate * up
        return np.matmul(hidden, self.w_down)


class TransformerBlock:
    def __init__(self, d_model: int, head_dim: int, d_ff: int):
        self.attn = AttentionHead(d_model, head_dim)
        self.ffn = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x):
        attn_out = self.attn.forward(x)
        h = x + attn_out
        ffn_out = self.ffn.forward(h)
        return h + ffn_out


class QwenMiniLM:
    def __init__(self, d_model: int, d_ff: int, vocab_size: int):
        self.embed = np.random.randn(vocab_size, d_model)
        self.block1 = TransformerBlock(d_model, 16, d_ff)
        self.block2 = TransformerBlock(d_model, 16, d_ff)
        self.lm_head = np.random.randn(d_model, vocab_size)

    def forward(self, input_ids):
        x = np.matmul(input_ids, self.embed)
        h1 = self.block1.forward(x)
        h2 = self.block2.forward(h1)
        logits = np.matmul(h2, self.lm_head)
        return torch.softmax(logits)

    def generate_token(self, input_ids):
        logits = self.forward(input_ids)
        return logits


if __name__ == '__main__':
    print("Initializing QwenMiniLM Transformer...")
    model = QwenMiniLM(64, 256, 128)
    input_data = np.zeros(1, 128)
    output = model.forward(input_data)
    print("Generation complete!")
