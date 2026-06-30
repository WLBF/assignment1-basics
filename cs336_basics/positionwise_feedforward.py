import torch
from .linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        assert d_k % 2 == 0, "d_k must be even"
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        i = torch.arange(max_seq_len).float()
        j = torch.arange(d_k // 2).float()
        freqs = self.theta ** (-2 * j / d_k)
        angles = torch.einsum("i,j->ij", i, freqs)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos

        out = torch.stack([rot1, rot2], dim=-1)

        return out.flatten(-2)
