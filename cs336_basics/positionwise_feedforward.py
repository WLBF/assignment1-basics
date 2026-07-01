import torch
from .linear import Linear
from torch import Tensor
from einops import rearrange, einsum
import einx
from jaxtyping import Bool, Float, Int

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

# Copyed from: assignment2-systems/blob/main/cs336-basics/cs336_basics/model.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
        self._freq_cis_cache: Float[Tensor, "2 context_length half_dim"]

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = torch.tensor(theta) ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None
    ) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        if pos_ids is not None:
            cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        # result = einx.id("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()
        result = torch.concat((x1_rot, x2_rot), dim=-1)
        return result

    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"
