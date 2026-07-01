import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from .attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm
from .embedding import Embedding
from .linear import Linear

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.size(-2)
        token_positions = torch.arange(x.shape[-2], device=x.device)
        nx = self.norm1(x)
        attn_output = self.self_attention.forward(nx, token_positions)
        x = x + attn_output
        nx = self.norm2(x)
        ff_output = self.feed_forward.forward(nx)
        x = x + ff_output
        return x

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size seq_len vocab_size"]:
        x = self.embedding(in_indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return x