import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from .softmax import softmax
from .linear import Linear
from .positionwise_feedforward import RotaryEmbedding

def scaled_dot_product_attention(
    query: Float[Tensor, " ... queries d_k"],
    key: Float[Tensor, " ... keys d_k"],
    value: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = query.size(-1)
    scores = torch.einsum("...qd,...kd->...qk", query, key) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = softmax(scores, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", attention, value)
    return output


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryEmbedding(max_seq_len, self.d_k, theta)
        else:
            self.rope = None

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.size(-2)

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scaled_attention = scaled_dot_product_attention(Q, K, V, mask)
        output = rearrange(scaled_attention, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.w_o(output)
