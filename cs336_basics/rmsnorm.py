import torch

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in__dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = torch.einsum("...d, d -> ...d", x / rms, self.weight)
        return result.to(in__dtype)
