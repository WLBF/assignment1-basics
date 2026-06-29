import torch

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        a = -3.0 * std
        b = 3.0 * std
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)
        self.weight = torch.nn.Parameter(
            tensor
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi, ...i -> ...o", self.weight, x)

