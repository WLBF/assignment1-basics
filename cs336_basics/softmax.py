import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract the max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum
