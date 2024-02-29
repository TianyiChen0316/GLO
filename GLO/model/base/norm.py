import torch

from model.nn import ZScore, MinMaxScaler


class SqrtZScore(ZScore):
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        view_args = (1, -1, *(1 for i in range(input.ndim - 2)))
        mean = self.mean.detach().view(*view_args)
        std : torch.Tensor = self.std.detach().view(*view_args)
        return (input - mean) / std.sqrt()


class SqrtMinMaxScaler(MinMaxScaler):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        view_args = (1, -1, *(1 for i in range(input.ndim - 2)))
        min = self.min.detach().view(*view_args)
        _range = self.range.detach().view(*view_args)
        return (input - min) / _range.sqrt()
