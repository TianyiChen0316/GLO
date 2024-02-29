import typing
import torch
from torch import nn as nn
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.dropout import _DropoutNd

import lib.syntax.view


class Dropout(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, training: typing.Union[None, bool] = None):
        super().__init__(p, inplace)
        self._training = training

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._training is not None:
            training = self._training
        else:
            training = self.training
        return torch.nn.functional.dropout(input, self.p, training, self.inplace)


class BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        bn_training: typing.Union[None, bool] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.bn_training = bn_training
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def _check_input_dim(self, input):
        return

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.bn_training is not None:
            bn_training = self.bn_training
        elif self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class ZScore(torch.nn.Module):
    def __init__(self, num_features: int, eps=1e-4):
        super().__init__()
        self._num_features = num_features
        self.mean = torch.nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.eps = eps

    def fit(self, values : torch.Tensor):
        mean = values.mean(dim=0, keepdim=False)
        std = values.std(dim=0, keepdim=False)
        self.set(mean, std)
        return self

    def set(
        self,
        mean : typing.Union[torch.Tensor, float, None] = None,
        std : typing.Union[torch.Tensor, float, None] = None,
    ):
        if mean is None:
            mean = 0.
        if std is None:
            std = 1.
        if not isinstance(mean, torch.Tensor):
            mean = torch.Tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.Tensor(std)
        if mean.numel() == 1:
            mean = mean.view(1).repeat(self._num_features)
        else:
            assert mean.ndim == 1 and mean.shape[0] == self._num_features, \
                f'Shape mismatch for mean [{", ".join(mean.shape)}] (expected [{self._num_features}])'
        if std.numel() == 1:
            std = std.view(1).repeat(self._num_features)
        else:
            assert std.ndim == 1 and std.shape[0] == self._num_features, \
                f'Shape mismatch for std [{", ".join(std.shape)}] (expected [{self._num_features}])'
        self.mean.data = mean.detach().clone().to(self.mean.device)
        self.std.data = std.nan_to_num(1.).clip(self.eps, None).detach().clone().to(self.std.device)
        return self

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        view_args = (1, -1, *(1 for i in range(input.ndim - 2)))
        mean = self.mean.detach().view(*view_args)
        std = self.std.detach().view(*view_args)
        return (input - mean) / std

class MinMaxScaler(torch.nn.Module):
    def __init__(self, num_features: int, eps=1e-4):
        super().__init__()
        self._num_features = num_features
        self.min = torch.nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.range = torch.nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.eps = eps

    def fit(self, values : torch.Tensor):
        min = values.min(dim=0, keepdim=False).values
        max = values.max(dim=0, keepdim=False).values
        self.set(min, max)
        return self

    def set(
        self,
        min: typing.Union[torch.Tensor, float, None] = None,
        max: typing.Union[torch.Tensor, float, None] = None,
    ):
        if min is None:
            min = 0.
        if max is None:
            max = 1.
        if not isinstance(min, torch.Tensor):
            min = torch.Tensor(min)
        if not isinstance(max, torch.Tensor):
            max = torch.Tensor(max)
        if min.numel() == 1:
            min = min.view(1).repeat(self._num_features)
        else:
            assert min.ndim == 1 and min.shape[0] == self._num_features, \
                f'Shape mismatch for min [{", ".join(min.shape)}] (expected [{self._num_features}])'
        if max.numel() == 1:
            max = max.view(1).repeat(self._num_features)
        else:
            assert max.ndim == 1 and max.shape[0] == self._num_features, \
                f'Shape mismatch for max [{", ".join(max.shape)}] (expected [{self._num_features}])'
        self.min.data = min.detach().clone().to(self.min.device)
        self.range.data = (max - min).clip(self.eps, None).detach().clone().to(self.range.device)
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        view_args = (1, -1, *(1 for i in range(input.ndim - 2)))
        min = self.min.detach().view(*view_args)
        _range = self.range.detach().view(*view_args)
        return (input - min) / _range


class TransposeBatchNorm1d(BatchNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.transpose(-1, -2)).transpose(-1, -2)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, input: torch.Tensor):
        return input.permute(*self.args)


class Transpose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, input: torch.Tensor):
        return input.transpose(*self.args)
