import torch

def log_transform_with_cap(value : torch.Tensor, min=None, max=None):
    new_value = value.clone()
    if min is not None:
        indices = value < min
        ori_value = value[value < min]
        new_value[indices] = -torch.log(1 + min - ori_value) + min
    if max is not None:
        indices = value > max
        ori_value = value[value > max]
        new_value[indices] = torch.log(1 - max + ori_value) + max
    return new_value

def log_transform_rev(value : torch.Tensor, min=None, max=None):
    new_value = value.clone()
    if min is not None:
        indices = value < min
        ori_value = value[value < min]
        new_value[indices] = 1 + min - (min - ori_value).exp()
    if max is not None:
        indices = value > max
        ori_value = value[value > max]
        new_value[indices] = (ori_value - max).exp() + max - 1
    return new_value

def capped_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    min = None,
    max = None,
    reduction: str = "mean",
):
    """
    When the bias is not between [*min*, *max*], the l2 mse loss is degraded to l1 loss.
    """
    values = input - target
    res = values.clamp(min, max) * values
    if reduction == 'mean':
        res = res.mean()
    elif reduction == 'sum':
        res = res.sum()
    return res
