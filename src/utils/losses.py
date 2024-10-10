from typing import Optional

import torch


def masked_regression_loss(
    pred: torch.Tensor, 
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    if mask is not None:
        loss = (pred - y).pow(2)[~mask].mean()
    else:
        loss = (pred - y).pow(2).mean()
    return loss

    # if mask is not None:
    #     loss = torch.where(
    #         mask,
    #         0.0,
    #         (pred - y).pow(2),
    #     ).sum() / (~mask).sum()
    # else:
    #     loss = (pred - y).pow(2).mean()
    # return loss