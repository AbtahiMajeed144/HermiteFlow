# --------------------------------------------------------
# HermiteFlow-VFI — Optimizer factory
# Adapted from GIMM-VFI's optimizer.py
# --------------------------------------------------------

import torch


def create_hermiteflow_optimizer(model, config):
    optimizer_type = config.type.lower()
    if not config.ft:
        param_dicts = model.parameters()
    else:
        # Fine-tuning mode: we freeze the AMT decoder and train the rest.
        # Just in case some AMT parameters are left unfrozen, we put them in a lower LR group.
        # The main parameters to train (CoefficientNet) get the primary LR.
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "amt_" in n and p.requires_grad
                ],
                "lr": config.init_lr * 0.01,
                "weight_decay": config.weight_decay * 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "amt_" not in n and p.requires_grad
                ]
            },
        ]
        if len(param_dicts[0]["params"]) == 0:
            print("No amt_ parameters to train (frozen). Training other parts.")

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=config.init_lr,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"{optimizer_type} invalid..")
    return optimizer


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if "hermite" in arch_type:
        optimizer = create_hermiteflow_optimizer(model, config.optimizer)
    else:
        raise ValueError(f"{arch_type} invalid..")
    return optimizer
