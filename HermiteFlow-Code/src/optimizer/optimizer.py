# --------------------------------------------------------
# HermiteFlow — optimizer factory
# Adapted from GIMM-VFI's optimizer.py
#
# Everything that is trainable is trained: CoeffNet (Phase 2),
# RefineNet (Phase 4) and SynthNet (Phase 5). The flow estimator
# (Phase 1) is frozen and Phase 3 has no parameters at all, so
# both are absent from the parameter list by construction.
# --------------------------------------------------------

import torch


def create_hermiteflow_optimizer(model, config):
    optimizer_type = config.type.lower()

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("no trainable parameters found")

    kwargs = dict(lr=config.init_lr, weight_decay=config.weight_decay)
    if optimizer_type in ("adamw", "adam"):
        kwargs["betas"] = tuple(config.betas)

    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, **kwargs)
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, momentum=0.9, **kwargs)

    raise ValueError(f"{optimizer_type} invalid..")


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if "hermite" in arch_type:
        return create_hermiteflow_optimizer(model, config.optimizer)
    raise ValueError(f"{arch_type} invalid..")
