# --------------------------------------------------------
# HermiteFlow-VFI — Model factory
# --------------------------------------------------------

from .ema import ExponentialMovingAverage
from .hermite_vfi import hermiteflow_r, hermiteflow_f
# GIMM-VFI, vendored verbatim from the upstream repo (see
# generalizable_INR/__init__.py) - the baseline trained from scratch on
# the same X4K split HermiteFlow trains on, not a different pipeline.
from .generalizable_INR import gimm, gimmvfi_f, gimmvfi_r


def create_model(config, ema=False):
    model_type = config.type.lower()
    if model_type == "hermiteflow_r":
        model = hermiteflow_r(config)
        model_ema = hermiteflow_r(config) if ema else None
    elif model_type == "hermiteflow_f":
        model = hermiteflow_f(config)
        model_ema = hermiteflow_f(config) if ema else None
    elif model_type == "gimm":
        model = gimm(config)
        model_ema = gimm(config) if ema else None
    elif model_type == "gimmvfi_f":
        model = gimmvfi_f(config)
        model_ema = gimmvfi_f(config) if ema else None
    elif model_type == "gimmvfi_r":
        model = gimmvfi_r(config)
        model_ema = gimmvfi_r(config) if ema else None
    else:
        raise ValueError(f"{model_type} is invalid..")

    if ema:
        mu = config.ema
        if config.ema_value is not None:
            mu = config.ema_value
        model_ema = ExponentialMovingAverage(model_ema, mu)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema
