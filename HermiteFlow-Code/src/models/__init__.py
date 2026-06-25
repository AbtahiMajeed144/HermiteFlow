# --------------------------------------------------------
# HermiteFlow-VFI — Model factory
# --------------------------------------------------------

from .ema import ExponentialMovingAverage
from .hermite_vfi import hermiteflow_r, hermiteflow_f


def create_model(config, ema=False):
    model_type = config.type.lower()
    if model_type == "hermiteflow_r":
        model = hermiteflow_r(config)
        model_ema = hermiteflow_r(config) if ema else None
    elif model_type == "hermiteflow_f":
        model = hermiteflow_f(config)
        model_ema = hermiteflow_f(config) if ema else None
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
