# --------------------------------------------------------
# HermiteFlow-VFI — Model package
# --------------------------------------------------------

from .hermiteflow_r import HermiteFlow_R
from .hermiteflow_f import HermiteFlow_F


def hermiteflow_r(config):
    return HermiteFlow_R(config)


def hermiteflow_f(config):
    return HermiteFlow_F(config)
