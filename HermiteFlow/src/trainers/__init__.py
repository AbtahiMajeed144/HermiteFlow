# --------------------------------------------------------
# HermiteFlow-VFI — Trainer factory
# --------------------------------------------------------

from .trainer_hermiteflow import Trainer as TrainerHermiteFlow


def create_trainer(config):
    if config.arch.type in ["hermiteflow_r", "hermiteflow_f"]:
        return TrainerHermiteFlow
    else:
        print(config.arch.type)
        raise ValueError("architecture type not supported")
