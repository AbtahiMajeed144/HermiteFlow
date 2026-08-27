# --------------------------------------------------------
# HermiteFlow-VFI — Trainer factory
# --------------------------------------------------------

from .trainer_hermiteflow import Trainer as TrainerHermiteFlow
# GIMM-VFI's own trainers, vendored verbatim - see trainer_gimm.py /
# trainer_gimmvfi.py for what "verbatim" means here (nothing changed).
from .trainer_gimm import Trainer as TrainerGIMM
from .trainer_gimmvfi import Trainer as TrainerGIMMVFI


def create_trainer(config):
    if config.arch.type in ["hermiteflow_r", "hermiteflow_f"]:
        return TrainerHermiteFlow
    elif config.arch.type in ["gimm"]:
        return TrainerGIMM
    elif config.arch.type in ["gimmvfi", "gimmvfi_f", "gimmvfi_r"]:
        return TrainerGIMMVFI
    else:
        print(config.arch.type)
        raise ValueError("architecture type not supported")
