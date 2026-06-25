# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------


class Profiler:
    opts_model_size = {"trainable-only", "transformer-block-only"}

    def __init__(self, logger):
        self._logger = logger

    def get_model_size(self, model, opt=None):
        if opt is None:
            self._logger.info(
                "[OPTION: ALL] #parameters: %.4fM",
                sum(p.numel() for p in model.parameters()) / 1e6,
            )
        else:
            assert (
                opt in self.opts_model_size
            ), f"{opt} is not in {self.opts_model_size}"

            if opt == "trainable-only":
                self._logger.info(
                    "[OPTION: %s] #parameters: %.4fM",
                    opt,
                    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
                )
            else:
                if hasattr(model, "blocks"):
                    self._logger.info(
                        "[OPTION: %s] #parameters: %.4fM",
                        opt,
                        sum(p.numel() for p in model.blocks.parameters()) / 1e6,
                    )

    def get_module_breakdown(self, model):
        self._logger.info("\n[MODULE BREAKDOWN]")
        
        # We want to iterate through top-level modules of the base model
        # If wrapped in DDP, use module.module
        base_model = model.module if hasattr(model, "module") else model
        
        total_trainable = 0
        total_frozen = 0
        
        for name, child in base_model.named_children():
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in child.parameters() if not p.requires_grad)
            
            total_trainable += trainable
            total_frozen += frozen
            
            total_mod = trainable + frozen
            if total_mod > 0:
                self._logger.info(
                    "  %-25s %8.4fM (trainable: %8.4fM, frozen: %8.4fM)", 
                    name + ":", 
                    total_mod / 1e6, 
                    trainable / 1e6, 
                    frozen / 1e6
                )
                
        total_all = total_trainable + total_frozen
        self._logger.info(
            "  %-25s %8.4fM (trainable: %8.4fM, frozen: %8.4fM)\n", 
            "TOTAL:", 
            total_all / 1e6, 
            total_trainable / 1e6, 
            total_frozen / 1e6
        )
