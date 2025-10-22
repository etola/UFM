# Training wrapper for the model
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from uniflowmatch.training.train_pl import train_pl_main


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["launch_mode"] == "train" or cfg["launch_mode"] == "validate":
        train_pl_main(cfg)
    else:
        raise ValueError(f"Unknown launch mode: {cfg['launch_mode']}")

    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
