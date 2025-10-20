# Training wrapper for the model
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from ufm_benchmarks import run_benchmark


@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    run_benchmark(cfg)

    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    main()
