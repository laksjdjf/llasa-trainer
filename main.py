import argparse
from omegaconf import OmegaConf
from modules.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config
    config = OmegaConf.load(config_path)
    main(config)