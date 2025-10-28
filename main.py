import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--diffuse", action="store_true", help="Diffusionモデルで学習を実行")
    args = parser.parse_args()
    config_path = args.config
    config = OmegaConf.load(config_path)
    if args.diffuse:
        from modules.llasa_diffusion import main
        main(config)
    else:
        from modules.train import main
        main(config)