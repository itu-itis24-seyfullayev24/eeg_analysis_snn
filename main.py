import yaml
import argparse
import torch


from src.snn_modeling.utils.model_builder import build_model
from train import run_training



def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(config).to(device)
    #print(model)

    run_training(config, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    main(args.config)