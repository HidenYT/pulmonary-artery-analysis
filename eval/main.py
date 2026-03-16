# cli.py
import argparse
from pathlib import Path
import torch

from classification.loader import load_resnet50
from core.config import ConfigService
from eval.evaluate_dataset import run_dataset_processing
from segmentation.loader import load_segnet
import pandas as pd

from eval.evaluate_dataset import run_dataset_processing

def main():
    parser = argparse.ArgumentParser(description="Run artery evaluation")
    parser.add_argument("--data", required=True, help="Root folder with scans")
    parser.add_argument("--output", default="results.csv", help="CSV output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = load_resnet50("weights/model_resnet_1stack", device)
    segmentator = load_segnet("weights/segnet_aug_dice_weights_50_epoch", device)


    config_service = ConfigService()

    results = run_dataset_processing(
        root_folder=args.data,
        config=config_service.load(),
        classifier=classifier,
        segmentator=segmentator,
        device=device
    )

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
