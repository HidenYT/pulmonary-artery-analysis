import json
import math
from pathlib import Path
import numpy as np

from core.processor import run_processing


def find_line_markup(mrk_path: Path):
    with open(mrk_path, "r") as f:
        data = json.load(f)

    if not data.get("markups"):
        return None

    markup = data["markups"][0]

    if markup.get("type") != "Line":
        return None

    p1 = markup["controlPoints"][0]["position"][:2]
    p2 = markup["controlPoints"][1]["position"][:2]

    return (np.array(p1), np.array(p2))


def line_length(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def find_image_and_markup(folder: Path):
    image_path = None
    markup_coords = None

    for file in folder.iterdir():

        if file.suffix == ".nrrd":
            image_path = file

        if file.name.endswith(".mrk.json"):
            coords = find_line_markup(file)
            if coords is not None:
                markup_coords = coords

    if image_path and markup_coords:
        return image_path, markup_coords

    return None


def collect_dataset(root: Path):
    dataset = []

    for folder in root.rglob("*"):
        if folder.is_dir():
            result = find_image_and_markup(folder)
            if result:
                dataset.append(result)

    return dataset


def run_dataset_processing(
    root_folder: str,
    config,
    classifier,
    segmentator,
    device
):
    root = Path(root_folder)

    dataset = collect_dataset(root)

    print(f"Found {len(dataset)} scans")

    results = []

    for image_path, (p1, p2) in dataset:

        gt_length = line_length(p1, p2)

        analysis_result = run_processing(
            str(image_path),
            config,
            classifier,
            segmentator,
            device
        )

        predicted_length = analysis_result.postanalysis_result.main_artery_d

        diff = abs(predicted_length - gt_length)

        results.append({
            "image": str(image_path),
            "gt_length": gt_length,
            "predicted_length": predicted_length,
            "difference": diff
        })

        print(
            f"{image_path.name} | "
            f"GT={gt_length:.2f} "
            f"PRED={predicted_length:.2f} "
            f"DIFF={diff:.2f}"
        )

    return results