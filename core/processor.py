from PIL import Image
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from core.config import Config
from pa_analysis.cv.processing import find_arteries_d
from pa_analysis.post_analysis.analyze import make_postanalysis
from pa_analysis.entity import PostAnalysisResult
from utils.cv import normalize_image, window_level


@dataclass
class ScanFullAnalysisResult:
    postanalysis_result: PostAnalysisResult
    result_image: Image.Image


import numpy as np

from utils.image_reader.image_reader import read_medical_image
from classification.classifier import classify_slices
from segmentation.segmentator import segment_slices
from segmentation.postprocessing import clean_mask, find_largest_mask



def run_processing(image_path: str, config: Config, classifier, segmentator, device) -> ScanFullAnalysisResult:
    volume, meta = read_medical_image(image_path)
    positive_slices = classify_slices(
        volume,
        classifier,
        device
    )
    if not positive_slices:
        return None
    masks = segment_slices(
        volume,
        positive_slices,
        segmentator,
        device
    )
    processed = []

    for idx, mask in masks:
        clean = clean_mask(mask)

        processed.append((idx, clean))

    slice_idx, best_mask = find_largest_mask(processed)
    cv_result = find_arteries_d(best_mask, config)
    img_to_show = window_level(volume[slice_idx], 2000, 0)
    img_to_show = normalize_image(img_to_show) * 255
    vis = cv.cvtColor(np.uint8(img_to_show), cv.COLOR_RGB2BGR)
    cv.line(vis, *cv_result.main_artery_points, (0, 0, 255), 1)
    cv.line(vis, *cv_result.left_artery_points, (0, 0, 255), 1)
    cv.line(vis, *cv_result.right_artery_points, (0, 0, 255), 1)
    img = Image.fromarray(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    post_analysis_result = make_postanalysis(cv_result, config)
    return ScanFullAnalysisResult(postanalysis_result=post_analysis_result, result_image=img)
