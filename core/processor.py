from PIL import Image
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from core.config import Config
from pa_analysis.cv.processing import find_arteries_d
from pa_analysis.post_analysis.analyze import make_postanalysis
from pa_analysis.entity import PostAnalysisResult


@dataclass
class ScanFullAnalysisResult:
    postanalysis_result: PostAnalysisResult
    result_image: Image.Image


def run_processing(image_path: str, mask_path: str, config: Config) -> ScanFullAnalysisResult:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    cv_result = find_arteries_d(mask)
    vis = cv.cvtColor(np.uint8(image), cv.COLOR_RGB2BGR)
    cv.line(vis, *cv_result.main_artery_points, (0, 0, 255), 1)
    cv.line(vis, *cv_result.left_artery_points, (0, 0, 255), 1)
    cv.line(vis, *cv_result.right_artery_points, (0, 0, 255), 1)
    img = Image.fromarray(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
    post_analysis_result = make_postanalysis(cv_result, config)
    return ScanFullAnalysisResult(postanalysis_result=post_analysis_result, result_image=img)
