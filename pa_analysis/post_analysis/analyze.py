from typing import Any

import numpy as np

from core.config import Config
from pa_analysis.entity import CVResult, PostAnalysisResult
from utils.image_reader.models import ImageMeta


def make_postanalysis(cv_result: CVResult, config: Config, meta: ImageMeta) -> PostAnalysisResult:
    pixel_to_mm_multiplier = np.array(meta.spacing[:2])  # x, y, z -> x, y

    main_1, main_2 = cv_result.main_artery_points
    main_d_mm = np.linalg.norm((main_1 - main_2) * pixel_to_mm_multiplier)
    main_problem = not(config.width_main_mm - config.diff_mm <= main_d_mm <= config.width_main_mm + config.diff_mm)
    
    left_1, left_2 = cv_result.left_artery_points
    left_d_mm = np.linalg.norm((left_1 - left_2) * pixel_to_mm_multiplier)
    left_problem = not(config.width_left_mm - config.diff_mm <= left_d_mm <= config.width_left_mm + config.diff_mm)
    
    right_1, right_2 = cv_result.right_artery_points
    right_d_mm = np.linalg.norm((right_1 - right_2) * pixel_to_mm_multiplier)
    right_problem = not(config.width_right_mm - config.diff_mm <= right_d_mm <= config.width_right_mm + config.diff_mm)

    return PostAnalysisResult(
        main_artery_d=main_d_mm,
        main_artery_problem=main_problem,
        left_artery_d=left_d_mm,
        left_artery_problem=left_problem,
        right_artery_d=right_d_mm,
        right_artery_problem=right_problem,
    )