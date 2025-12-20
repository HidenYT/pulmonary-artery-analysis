from typing import Any

import numpy as np

from core.config import Config
from pa_analysis.entity import CVResult, PostAnalysisResult


def make_postanalysis(cv_result: CVResult, config: Config) -> PostAnalysisResult:
    main_1, main_2 = cv_result.main_artery_points
    main_d = np.sqrt(np.sum((main_1-main_2)**2))
    main_problem = False
    if not (config.width_main_px - config.diff_px <= main_d <= config.width_main_px + config.diff_px):
        main_problem = True
    
    left_1, left_2 = cv_result.left_artery_points
    left_d = np.sqrt(np.sum((left_1-left_2)**2))
    left_problem = False
    if not (config.width_left_px - config.diff_px <= left_d <= config.width_left_px + config.diff_px):
        left_problem = True
    
    right_1, right_2 = cv_result.right_artery_points
    right_d = np.sqrt(np.sum((right_1-right_2)**2))
    right_problem = False
    if not (config.width_right_px - config.diff_px <= right_d <= config.width_right_px + config.diff_px):
        right_problem = True

    return PostAnalysisResult(
        main_artery_d=main_d,
        main_artery_problem=main_problem,
        left_artery_d=left_d,
        left_artery_problem=left_problem,
        right_artery_d=right_d,
        right_artery_problem=right_problem,
    )