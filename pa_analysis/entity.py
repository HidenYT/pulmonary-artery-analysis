from dataclasses import dataclass
import numpy as np


@dataclass
class PostAnalysisResult:
    main_artery_d: float
    main_artery_problem: bool
    left_artery_d: float
    left_artery_problem: bool
    right_artery_d: float
    right_artery_problem: bool


@dataclass
class CVResult:
    main_artery_points: tuple[np.ndarray, np.ndarray]
    left_artery_points: tuple[np.ndarray, np.ndarray]
    right_artery_points: tuple[np.ndarray, np.ndarray]
