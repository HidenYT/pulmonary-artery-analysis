from dataclasses import dataclass
import numpy as np

@dataclass
class ImageMeta:
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

@dataclass
class ArteryResult:
    diameters: list[float]
    mask: np.ndarray
    slice_index: int
    pixel_spacing: tuple[float, float]