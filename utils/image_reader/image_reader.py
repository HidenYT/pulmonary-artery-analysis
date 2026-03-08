import SimpleITK as sitk
import numpy as np
from pathlib import Path
from utils.image_reader.models import ImageMeta


def read_medical_image(path: str) -> tuple[np.ndarray, ImageMeta]:

    p = Path(path)

    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(path)

    arr = sitk.GetArrayFromImage(image)  # (z,y,x)

    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    meta = ImageMeta(spacing=spacing, origin=origin)

    return arr, meta