from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path
import base64
from io import BytesIO
from PIL.Image import Image
from dataclasses import dataclass

from core.config import Config
from core.processor import ScanFullAnalysisResult

@dataclass
class ReportArteryData:
    d: float
    problem: bool


@dataclass
class ReportData:
    main_artery: ReportArteryData
    left_artery: ReportArteryData
    right_artery: ReportArteryData
    doctor_full_name: str
    image_b64: str


def image_to_base64(pil_image: Image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_pdf(result: ScanFullAnalysisResult, output_pdf: str, config: Config):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.html")

    image_base64 = image_to_base64(result.result_image)

    report_data = prepare_report_data(result, config, image_base64)
    html_content = template.render(report_data=report_data)

    HTML(string=html_content, base_url=".").write_pdf(output_pdf)


def prepare_report_data(result: ScanFullAnalysisResult, config: Config, img_b64: str) -> ReportData:
    return ReportData(
        main_artery=ReportArteryData(
            d=result.postanalysis_result.main_artery_d/config.pixels_in_mm,
            problem=result.postanalysis_result.main_artery_problem,
        ),
        left_artery=ReportArteryData(
            d=result.postanalysis_result.left_artery_d/config.pixels_in_mm,
            problem=result.postanalysis_result.left_artery_problem,
        ),
        right_artery=ReportArteryData(
            d=result.postanalysis_result.right_artery_d/config.pixels_in_mm,
            problem=result.postanalysis_result.right_artery_problem,
        ),
        doctor_full_name=config.doctor_full_name,
        image_b64=img_b64,
    )