import json
from pathlib import Path
from pydantic import BaseModel, Field, computed_field

CONFIG_PATH = Path("data/config.json")


class Config(BaseModel):
    max_width_main_mm: float = Field(27, description="Макс. диаметр главной артерии, мм")
    max_width_left_mm: float = Field(27, description="Макс. диаметр левой артерии, мм")
    max_width_right_mm: float = Field(27, description="Макс. диаметр правой артерии, мм")

    doctor_full_name: str = Field("", description="Имя врача")

    normal_vector_window_sz: int = Field(10, description="Окно вычисления вектора диаметра артерии")
    skeleton_center_points_remove_ratio: float = Field(0.3, description="Доля удаляемых точек скелета артерии от центра")
    skeleton_points_remove_from_edge_ratio: float = Field(0.2, description="Доля удаляемых точек скелета артерии с краю")
    classification_threshold: float = Field(0.51, description="Порог классификации")


class ConfigService:
    def __init__(self):
        self._config = self.load()

    def load(self) -> Config:
        if not CONFIG_PATH.exists():
            cfg = Config()
            self.save(cfg)
            return cfg
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return Config(**json.load(f))

    def save(self, config: Config):
        CONFIG_PATH.parent.mkdir(exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(exclude_computed_fields=True), f, indent=4)

    def update(self, key, value):
        self._config = self._config.model_copy(update={key: value})
        self.save(self._config)
