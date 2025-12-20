import json
from pathlib import Path
from pydantic import BaseModel, computed_field

CONFIG_PATH = Path("data/config.json")


class Config(BaseModel):
    pixels_in_mm: float = 1.7
    width_main_mm: float = 27
    width_left_mm: float = 27
    width_right_mm: float = 27
    diff_mm: float = 1

    doctor_full_name: str = ""

    @computed_field
    @property
    def width_main_px(self) -> float:
        return self.pixels_in_mm * self.width_main_mm

    @computed_field
    @property
    def width_left_px(self) -> float:
        return self.pixels_in_mm * self.width_left_mm

    @computed_field
    @property
    def width_right_px(self) -> float:
        return self.pixels_in_mm * self.width_right_mm

    @computed_field
    @property
    def diff_px(self) -> float:
        return self.pixels_in_mm * self.diff_mm


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
