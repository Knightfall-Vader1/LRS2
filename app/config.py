import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default, cast=None):
    value = os.getenv(f"LRS_{name}")
    if value is None or value == "":
        return default
    if cast is None:
        return value
    return cast(value)


@dataclass(frozen=True)
class Settings:
    app_name: str = _env("APP_NAME", "LRS2.0")
    models_dir: Path = _env("MODELS_DIR", Path("models"), Path)
    yolo_weights: Path = _env("YOLO_WEIGHTS", Path("models/plate_detector.pt"), Path)
    trocr_weights_dir: Path = _env("TROCR_WEIGHTS_DIR", Path("models/trocr_fixed"), Path)
    authorized_plates_path: Path = _env(
        "AUTHORIZED_PLATES_PATH", Path("authorized_plates.json"), Path
    )
    input_size: int = _env("INPUT_SIZE", 640, int)
    confidence_threshold: float = _env("CONFIDENCE_THRESHOLD", 0.3, float) #0.0127 is the best yet


settings = Settings()
