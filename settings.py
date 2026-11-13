"""Configuration locale des chemins utilisés dans les notebooks."""

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).resolve().parent
    DATA_PATH: Path = Field(
        default_factory=lambda: Path(os.getenv("CHURN_DATA_PATH") or "data/raw/churn_clients.csv"),
        validate_default=True,
    )
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "output"
    FEATURES_DIR: Path = PROJECT_ROOT / "data" / "interim" / "features"

    @property
    def FEATURE_CONTRACT_PATH(self) -> Path:
        return self.FEATURES_DIR / "feature_contract.json"

    MLFLOW_DIR: Path = PROJECT_ROOT / ".mlflow"
    MLFLOW_EXPERIMENT: str = "telco-churn"

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return f"sqlite:///{(self.MLFLOW_DIR / 'mlflow.db').as_posix()}"

    @field_validator("DATA_PATH", "PROCESSED_DIR", "OUTPUT_DIR", "FEATURES_DIR", "MLFLOW_DIR")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            path = cls.PROJECT_ROOT / path
        return path.resolve()


settings = Settings()
