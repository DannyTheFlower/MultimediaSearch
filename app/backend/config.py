from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    PROJECT_ROOT: str = str(Path(__file__).resolve().parents[2]).split()[-1]  # For future: use Path instead of strings

    # Paths
    UPLOAD_FOLDER: str = PROJECT_ROOT + "/uploaded_files"
    MEDIA_FOLDER: str = PROJECT_ROOT + "/media"

    # Model
    EMBEDDER_NAME: str = "deepvk/USER-bge-m3"

    # OCR and Chart2Text Options
    INCLUDE_OCR: bool = True
    INCLUDE_VLM: bool = False

    # Other constants
    CHUNK_SIZE: int = 250
    OVERLAP: int = 10
    TOO_FEW_CHARS: int = 50

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_TEXT_COLLECTION: str = "multimedia_data_text"
    QDRANT_CAPTION_COLLECTION: str = "multimedia_data_captions"
    THRESHOLD: float = 0.5
    VECTOR_SIZE: int = 1024
    DISTANCE: str = "Cosine"

    # LLM settings
    LLM_MODEL_NAME: str = "YOUR_MODEL_NAME"
    LLM_IAM_TOKEN: str = "YOUR_TOKEN"

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT + '/.env', env_file_encoding='utf-8')


config = Config()
