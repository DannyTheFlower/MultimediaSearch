from pathlib import Path
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    PROJECT_ROOT: str = str(Path(__file__).resolve().parents[2]).split()[-1]  # For future: use Path instead of strings

    # Paths
    UPLOAD_FOLDER: str = PROJECT_ROOT + "/uploaded_files"
    MEDIA_FOLDER: str = PROJECT_ROOT + "/media"
    QDRANT_BACKUP: str = PROJECT_ROOT + "/app/backend/backup/qdrant_backup.snapshot"

    # Model
    EMBEDDER_NAME: str = "deepvk/USER-bge-m3"

    # OCR and Chart2Text Options
    INCLUDE_OCR: bool = True
    INCLUDE_CHART2TEXT: bool = False

    # Other constants
    CHUNK_SIZE: int = 250
    OVERLAP: int = 10
    TOO_FEW_CHARS: int = 50

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "multimedia_data"
    VECTOR_SIZE: int = 1024
    DISTANCE: str = "Cosine"


config = Config()
