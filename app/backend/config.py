from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Paths
    UPLOAD_FOLDER: str = "uploaded_files"
    MEDIA_FOLDER: str = "media"
    DATA_PATH: str = "app/backend/data/indexed_data.csv"
    INDEX_PATH: str = "app/backend/data/index.faiss"
    EMBEDDINGS_PATH: str = "app/backend/data/embeddings.npz"
    SAVE_INDEX_PATH: str = "app/backend/data/index.faiss"
    SAVE_EMBEDDINGS_PATH: str = "app/backend/data/embeddings.npz"
    TEMP_DATA_CSV: str = "app/backend/data/temp_data.csv"
    CSV_FILE_PATH: str = "indexed_data.csv"

    # Model and Tokens
    EMBEDDER_NAME: str = "sergeyzh/rubert-tiny-turbo"
    MODEL_NAME: str = "model_name"
    IAM_TOKEN: str = "iam_token"

    # OCR and Chart2Text Options
    INCLUDE_OCR: bool = True
    INCLUDE_CHART2TEXT: bool = False

    # Other constants
    CHUNK_SIZE: int = 250
    OVERLAP: int = 10
    TOO_FEW_CHARS: int = 50


config = Config()
