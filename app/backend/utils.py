import os
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


def detect_language(text: str) -> str:
    """
    Detects whether the text is in English or Russian.

    :param text: The text to analyze.
    :return: 'en' for English, 'ru' for Russian.
    """
    english_letters = 0
    russian_letters = 0
    for char in text:
        if char.isalpha() and char.isascii():
            english_letters += 1
        elif char.isalpha() and not char.isascii():
            russian_letters += 1
    language = 'en' if english_letters > russian_letters else 'ru'
    return language


def get_file_extension(text: str) -> str:
    """
    Extracts the file extension from a filename.

    :param text: The filename.
    :return: The file extension.
    """
    return os.path.splitext(text)[1]


def migrate_to_qdrant(
        data: pd.DataFrame,
        embeddings: np.ndarray,
        collection_name: str = "multimedia_data",
        host: str = "localhost",
        port: int = 6333,
        batch_size: int = 100,
):
    dim = embeddings.shape[1]
    client = QdrantClient(host=host, port=port)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for i in range(0, len(data), batch_size):
        points = []

        for j in range(i, min(i + batch_size, len(data))):
            payload = {
                "filename": str(data.iloc[j]["filename"]),
                "text": str(data.iloc[j]["text"]),
                "n_slide": int(data.iloc[j]["n_slide"]),
            }
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[j].tolist(),
                payload=payload,
            )
            points.append(point)

        client.upsert(collection_name=collection_name, points=points)
        print(f"Batch {i // batch_size + 1}/{len(data) // batch_size + (len(data) % batch_size > 0)}. Upserted {len(points)} points.")
