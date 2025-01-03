import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from .config import config
from typing import Tuple, List, Dict
import uuid
from functools import lru_cache


DATA_VERSION = 0


@lru_cache(maxsize=1)
def load_retrieval_resources(
        data_version: int = 0,
        embedder_name: str = config.EMBEDDER_NAME,
        qdrant_host: str = config.QDRANT_HOST,
        qdrant_port: int = config.QDRANT_PORT,
        qdrant_text_collection: str = config.QDRANT_TEXT_COLLECTION,
        qdrant_caption_collection: str = config.QDRANT_CAPTION_COLLECTION,
        dim: int = config.VECTOR_SIZE,
        distance: str = config.DISTANCE,
) -> Tuple[QdrantClient, SentenceTransformer]:
    """
    Loads or creates retrieval resources: qdrant client and embedder.

    :param data_version: Version of the backup for cache invalidation.
    :param embedder_name: Name of the sentence transformer model.
    :param qdrant_host: Address of the qdrant server.
    :param qdrant_port: Port of the qdrant server.
    :param qdrant_collection_name: Name of the qdrant collection.
    :param dim: Dimensionality of the vector space.
    :param distance: Distance metric.
    :return: Tuple containing qdrant client and embedder.
    """
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    if not client.collection_exists(qdrant_text_collection):
        client.create_collection(
            collection_name=qdrant_text_collection,
            vectors_config=VectorParams(size=dim, distance=distance)
        )
    if not client.collection_exists(qdrant_caption_collection):
        client.create_collection(
            collection_name=qdrant_caption_collection,
            vectors_config=VectorParams(size=dim, distance=distance)
        )

    embedder = SentenceTransformer(embedder_name)
    return client, embedder


def find_similar_neighbors(query: str, k: int = 3) -> List[Dict]:
    """
    Finds similar neighbors for the given query using FAISS index.

    :param query: The search query.
    :param k: Number of neighbors to retrieve.
    :return: List of similar backup entries.
    """
    client, embedder = load_retrieval_resources(DATA_VERSION)

    query_vector = embedder.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)

    # Stage 1: searching among text on slides
    search_results = client.search(
        collection_name=config.QDRANT_TEXT_COLLECTION,
        query_vector=query_vector.tolist(),
        limit=k,
    )
    neighbors = []
    for res in search_results:
        doc = {
            "filename": res.payload.get("filename", ""),
            "n_slide": res.payload.get("n_slide", None),
            "text": res.payload.get("text", ""),
            "score": res.score,
            "source": "text_collection",
        }
        neighbors.append(doc)

    # Stage 2: if there are not enough good results, we're searching among extracted descriptions
    neighbors = list(filter(lambda x: x["score"] > config.THRESHOLD, neighbors))
    if len(neighbors) < k:
        captions_search_results = client.search(
            collection_name=config.QDRANT_CAPTION_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=k - len(neighbors),
        )
        for res in captions_search_results:
            doc = {
                "filename": res.payload.get("filename", ""),
                "n_slide": res.payload.get("n_slide", None),
                "text": res.payload.get("text", ""),
                "score": res.score,
                "source": "caption_collection",
            }
            neighbors.append(doc)

    print("scores:", ', '.join([str(n["score"]) for n in neighbors]))
    return neighbors


def index_new_data(data_rows: List[Dict]):
    """
    Indexes new data in qdrant collection.

    :param data_rows: List of dicts like {"filename": str, "n_slide": int or None, "text": str}.
    """
    def batch_upsert(client, collection_name, points, batch_size=10):
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Upserted batch {i // batch_size + 1} with {len(batch)} points.")

    global DATA_VERSION
    client, embedder = load_retrieval_resources(DATA_VERSION)

    new_texts = [row["text"] for row in data_rows]
    new_embeddings = embedder.encode(new_texts, convert_to_numpy=True, batch_size=16)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    points = []
    for i, row in enumerate(data_rows):
        payload = {
            "filename": row["filename"],
            "n_slide": row["n_slide"],
            "text": row["text"]
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=new_embeddings[i].tolist(),
                payload=payload
            )
        )

    batch_upsert(client, config.QDRANT_COLLECTION_NAME, points)
    DATA_VERSION += 1
    load_retrieval_resources(DATA_VERSION)
