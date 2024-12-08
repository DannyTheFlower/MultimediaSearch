import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import streamlit as st
from backend.config import config
from typing import Tuple, List, Dict
import time
import uuid


@st.cache_resource(show_spinner=False)
def load_retrieval_resources(
        data_version: int = 0,
        embedder_name: str = config.EMBEDDER_NAME,
        qdrant_host: str = config.QDRANT_HOST,
        qdrant_port: int = config.QDRANT_PORT,
        qdrant_collection_name: str = config.QDRANT_COLLECTION_NAME,
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
    if not client.collection_exists(qdrant_collection_name):
        client.create_collection(
            collection_name=qdrant_collection_name,
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
    client, embedder = load_retrieval_resources(st.session_state["DATA_VERSION"])

    query_vector = embedder.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)

    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    search_results = qdrant_client.search(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=k
    )

    neighbors = []
    for res in search_results:
        doc = {
            "filename": res.payload.get("filename", ""),
            "n_slide": res.payload.get("n_slide", None),
            "text": res.payload.get("text", "")
        }
        neighbors.append(doc)
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

    client, embedder = load_retrieval_resources(st.session_state["DATA_VERSION"])

    new_texts = [row["text"] for row in data_rows]
    new_embeddings = embedder.encode(new_texts, convert_to_numpy=True, batch_size=16)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    points = []
    for i, row in enumerate(data_rows):
        payload = {
            "filename": row["filename"],
            "text": row["text"]
        }
        if row.get("n_slide") is not None:
            payload["n_slide"] = row["n_slide"]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=new_embeddings[i].tolist(),
                payload=payload
            )
        )

    batch_upsert(client, config.QDRANT_COLLECTION_NAME, points)
    st.session_state["DATA_VERSION"] += 1
    load_retrieval_resources(st.session_state["DATA_VERSION"])
