import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from backend.config import config
from typing import Tuple, Any, List


@st.cache_resource(show_spinner=False)
def load_retrieval_resources(
        data_version: int = 0,
        data_path: str = config.DATA_PATH,
        embedder_name: str = config.EMBEDDER_NAME,
        index_path: str = config.INDEX_PATH,
        embeddings_path: str = config.EMBEDDINGS_PATH,
        save_index_path: str = config.SAVE_INDEX_PATH,
        save_embeddings_path: str = config.SAVE_EMBEDDINGS_PATH
) -> Tuple[pd.DataFrame, SentenceTransformer, faiss.IndexFlatIP, np.ndarray]:
    """
    Loads or creates retrieval resources including data, embedder, index, and embeddings.

    :param data_version: Version of the data for cache invalidation.
    :param data_path: Path to the data CSV file.
    :param embedder_name: Name of the sentence transformer model.
    :param index_path: Path to the FAISS index file.
    :param embeddings_path: Path to the embeddings file.
    :param save_index_path: Path to save the FAISS index.
    :param save_embeddings_path: Path to save the embeddings.
    :return: Tuple containing data, embedder, index, and embeddings.
    """
    data = pd.read_csv(data_path)
    embedder = SentenceTransformer(embedder_name)
    if os.path.exists(index_path) and os.path.exists(embeddings_path):
        index = faiss.read_index(index_path)
        embeddings = np.load(embeddings_path)['embeddings']
    else:
        embeddings = embedder.encode(data['text'].values, convert_to_numpy=True, batch_size=16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        if save_index_path:
            faiss.write_index(index, save_index_path)
        if save_embeddings_path:
            np.savez(save_embeddings_path, embeddings=embeddings)
    return data, embedder, index, embeddings


def find_similar_neighbors(query: str, k: int = 3) -> List[pd.Series]:
    """
    Finds similar neighbors for the given query using FAISS index.

    :param query: The search query.
    :param k: Number of neighbors to retrieve.
    :return: List of similar data entries.
    """
    data, embedder, index, embeddings = load_retrieval_resources(st.session_state["DATA_VERSION"])

    query_vector = embedder.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)
    distances, neighbors = index.search(np.array([query_vector], dtype=np.float32), k)
    similar_neighbors = [data.iloc[idx] for idx in neighbors.flatten()]
    return similar_neighbors


def index_new_data(
        new_data_csv_filepath: str = config.TEMP_DATA_CSV,
        save_index_path: str = config.SAVE_INDEX_PATH,
        save_embeddings_path: str = config.SAVE_EMBEDDINGS_PATH
):
    """
    Indexes new data and updates the FAISS index and embeddings.

    :param new_data_csv_filepath: Path to the new data CSV file.
    :param save_index_path: Path to save the updated FAISS index.
    :param save_embeddings_path: Path to save the updated embeddings.
    """
    data, embedder, index, embeddings = load_retrieval_resources(st.session_state["DATA_VERSION"])

    new_data = pd.read_csv(new_data_csv_filepath)
    new_texts = new_data['text'].values
    new_embeddings = embedder.encode(new_texts, convert_to_numpy=True, batch_size=16)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    index.add(new_embeddings.astype(np.float32))
    faiss.write_index(index, save_index_path)

    embeddings = np.vstack((embeddings, new_embeddings))
    np.savez(save_embeddings_path, embeddings=embeddings)

    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv(config.DATA_PATH, index=False)

    os.remove(new_data_csv_filepath)
    st.session_state["DATA_VERSION"] += 1
    load_retrieval_resources(st.session_state["DATA_VERSION"])
