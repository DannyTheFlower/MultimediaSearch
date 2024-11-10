import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_retrieval_resources(
        data_version: int = 0,
        data_path='app/backend/indexed_data.csv',
        embedder_name='sergeyzh/rubert-tiny-turbo',
        index_path: str = 'app/backend/index.faiss',
        embeddings_path: str = 'app/backend/embeddings.npz',
        save_index_path: str = 'app/backend/index.faiss',
        save_embeddings_path: str = 'app/backend/embeddings.npz'
):
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


def find_similar_neighbors(query, k=3):
    data, embedder, index, embeddings = load_retrieval_resources(st.session_state["DATA_VERSION"])

    query_vector = embedder.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)
    distances, neighbors = index.search(np.array([query_vector], dtype=np.float32), k)
    similar_neighbors = [data.iloc[idx] for idx in neighbors.flatten()]
    return similar_neighbors


def index_new_data(
        new_data_csv_filepath='app/backend/temp_data.csv',
        save_index_path='app/backend/index.faiss',
        save_embeddings_path='app/backend/embeddings.npz'
):
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
    data.to_csv('app/backend/indexed_data.csv', index=False)

    os.remove(new_data_csv_filepath)
    st.session_state["DATA_VERSION"] += 1
    load_retrieval_resources(st.session_state["DATA_VERSION"])
