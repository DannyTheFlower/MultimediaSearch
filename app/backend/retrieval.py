import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


DATA = None
EMBEDDER = None
INDEX = None


def init_all(data_path='app/backend/indexed_data.csv', embedder_name='sergeyzh/rubert-tiny-turbo', index_path=None, save_index_path=None):
    global DATA, EMBEDDER, INDEX
    DATA = pd.read_csv(data_path)
    EMBEDDER = SentenceTransformer(embedder_name)
    if index_path is None:
        embeddings = EMBEDDER.encode(DATA['text'].values, convert_to_numpy=True, batch_size=16)
        INDEX = faiss.IndexFlatIP(embeddings.shape[1])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        INDEX.add(embeddings.astype(np.float32))
        if save_index_path:
            faiss.write_index(INDEX, save_index_path)
    else:
        INDEX = faiss.read_index(index_path)
        

def find_similar_neighbors(query, k=3):
    global DATA, EMBEDDER, INDEX
    if DATA is None or EMBEDDER is None or INDEX is None:
        init_all()
    query_vector = EMBEDDER.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)
    distances, neighbors = INDEX.search(np.array([query_vector], dtype=np.float32), k)
    similar_neighbors = [DATA.iloc[idx] for idx in neighbors.flatten()]
    return similar_neighbors
