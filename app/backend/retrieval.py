import faiss
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


DATA = None
EMBEDDER = None
INDEX = None


def init_all(data_path='app/backend/indexed_data.csv', embedder_name='sergeyzh/rubert-tiny-turbo', index_path=None, save_index_path=None):
    global DATA, EMBEDDER, INDEX
    DATA = pd.read_csv(data_path)
    EMBEDDER = SentenceTransformer(embedder_name)
    if index_path is None:
        embeddings = EMBEDDER.encode(DATA['text'].values, convert_to_numpy=True)
        INDEX = faiss.IndexFlatIP(embeddings.shape[1])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        INDEX.add(embeddings.astype(np.float32))
        if save_index_path:
            with open(save_index_path, 'wb') as f:
                pickle.dump(INDEX, f)
    else:
        with open(index_path, 'rb') as f:
            INDEX = pickle.load(f)
        

def find_similar_neighbors(query, k=3):
    global DATA, EMBEDDER, INDEX
    if DATA is None or EMBEDDER is None or INDEX is None:
        init_all()
    query_vector = EMBEDDER.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)
    distances, neighbors = INDEX.search(np.array([query_vector], dtype=np.float32), k)
    similar_neighbors = [DATA.iloc[idx] for idx in neighbors.flatten()]
    return similar_neighbors
