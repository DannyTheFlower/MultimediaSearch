import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


DATA = None
EMBEDDER = None
INDEX = None
EMBEDDINGS = None 


def init_all(data_path='app/backend/indexed_data.csv', embedder_name='sergeyzh/rubert-tiny-turbo', index_path='app/backend/index.faiss', embeddings_path='app/backend/embeddings.npz',
             save_index_path='app/backend/index.faiss', save_embeddings_path='app/backend/embeddings.npz'):
    global DATA, EMBEDDER, INDEX, EMBEDDINGS
    DATA = pd.read_csv(data_path)
    EMBEDDER = SentenceTransformer(embedder_name)
    
    if (INDEX is None or EMBEDDINGS is None) and index_path and embeddings_path:
        INDEX = faiss.read_index(index_path)
        EMBEDDINGS = np.load(embeddings_path)['embeddings']
    else:
        EMBEDDINGS = EMBEDDER.encode(DATA['text'].values, convert_to_numpy=True, batch_size=16)
        EMBEDDINGS = EMBEDDINGS / np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
        INDEX = faiss.IndexFlatIP(EMBEDDINGS.shape[1])
        INDEX.add(EMBEDDINGS.astype(np.float32))
        
        if save_index_path:
            faiss.write_index(INDEX, save_index_path)
        if save_embeddings_path:
            np.savez(save_embeddings_path, embeddings=EMBEDDINGS)
            

def find_similar_neighbors(query, k=3):
    global DATA, EMBEDDER, INDEX
    query_vector = EMBEDDER.encode([query]).flatten()
    query_vector /= np.linalg.norm(query_vector)
    distances, neighbors = INDEX.search(np.array([query_vector], dtype=np.float32), k)
    similar_neighbors = [DATA.iloc[idx] for idx in neighbors.flatten()]
    return similar_neighbors


def index_new_data(new_data_csv_filepath='app/backend/indexed_data.csv', save_index_path='app/backend/index.faiss', save_embeddings_path='app/backend/embeddings.npz'):
    global DATA, EMBEDDER, INDEX, EMBEDDINGS
    
    new_data = pd.read_csv(new_data_csv_filepath)
    new_texts = new_data['text'].values
    new_embeddings = EMBEDDER.encode(new_texts, convert_to_numpy=True, batch_size=16)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    
    INDEX.add(new_embeddings.astype(np.float32))
    faiss.write_index(INDEX, save_index_path)
    
    EMBEDDINGS = np.vstack((EMBEDDINGS, new_embeddings)) 
    np.savez(save_embeddings_path, embeddings=EMBEDDINGS)
     