import faiss
import numpy as np
from utils import load_model_and_data

def build_faiss_index():
    model, dataset = load_model_and_data()
    texts = [d["text"] for d in dataset]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "quote_index.faiss")

if __name__ == "__main__":
    build_faiss_index()
