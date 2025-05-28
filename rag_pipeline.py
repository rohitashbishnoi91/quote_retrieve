import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils import load_model_and_data

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=lm_model, tokenizer=tokenizer, max_new_tokens=128)

# Build the index at startup (in memory)
embedder, dataset = load_model_and_data()
texts = [d["text"] for d in dataset]
embeddings = [embedder.encode(text, convert_to_numpy=True) for text in texts]
nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)

def retrieve_and_generate(query):
    query_vec = embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = nn.kneighbors(query_vec)
    retrieved = [texts[i] for i in indices[0]]

    prompt = f"Answer this query using the following quotes:\n{retrieved}\nQuery: {query}"
    response = generator(prompt, do_sample=True, temperature=0.7)
    answer = response[0]['generated_text'], retrieved
    return answer