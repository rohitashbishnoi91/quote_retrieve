import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils import load_model_and_data

# Load a lightweight HuggingFace model for local generation
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lm_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=lm_model, tokenizer=tokenizer, max_new_tokens=128)

def retrieve_and_generate(query):
    embedder, dataset = load_model_and_data()
    texts = [d["text"] for d in dataset]
    index = faiss.read_index("quote_index.faiss")
    query_vec = embedder.encode(query, convert_to_numpy=True)
    _, I = index.search(np.array([query_vec]), k=5)
    retrieved = [texts[i] for i in I[0]]

    prompt = f"Answer this query using the following quotes:\n{retrieved}\nQuery: {query}"
    response = generator(prompt, do_sample=True, temperature=0.7)
    answer = response[0]['generated_text'], retrieved
    return answer