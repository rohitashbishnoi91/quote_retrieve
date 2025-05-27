from sentence_transformers import SentenceTransformer
from data_prep import load_and_preprocess

def load_model_and_data():
    model = SentenceTransformer("fine_tuned_quote_model")
    dataset = load_and_preprocess()
    return model, dataset
