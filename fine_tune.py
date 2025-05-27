from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from data_prep import load_and_preprocess

def fine_tune_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    dataset = load_and_preprocess()
    train_data = [InputExample(texts=["quote", d["text"]], label=1.0) for d in dataset.select(range(1000))]
    train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_loader, loss)], epochs=1, warmup_steps=100)
    model.save("fine_tuned_quote_model")

if __name__ == "__main__":
    fine_tune_model()
