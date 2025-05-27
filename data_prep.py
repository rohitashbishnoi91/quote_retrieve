from datasets import load_dataset

def load_and_preprocess():
    dataset = load_dataset("Abirate/english_quotes")
    def preprocess(example):
        if example['quote'] and example['author']:
            tags = ', '.join(example.get('tags', []))
            text = f"{example['quote']} - {example['author']} ({tags})"
            return {"text": text.lower()}
        return None

    processed = dataset["train"].map(preprocess)
    return processed.filter(lambda x: x is not None)
