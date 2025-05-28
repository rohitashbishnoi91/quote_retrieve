# Semantic Quote Retrieval System (RAG-based)

This project is a **semantic quote retrieval system** powered by Retrieval Augmented Generation (RAG). It allows users to query a large collection of English quotes and receive relevant, context-aware responses using state-of-the-art NLP models.

## Demo

[Click here to try the live app!](https://rohitashbishnoi91-quote-retrieve-app-pfgk2e.streamlit.app/)

## Features

- **Semantic search** over a quotes dataset using sentence embeddings
- **Retrieval Augmented Generation (RAG)** pipeline for context-aware answers
- **Streamlit web app** for interactive querying
- **Model fine-tuning** on the Abirate/english_quotes dataset
- **Evaluation** using RAGAS
- **Local LLM generation** (no API key required for generation)

## How It Works

1. **Data Preparation:**  
   The Abirate/english_quotes dataset is downloaded and preprocessed.

2. **Model Fine-Tuning:**  
   A sentence-transformer model is fine-tuned on the quotes for better semantic retrieval.

3. **RAG Pipeline:**  
   - Quotes are embedded and indexed using scikit-learn's NearestNeighbors.
   - The top relevant quotes are retrieved for a user query.
   - A local language model (e.g., distilgpt2) generates a context-aware answer.

4. **Streamlit App:**  
   Users can enter natural language queries and view answers, retrieved quotes, and sources.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rohitashbishnoi91/quote_retrieve.git
cd quote_retrieve
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## File Structure

- `app.py` — Streamlit web interface
- `rag_pipeline.py` — Retrieval and generation logic
- `fine_tune.py` — Model fine-tuning script
- `data_prep.py` — Data loading and preprocessing
- `requirements.txt` — Python dependencies

## Usage

- Open the Streamlit app in your browser.
- Enter a query (e.g., "quotes about hope by Oscar Wilde").
- View the generated answer and the most relevant quotes.

## Customization

- To use a different language model, change `MODEL_NAME` in `rag_pipeline.py`.
- To fine-tune on more data, adjust the scripts in `fine_tune.py` and `data_prep.py`.

## Evaluation

- RAG output can be evaluated using RAGAS (see `evaluate_ragas.py`).



---

**Project by [Rohitash Bishnoi]**