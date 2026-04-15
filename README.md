# Semantic Book Recommendor

End-to-end NLP project on the **7k books metadata** dataset: exploration, category simplification, emotion scoring, vector search, and a **Gradio** recommendation dashboard.

## Project Overview

This repo walks through a practical recommendation workflow:

1. Explore and clean the dataset.
2. Normalize noisy categories into simpler labels.
3. Infer emotions from book descriptions.
4. Build semantic retrieval with embeddings + Chroma.
5. Serve recommendations in a Gradio UI.

## Repo Contents

| File | Purpose |
|------|---------|
| [`data-exploration.ipynb`](data-exploration.ipynb) | Data loading, profiling, and visualization. |
| [`text-classification.ipynb`](text-classification.ipynb) | Category simplification with transformer-based classification. |
| [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb) | Emotion scoring for each book description. |
| [`vector-search.ipynb`](vector-search.ipynb) | Build semantic search with embeddings and Chroma. |
| [`gradio-dashboard.py`](gradio-dashboard.py) | Interactive semantic recommendation interface. |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

## Dataset

The raw source is Kaggle: [7k books with metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (`dylanjcastillo/7k-books-with-metadata`).

Download/auth is handled in notebook workflow via `kagglehub`. See [Kaggle API docs](https://www.kaggle.com/docs/api) if authentication is needed.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=semantic-book-nlp --display-name="Semantic Book NLP"
```

Optional: configure environment variables.

```bash
cp .example.env .env
# Edit .env and set: HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Recommended Run Order

Run notebooks in this order for reproducible outputs:

1. [`data-exploration.ipynb`](data-exploration.ipynb)
2. [`text-classification.ipynb`](text-classification.ipynb)
3. [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb)
4. [`vector-search.ipynb`](vector-search.ipynb)

These steps create and/or rely on intermediate artifacts such as:
- `books_with_emotions.csv`
- `tagged_description.txt`

## Run the Dashboard

After generating the required files, launch:

```bash
python gradio-dashboard.py
```

Then open the local URL printed in terminal (usually `http://127.0.0.1:7860`).

## Tech Stack

- Data: `pandas`, `numpy`, `matplotlib`, `seaborn`
- NLP/ML: `torch`, `transformers`, `tqdm`
- Retrieval: `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`, `chromadb`, `sentence-transformers`
- App/UI: `gradio`
- Utilities: `python-dotenv`, `kagglehub`, `jupyter`, `ipykernel`

## Troubleshooting

- **`chunk_size must be > 0`**: ensure your splitter config uses a positive value.
- **Missing model/token errors**: add `HUGGINGFACEHUB_API_TOKEN` in `.env`.
- **Slow first run**: expected; model weights and embeddings are downloaded/cached.
