# Semantic Book NLP

Exploratory and modeling notebooks for a **~7k book metadata** corpus: cleaning and simplifying categories, **Hugging Face** text classification, **emotion** scoring from descriptions, and **semantic search** over book text with **LangChain** and **Chroma**.

## What’s in this repo

| Notebook | Purpose |
|----------|---------|
| [`data-exploration.ipynb`](data-exploration.ipynb) | Load the Kaggle dataset with `kagglehub`, profile fields, and visualize distributions with pandas, seaborn, and matplotlib. |
| [`text-classification.ipynb`](text-classification.ipynb) | Map messy `categories` into a smaller set of **simple categories** using a transformers **zero-shot** classifier. |
| [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb) | Run an **emotion** classification pipeline on book text (DistilRoBERTa-based model). |
| [`vector-search.ipynb`](vector-search.ipynb) | Chunk `tagged_description.txt`, embed with **sentence-transformers** (`all-MiniLM-L6-v2`), store vectors in **Chroma**, and query semantically. |

Supporting CSV and text files (`books_cleaned.csv`, `books_with_categories.csv`, `tagged_description.txt`) are produced or consumed along that pipeline.

## Dataset

Raw tables come from the Kaggle dataset **[7k books with metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)** (`dylanjcastillo/7k-books-with-metadata`). The exploration notebook downloads it via [`kagglehub`](https://github.com/Kaggle/kagglehub).

## Requirements

- **Python 3.11+** (3.13 is used in development; adjust pins if needed.)
- Enough disk and RAM for **PyTorch**, **transformers**, and embedding models (first runs download weights from the Hugging Face Hub).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=semantic-book-nlp --display-name="Semantic Book NLP"
```

### Hugging Face token (optional but recommended)

Some Hub usage is smoother with a token. Copy the example env file and add your token:

```bash
cp .example.env .env
# Edit .env: HUGGINGFACEHUB_API_TOKEN=your_token_here
```

In notebooks that call `load_dotenv()`, secrets are read from `.env` (that file is gitignored).

### Kaggle download

If you use `kagglehub` to pull the dataset, follow [Kaggle’s authentication docs](https://www.kaggle.com/docs/api) so the CLI or `kagglehub` can access your account.

## Tech stack

pandas · numpy · matplotlib · seaborn · Jupyter · **PyTorch** · **transformers** · **tqdm** · **LangChain** (community, text splitters, Hugging Face integrations) · **Chroma** · **sentence-transformers** · **python-dotenv** · **kagglehub**
