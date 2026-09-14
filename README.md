python -m venv dissertation
dissertation\Scripts\activate
set PYTHONSAFEPATH=1
python -m uvicorn backend.main:app --reload

all-MiniLM-L6-v2
BAAI/bge-base-en-v1.5
text-embedding-3-small
FAISS
chunk size = 800
overlap = 150
Top-K = 5