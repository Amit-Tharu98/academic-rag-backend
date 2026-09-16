python -m venv dissertation
dissertation\Scripts\activate
python -m uvicorn backend.main:app --reload


# Academic RAG Backend

FastAPI backend for the MSc Computer Science dissertation project:

**Comparative Evaluation of Embedding Models for Retrieval-Augmented Academic Document Analysis**

This repository contains the research and API backend for an academic-document Retrieval-Augmented Generation (RAG) prototype.

## Features

- Academic PDF processing and chunking.
- Fixed research corpus retrieval.
- Temporary multi-PDF collections.
- Sentence Transformer embeddings.
- BGE embeddings.
- OpenAI embeddings.
- FAISS semantic vector retrieval.
- RAG question answering.
- Full collection summarisation.
- Topic-focused summarisation.
- Three-model comparison.
- Retrieval and generation research-result endpoints.
- Source filename/page tracking.
- Retrieval timing and similarity statistics.

## Embedding models

The implementation compares:

1. Sentence Transformer
   - `sentence-transformers/all-MiniLM-L6-v2`

2. BGE
   - `BAAI/bge-base-en-v1.5`

3. OpenAI
   - `text-embedding-3-small`

The OpenAI model requires an API key. The Sentence Transformer and BGE models are downloaded locally through Sentence Transformers/Hugging Face when needed.

## Technology

Important pinned dependencies include:

- Python
- FastAPI 0.141.1
- Uvicorn 0.52.1
- FAISS CPU 1.15.0
- OpenAI 2.53.0
- PyMuPDF 1.28.0
- NumPy 2.5.1
- Pandas 3.0.5
- scikit-learn 1.9.0
- sentence-transformers 5.6.1
- PyTorch 2.13.0
- Transformers 5.14.1
- python-dotenv 1.2.2
- python-multipart 0.0.32
- rouge-score 0.1.2

See `requirements.txt` for the complete pinned environment.

## Prerequisites

Install:

1. Git
2. Python compatible with the pinned packages
3. Internet access for initial Python/model installation
4. A valid OpenAI API key

Because this repository pins recent package versions, using the same Python version used by the project environment is safest. If a dependency has no wheel for your Python version, create the environment with a Python version supported by the pinned `torch`, `faiss-cpu` and other packages.

Check Python:

```bash
python --version
```

On macOS/Linux you may need:

```bash
python3 --version
```

## Clone the repository

```bash
git clone <BACKEND_REPOSITORY_URL>
cd academic-rag-backend
```

Replace `<BACKEND_REPOSITORY_URL>` with this repository's URL.

Important: run all backend commands from the repository root. Several project paths are relative to the root directory.

## Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When activated, the terminal normally shows `(.venv)`.

## Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Install dependencies

```bash
pip install -r requirements.txt
```

Installation may take some time because PyTorch, FAISS, Transformers and Sentence Transformers are substantial dependencies.

## Configure the OpenAI API key

Create a file named:

```text
.env
```

in the repository root.

Add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own key.

The backend calls `load_dotenv()` and reads the key using:

```python
os.getenv("OPENAI_API_KEY")
```

If the variable is missing, OpenAI embedding functionality cannot start.

Never commit the real `.env` file.

A useful repository file is `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This example may be committed because it contains no real secret.

## Recommended `.gitignore`

Ensure secrets and generated files are excluded as appropriate:

```gitignore
.env
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
temp_uploads/
```

Do not blindly ignore research data/results that are intentionally required for reproducing the fixed-corpus evaluation or for serving the Research Results page. Decide which research artifacts are part of the reproducible repository before excluding them.

## Start the backend

From the repository root, with the virtual environment activated:

```bash
python -m uvicorn backend.main:app --reload
```

The API should run at:

```text
http://127.0.0.1:8000
```

## Verify the server

Open:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "Academic RAG API is running."
}
```

Health endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/health
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## Connect the Angular frontend

The backend CORS configuration allows the local Angular frontend at:

```text
http://localhost:4200
```

The frontend development environment should use:

```text
http://127.0.0.1:8000/api
```

Start the backend before the frontend.

## Main API groups

### Fixed research corpus

```text
POST /api/ask
POST /api/compare
```

### Uploaded PDF collections

```text
POST /api/upload
POST /api/upload/ask
POST /api/upload/compare
POST /api/upload/summary
POST /api/upload/summary/compare
```

### Research results

```text
GET /api/research/overview
GET /api/research/retrieval
GET /api/research/generation
GET /api/research/retrieval/details
GET /api/research/generation/details
```

Use Swagger at `/docs` to inspect the current request/response schemas.

## PDF upload rules

The backend validates uploads server-side.

Current limits:

```text
Maximum files per collection: 10
Maximum size per PDF:         10 MB
Maximum combined size:        50 MB
File type:                    PDF only
```

Empty PDFs are rejected.

These limits apply to the interactive uploaded-document demonstrator and do not modify the controlled research benchmark.

## Chunking

Uploaded PDFs use:

```text
Chunk size:     800 characters
Chunk overlap:  150 characters
```

Multiple PDFs uploaded in one request are processed into a single temporary collection under one `document_id`.

The collection stores document metadata and combined chunks while preserving source filenames/page information for retrieval evidence.

## Embedding behaviour

### Sentence Transformer

```text
sentence-transformers/all-MiniLM-L6-v2
```

### BGE

```text
BAAI/bge-base-en-v1.5
```

BGE query encoding adds the retrieval instruction:

```text
Represent this sentence for searching relevant passages:
```

### OpenAI

```text
text-embedding-3-small
```

OpenAI document embeddings are requested in batches of 100.

## Vector retrieval

The embedding classes return raw vectors. The project's vector-store layer normalises document and query vectors and uses FAISS inner-product search.

For L2-normalised vectors, inner product is equivalent to cosine similarity.

Raw similarity magnitudes should not be directly interpreted as cross-model performance because different embedding spaces can have different score distributions.

For research conclusions, use controlled metrics such as Recall@K, Top-K Accuracy, MRR, generation metrics and retrieval latency.

## First-run model downloads

The first use of Sentence Transformer or BGE may download model files from Hugging Face.

Models:

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-base-en-v1.5
```

Therefore:

- the first model load may be slow;
- internet access is required unless the model is already cached;
- subsequent runs normally reuse the local Hugging Face cache.

## OpenAI usage and cost

OpenAI-dependent features require:

- a valid `OPENAI_API_KEY`;
- network access;
- an OpenAI account/project with API access and available billing/credits as applicable.

Embedding requests use `text-embedding-3-small`. Other OpenAI-dependent generation behaviour is configured in the project's generator code.

## Fixed research corpus

The controlled dissertation evaluation uses a fixed academic-document corpus, separate from temporary user uploads.

The completed experiment used:

```text
30 academic PDFs
414 pages
2,762 chunks
90 evaluation questions
```

For a cloned repository to reproduce fixed-corpus functionality, the required corpus-derived chunks, vector indexes, evaluation files and/or result CSVs must be present in the repository (or separately supplied) at the paths expected by the code.

If large research artifacts are intentionally excluded from GitHub, document where to obtain them and how to rebuild them. A README cannot make the fixed-corpus endpoints reproducible if the required corpus/index files are absent.

## Research evaluation

Retrieval evaluation includes:

- Precision@K
- Recall@K
- Top-K Accuracy
- Mean Reciprocal Rank (MRR)
- Retrieval time

Generation evaluation includes:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Research result files include outputs such as:

```text
results/retrieval_results.csv
results/retrieval_summary.csv
results/generation_results.csv
results/generation_summary.csv
```

Keep the result files required by `research_results_service.py` if you want the Research Results API/page to work immediately after cloning.

## Temporary uploaded collections

Interactive uploads are stored under the project's temporary upload area.

Conceptually:

```text
PDF files
   |
   v
Text extraction
   |
   v
800/150 chunking
   |
   v
Collection chunks
   |
   v
Embedding model
   |
   v
FAISS index
   |
   v
Top-K retrieval
   |
   v
RAG answer / topic summary
```

Temporary uploaded data should normally not be committed to Git.

## Summarisation

The backend supports:

### Full collection summary

Summarises an uploaded PDF or collection. For multi-document collections, the service synthesises the collection while preserving document boundaries.

### Topic-focused summary

Uses embedding retrieval to find chunks relevant to a user-supplied topic and generates a focused summary.

Because topic-focused summarisation involves retrieval, it can be compared across embedding models.

## Run without auto-reload

For a normal local server:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For access from another device on the local network:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Only expose a development server to networks you trust.

## Render deployment

The deployed service uses a start command of the form:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set secrets such as `OPENAI_API_KEY` in the hosting platform's environment-variable settings, never in GitHub.

The current production frontend origin allowed by CORS is:

```text
https://academic-rag-frontend.netlify.app
```

If you deploy your own frontend domain, add that origin to the backend CORS configuration.

## Common problems

### `ModuleNotFoundError`

Confirm you are in the repository root and the virtual environment is active.

Then:

```bash
pip install -r requirements.txt
```

### `uvicorn` command not found

Use:

```bash
python -m uvicorn backend.main:app --reload
```

### `OPENAI_API_KEY is missing from .env`

Create `.env` in the repository root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Restart the backend after creating/changing it.

### OpenAI authentication or quota error

Check that:

- the key is correct;
- the key has not been revoked;
- API access/billing is available;
- `.env` is being loaded from the expected working directory.

### Sentence Transformer/BGE model download fails

Check internet access and retry. Hugging Face model files are downloaded on first use.

### FAISS or PyTorch installation fails

This usually indicates a Python/OS/package compatibility problem.

Check your Python version and use a version for which the exact pinned packages in `requirements.txt` provide compatible distributions.

Do not randomly change package versions if your goal is to reproduce the dissertation environment; create a compatible Python environment first.

### `/api/research/...` fails after cloning

Check whether the required files under `results/`, the fixed-corpus data, and any expected indexes/chunks were included in the clone.

Research endpoints depend on repository artifacts expected by the corresponding services.

### Fixed-corpus Ask Question fails

The fixed-corpus retriever needs the preprocessed corpus/index artifacts expected by `AcademicRetriever`.

If these large/generated files are not committed, rebuild or obtain them before using fixed-corpus endpoints.

### Uploaded PDF endpoints work but fixed-corpus endpoints do not

This usually means the temporary upload pipeline is functioning but the fixed research corpus/index artifacts are missing.

### Frontend CORS error

For local Angular development, use:

```text
http://localhost:4200
```

The supplied backend already permits this origin.

If you use a different frontend origin, update CORS in `backend/main.py`.

### Port 8000 is already in use

Run:

```bash
python -m uvicorn backend.main:app --reload --port 8001
```

Then change the frontend development API URL to:

```text
http://127.0.0.1:8001/api
```

### First request is slow

This is normal when an embedding model has to be downloaded or loaded into memory, or when an uploaded collection's vector index has to be built.

## Full local setup checklist

For a new machine:

```text
1. Install Git.
2. Install a compatible Python version.
3. Clone the backend.
4. Create and activate .venv.
5. Upgrade pip.
6. Install requirements.txt.
7. Create .env.
8. Set OPENAI_API_KEY.
9. Confirm required fixed-corpus/research artifacts exist.
10. Start FastAPI.
11. Open http://127.0.0.1:8000/docs.
12. Clone the Angular frontend.
13. Install Node.js/npm.
14. Run npm install.
15. Confirm environment.development.ts points to http://127.0.0.1:8000/api.
16. Run npm start.
17. Open http://localhost:4200.
18. Test Research Results, Ask Question, Upload & Analyse and Compare Models.
```

## Quick start

### Windows

```bash
git clone <BACKEND_REPOSITORY_URL>
cd academic-rag-backend

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Start:

```bash
python -m uvicorn backend.main:app --reload
```

### macOS/Linux

```bash
git clone <BACKEND_REPOSITORY_URL>
cd academic-rag-backend

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Start:

```bash
python -m uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Security

Never commit:

- `.env`
- real OpenAI API keys
- passwords
- personal access tokens
- private credentials

If a secret has ever been committed, removing it from the latest file is not enough: revoke/rotate the secret and clean the repository history where appropriate.

## Research scope

This system is an MSc research prototype and demonstrator, not a production-scale document-management service.

Interactive upload limits are intentionally bounded for predictable processing and resource use. They do not alter the controlled embedding-model evaluation.

## Author

Amit Chaudhary  
MSc Computer Science  
University College Birmingham
