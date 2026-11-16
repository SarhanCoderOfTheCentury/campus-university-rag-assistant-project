# Campus Assistant

A Streamlit and LangChain knowledge assistant built from the supplied `dsu_rag_bundle.zip`. It uses a local FAISS index, Gemini query embeddings, hybrid semantic/BM25 retrieval, policy-aware filtering, and cited answers. The chat is intentionally one screen; evidence is available under each answer.

> The supplied documents describe themselves as a **fictional classroom DSU dataset**. Their fees, policies, dates, contacts, and offerings are not verified university facts. The app labels this clearly; do not use it for actual admissions decisions.

## Run locally

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put your Gemini API key in .env
streamlit run streamlit_app.py
```

For Streamlit Community Cloud, choose branch `main`, main file `streamlit_app.py`, Python 3.12, and add the following under app **Secrets**:

```toml
GEMINI_API_KEY = "your-key-here"
GEMINI_MODEL = "gemini-3.5-flash-lite"
```

Never commit your key. A key rotation invalidates the cached Gemini clients. A model change affects generation only; an embedding model or dimension change requires rebuilding all vectors.

## Data and retrieval

| Stage | Implementation and check |
| --- | --- |
| Data | 45 self-contained Markdown-section chunks from five supplied documents; `data/chunks.jsonl` |
| Vectors | 45 × 768 float32 vectors with ordered IDs in `data/embeddings.npz` |
| Embedding contract | `gemini-embedding-001`, 768 dimensions, `RETRIEVAL_DOCUMENT` for stored vectors and `RETRIEVAL_QUERY` for questions, L2 normalization |
| Index | FAISS `IndexFlatIP`, reconstructed locally at startup from the matrix; no pickle load |
| Search | Semantic search plus BM25 merged by reciprocal rank fusion; catalog questions include all nine program sections |
| Policies | Current rules exclude archived chunks; explicit comparisons retrieve both v3.0 and v2.1 |
| Generation | LangGraph route → retrieve → answer; Gemini structured claims must carry exact quotations from retrieved chunks |
| Display | Responses render as Markdown with numbered citation markers and a collapsed source panel. Validated text appears progressively after generation. |

The loader checks the manifest model, tasks, dimensions, record count, vector shape, ordered chunk IDs, finite values, and vector norms. It stops if any part of the bundle does not match. The `chroma_db/` directory in the original archive is deliberately not shipped: this project builds a local FAISS store as requested. The bundle's `checkpoint.jsonl` and scripts are not required for querying and are not copied. The excluded test document is also not part of retrieval.

## Features

- One-page responsive chat with DM Sans, suggested questions, and readable grouped program listings.
- Follow-up questions can resolve references against the last four answered turns; conversation text is never treated as factual evidence.
- Source quotations, file names, and policy versions can be inspected under each response.
- New conversation and Markdown transcript download in the collapsed sidebar.
- Current versus archived policy questions preserve version and effective-date distinctions.
- Missing evidence produces an explicit refusal, and provider errors are not presented as factual answers.

## Verify

```bash
python -m pytest -q
```

The tests verify the bundled vector alignment, FAISS search and policy filtering, citation rejection, and app boot. For a live smoke check with a valid key, ask:

1. “What programs are offered?” — should include all nine documented program sections with citations.
2. “Compare old and current attendance rules” — should distinguish archived v2.1 from current v3.0.
3. “What is the president's personal phone number?” — should decline if no supporting document contains it.

The supplied bundle reports five successful semantic checks, but that report is not a substitute for the tests or live checks performed on this application. The exact claim-to-quotation validator verifies the quotation exists; it cannot prove all inferences are logically entailed. Answers should be checked against displayed sources.

## Project layout

```text
streamlit_app.py     Chat UI, session memory, suggestions and transcript export
rag/agent.py         LangGraph flow, Gemini generation and citation validation
rag/retrieval.py     Policy-aware FAISS + BM25 retrieval
rag/store.py         Bundle verification and local FAISS construction
rag/embeddings.py    Compatible Gemini query embeddings
data/                The supplied chunks, vectors and manifest
tests/               Local integration checks
```
