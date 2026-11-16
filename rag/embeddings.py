"""The query MUST use the same model, dimensions and normalization as the corpus."""
import hashlib
from pathlib import Path
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

MODEL = 'gemini-embedding-001'
DIMENSIONS = 768


def normalize(values):
    v = np.asarray(values, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError('Embedding contains non-finite values.')
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if (norms < 1e-10).any():
        raise ValueError('Embedding is a zero vector.')
    return v / norms


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, cache_dir: Path | None = None):
        if not api_key:
            raise ValueError('Set GEMINI_API_KEY in your environment or Streamlit secrets.')
        self.client = GoogleGenerativeAIEmbeddings(
            model=MODEL, api_key=api_key, output_dimensionality=DIMENSIONS,
            request_options={'timeout': 45}, vertexai=False)
        self.cache_dir = cache_dir

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = self.client.embed_documents(texts, task_type='RETRIEVAL_DOCUMENT',
            output_dimensionality=DIMENSIONS, batch_size=20)
        return normalize(values).tolist()

    def embed_query(self, text: str) -> list[float]:
        # The optional disk cache is used by evaluation CLI only; the web app
        # never persists visitor questions. Key includes the entire embedding contract.
        path = None
        if self.cache_dir:
            digest = hashlib.sha256((MODEL + f':{DIMENSIONS}:RETRIEVAL_QUERY:' + text).encode()).hexdigest()
            path = self.cache_dir / (digest + '.npy')
            if path.exists():
                v = np.load(path, allow_pickle=False)
                if v.shape == (DIMENSIONS,):
                    return normalize(v).tolist()
        value = normalize(self.client.embed_query(text,
            task_type='RETRIEVAL_QUERY', output_dimensionality=DIMENSIONS))
        if value.shape != (DIMENSIONS,):
            raise ValueError('Query embedding dimension mismatch.')
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value, allow_pickle=False)
        return value.tolist()
