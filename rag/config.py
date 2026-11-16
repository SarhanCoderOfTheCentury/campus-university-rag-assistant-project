from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
REFUSAL = 'I could not find enough information in the provided documents'

@dataclass(frozen=True)
class Settings:
    corpus: str = 'dsu'
    k: int = 5
    search: str = 'hybrid'
    chunk_size: int = 1000
    chunk_overlap: int = 150
    context_tokens: int = 16000
    rewrite: bool = False
    rerank: bool = False
    model: str = field(default_factory=lambda: os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'))

    def __post_init__(self):
        if self.corpus != 'dsu':
            raise ValueError('This app uses the supplied DSU dataset.')
        if not 1 <= self.k <= 20:
            raise ValueError('k must be between 1 and 20.')
        if self.search not in ('semantic', 'keyword', 'hybrid'):
            raise ValueError('Unknown search method.')
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError('Overlap must be smaller than chunk size.')
        if not 100 <= self.chunk_size <= 5000:
            raise ValueError('Chunk size must be 100-5000 characters.')
