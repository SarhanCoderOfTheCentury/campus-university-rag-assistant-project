"""Build a local FAISS index from verified, precomputed Gemini vectors."""
import hashlib
import json

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from .config import ROOT
from .embeddings import DIMENSIONS, MODEL, normalize


def load_corpus(settings, embeddings):
    base = ROOT / 'data'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['embedding_model'] != MODEL or manifest['vector_dim'] != DIMENSIONS
            or manifest['task_type_index'] != 'RETRIEVAL_DOCUMENT'
            or manifest['task_type_query'] != 'RETRIEVAL_QUERY' or not manifest['normalized']):
        raise ValueError('Embedding contract does not match the supplied corpus.')
    records = [json.loads(line) for line in (base / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    with np.load(base / 'embeddings.npz', allow_pickle=False) as bundle:
        vectors = np.asarray(bundle['embeddings'], dtype=np.float32)
        vector_ids = [str(x) for x in bundle['chunk_ids']]
    ids = [r['chunk_id'] for r in records]
    if len(ids) != len(set(ids)) or ids != vector_ids or vectors.shape != (len(ids), DIMENSIONS):
        raise ValueError('Chunk IDs, vector order, count, or dimensions do not match.')
    if len(ids) != manifest['num_chunks'] or len(ids) != manifest['num_vectors']:
        raise ValueError('Manifest count does not match the supplied data.')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.02):
        raise ValueError('Invalid or unnormalized vectors in embedding bundle.')
    chunks = [Document(page_content=r['text'], metadata={
        'id': r['chunk_id'], 'source': r['source_file'], 'filename': r['source_file'].split('/')[-1],
        'section': r['section_title'], 'doc_type': r['doc_type'], 'version': r['version'],
        'effective_date': r['effective_date'], 'status': r['status'], 'source_url': '',
        'position': r['section_index']}) for r in records]
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(normalize(vectors))
    store = FAISS(embeddings, index, InMemoryDocstore(dict(zip(ids, chunks))),
        dict(enumerate(ids)), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    fingerprint = hashlib.sha256((base/'chunks.jsonl').read_bytes() + (base/'embeddings.npz').read_bytes()).hexdigest()
    return store, chunks, {'rows': len(chunks), 'dimensions': DIMENSIONS, 'model': MODEL, 'fingerprint': fingerprint}
