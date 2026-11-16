"""Module 09 ideas: semantic + BM25, RRF, rewrite, re-rank, context budget."""
import re
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi


def words(text):
    return re.findall(r'\w+', text.lower())


def policy_intent(question):
    q = question.lower()
    old = bool(re.search(r'\b(old|previous|archived|formerly|before|used to|v2[.]1|purani|pehle)\b', q))
    comparison = bool(re.search(r'\b(compare|comparison|versus|vs|changed|change|difference|then and now)\b', q))
    return 'compare' if comparison else ('archived' if old else 'current')


def allowed(doc, intent):
    status = doc.metadata.get('status', '')
    if intent == 'current':
        return not status.startswith('archived')
    if intent == 'archived':
        # Reference information remains accessible, current policy does not.
        return status != 'current'
    return True


def rrf(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (constant + rank)
    return sorted(scores, key=lambda i: (-scores[i], i)), dict(scores)


def search(store, chunks, embeddings, query, k=5, method='hybrid', intent='current', reranker=None):
    eligible = [i for i, d in enumerate(chunks) if allowed(d, intent)]
    if not eligible:
        return [], {'method': method, 'eligible': 0}
    pool = min(len(eligible), max(k * 4, 20))
    semantic, cosine = [], {}
    if method != 'keyword':
        qv = np.asarray(embeddings.embed_query(query), dtype=np.float32)
        # Search the complete tiny corpus, then FILTER BEFORE top-k. Avoid
        # old-policy candidates crowding out valid current records.
        scores, indices = store.index.search(qv[None, :], len(chunks))
        valid = set(eligible)
        for score, i in zip(scores[0], indices[0]):
            if int(i) in valid:
                cosine[int(i)] = float(score)
                semantic.append(int(i))
        semantic = semantic[:pool]
    lexical = []
    bm_scores = {}
    if method != 'semantic':
        bm = BM25Okapi([words(chunks[i].page_content) or ['_'] for i in eligible])
        values = bm.get_scores(words(query))
        bm_scores = {i: float(s) for i, s in zip(eligible, values)}
        lexical = sorted((i for i in eligible if bm_scores[i] > 0), key=lambda i: -bm_scores[i])[:pool]
    if method == 'hybrid':
        ranking, fused = rrf([semantic, lexical])
    else:
        ranking, fused = (semantic if method == 'semantic' else lexical), {}
    if reranker and ranking:
        ranking = reranker(query, ranking[:pool], chunks)
    # Explicit old/new comparison allocates one slot per policy version if
    # candidates exist. k=1 cannot cover two documents and remains a measured failure.
    if intent == 'compare' and k >= 2:
        reserved = []
        for prefix in ('current', 'archived'):
            match = next((i for i in ranking if chunks[i].metadata.get('status','').startswith(prefix)), None)
            if match is not None:
                reserved.append(match)
        ranking = reserved + [i for i in ranking if i not in reserved]
    selected, seen = [], set()
    for i in ranking:
        c = chunks[i]
        # Remove exact duplicates only; separate windows from one parent may
        # contain complementary rules and must not be collapsed blindly.
        key = (c.metadata['source'], c.page_content)
        if key in seen:
            continue
        seen.add(key)
        selected.append({'document': c, 'cosine': cosine.get(i),
            'bm25': bm_scores.get(i), 'rrf': fused.get(i), 'index': i})
        if len(selected) >= k:
            break
    trace = {'method': method, 'eligible': len(eligible), 'intent': intent,
        'semantic_ids': [chunks[i].metadata['id'] for i in semantic],
        'keyword_ids': [chunks[i].metadata['id'] for i in lexical]}
    return selected, trace


def pack_context(hits, token_budget=6000):
    # UTF-8 bytes upper-bound the token count for byte-based tokenizers.
    # Reserve 500 tokens for JSON wrappers. This is conservative, not an
    # assertion of an exact Gemini token count.
    budget = max(0, token_budget - 500)
    used, evidence = 0, []
    for hit in hits:
        d = hit['document']
        size = len(d.page_content.encode('utf-8')) + 250
        if used + size > budget:
            continue
        used += size
        evidence.append({'id': d.metadata['id'], 'filename': d.metadata['source'],
            'version': d.metadata.get('version'), 'status': d.metadata.get('status'),
            'effective_date': d.metadata.get('effective_date'),
            'source_url': d.metadata.get('source_url'), 'text': d.page_content})
    return evidence, used
