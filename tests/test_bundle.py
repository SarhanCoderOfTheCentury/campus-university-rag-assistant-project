from rag.config import Settings
from rag.store import load_corpus
from rag.retrieval import search
from rag.agent import Answer, Claim, Citation, RAGAgent, validate_answer


class StubEmbeddings:
    def embed_query(self, text):
        import numpy as np
        with np.load('data/embeddings.npz', allow_pickle=False) as bundle:
            return bundle['embeddings'][10].tolist()


def test_bundle_and_local_faiss():
    emb = StubEmbeddings()
    store, chunks, manifest = load_corpus(Settings(), emb)
    assert (manifest['rows'], manifest['dimensions'], store.index.ntotal) == (45, 768, 45)
    hits, _ = search(store, chunks, emb, 'software engineering program', k=5)
    assert hits and all(h['document'].metadata['status'] == 'current' for h in hits)


def test_policy_scope_and_exact_citations():
    emb = StubEmbeddings()
    store, chunks, _ = load_corpus(Settings(), emb)
    old, _ = search(store, chunks, emb, 'previous attendance policy', k=5, intent='archived')
    assert old and all(h['document'].metadata['status'] == 'archived' for h in old)
    evidence = [{'id': 'x', 'text': 'Attendance requires 80 percent.', 'filename': 'policy.md',
        'status': 'current', 'version': '3.0', 'source_url': ''}]
    answer = Answer(insufficient=False, claims=[Claim(text='Attendance requires 80 percent.',
        citations=[Citation(chunk_id='x', quote='Attendance requires 80 percent.')])])
    assert validate_answer(answer, evidence)['status'] == 'answered'
    answer.claims[0].citations[0].quote = 'fabricated quotation'
    assert validate_answer(answer, evidence)['status'] == 'invalid_citations'


def test_catalog_covers_every_supplied_program_without_generation():
    settings = Settings(k=6)
    emb = StubEmbeddings()
    store, chunks, _ = load_corpus(settings, emb)
    agent = RAGAgent.__new__(RAGAgent)
    agent.store, agent.chunks, agent.embeddings, agent.settings = store, chunks, emb, settings
    state = {'question': 'What programs are offered?', 'query': 'What programs are offered?', 'intent': 'current'}
    state.update(agent.retrieve(state))
    result = agent.answer(state)['result']
    assert result['status'] == 'answered'
    assert len(result['sources']) == 9
    assert 'BS Software Engineering' in result['answer']
    assert 'Executive MBA' in result['answer']
