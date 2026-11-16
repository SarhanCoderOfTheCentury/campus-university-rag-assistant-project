"""Single-page Streamlit interface for the supplied, locally indexed knowledge base."""
import hashlib
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from rag.config import ROOT, Settings
from rag.embeddings import GeminiEmbeddings
from rag.store import load_corpus
from rag.agent import RAGAgent, safe_error

load_dotenv(ROOT / '.env')
st.set_page_config(page_title='Campus Assistant', page_icon='🎓', layout='centered', initial_sidebar_state='collapsed')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [data-testid="stApp"], button, input, textarea {font-family:'DM Sans',Arial,sans-serif}
.block-container {max-width:820px;padding-top:3rem;padding-bottom:5rem}
h1 {font-size:2.3rem!important;letter-spacing:-.045em}
[data-testid="stChatMessage"] {border-radius:18px;padding:1rem 1.2rem;margin-bottom:.75rem}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {line-height:1.65}
[data-testid="stChatMessage"] li {margin-bottom:.45rem}
[data-testid="stChatMessage"] ul {padding-left:1.35rem}
div.stButton button {border-radius:20px}
</style>''', unsafe_allow_html=True)


def secret(name):
    if os.getenv(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, ''))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ''


@st.cache_resource(show_spinner=False)
def load_agent(settings, key_revision, _key):
    embeddings = GeminiEmbeddings(_key)
    store, chunks, manifest = load_corpus(settings, embeddings)
    return RAGAgent(settings, embeddings, store, chunks, api_key=_key), manifest


def progressive(text):
    # Every word has already passed evidence and citation validation.
    for word in re.findall(r'\S+\s*|\s+', text):
        yield word
        time.sleep(.012)


def display(result, animate=False):
    if result['status'] == 'api_error':
        st.error(result['answer'])
    elif animate and result['status'] == 'answered':
        st.write_stream(progressive(result['answer']))
    else:
        st.markdown(result['answer'])
    if result.get('sources'):
        with st.expander(f"View sources · {len(result['sources'])}"):
            for i, source in enumerate(result['sources'], 1):
                st.markdown(f"**[{i}] {source['filename']} · {source['version']}**")
                st.caption(source['quote'])


st.session_state.setdefault('messages', [])
with st.sidebar:
    st.markdown('### Your conversation')
    if st.button('New conversation', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        transcript = '\n\n'.join(f"**You:** {m['question']}\n\n**Assistant:** {m['result']['answer']}"
            for m in st.session_state.messages)
        st.download_button('Download conversation', transcript, file_name='campus-conversation.md',
            mime='text/markdown', use_container_width=True)
    st.divider()
    st.caption('Answers come from the supplied classroom dataset. Confirm real university details with DSU.')

st.title('Campus Assistant')
st.caption('Ask about programs, fees, admissions, and academic policies.')

key = secret('GEMINI_API_KEY')
if not key:
    st.warning('Set GEMINI_API_KEY in Streamlit app secrets to start asking questions.')
    st.stop()
settings = Settings(model=secret('GEMINI_MODEL') or 'gemini-3.5-flash-lite', k=6)
try:
    revision = hashlib.sha256(key.encode()).hexdigest()[:16]
    agent, manifest = load_agent(settings, revision, key)
except Exception as exc:
    st.error(safe_error(exc))
    st.stop()

suggested = None
if not st.session_state.messages:
    st.write('Explore the knowledge base with a question, or try one of these:')
    prompts = ['What programs are offered?', 'Compare old and current attendance rules', 'How do I apply?']
    cols = st.columns(3)
    for i, prompt in enumerate(prompts):
        if cols[i].button(prompt, use_container_width=True):
            suggested = prompt

for item in st.session_state.messages:
    with st.chat_message('user'):
        st.write(item['question'])
    with st.chat_message('assistant'):
        display(item['result'])

question = suggested or st.chat_input('Ask a question…', max_chars=3000)
if question and question.strip():
    with st.chat_message('user'):
        st.write(question)
    with st.chat_message('assistant'):
        with st.spinner('Checking the documents…'):
            history = [{'question': m['question'], 'answer': m['result']['answer'][:600]}
                for m in st.session_state.messages[-4:] if m['result']['status'] == 'answered']
            result = agent.ask(question, history=history)
        display(result, animate=True)
    st.session_state.messages.append({'question': question, 'result': result})
