from dotenv import load_dotenv
load_dotenv()
import sys
import os
import tempfile
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.document_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import embed_documents
from app.vector_store import VectorStore
from app.rag_pipeline import ask


st.set_page_config(
    page_title="Nexus Research",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --blue: #2f6bff;
    --blue-dark: #1f4dcc;
    --ink: #101828;
    --ink-soft: #344054;
    --muted: #667085;
    --canvas: #f4f7fb;
    --white: #ffffff;
    --line: #e4e9f2;
    --blue-pale: #edf3ff;
    --green: #0e9f6e;
    --orange: #f79009;
}

.stApp { background:var(--canvas); color:var(--ink); font-family:'Space Grotesk',sans-serif; }
.block-container { max-width:1180px; padding:1.25rem 2.5rem 3rem; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

/* Header */
.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:2.2rem; }
.brand { display:flex; align-items:center; gap:.7rem; color:var(--ink); font-size:1rem; font-weight:700; letter-spacing:-.04em; }
.brand-mark { display:grid; place-items:center; width:30px; height:30px; border-radius:9px; background:var(--ink); color:#fff; font-size:.85rem; font-weight:700; }
.brand-mark:after { content:''; width:7px; height:7px; border-radius:50%; background:var(--blue); position:absolute; margin:17px 0 0 17px; border:2px solid var(--ink); }
.top-meta { display:flex; align-items:center; gap:1rem; color:#8993a4; font:500 .62rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
.online { display:flex; align-items:center; gap:.4rem; }
.online-dot { width:6px; height:6px; border-radius:50%; background:#20b47a; }

/* Steps */
.stepper { display:flex; align-items:center; gap:.7rem; margin-bottom:2.2rem; }
.step { display:flex; align-items:center; gap:.4rem; color:#98a2b3; font-size:.7rem; font-weight:600; }
.step.active { color:var(--blue); }
.step.done { color:var(--green); }
.step-num { display:grid; place-items:center; width:24px; height:24px; border:1px solid #d0d5dd; border-radius:50%; font:500 .63rem 'DM Mono',monospace; }
.step.active .step-num { background:var(--blue); color:#fff; border-color:var(--blue); }
.step.done .step-num { background:#e6f7f0; color:var(--green); border-color:#b7ebd6; }
.step-line { width:52px; height:1px; background:#dfe4ec; }

/* Upload screen */
.upload-grid { display:grid; grid-template-columns:1.05fr .95fr; gap:5rem; align-items:center; min-height:510px; }
.kicker { color:var(--blue); font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.14em; }
.upload-copy h1 { color:var(--ink); font-size:clamp(3rem,5.7vw,5.2rem); line-height:.94; letter-spacing:-.085em; margin:.8rem 0 1.15rem; font-weight:700; }
.upload-copy h1 span { color:var(--blue); }
.upload-copy p { max-width:430px; color:var(--muted); font-size:1rem; line-height:1.65; margin:0; }
.project-note { max-width:430px; color:#98a2b3; font-size:.78rem !important; line-height:1.55 !important; margin:.85rem 0 0 !important; }
.feature-list { display:flex; gap:.55rem; margin-top:2rem; flex-wrap:wrap; }
.feature { display:flex; align-items:center; gap:.4rem; color:#667085; background:#fff; border:1px solid var(--line); border-radius:999px; padding:.48rem .7rem; font-size:.68rem; }
.feature-check { color:var(--green); font-weight:700; }
.upload-panel { background:var(--white); border:1px solid var(--line); border-radius:20px; padding:1.3rem; box-shadow:0 18px 45px #162b4d0d; }
.upload-container { background:var(--white); border:1px solid var(--line); border-radius:20px; padding:1.1rem 1.2rem .85rem; box-shadow:0 18px 45px #162b4d0d; }
.panel-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; }
.panel-title { color:var(--ink); font-size:1.05rem; font-weight:700; letter-spacing:-.03em; }
.panel-subtitle { color:#98a2b3; font-size:.72rem; margin-top:.25rem; }
.file-type { color:#8590a2; font:500 .6rem 'DM Mono',monospace; border:1px solid var(--line); border-radius:6px; padding:.33rem .45rem; }
[data-testid="stFileUploader"] { background:#f9fbff; border:1px dashed #b4c6e6; border-radius:13px; padding:.15rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--white); border:1px solid var(--line); border-radius:20px; padding:1.1rem 1.2rem .85rem; box-shadow:0 18px 45px #162b4d0d; }
[data-testid="stFileUploaderDropzone"] { background:transparent; }
[data-testid="stFileUploaderDropzoneInstructions"] div { color:#667085; font-size:.75rem; }
.upload-bottom { display:flex; justify-content:space-between; gap:.5rem; color:#98a2b3; font-size:.65rem; margin-top:.8rem; }

/* Processing */
.processing { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:520px; text-align:center; }
.processing-ring { display:grid; place-items:center; width:72px; height:72px; border:2px solid #cbd9f5; border-top-color:var(--blue); border-radius:50%; animation:spin 1s linear infinite; }
.processing-ring span { color:var(--blue); font-weight:700; animation:counterspin 1s linear infinite; }
.processing h2 { margin:1.2rem 0 .45rem; font-size:1.35rem; letter-spacing:-.04em; }
.processing p { color:var(--muted); font-size:.8rem; }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes counterspin { to { transform:rotate(-360deg); } }

/* Ask screen */
.ask-wrap { max-width:920px; margin:0 auto; }
.ask-heading { display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:1.4rem; }
.ask-heading h1 { color:var(--ink); font-size:clamp(2.4rem,4vw,4rem); line-height:.95; letter-spacing:-.08em; margin:0; }
.ask-heading h1 span { color:var(--blue); }
.library-tag { display:flex; align-items:center; gap:.45rem; color:var(--green); background:#e9f8f1; border:1px solid #c9eedf; border-radius:999px; padding:.45rem .65rem; font:500 .61rem 'DM Mono',monospace; white-space:nowrap; }
.library-tag:before { content:'✓'; font-size:.75rem; }
.ask-label { color:var(--blue); font:500 .62rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; margin:1.5rem 0 .55rem; }
[data-testid="stTextInput"] input { background:#fff; color:var(--ink); border:1px solid #cfd8e7; border-radius:10px; padding:.82rem .95rem; font-size:.9rem; box-shadow:0 4px 14px #1d3a6810; }
[data-testid="stTextInput"] input:focus { border-color:var(--blue); box-shadow:0 0 0 3px #2f6bff22; }
.ask-controls [data-testid="stCheckbox"] label { color:#667085 !important; font-size:.72rem; }
.ask-controls .stButton > button { background:var(--blue); border-color:var(--blue); color:#fff; }
.ask-controls .stButton > button:hover { background:#5687ff; border-color:#5687ff; }
.quick-label { color:#98a2b3; font:500 .61rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.11em; margin:1.2rem 0 .55rem; }
.quick-button .stButton > button { background:#fff; color:#475467; border:1px solid var(--line); font-size:.69rem; font-weight:500; padding:.45rem .55rem; }
.quick-button .stButton > button:hover { color:var(--blue); border-color:#b4c6e6; background:var(--blue-pale); }

/* Results */
.result-bar { display:flex; justify-content:space-between; align-items:center; margin:1.55rem 0 .65rem; }
.result-title { color:var(--ink); font-size:.9rem; font-weight:700; }
.result-label { color:#98a2b3; font:500 .6rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
.answer-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; }
.answer-question { color:#98a2b3; font:500 .6rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.55rem; }
.answer-text { color:#344054; font-size:.92rem; line-height:1.75; }
.source-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.65rem; }
.source-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.8rem; }
.source-number { color:var(--blue); font:500 .6rem 'DM Mono',monospace; }
.source-name { color:var(--ink); font-size:.72rem; font-weight:700; line-height:1.35; margin:.35rem 0; word-break:break-word; }
.source-page { color:#8993a4; font-size:.65rem; }
.source-score { color:var(--green); font:500 .58rem 'DM Mono',monospace; margin-top:.5rem; }
.empty { color:#98a2b3; background:#fff; border:1px dashed #cbd5e1; border-radius:14px; text-align:center; padding:1.2rem; font-size:.75rem; }
.history-row { display:flex; gap:.7rem; }
.history-item { flex:1; min-width:0; border-top:1px solid var(--line); padding-top:.6rem; color:#667085; font-size:.68rem; line-height:1.4; }
.history-time { color:#98a2b3; font:500 .57rem 'DM Mono',monospace; margin-top:.25rem; }

.stButton > button { border-radius:9px; font-weight:600; transition:all .18s ease; }
.stButton > button:hover { transform:translateY(-1px); }
[data-testid="stExpander"] { border-color:var(--line); border-radius:10px; }
@media (max-width:800px) {
    .block-container { padding-left:1rem; padding-right:1rem; }
    .upload-grid { grid-template-columns:1fr; gap:1.5rem; min-height:auto; }
    .upload-copy h1 { font-size:3.4rem; }
    .ask-heading { display:block; }
    .library-tag { display:inline-flex; margin-top:.8rem; }
    .source-grid { grid-template-columns:1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# State
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []
if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None
if "pending_files" not in st.session_state:
    st.session_state.pending_files = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "history" not in st.session_state:
    st.session_state.history = []


def process_documents(files):
    all_chunks = []
    names = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as file:
                file.write(uploaded_file.getvalue())
            pages = load_pdf(path)
            all_chunks.extend(chunk_documents(pages))
            names.append(uploaded_file.name)

    if not all_chunks:
        raise ValueError("No extractable text found in the uploaded PDFs.")

    embedded_chunks = embed_documents(all_chunks)
    store = VectorStore(dimension=len(embedded_chunks[0]["embedding"]))
    store.add_chunks(embedded_chunks)
    st.session_state.vector_store = store
    st.session_state.indexed_documents = names
    return len(names), len(all_chunks)


def signature(files):
    return tuple((file.name, file.size) for file in files)


def stepper(active):
    steps = [("upload", "1", "Upload"), ("processing", "2", "Prepare"), ("ask", "3", "Ask")]
    html = '<div class="stepper">'
    for index, (name, number, label) in enumerate(steps):
        state = "active" if active == name else ("done" if (active == "ask" or active == "processing" and index == 0) else "")
        html += f'<div class="step {state}"><span class="step-num">{"✓" if state == "done" else number}</span>{label}</div>'
        if index < 2:
            html += '<div class="step-line"></div>'
    st.markdown(html + '</div>', unsafe_allow_html=True)


# Global header
st.markdown('<div class="topbar"><div class="brand"><div class="brand-mark">N</div> NEXUS RESEARCH</div><div class="top-meta"><span class="online"><span class="online-dot"></span> session ready</span><span>private</span></div></div>', unsafe_allow_html=True)


# Upload screen: file selection automatically moves into preparation.
if st.session_state.stage == "upload":
    stepper("upload")
    copy_column, upload_column = st.columns([1.05, .95], gap="large")
    with copy_column:
        st.markdown('<div class="upload-copy"><div class="kicker">AI research workspace</div><h1>Turn documents<br>into <span>clarity.</span></h1><p>Upload your papers, reports, or notes to begin.</p><p class="project-note">Nexus transforms your documents into a focused research assistant that finds relevant passages, explains the key ideas, and keeps every answer connected to its source.</p></div>', unsafe_allow_html=True)
    with upload_column:
        with st.container(border=True):
            st.markdown('<div class="panel-top"><div><div class="panel-title">Add your documents</div><div class="panel-subtitle">Select one or more PDF files to begin</div></div><div class="file-type">PDF</div></div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Upload", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        current_signature = signature(uploaded_files)
        if current_signature != st.session_state.indexed_signature:
            st.session_state.pending_files = uploaded_files
            st.session_state.stage = "processing"
            st.rerun()


# Automatic preparation screen.
elif st.session_state.stage == "processing":
    stepper("processing")
    st.markdown('<div class="processing"><div class="processing-ring"><span>N</span></div><h2>Preparing your research space</h2><p>Reading pages and organizing searchable evidence.</p></div>', unsafe_allow_html=True)
    files = st.session_state.pending_files
    if files:
        with st.status("Processing your documents…", expanded=True) as status:
            status.write("Extracting text from pages")
            document_count, chunk_count = process_documents(files)
            status.write(f"Created {chunk_count} searchable passages")
            status.update(label=f"Ready · {document_count} document(s)", state="complete", expanded=False)
        st.session_state.indexed_signature = signature(files)
        st.session_state.stage = "ask"
        st.rerun()
    else:
        st.error("The uploaded files are no longer available. Please refresh and try again.")


# Ask screen.
else:
    stepper("ask")
    
    # Check for API key (Required for LLM generation)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    
    if not api_key:
        st.warning("⚠️ API Key is missing! The LLM will not be able to generate answers until this is set in the `.env` file or Streamlit secrets.")
        
    st.markdown('<div class="ask-wrap"><div class="ask-heading"><h1>Ask your<br><span>documents.</span></h1><div class="library-tag">' + str(len(st.session_state.indexed_documents)) + ' documents ready</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ask-label">Research question</div>', unsafe_allow_html=True)
    question = st.text_input("Question", value=st.session_state.question, placeholder="e.g. What are the main findings?", label_visibility="collapsed")
    st.markdown('<div class="ask-controls">', unsafe_allow_html=True)
    ask_col, search_col = st.columns([1, 1])
    with ask_col:
        ask_clicked = st.button("Find answer  →", type="primary", use_container_width=True)
    with search_col:
        use_reranker = st.checkbox("Deeper search", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="quick-label">Start with a prompt</div>', unsafe_allow_html=True)
    quick_cols = st.columns(3)
    prompts = ["Summarize the main findings", "What supports the conclusion?", "What are the limitations?"]
    for col, prompt in zip(quick_cols, prompts):
        with col:
            if st.button(prompt, key=f"prompt_{prompt}", use_container_width=True):
                st.session_state.question = prompt
                st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Finding the clearest evidence…"):
                try:
                    result = ask(question.strip(), st.session_state.vector_store, top_k=3, use_reranker=use_reranker)
                    st.session_state.last_result = result
                    st.session_state.question = question.strip()
                    st.session_state.history.insert(0, {"question": question.strip(), "time": datetime.now().strftime("%H:%M")})
                    st.session_state.history = st.session_state.history[:5]
                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")

    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown('<div class="result-bar"><div class="result-title">Answer</div><div class="result-label">grounded in your sources</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card"><div class="answer-question">{st.session_state.question}</div><div class="answer-text">{result.get("answer", "No answer returned.")}</div></div>', unsafe_allow_html=True)
        sources = result.get("sources", [])
        if sources:
            st.markdown('<div class="result-bar"><div class="result-title">Sources</div><div class="result-label">evidence trail</div></div>', unsafe_allow_html=True)
            cards = '<div class="source-grid">'
            for index, source in enumerate(sources, 1):
                score = source.get("score", source.get("reranker_score", 0))
                cards += f'<div class="source-card"><div class="source-number">SOURCE {index:02d}</div><div class="source-name">{source.get("source", "Unknown document")}</div><div class="source-page">Page {source.get("page_number", "—")}</div><div class="source-score">match · {score:.3f}</div></div>'
            st.markdown(cards + '</div>', unsafe_allow_html=True)
            with st.expander("View retrieved passages"):
                for index, source in enumerate(sources, 1):
                    st.markdown(f"**{index}. {source.get('source', 'Unknown')} · page {source.get('page_number', '—')}**")
                    st.caption(source.get("text", "No passage text available."))
    else:
        st.markdown('<div class="result-bar"><div class="result-title">Answer</div><div class="result-label">waiting for your question</div></div><div class="empty">Ask a question above and your grounded answer will appear here.</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="result-bar"><div class="result-title">Recent questions</div><div class="result-label">this session</div></div>', unsafe_allow_html=True)
        history = '<div class="history-row">'
        for item in st.session_state.history[:3]:
            history += f'<div class="history-item">{item["question"]}<div class="history-time">{item["time"]}</div></div>'
        st.markdown(history + '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
