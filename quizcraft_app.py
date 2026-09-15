"""Streamlit UI for testing QuizCraft RAG quiz generation."""
from __future__ import annotations

import html
import os
import tempfile
import uuid

import streamlit as st

# Teammate's baseline generation
import generation
from generation import GenerationError, generate_baseline_quiz

# Tested RAG backend
from rag.loaders import load_document
from rag.chunker import chunk_document
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import DocumentRetriever

# Tested Gemini/quiz backend
from llm.gemini import GeminiClient
from quiz.generator import QuizGenerator
from quiz.scorer import calculate_score
from llm.schemas import Quiz as RAGQuiz


AVAILABLE_FLASH_MODEL = "gemini-3.6-flash"


st.set_page_config(
    page_title="QuizCraft RAG",
    page_icon="Q",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --surface: #111827;
        --surface-raised: #182235;
        --surface-soft: #202c42;
        --border: #2d3a52;
        --text: #f5f7fb;
        --muted: #a8b4c7;
        --brand: #8b5cf6;
        --brand-bright: #a78bfa;
        --success: #51d49c;
    }
    .stApp {
        background:
            radial-gradient(circle at 82% -8%, rgba(139, 92, 246, .22), transparent 30rem),
            radial-gradient(circle at 8% 18%, rgba(34, 211, 238, .09), transparent 25rem),
            #0b1120;
        color: var(--text);
    }
    [data-testid="stSidebar"] { background: rgba(17, 24, 39, .94); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h2 { color: var(--text); font-size: 1.1rem; }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label, .stRadio label, .stFileUploader label { color: var(--muted) !important; }
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
        background: var(--surface-raised) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }
    [data-baseweb="select"] * { color: var(--text) !important; }
    .stButton > button {
        min-height: 2.75rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(120deg, #7c3aed, #a855f7) !important;
        color: white !important;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(124, 58, 237, .28);
    }
    .stButton > button:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .quiz-shell {
        max-width: 960px;
        margin: 0 auto;
        padding: 1.75rem 0 4rem;
    }
    .brand-kicker { color: var(--brand-bright); font-size: .75rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    .quiz-title {
        font-size: clamp(2.25rem, 5vw, 3.5rem);
        line-height: 1.05;
        letter-spacing: -.055em;
        font-weight: 800;
        margin: .35rem 0 .55rem;
    }
    .quiz-subtitle {
        color: var(--muted);
        font-size: 1.04rem;
        margin-bottom: 1.75rem;
    }
    .context-panel { border: 1px solid var(--border); background: rgba(24, 34, 53, .7); padding: 1rem 1.15rem; border-radius: 14px; margin-bottom: 1.2rem; }
    .question-card {
        background: linear-gradient(145deg, rgba(32, 44, 66, .94), rgba(17, 24, 39, .96));
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 1.35rem;
        margin: .95rem 0;
        box-shadow: 0 16px 35px rgba(0, 0, 0, .16);
    }
    .question-number { color: var(--brand-bright); font-size: .77rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; margin-bottom: .45rem; }
    .question-title {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .option {
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: .7rem .8rem;
        margin: .45rem 0;
        background: rgba(11, 17, 32, .42);
        color: #d7deeb;
    }
    .answer {
        border-color: rgba(81, 212, 156, .75);
        background: rgba(30, 113, 81, .24);
        color: #c8f5db;
        font-weight: 700;
    }
    .meta {
        margin-top: 1rem;
        padding-top: .85rem;
        border-top: 1px solid rgba(45, 58, 82, .85);
        color: var(--muted);
        font-size: .92rem;
    }
    .meta strong { color: #dbe4f2; }
    [data-testid="stAlert"] { border-radius: 12px; }
    h3 { color: var(--text) !important; margin-top: 1.75rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Heavy backend resources — cached so they survive Streamlit reruns
# ---------------------------------------------------------------------------
if "rag_collection_name" not in st.session_state:
    st.session_state.rag_collection_name = (
        f"streamlit_quizcraft_{uuid.uuid4().hex}"
    )
@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_vector_store(collection_name: str):
    return VectorStore(collection_name=collection_name)

@st.cache_resource
def get_retriever(_embedder, _vector_store):
    return DocumentRetriever(_embedder, _vector_store)


# ---------------------------------------------------------------------------
# Baseline rendering (teammate's original approach — answers shown inline)
# ---------------------------------------------------------------------------
def render_baseline_question_card(index: int, item: dict) -> None:
    """Render a baseline-mode question card exactly as the teammate designed."""
    options_html = []
    for option in item["options"]:
        css_class = "option answer" if option == item["correct_answer"] else "option"
        options_html.append(f"<div class='{css_class}'>{html.escape(option)}</div>")

    st.markdown(
        f"""
        <div class="question-card">
            <div class="question-number">Question {index}</div>
            <div class="question-title">{html.escape(item["question"])}</div>
            {"".join(options_html)}
            <div class="meta"><strong>Explanation:</strong> {html.escape(item["explanation"])}</div>
            <div class="meta"><strong>Source:</strong> {html.escape(item["source_snippet"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# RAG file processing
# ---------------------------------------------------------------------------
def process_uploaded_files():
    """Process every uploaded file through the RAG pipeline."""
    uploaded_files = st.session_state.get("rag_files")

    if not uploaded_files:
        vector_store = get_vector_store(
            st.session_state.rag_collection_name
        )
        vector_store.clear()

        st.session_state.file_indexed = False
        st.session_state.chunk_map = {}
        st.session_state.indexed_filenames = []

        clear_quiz()
        return

    embedder = get_embedder()
    vector_store = get_vector_store(
        st.session_state.rag_collection_name
    )
    retriever = get_retriever(embedder, vector_store)

    with st.spinner("Processing documents…"):
        try:
            # Clear previous index
            vector_store.clear()

            total_chunks = 0
            all_chunk_map = {}
            filenames = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_{uploaded_file.name}",
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    document = load_document(tmp_path)

                    # Keep the original filename for display/source metadata.
                    document.source = uploaded_file.name

                    chunks = chunk_document(
                        document,
                        chunk_size=st.session_state.get("chunk_size", 800),
                        overlap=st.session_state.get("chunk_overlap", 150),
                    )

                    if not chunks:
                        st.warning(
                            f"No text extracted from {uploaded_file.name}."
                        )
                        continue

                    # Give every upload a unique ID so duplicate filenames
                    # cannot produce duplicate chunk IDs.
                    upload_id = uuid.uuid4().hex[:12]

                    for c in chunks:
                        c["chunk_id"] = f"{upload_id}_{c['chunk_id']}"
                        all_chunk_map[c["chunk_id"]] = c

                    retriever.index_documents(chunks)
                    total_chunks += len(chunks)
                    filenames.append(uploaded_file.name)

                except (FileNotFoundError, ValueError) as e:
                    st.error(
                        f"Error processing {uploaded_file.name}: {e}"
                    )

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if total_chunks > 0:
                st.session_state.chunk_map = all_chunk_map
                st.session_state.file_indexed = True
                st.session_state.indexed_filenames = filenames

                st.success(
                    f"Indexed {total_chunks} chunks from: "
                    f"{', '.join(filenames)}"
                )
            else:
                st.session_state.file_indexed = False
                st.error("No valid text found in any uploaded file.")

        except Exception as e:
            st.error(f"Error during indexing: {e}")
            st.session_state.file_indexed = False


# ---------------------------------------------------------------------------
# Quiz state helpers
# ---------------------------------------------------------------------------
def reset_quiz_answers():
    """Clear answers and submission state, keeping the current quiz."""
    st.session_state.quiz_submitted = False
    st.session_state.user_answers = {}
    st.session_state.quiz_results = None

    # Remove Streamlit radio-widget state so Retake really starts fresh.
    for key in list(st.session_state.keys()):
        if key.startswith("q_"):
            del st.session_state[key]


def clear_quiz():
    """Clear the entire quiz along with answers."""
    st.session_state.pop("rag_quiz", None)
    st.session_state.pop("baseline_questions", None)
    reset_quiz_answers()


# ---------------------------------------------------------------------------
# Quiz generation
# ---------------------------------------------------------------------------
def generate_quiz() -> None:
    api_key = st.session_state.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Add a Gemini API key in the sidebar or set GEMINI_API_KEY.")
        return

    mode = st.session_state.get("mode", "Grounded RAG")

    try:
        if mode == "Grounded RAG":
            _generate_rag_quiz(api_key)
        else:
            _generate_baseline_quiz(api_key)
    except Exception as exc:
        st.error(f"Failed to generate quiz: {exc}")


def _generate_rag_quiz(api_key: str) -> None:
    """Run the full RAG pipeline: retrieve → generate → store quiz."""
    if not st.session_state.get("file_indexed"):
        st.error("Please upload and index documents before generating a quiz.")
        return

    topic = st.session_state.get("topic", "").strip()
    if not topic:
        st.error("Please enter a topic.")
        return

    embedder = get_embedder()
    vector_store = get_vector_store(
        st.session_state.rag_collection_name
    )
    retriever = get_retriever(embedder, vector_store)

    with st.spinner("Retrieving context and generating quiz…"):
        context_chunks = retriever.get_relevant_context(
            query=topic,
            top_k=st.session_state.get("retrieval_k", st.session_state.count * 2),
        )

        if not context_chunks:
            st.error("No relevant context found for this topic in the uploaded documents.")
            return

        gemini_client = GeminiClient(api_key=api_key, model=AVAILABLE_FLASH_MODEL)
        quiz_generator = QuizGenerator(gemini_client=gemini_client)

        quiz = quiz_generator.generate(
            topic=topic,
            context_chunks=context_chunks,
            question_count=st.session_state.count,
            difficulty=st.session_state.difficulty.lower(),
        )

        # Store quiz and reset interactive state
        st.session_state.rag_quiz = quiz
        st.session_state.pop("baseline_questions", None)
        reset_quiz_answers()


def _generate_baseline_quiz(api_key: str) -> None:
    """Use the teammate's baseline generation."""
    topic = st.session_state.get("topic", "").strip()
    if not topic:
        st.error("Please enter a topic.")
        return

    generation.MODEL_NAME = AVAILABLE_FLASH_MODEL
    client = generation.init_client(api_key)

    questions = generate_baseline_quiz(
        client,
        topic=topic,
        count=st.session_state.count,
        difficulty=st.session_state.difficulty,
    )

    st.session_state.baseline_questions = questions
    st.session_state.pop("rag_quiz", None)
    reset_quiz_answers()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("quiz_submitted", False),
    ("user_answers", {}),
    ("quiz_results", None),
    ("file_indexed", False),
    ("chunk_map", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Quiz Settings")
    st.text_input("Gemini API Key", type="password", key="api_key")
    st.radio("Mode", ["Grounded RAG", "Baseline"], key="mode")
    st.text_input("Topic", value="Photosynthesis", key="topic")
    st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="difficulty")
    st.slider("Question count", min_value=1, max_value=10, value=5, key="count")
    if st.session_state.get("mode") == "Grounded RAG":
        with st.expander("Advanced RAG settings"):
            st.slider(
                "Chunk size (characters)",
                min_value=200, max_value=2000, value=800, step=100,
                key="chunk_size",
                help="Larger chunks give more context per question but reduce retrieval precision.",
            )
            st.slider(
                "Chunk overlap (characters)",
                min_value=0, max_value=500, value=150, step=50,
                key="chunk_overlap",
                help="Overlap helps avoid cutting a fact in half at a chunk boundary.",
            )
            st.slider(
                "Chunks retrieved per quiz",
                min_value=2, max_value=20, value=10, step=1,
                key="retrieval_k",
                help="More retrieved chunks give the LLM more material to draw from.",
            )
            st.caption("Changing chunk size/overlap only applies on your next file upload — re-upload to reprocess.")
    st.button(
        "Generate Quiz",
        type="primary",
        use_container_width=True,
        on_click=generate_quiz,
    )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.markdown('<div class="quiz-shell">', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-kicker">Grounded learning studio</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="quiz-title">QuizCraft RAG</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="quiz-subtitle">Generate schema-validated quizzes from documents or general knowledge.</div>',
    unsafe_allow_html=True,
)

mode = st.session_state.get("mode", "Grounded RAG")

# ---------------------------------------------------------------------------
# Mode-specific source panel
# ---------------------------------------------------------------------------
if mode == "Grounded RAG":
    st.markdown(
        '<div class="context-panel">Upload your source documents (PDF or TXT). '
        "The RAG pipeline will extract, chunk, embed, and index the content for "
        "semantic retrieval.</div>",
        unsafe_allow_html=True,
    )
    st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="rag_files",
        on_change=process_uploaded_files,
    )

    if st.session_state.get("file_indexed"):
        filenames = st.session_state.get("indexed_filenames", [])
        chunk_count = len(st.session_state.get("chunk_map", {}))
        st.success(
            f"✅ {chunk_count} chunks indexed from: {', '.join(filenames)}"
        )
    elif st.session_state.get("rag_files"):
        st.info("Click the file uploader or re-upload to process your files.")
    else:
        st.info("Upload a PDF or TXT file to get started with Grounded RAG mode.")
else:
    st.info(
        "Baseline mode uses general knowledge and marks every source as General Knowledge."
    )


# ---------------------------------------------------------------------------
# Quiz display
# ---------------------------------------------------------------------------
rag_quiz: RAGQuiz | None = st.session_state.get("rag_quiz")
baseline_questions: list | None = st.session_state.get("baseline_questions")
chunk_map: dict = st.session_state.get("chunk_map", {})

if rag_quiz:
    # -----------------------------------------------------------------------
    # Grounded RAG interactive quiz
    # -----------------------------------------------------------------------
    st.subheader(f"Generated Quiz: {rag_quiz.topic}")

    # Score banner (only after submission)
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        res = st.session_state.quiz_results
        st.markdown(
            "<h3 style='text-align: center;'>Quiz Complete</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h1 style='text-align: center; color: var(--brand-bright);'>"
            f"{res['score']} / {res['total']}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h2 style='text-align: center;'>{res['percentage']:.0f}%</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr>", unsafe_allow_html=True)

    for index, item in enumerate(rag_quiz.questions, start=1):
        # Question header
        st.markdown(
            f"""
            <div class="question-card">
                <div class="question-number">Question {index}</div>
                <div class="question-title">{html.escape(item.question)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.quiz_submitted and st.session_state.quiz_results:
            # --- Post-submission: show results ---
            result = st.session_state.quiz_results["per_question_results"][index - 1]

            if result["is_correct"]:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")

            st.write(f"**Your answer:** {result['selected_answer'] or '(unanswered)'}")
            st.write(f"**Correct answer:** {result['correct_answer']}")

            st.markdown(
                f'<div class="meta"><strong>Explanation:</strong> '
                f'{html.escape(result["explanation"])}</div>',
                unsafe_allow_html=True,
            )

            # Source display
            source_chunk = chunk_map.get(result["source_chunk_id"])
            if source_chunk:
                source_name = source_chunk.get("source", "Unknown file")
                start_page = source_chunk.get("start_page")
                end_page = source_chunk.get("end_page")

                page_info = ""
                if start_page is not None:
                    if start_page == end_page or end_page is None:
                        page_info = f" (Page {start_page})"
                    else:
                        page_info = f" (Pages {start_page}–{end_page})"

                source_label = f"{source_name}{page_info}"
                source_text = source_chunk.get("text", "")
            else:
                source_label = "Unknown Source"
                source_text = "Source text not found."

            st.write(f"**Source:** {source_label}")
            with st.expander("View Source"):
                st.write(source_text)
        else:
            # --- Pre-submission: interactive radio ---
            radio_key = f"q_{index}"
            selected = st.radio(
                label=f"Select your answer for Question {index}",
                options=item.options,
                index=None,
                key=radio_key,
                label_visibility="collapsed",
            )
            if selected is not None:
                st.session_state.user_answers[index] = selected

    # Action buttons
    if not st.session_state.quiz_submitted:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Quiz", type="primary", use_container_width=True):
            n_answered = len(st.session_state.user_answers)
            n_total = len(rag_quiz.questions)
            if n_answered < n_total:
                st.warning(
                    f"You have answered {n_answered} of {n_total} questions. "
                    f"Please answer all questions before submitting."
                )
            else:
                st.session_state.quiz_results = calculate_score(
                    rag_quiz, st.session_state.user_answers
                )
                st.session_state.quiz_submitted = True
                st.rerun()
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retake Quiz", use_container_width=True):
                reset_quiz_answers()
                st.rerun()
        with col2:
            if st.button("Generate New Quiz", type="primary", use_container_width=True):
                clear_quiz()
                st.rerun()

elif baseline_questions:
    # -----------------------------------------------------------------------
    # Baseline mode — teammate's original rendering
    # -----------------------------------------------------------------------
    st.subheader("Generated Quiz")
    for index, item in enumerate(baseline_questions, start=1):
        render_baseline_question_card(index, item)

else:
    st.info("Choose settings in the sidebar, then click Generate Quiz.")

st.markdown("</div>", unsafe_allow_html=True)
