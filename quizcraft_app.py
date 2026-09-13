"""Streamlit UI for testing QuizCraft RAG quiz generation."""

from __future__ import annotations

import html
import os

import streamlit as st

import generation
from generation import GenerationError, generate_baseline_quiz, generate_grounded_quiz


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
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label, .stRadio label { color: var(--muted) !important; }
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


def render_question_card(index: int, item: dict) -> None:
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


def generate_quiz() -> None:
    api_key = st.session_state.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Add a Gemini API key in the sidebar or set GEMINI_API_KEY.")
        return

    generation.MODEL_NAME = AVAILABLE_FLASH_MODEL
    client = generation.init_client(api_key)

    try:
        if st.session_state.mode == "Grounded RAG":
            chunks = [
                chunk.strip()
                for chunk in st.session_state.context_text.split("\n\n")
                if chunk.strip()
            ]
            questions = generate_grounded_quiz(
                client,
                context_chunks=chunks,
                topic=st.session_state.topic,
                count=st.session_state.count,
                difficulty=st.session_state.difficulty,
            )
        else:
            questions = generate_baseline_quiz(
                client,
                topic=st.session_state.topic,
                count=st.session_state.count,
                difficulty=st.session_state.difficulty,
            )
    except (GenerationError, ValueError) as exc:
        st.error(str(exc))
        return

    st.session_state.questions = questions


with st.sidebar:
    st.header("Quiz Settings")
    st.text_input("Gemini API Key", type="password", key="api_key")
    st.radio("Mode", ["Grounded RAG", "Baseline"], key="mode")
    st.text_input("Topic", value="Photosynthesis", key="topic")
    st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="difficulty")
    st.slider("Question count", min_value=1, max_value=10, value=5, key="count")
    st.button("Generate Quiz", type="primary", use_container_width=True, on_click=generate_quiz)


st.markdown('<div class="quiz-shell">', unsafe_allow_html=True)
st.markdown('<div class="brand-kicker">Grounded learning studio</div>', unsafe_allow_html=True)
st.markdown('<div class="quiz-title">QuizCraft RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="quiz-subtitle">Generate schema-validated quizzes from documents or general knowledge.</div>',
    unsafe_allow_html=True,
)

if "context_text" not in st.session_state:
    st.session_state.context_text = (
        "Photosynthesis takes place in chloroplasts. Plants use light energy to convert carbon dioxide and water into glucose and oxygen.\n\n"
        "Chlorophyll is the pigment that absorbs light energy for photosynthesis."
    )

if st.session_state.get("mode", "Grounded RAG") == "Grounded RAG":
    st.markdown('<div class="context-panel">Paste your source material below. Separate notes with a blank line for better grounding.</div>', unsafe_allow_html=True)
    st.text_area(
        "Context chunks",
        key="context_text",
        height=160,
        help="Separate chunks with a blank line.",
    )
else:
    st.info("Baseline mode uses general knowledge and marks every source as General Knowledge.")

questions = st.session_state.get("questions", [])
if questions:
    st.subheader("Generated Quiz")
    for index, item in enumerate(questions, start=1):
        render_question_card(index, item)
else:
    st.info("Choose settings in the sidebar, then click Generate Quiz.")

st.markdown("</div>", unsafe_allow_html=True)
