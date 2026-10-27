from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mcqgen.documents import extract_uploaded_text
from mcqgen.evaluation import evaluate_questions
from mcqgen.llm_generator import generate_with_openai
from mcqgen.local_generator import generate_from_chunks
from mcqgen.retrieval import HybridRetriever, SemanticRetriever
from mcqgen.storage import DEFAULT_DB, list_generations, load_generation, save_generation

ROOT = Path(__file__).parent
SAMPLE_TEXT = (ROOT / "sample_data" / "sample_chapter.txt").read_text(encoding="utf-8")
load_dotenv(ROOT / ".env")
try:
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
    if not os.getenv("OPENAI_MODEL") and "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = str(st.secrets["OPENAI_MODEL"])
except Exception:
    pass

st.set_page_config(page_title="StudyForge | Source-grounded MCQs", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1180px;}
[data-testid="stMetric"] {background: rgba(127,127,127,.07); border: 1px solid rgba(127,127,127,.18); padding: .75rem; border-radius: .75rem;}
.question-card {padding: .85rem 1rem; border: 1px solid rgba(127,127,127,.25); border-radius: .75rem; margin-bottom: .65rem;}
.small-muted {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📚 StudyForge")
st.subheader("Build practice questions from your own study material")
st.caption("Source-aware MCQ generation · hybrid/semantic retrieval · transparent evaluation · quiz mode")

# Apply deferred updates before creating the source text widget.
# Streamlit locks a widget's keyed state after the widget is instantiated.
if "pending_source_text" in st.session_state:
    st.session_state["source_text"] = st.session_state.pop("pending_source_text")

# Stable session state
for key, value in {
    "source_text": SAMPLE_TEXT,
    "questions": [],
    "source_for_results": "",
    "retrieved_chunks": [],
    "generation_mode": "",
    "evaluation": None,
    "focus_used": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Generation settings")
    n_questions = st.slider("Number of questions", min_value=3, max_value=12, value=6)
    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"], index=1)
    question_type = st.selectbox("Question style", ["Mixed", "Recall", "Conceptual", "Application"], index=0)
    focus = st.text_input("Focus topic (optional)", placeholder="e.g., overfitting, transformers, SQL joins")
    retrieval_mode = st.selectbox(
        "Retrieval strategy",
        ["Hybrid lexical (recommended)", "Semantic embeddings (optional)"],
        help="Hybrid works offline. Semantic retrieval requires the optional dependency and downloads a model the first time it runs."
    )
    st.divider()
    api_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    generation_mode_ui = st.radio("Question generation", ["Offline extractive", "OpenAI RAG (optional)"] if api_configured else ["Offline extractive", "OpenAI RAG (optional; key needed)"], index=0)
    if not api_configured:
        st.caption("No API key detected. Offline mode is available. Add OPENAI_API_KEY in a private .env file to use the optional LLM mode.")
    st.caption("Offline mode produces Foundational cloze/recall questions. Difficulty and question-style selections are applied only in OpenAI RAG mode.")
    st.divider()
    st.caption("Privacy: source material and history stay on this machine. Saving a run is optional.")

upload = st.file_uploader("Upload study material", type=["txt", "pdf", "docx"], help="Text-based PDFs work best. Scanned PDFs require OCR first.")
if upload is not None:
    if st.session_state.get("last_upload_name") != upload.name:
        try:
            extracted = extract_uploaded_text(upload.name, upload.getvalue())
            if extracted.strip():
                st.session_state.source_text = extracted
                st.session_state.last_upload_name = upload.name
                st.success(f"Loaded {upload.name} ({len(extracted.split())} words).")
            else:
                st.warning("No selectable text was found. Scanned documents may need OCR first.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

col_title, col_sample, col_clear = st.columns([6, 1, 1])
with col_title:
    st.markdown("#### Source material")
with col_sample:
    if st.button("Load sample", use_container_width=True):
        st.session_state.source_text = SAMPLE_TEXT
        st.rerun()
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.source_text = ""
        st.rerun()
source_text = st.text_area(
    "Paste or edit notes", key="source_text", height=245,
    placeholder="Paste a chapter, lecture notes, or technical documentation here…",
    help="Best results come from factual text with clear definitions and relationships."
)
word_count = len(source_text.split())
st.caption(f"{word_count:,} words · {len(source_text):,} characters")

with st.expander("How StudyForge works"):
    st.markdown("""
    1. **Chunking:** split source text into bounded, overlapping passages with stable IDs.
    2. **Retrieval:** rank useful passages using hybrid lexical scoring or optional semantic embeddings.
    3. **Generation:** create extractive cloze questions offline, or ask an optional LLM to generate questions from retrieved passages.
    4. **Validation:** require four distinct options and an answer matching one option. LLM questions must also cite a valid source chunk and provide an exact evidence quote.
    5. **Evaluation:** surface duplicates, missing evidence, source-ID problems, and other review flags.
    6. **Practice:** hide the key until the user submits quiz answers.
    """)

if st.button("Generate question set", type="primary", use_container_width=True):
    clean_source = source_text.strip()
    if len(clean_source.split()) < 35:
        st.error("Please provide at least 35 words of study material.")
    elif len(clean_source) > 250_000:
        st.error("Please keep source material below 250,000 characters for this student version.")
    elif generation_mode_ui.startswith("OpenAI") and not api_configured:
        st.error("OpenAI mode is unavailable until OPENAI_API_KEY is configured in your private .env file.")
    else:
        try:
            with st.spinner("Retrieving relevant passages and generating questions…"):
                if retrieval_mode.startswith("Semantic"):
                    retriever = SemanticRetriever(clean_source)
                else:
                    retriever = HybridRetriever(clean_source)
                query = focus.strip() or "key concepts definitions processes causes effects and important facts"
                retrieved = retriever.retrieve(query, k=min(8, max(4, n_questions)))
                if not retrieved:
                    raise RuntimeError("No relevant source passages could be retrieved. Try a longer source.")
                if generation_mode_ui.startswith("OpenAI"):
                    questions = generate_with_openai(
                        retrieved, n_questions=n_questions, difficulty=difficulty,
                        focus=focus, question_type=question_type,
                    )
                    mode_label = f"OpenAI RAG + {retrieval_mode.split(' (')[0]}"
                else:
                    questions = generate_from_chunks(retrieved, n_questions=n_questions)
                    mode_label = f"Offline extractive + {retrieval_mode.split(' (')[0]}"
                if not questions:
                    raise RuntimeError("No valid questions were produced. Try more detailed source material, a different focus topic, or the optional LLM mode.")
                st.session_state.questions = questions
                st.session_state.source_for_results = clean_source
                st.session_state.retrieved_chunks = [
                    {"chunk_id": chunk.chunk_id, "text": chunk.text, "score": chunk.score,
                     "tfidf_score": chunk.tfidf_score, "lexical_score": chunk.lexical_score, "reason": chunk.reason}
                    for chunk in retrieved
                ]
                st.session_state.generation_mode = mode_label
                st.session_state.focus_used = focus.strip()
                st.session_state.evaluation = evaluate_questions(
                    questions, clean_source, {chunk.chunk_id for chunk in retrieved}
                )
                st.session_state.pop("quiz_submitted", None)
                st.session_state.pop("quiz_answers", None)
        except Exception as exc:
            st.error(str(exc))
            if retrieval_mode.startswith("Semantic"):
                st.info("If the semantic retrieval package is missing, run: pip install -r requirements-semantic.txt")
            if generation_mode_ui.startswith("OpenAI"):
                st.caption("Check your API key, billing/settings, network access, and configured model name.")

questions = st.session_state.questions
if questions:
    st.divider()
    st.header("Your question set")
    st.caption(f"Mode: {st.session_state.generation_mode} · Focus: {st.session_state.focus_used or 'Broad coverage'} · Review flagged items before relying on them.")
    report = st.session_state.evaluation or evaluate_questions(questions, st.session_state.source_for_results)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Questions", report["total_questions"])
    metric_cols[1].metric("Format valid", f"{report['format_compliance_pct']}%")
    metric_cols[2].metric("Evidence match", f"{report['evidence_in_source_pct']}%")
    metric_cols[3].metric("Answer text in source", f"{report['answer_terms_in_source_pct']}%")
    metric_cols[4].metric("Duplicates", report["duplicate_questions"])
    st.caption("These checks are heuristics—not proof of factual correctness or educational quality.")

    tabs = st.tabs(["Practice quiz", "Review questions", "Evaluation report", "Retrieved sources"])
    with tabs[0]:
        st.markdown("Answer the questions first. The answer key stays hidden until you submit.")
        with st.form("practice_quiz"):
            chosen_answers: dict[str, str] = {}
            for idx, item in enumerate(questions, start=1):
                st.markdown(f"**Q{idx}. {item['question']}**")
                chosen_answers[str(idx)] = st.radio(
                    f"Choose an answer for question {idx}", item["options"],
                    index=None, key=f"quiz_{idx}", label_visibility="collapsed"
                )
                st.write("")
            submitted = st.form_submit_button("Submit answers", type="primary")
        if submitted:
            st.session_state.quiz_answers = chosen_answers
            st.session_state.quiz_submitted = True
        if st.session_state.get("quiz_submitted"):
            saved_answers = st.session_state.get("quiz_answers", {})
            attempted = sum(bool(saved_answers.get(str(i))) for i in range(1, len(questions) + 1))
            correct = sum(saved_answers.get(str(i)) == item["answer"] for i, item in enumerate(questions, start=1))
            st.success(f"Score: {correct}/{len(questions)} ({round(100 * correct / len(questions))}%) · Answered {attempted}/{len(questions)}")
            for idx, item in enumerate(questions, start=1):
                picked = saved_answers.get(str(idx))
                if picked == item["answer"]:
                    st.write(f"✅ Q{idx}: Correct")
                else:
                    st.write(f"❌ Q{idx}: Correct answer — **{item['answer']}**")
                if item.get("explanation"):
                    st.caption(item["explanation"])

    with tabs[1]:
        for idx, item in enumerate(questions, start=1):
            with st.container(border=True):
                st.markdown(f"**Q{idx}. {item['question']}**")
                badges = f"{item.get('difficulty', 'Not labelled')} · {item.get('question_type', 'Question')} · Sources: {', '.join(item.get('source_ids', [])) or 'not supplied'}"
                st.caption(badges)
                for letter, option in zip("ABCD", item.get("options", [])):
                    st.write(f"{letter}. {option}")
                with st.expander("Answer, explanation and evidence"):
                    st.success(f"Correct answer: {item['answer']}")
                    st.write(item.get("explanation") or "No explanation was supplied.")
                    st.markdown("**Evidence from source**")
                    st.write(item.get("evidence") or "No evidence quote supplied.")
                item_flags = next((row["flags"] for row in report.get("flags", []) if row["question_number"] == idx), [])
                if item_flags:
                    st.warning("Review: " + "; ".join(item_flags))

    with tabs[2]:
        a, b, c = st.columns(3)
        a.metric("Source IDs valid", f"{report['source_id_validity_pct']}%")
        b.metric("Evidence quotes matched", f"{report['evidence_in_source_pct']}%")
        c.metric("Review flags", len(report.get("flags", [])))
        st.warning(report["warning"])
        if report.get("flags"):
            for flagged in report["flags"]:
                st.write(f"**Q{flagged['question_number']}** — " + "; ".join(flagged["flags"]))
        else:
            st.info("No heuristic flags were triggered. Human review is still required.")

    with tabs[3]:
        for chunk in st.session_state.retrieved_chunks:
            with st.expander(f"{chunk['chunk_id']} · hybrid/semantic score {chunk['score']:.3f}"):
                st.caption(chunk.get("reason", "retrieved source passage"))
                st.write(chunk["text"])
                if chunk.get("tfidf_score", 0) or chunk.get("lexical_score", 0):
                    st.caption(f"TF-IDF component: {chunk['tfidf_score']:.3f} · Query-token coverage: {chunk['lexical_score']:.3f}")

    st.divider()
    st.subheader("Export and save")
    col_json, col_csv, col_md = st.columns(3)
    with col_json:
        st.download_button("Download JSON", data=json.dumps(questions, indent=2, ensure_ascii=False), file_name="studyforge_questions.json", mime="application/json", use_container_width=True)
    with col_csv:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["question", "option_A", "option_B", "option_C", "option_D", "answer", "explanation", "evidence", "source_ids", "difficulty", "question_type"])
        writer.writeheader()
        for item in questions:
            options = item.get("options", [])
            writer.writerow({"question": item.get("question", ""), "option_A": options[0] if len(options) > 0 else "", "option_B": options[1] if len(options) > 1 else "", "option_C": options[2] if len(options) > 2 else "", "option_D": options[3] if len(options) > 3 else "", "answer": item.get("answer", ""), "explanation": item.get("explanation", ""), "evidence": item.get("evidence", ""), "source_ids": ";".join(item.get("source_ids", [])), "difficulty": item.get("difficulty", ""), "question_type": item.get("question_type", "")})
        st.download_button("Download CSV", data=output.getvalue(), file_name="studyforge_questions.csv", mime="text/csv", use_container_width=True)
    with col_md:
        md_lines = ["# StudyForge Question Set", "", f"Mode: {st.session_state.generation_mode}", ""]
        for idx, item in enumerate(questions, start=1):
            md_lines += [f"## Q{idx}. {item['question']}", ""]
            md_lines += [f"- {letter}. {option}" for letter, option in zip("ABCD", item["options"])]
            md_lines += [f"- **Answer:** {item['answer']}", f"- **Explanation:** {item.get('explanation', '')}", f"- **Evidence:** {item.get('evidence', '')}", ""]
        st.download_button("Download Markdown", data="\n".join(md_lines), file_name="studyforge_study_set.md", mime="text/markdown", use_container_width=True)

    with st.expander("Save this set to local history (optional)"):
        title = st.text_input("Study-set title", value=st.session_state.focus_used or "My StudyForge question set")
        st.caption("This stores your source text and questions in a SQLite file on this computer. Do not save sensitive notes on shared devices.")
        if st.button("Save to this computer"):
            try:
                saved_id = save_generation(title, st.session_state.source_for_results, st.session_state.generation_mode, questions)
                st.success(f"Saved locally as study set #{saved_id}.")
            except Exception as exc:
                st.error(f"Could not save this set: {exc}")

with st.expander("Local study history"):
    try:
        history = list_generations(DEFAULT_DB, limit=15)
        if history:
            labels = {f"#{item['id']} · {item['title']} · {item['question_count']} questions · {item['created_at'][:10]}": item['id'] for item in history}
            choice = st.selectbox("Saved study sets", list(labels))
            if st.button("Load selected set"):
                saved = load_generation(labels[choice], DEFAULT_DB)
                if saved:
                    st.session_state.questions = saved["questions"]
                    st.session_state.source_for_results = saved["source_text"]
                    st.session_state["pending_source_text"] = saved["source_text"]
                    st.session_state.generation_mode = saved["mode"] + " · loaded from history"
                    st.session_state.retrieved_chunks = []
                    st.session_state.evaluation = evaluate_questions(saved["questions"], saved["source_text"])
                    st.session_state.pop("quiz_submitted", None)
                    st.rerun()
        else:
            st.caption("No saved study sets yet.")
    except Exception as exc:
        st.caption(f"History is unavailable: {exc}")

st.divider()
st.caption("StudyForge is a student project. Automatic checks are heuristic. Verify every question and answer before using them for exams or formal assessment.")
