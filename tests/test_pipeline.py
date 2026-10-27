from pathlib import Path

from mcqgen.chunking import make_chunks
from mcqgen.evaluation import evaluate_questions
from mcqgen.local_generator import _terms, generate_from_chunks, generate_local_mcqs
from mcqgen.retrieval import HybridRetriever, RetrievedChunk
from mcqgen.schemas import normalize_question
from mcqgen.storage import list_generations, load_generation, save_generation

SAMPLE = """Machine learning uses data to find patterns and make predictions. Supervised learning uses labeled data to train a model. Classification predicts categories, while regression predicts numeric values. Unsupervised learning studies data without labels and can use clustering to group similar examples. Overfitting happens when a model memorizes details from training data and performs poorly on new examples. A test set estimates how well a trained model generalizes to unseen data. Precision measures the fraction of predicted positives that are truly positive, while recall measures how many actual positive cases are identified."""


def test_chunker_returns_stable_ids():
    chunks = make_chunks(SAMPLE, max_words=30)
    assert chunks
    assert chunks[0].chunk_id == "C01"
    assert all(chunk.text.strip() for chunk in chunks)


def test_hybrid_retriever_returns_scored_sources():
    retriever = HybridRetriever(SAMPLE, max_words=30)
    results = retriever.retrieve("overfitting and test data", k=2)
    assert 1 <= len(results) <= 2
    assert all(result.chunk_id.startswith("C") for result in results)
    assert all(0 <= result.score <= 1 for result in results)
    assert any("overfitting" in result.text.lower() for result in results)


def test_normalize_question_rejects_duplicate_options():
    item = normalize_question({
        "question": "Which concept uses labels?",
        "options": ["supervised learning", "Regression", "regression", "clustering"],
        "answer": "supervised learning",
    })
    assert item is None


def test_schema_preserves_citation_metadata():
    item = normalize_question({
        "question": "Which method predicts categories?",
        "options": ["Classification", "Regression", "Clustering", "Sampling"],
        "answer": "classification", "source_ids": ["C01"],
        "evidence": "Classification predicts categories.", "question_type": "Recall",
    })
    assert item is not None
    assert item["answer"] == "Classification"
    assert item["source_ids"] == ["C01"]
    assert item["evidence"].startswith("Classification")


def test_local_generator_outputs_valid_mcqs():
    questions = generate_local_mcqs(SAMPLE, n_questions=3)
    assert questions
    for item in questions:
        assert len(item["options"]) == 4
        assert len({option.casefold() for option in item["options"]}) == 4
        assert item["answer"] in item["options"]
        assert item["question"]
        assert item["source_ids"]
        assert item["evidence"]


def test_chunk_generator_keeps_source_ids():
    chunk = RetrievedChunk("C99", SAMPLE, 0.8, reason="test")
    questions = generate_from_chunks([chunk], n_questions=2)
    assert questions
    assert all(item["source_ids"] == ["C99"] for item in questions)


def test_evaluation_reports_evidence_and_source_id_metrics():
    questions = generate_local_mcqs(SAMPLE, n_questions=2)
    report = evaluate_questions(questions, SAMPLE, {"C01", "C02", "C03"})
    assert report["total_questions"] == len(questions)
    assert report["format_compliance_pct"] == 100.0
    assert report["evidence_in_source_pct"] == 100.0
    assert report["source_id_validity_pct"] == 100.0


def test_evaluation_flags_missing_evidence():
    item = {
        "question": "What does the model learn from?",
        "options": ["Data", "Clouds", "Numbers", "Nothing"],
        "answer": "Data", "evidence": "not in source", "source_ids": ["BAD"],
    }
    report = evaluate_questions([item], SAMPLE, {"C01"})
    assert report["evidence_in_source_pct"] == 0.0
    assert report["source_id_validity_pct"] == 0.0
    assert report["flags"]


def test_local_history_round_trip(tmp_path: Path):
    db = tmp_path / "history.sqlite3"
    questions = generate_local_mcqs(SAMPLE, n_questions=1)
    saved_id = save_generation("Test set", SAMPLE, "offline", questions, db_path=db)
    listed = list_generations(db_path=db)
    loaded = load_generation(saved_id, db_path=db)
    assert listed[0]["title"] == "Test set"
    assert loaded is not None
    assert loaded["source_text"] == SAMPLE
    assert loaded["questions"][0]["question"] == questions[0]["question"]


def test_empty_text_retriever_returns_empty():
    assert HybridRetriever("").retrieve("anything") == []


def test_offline_terms_are_contiguous_and_do_not_join_across_stop_words():
    text = "A machine learning workflow begins with selecting useful features, splitting the data into training and test sets."
    terms = {term for term, _ in _terms(text)}
    assert "machine learning workflow" in terms
    assert "features splitting data" not in terms


def test_offline_questions_are_labelled_honestly_as_cloze_recall():
    questions = generate_local_mcqs(SAMPLE, n_questions=3)
    assert questions
    assert all(item["difficulty"] == "Foundational" for item in questions)
    assert all(item["question_type"] == "Cloze / recall" for item in questions)
