"""Explainable heuristic quality checks; these are not a substitute for human review."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).casefold()).strip()


def evaluate_questions(questions: list[dict[str, Any]], source_text: str, valid_source_ids: set[str] | None = None) -> dict[str, Any]:
    if not questions:
        return {
            "total_questions": 0, "format_compliance_pct": 0.0,
            "answer_terms_in_source_pct": 0.0, "evidence_in_source_pct": 0.0,
            "source_id_validity_pct": 0.0, "duplicate_questions": 0,
            "flags": [],
            "warning": "No valid questions were generated. Try a longer or more specific source.",
        }

    normalized = [_normalize(item.get("question", "")) for item in questions]
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count)
    src = _normalize(source_text)
    raw_source = source_text.casefold()
    structurally_valid = answer_in_source = evidence_in_source = source_id_valid = 0
    flags: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        item_flags: list[str] = []
        question = str(item.get("question", "")).strip()
        options = item.get("options", [])
        answer = str(item.get("answer", "")).strip()
        options_valid = (
            isinstance(options, list) and len(options) == 4
            and len({_normalize(option) for option in options}) == 4
            and any(_normalize(option) == _normalize(answer) for option in options)
        )
        if question and options_valid:
            structurally_valid += 1
        else:
            item_flags.append("Invalid question/option structure")

        answer_norm = _normalize(answer)
        if answer_norm and answer_norm in src:
            answer_in_source += 1
        else:
            item_flags.append("Answer text not found literally in source")

        evidence = str(item.get("evidence", "")).strip()
        evidence_norm = _normalize(evidence)
        if evidence_norm and evidence_norm in src:
            evidence_in_source += 1
        else:
            item_flags.append("Evidence quote not found literally in source")

        ids = item.get("source_ids", [])
        if not valid_source_ids or (isinstance(ids, list) and ids and all(sid in valid_source_ids for sid in ids)):
            source_id_valid += 1
        else:
            item_flags.append("Missing or invalid source chunk ID")

        qlen = len(question.split())
        if qlen < 5:
            item_flags.append("Question may be too short")
        if qlen > 55:
            item_flags.append("Question may be too long")
        option_lengths = [len(str(option).split()) for option in options] if isinstance(options, list) else []
        if option_lengths and max(option_lengths) > 0 and max(option_lengths) >= 3 * max(1, min(option_lengths)):
            item_flags.append("Options differ greatly in length; review for answer clues")
        if _normalize(question) in counts and counts[_normalize(question)] > 1:
            item_flags.append("Duplicate question")
        if item_flags:
            flags.append({"question_number": index, "flags": item_flags})

    total = len(questions)
    return {
        "total_questions": total,
        "format_compliance_pct": round(100 * structurally_valid / total, 1),
        "answer_terms_in_source_pct": round(100 * answer_in_source / total, 1),
        "evidence_in_source_pct": round(100 * evidence_in_source / total, 1),
        "source_id_validity_pct": round(100 * source_id_valid / total, 1),
        "duplicate_questions": duplicates,
        "flags": flags,
        "warning": "These are transparent heuristic checks, not a correctness guarantee. Literal matches can miss paraphrases, and a source quote does not prove the answer is correct. Review every question.",
    }
