"""Validation helpers for generated multiple-choice questions."""
from __future__ import annotations

from typing import Any


def normalize_question(item: dict[str, Any], default_source_ids: list[str] | None = None) -> dict[str, Any] | None:
    """Normalize an untrusted generation result to a predictable schema."""
    question = str(item.get("question", "")).strip()
    raw_options = item.get("options", [])
    answer = str(item.get("answer", "")).strip()
    if not question or not isinstance(raw_options, list):
        return None

    options: list[str] = []
    seen: set[str] = set()
    for raw in raw_options:
        option = str(raw).strip()
        key = option.casefold()
        if option and key not in seen:
            options.append(option)
            seen.add(key)
    if len(options) != 4:
        return None
    answer_match = next((option for option in options if option.casefold() == answer.casefold()), None)
    if answer_match is None:
        return None

    raw_ids = item.get("source_ids", default_source_ids or [])
    if isinstance(raw_ids, str):
        source_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        source_ids = list(dict.fromkeys(str(sid).strip() for sid in raw_ids if str(sid).strip()))
    else:
        source_ids = list(default_source_ids or [])

    return {
        "question": question,
        "options": options,
        "answer": answer_match,
        "explanation": str(item.get("explanation", "")).strip(),
        "source_ids": source_ids,
        "evidence": str(item.get("evidence", "")).strip(),
        "difficulty": str(item.get("difficulty", "Intermediate")).strip(),
        "question_type": str(item.get("question_type", "Recall")).strip(),
    }
