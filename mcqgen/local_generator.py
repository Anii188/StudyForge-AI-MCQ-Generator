"""Transparent offline cloze-question generator.

The offline generator only creates source-grounded cloze/recall questions. Its
heuristic distractors are drawn from contiguous phrases found in the source and
must still be reviewed by a human. Conceptual/application questions are best
handled by the optional LLM-backed generator.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .retrieval import RetrievedChunk
from .schemas import normalize_question

# These words usually make a phrase an incomplete clause rather than a useful
# answer option. Keep this list conservative: domain terms such as "learning"
# and "training" can be nouns and should remain available inside compounds.
_VERB_LIKE = set("""
    enable enables enabled use uses used predict predicts predicted represent
    represents represented group groups grouped learn learns learned happen
    happens happened occur occurs occurred begin begins began collect collects
    collecting clean cleans cleaning select selects selecting split splits
    splitting evaluate evaluates evaluating attempt attempts attempting preserve
    preserves train trains trainswith measure measures measuring include includes
    including estimate estimates estimating identify identifies identifying study
    studies studying work works working compare compares comparing help helps
    helping capture captures capturing perform performs performing influence
    influences influencing result results resultin resultinfluence indicate indicates
    classify classifies classify predict make makes decide decides decision
    decide affect affects affect influence accidentally
""".split())

# Generic words are poor standalone options, but remain valid inside phrases
# such as "training data" or "validation set".
_GENERIC_SINGLE_WORDS = set("""
    data model models system systems case cases example examples information
    result results feature features set sets metric metrics score value values
    learning training machine positive negative actual new different common useful
    important possible available specific similar simple final numeric predicted
    accuracy
""".split())

# Common adjective/adverb endings that are particularly likely to create a
# fragment when used as the final token of a multiword answer.
_WEAK_ENDINGS = ("ly",)


def _terms(text: str, max_terms: int = 350) -> list[tuple[str, int]]:
    """Extract contiguous, source-grounded 1–3 word phrases.

    Unlike stop-word-filtered n-gram vectorization, this preserves word order
    and punctuation boundaries so it does not invent combinations like
    "features splitting data" by joining tokens around removed stop words.
    """
    counts: Counter[str] = Counter()
    normalized = re.sub(r"\s+", " ", text).strip()

    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        # Do not build a phrase across a comma/semicolon/colon: those commonly
        # separate independent clauses and would otherwise create fragments.
        for segment in re.split(r"[,;:—–]+", sentence):
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b", segment.casefold())
            for size in (3, 2, 1):
                for start in range(len(words) - size + 1):
                    phrase_words = words[start:start + size]
                    phrase = " ".join(phrase_words)
                    if len(phrase) < 4:
                        continue
                    # Reject phrases containing stop words rather than deleting
                    # stop words and accidentally joining non-adjacent words.
                    if any(word in ENGLISH_STOP_WORDS for word in phrase_words):
                        continue
                    if any(word in _VERB_LIKE for word in phrase_words):
                        continue
                    if any(word.endswith(_WEAK_ENDINGS) for word in phrase_words):
                        continue
                    if size == 1 and phrase in _GENERIC_SINGLE_WORDS:
                        continue
                    if size > 1 and phrase_words[-1] in {
                        "specific", "similar", "useful", "important", "possible",
                        "actual", "common", "different", "simple", "final",
                        "numeric", "available", "new", "positive", "negative",
                        "predicted", "imbalanced",
                    }:
                        continue
                    counts[phrase] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0].split()), -len(item[0]), item[0]))
    return ranked[:max_terms]


def _whole_phrase_in(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.IGNORECASE) is not None


def _phrase_candidates(sentence: str, candidates: list[tuple[str, int]]) -> list[tuple[str, int]]:
    present = [(term, count) for term, count in candidates if _whole_phrase_in(sentence, term)]
    return sorted(present, key=lambda item: (-len(item[0].split()), -item[1], -len(item[0]), item[0]))


def _distractors_for(
    answer: str,
    sentence: str,
    candidates: list[tuple[str, int]],
) -> list[str]:
    """Rank plausible alternate concepts without using phrases in the stem."""
    answer_tokens = set(answer.split())
    answer_words = answer.split()
    pool: list[tuple[str, int]] = []

    def aligned_overlap(term: str) -> int:
        """Prefer parallel compounds like supervised/unsupervised learning."""
        term_words = term.split()
        shared_from_end = 0
        for answer_word, term_word in zip(reversed(answer_words), reversed(term_words)):
            if answer_word != term_word:
                break
            shared_from_end += 1
        return shared_from_end

    for term, count in candidates:
        if term.casefold() == answer.casefold():
            continue
        if _whole_phrase_in(sentence, term):
            continue
        # Avoid choices that are merely a subphrase of the answer or vice versa.
        term_tokens = set(term.split())
        if term_tokens.issubset(answer_tokens) or answer_tokens.issubset(term_tokens):
            continue
        # If two options share a concept word, prefer matching compounds (for
        # example, "supervised learning" over the fragment "learning workflow").
        if answer_tokens.intersection(term_tokens) and aligned_overlap(term) == 0:
            continue
        pool.append((term, count))

    pool.sort(key=lambda item: (
        -aligned_overlap(item[0]),
        -len(answer_tokens.intersection(item[0].split())),
        abs(len(item[0].split()) - len(answer_tokens)),
        -item[1],
        abs(len(item[0]) - len(answer)),
        item[0],
    ))
    return [term for term, _ in pool]


def generate_from_chunks(chunks: Iterable[RetrievedChunk], n_questions: int = 5, seed: int = 23) -> list[dict]:
    """Generate source-linked cloze items from retrieved chunks; no API required."""
    if n_questions <= 0:
        return []

    chunk_list = list(chunks)
    rng = random.Random(seed)
    generated: list[dict] = []
    seen_questions: set[str] = set()

    for chunk in chunk_list:
        text = re.sub(r"\s+", " ", chunk.text).strip()
        if len(text.split()) < 18:
            continue
        candidates = _terms(text)
        if len(candidates) < 5:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if len(generated) >= n_questions:
                break
            sentence = sentence.strip()
            if len(sentence.split()) < 7:
                continue

            present = _phrase_candidates(sentence, candidates)
            if not present:
                continue

            # Prefer meaningful multiword concepts over isolated generic words.
            answer = present[0][0]
            masked = re.sub(
                r"(?<!\w)" + re.escape(answer) + r"(?!\w)",
                "____", sentence, count=1, flags=re.IGNORECASE,
            )
            question = "Complete the statement: " + masked
            if question.casefold() in seen_questions:
                continue

            distractor_pool = _distractors_for(answer, sentence, candidates)
            if len(distractor_pool) < 3:
                # Failing to produce three reasonable alternatives is better
                # than fabricating word fragments. Skip this item instead.
                continue

            # Keep choices reasonably close in length, while permitting a
            # related two-word term when a three-word answer is selected.
            answer_words = len(answer.split())
            length_matched = [term for term in distractor_pool if abs(len(term.split()) - answer_words) <= 1]
            chosen_pool = length_matched if len(length_matched) >= 3 else distractor_pool
            options = [answer, *chosen_pool[:20][:3]]
            if len({option.casefold() for option in options}) != 4:
                continue
            rng.shuffle(options)

            item = normalize_question({
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": f"The source sentence states: {sentence}",
                "source_ids": [chunk.chunk_id],
                "evidence": sentence,
                "difficulty": "Foundational",
                "question_type": "Cloze / recall",
            })
            if item:
                generated.append(item)
                seen_questions.add(question.casefold())

        if len(generated) >= n_questions:
            break

    return generated[:n_questions]


def generate_local_mcqs(text: str, n_questions: int = 5, seed: int = 23) -> list[dict]:
    """Backwards-compatible wrapper for tests and simple usage."""
    if len(text.split()) < 35:
        raise ValueError("Please provide at least 35 words so the generator has enough context.")
    from .chunking import make_chunks
    chunks = make_chunks(text, max_words=200, overlap_sentences=0)
    wrapped = [RetrievedChunk(c.chunk_id, c.text, 0.0) for c in chunks]
    result = generate_from_chunks(wrapped, n_questions=n_questions, seed=seed)
    if len(result) < min(1, n_questions):
        raise ValueError("Not enough candidate phrases were found. Try a longer text with factual sentences.")
    return result
