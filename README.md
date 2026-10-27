# StudyForge — Source-Grounded MCQ Generator

StudyForge turns study notes into practice multiple-choice questions, shows the passages used to create them, runs transparent quality checks, and lets learners test themselves before revealing the answer key.

> **Student project / research prototype:** generated questions and automatic checks can be wrong. Review the question, answer, distractors, and evidence before relying on them for assessment.

## What's new in this advanced version

- **Hybrid retrieval by default:** combines TF-IDF similarity with query-token coverage; no external service or model download required.
- **Optional semantic retrieval:** use Sentence Transformers embeddings for meaning-based retrieval rather than exact word overlap.
- **RAG-grounded LLM mode:** optional OpenAI generation over retrieved passages. The prompt treats source text as data, requires source IDs and a verbatim evidence quote, and rejects results whose evidence does not appear in the cited passage.
- **Offline mode:** creates extractive cloze questions without an API key. It's deliberately basic and its distractors may be weak.
- **Transparent review report:** checks format compliance, duplicate prompts, answer-term matches, source-ID validity, evidence quote matches, and simple answer-option patterns.
- **Practice quiz:** submit answers before the application reveals the answer key and explanations.
- **Exports:** JSON, CSV, and Markdown study set.
- **Optional local history:** save question sets to SQLite on your own computer. Saving is user-initiated and the DB is excluded from Git.
- **Tests:** retrieval, schema normalization, citation metadata, local generation, evidence checks, and local history.

## Stack

Python · Streamlit · scikit-learn · NumPy · pypdf · python-docx · optional OpenAI API · optional Sentence Transformers · SQLite · pytest

## Quick start — Windows

Open PowerShell in the extracted ai-mcq-generator folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application opens at a local URL printed in the terminal (usually `http://localhost:8501`). In offline mode, you do not need an API key.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Optional feature: semantic retrieval

Semantic mode uses `sentence-transformers/all-MiniLM-L6-v2`. Install the optional dependencies separately:

```bash
pip install -r requirements-semantic.txt
```

The first semantic retrieval run downloads the embedding model. That download can be relatively large and may take a while, so the hybrid retriever is the default. You can use semantic retrieval without enabling the OpenAI generation API.

## Optional feature: OpenAI question generation

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` in `.env` and keep the file private.
3. Optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`).
4. Restart Streamlit and select the OpenAI RAG mode.

API usage may incur charges. Never commit `.env`, publish your API key, or place secrets in source code. On Streamlit Community Cloud, configure secrets in the app settings instead of committing a `.env` file.

## Pipeline overview

```text
TXT / text PDF / DOCX / pasted notes
                 |
                 v
     Clean text + sentence chunking
                 |
                 v
      Retrieval (choose one mode)
       /                    \
      v                      v
Hybrid lexical            Semantic embeddings
TF-IDF + token coverage   Sentence Transformers
       \                    /
        v                  v
      Top source passages with stable chunk IDs
                 |
       +---------+----------+
       |                    |
       v                    v
Offline cloze          Optional OpenAI RAG
       |                    |
       +---------+----------+
                 v
     Schema + source evidence validation
                 |
                 v
   Quality flags + practice quiz + exports
                 |
                 v
      Optional local SQLite history
```

## What the evaluation scores mean

- **Format valid:** has a question, exactly four distinct options, and an answer that matches one option.
- **Evidence match:** the evidence quote appears in the original source after basic whitespace/punctuation normalization. This does not prove that the answer follows from the quote.
- **Answer text in source:** the answer string occurs literally in the source. Paraphrases can fail this check, so low scores are not automatically errors.
- **Source ID valid:** each question references source chunk IDs selected for the generation call.
- **Review flags:** transparent heuristics highlight missing evidence, invalid IDs, possible duplicate questions, and some structural patterns.

None of these scores is a calibrated factuality metric. Human review remains necessary. In particular, the offline mode uses extractive fill-in-the-blank prompts and heuristic distractors; it is a fallback, not equivalent to an advanced question-generation model.

## Project layout

```text
ai-mcq-generator/
├── app.py
├── mcqgen/
│   ├── chunking.py
│   ├── documents.py
│   ├── evaluation.py
│   ├── llm_generator.py
│   ├── local_generator.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── storage.py
├── sample_data/sample_chapter.txt
├── tests/test_pipeline.py
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-semantic.txt
└── README.md
```

## Run automated tests

```bash
pytest -q
```

## Publish to GitHub

Create an empty public repository (for example, `studyforge-mcq-generator`), then run these commands from this folder after installing Git and configuring your GitHub identity:

```bash
git init
git add .
git commit -m "Upgrade StudyForge with RAG and question evaluation"
git branch -M main
git remote add origin https://github.com/Anii188/studyforge-mcq-generator.git
git push -u origin main
```

Replace the sample remote URL with the exact repository URL shown by GitHub. Do not create a public repository containing `.env`, API keys, private study materials, or the local SQLite history file.

## Reference and attribution

This is an independent student implementation inspired by the general task of generating MCQs from educational text. It was informed by the workflow described in [Rohan-Thoma's Multiple-choice-question-generator-using-NLP notebook](https://github.com/Rohan-Thoma/Multiple-choice-question-generator-using-NLP/blob/main/Multiple_Choice_Question_generator.ipynb), which discusses question generation and distractor construction. This project does not copy that notebook's code. Review the original repository's license separately before reusing any code or assets.

## Resume entry — use only after running and understanding the code

**StudyForge — Source-Grounded MCQ Generator** | Python, Streamlit, scikit-learn, RAG, Sentence Transformers (optional)

- Built a Streamlit application that turns uploaded study material into practice MCQs with retrieved-source evidence and JSON/CSV/Markdown exports.
- Implemented hybrid TF-IDF retrieval, optional semantic embeddings, and an optional LLM generation path with source-ID and evidence-quote validation.
- Added heuristic question-quality flags, an interactive practice quiz, SQLite-based local history, and automated tests.

Only list the components you have actually installed, run, tested, and can explain in an interview.
