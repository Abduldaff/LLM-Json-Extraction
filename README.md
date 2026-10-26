# Structured LLM Prompt Pipeline & JSON Extractor

A small, production-minded Python CLI for turning unstructured invoice or resume text into strictly validated JSON.

## What it demonstrates

- Parameterized, task-specific prompt templates with a few-shot example
- Pydantic schemas with required fields and field-level validation
- An OpenAI structured-output provider and an offline deterministic provider
- Helpful validation errors and machine-readable JSON output
- Sample documents and automated tests

## Quick start

```powershell
cd structured-llm-json-extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m extractor samples/invoice.txt --document-type invoice
```

The command defaults to **offline mode**, so the example runs without credentials.

## Using OpenAI

```powershell
Copy-Item .env.example .env
# Edit .env and add OPENAI_API_KEY
python -m extractor samples/invoice.txt --document-type invoice --provider openai
```

The OpenAI provider requests a JSON schema response and validates it again locally with Pydantic. Set `OPENAI_MODEL` in `.env` to choose a compatible model.

## Usage

```text
python -m extractor INPUT_FILE --document-type {invoice,resume} [--provider offline|openai] [--output result.json]
```

This project deliberately does not send content anywhere unless `--provider openai` is chosen.
