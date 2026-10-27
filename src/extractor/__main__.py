from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pydantic import ValidationError

from extractor.providers import ExtractionError, provider_for


def extraction_main() -> int:
    parser = argparse.ArgumentParser(description="Extract validated JSON from invoices or resumes.")
    parser.add_argument("input", type=Path, help="UTF-8 text document")
    parser.add_argument("--document-type", choices=("invoice", "resume"), required=True)
    parser.add_argument("--provider", choices=("offline", "openai"), default="offline")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args()
    try:
        result = provider_for(args.provider).extract(args.document_type, args.input.read_text(encoding="utf-8"))
        payload = result.model_dump_json(indent=2)
    except (OSError, ExtractionError, ValidationError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Validated JSON written to {args.output}")
    else:
        print(payload)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        from extractor.rag import rag_main

        return rag_main(sys.argv[2:])
    return extraction_main()


if __name__ == "__main__":
    raise SystemExit(main())
