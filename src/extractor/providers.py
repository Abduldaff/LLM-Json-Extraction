from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from dotenv import load_dotenv

from extractor.prompts import build_messages
from extractor.schemas import SCHEMAS, Document


class ExtractionError(RuntimeError):
    pass


class Provider(ABC):
    @abstractmethod
    def extract(self, document_type: str, text: str) -> Document:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ExtractionError("OPENAI_API_KEY is missing. Add it to .env or use --provider offline.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ExtractionError("Install optional dependency: pip install -e '.[openai]'") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def extract(self, document_type: str, text: str) -> Document:
        schema = SCHEMAS[document_type]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(document_type, text),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": document_type, "strict": True, "schema": schema.model_json_schema()},
            },
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ExtractionError("The model returned an empty response.")
        return schema.model_validate_json(content)


class OfflineProvider(Provider):
    """Small demo parser; it keeps the project runnable without network credentials."""

    def extract(self, document_type: str, text: str) -> Document:
        if document_type == "invoice":
            return SCHEMAS[document_type].model_validate(self._invoice(text))
        return SCHEMAS[document_type].model_validate(self._resume(text))

    @staticmethod
    def _money(value: str) -> Decimal:
        return Decimal(re.sub(r"[^0-9.]", "", value))

    def _invoice(self, text: str) -> dict[str, Any]:
        find = lambda pattern: re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        number = find(r"invoice\s*:\s*([^\n]+)")
        total = find(r"total(?:\s+due)?\s*:\s*([^\n]+)")
        currencies = re.findall(r"\b(USD|EUR|GBP|INR|JPY|CAD|AUD)\b", text, re.IGNORECASE)
        item_matches = re.findall(r"^(.+?)\s{2,}\$([\d,.]+)", text, re.MULTILINE)
        line_items = [
            {"description": name.strip(), "quantity": 1, "unit_price": self._money(amount), "line_total": self._money(amount)}
            for name, amount in item_matches
            if not name.lower().startswith(("tax", "total"))
        ]
        tax_match = find(r"^tax\s+\$?([\d,.]+)")
        tax = self._money(tax_match.group(1)) if tax_match else Decimal("0")
        calculated_subtotal = sum((item["line_total"] for item in line_items), Decimal("0"))
        total_value = self._money(total.group(1)) if total else calculated_subtotal + tax
        return {"invoice_number": number.group(1).strip() if number else "UNKNOWN", "vendor_name": None,
                "customer_name": (find(r"bill to:\s*([^\n]+)").group(1).strip() if find(r"bill to:\s*([^\n]+)") else None),
                "issue_date": (find(r"issued:\s*(\d{4}-\d{2}-\d{2})").group(1) if find(r"issued:\s*(\d{4}-\d{2}-\d{2})") else None),
                "due_date": (find(r"due date:\s*(\d{4}-\d{2}-\d{2})").group(1) if find(r"due date:\s*(\d{4}-\d{2}-\d{2})") else None),
                "currency": currencies[-1].upper() if currencies else "USD", "line_items": line_items or [{"description": "Unspecified charge", "quantity": 1, "unit_price": total_value, "line_total": total_value}],
                "subtotal": calculated_subtotal or total_value - tax, "tax": tax, "total": total_value}

    def _resume(self, text: str) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        phone = re.search(r"\+?[\d][\d ()-]{6,}\d", text)
        skills = re.search(r"skills:\s*(.+)", text, re.IGNORECASE)
        return {"full_name": lines[0] if lines else "Unknown", "email": email.group(0) if email else None,
                "phone": phone.group(0) if phone else None, "summary": None,
                "skills": [s.strip() for s in skills.group(1).split(",")] if skills else [], "experience": []}


def provider_for(name: str) -> Provider:
    return OpenAIProvider() if name == "openai" else OfflineProvider()
