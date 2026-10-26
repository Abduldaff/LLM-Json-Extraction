from __future__ import annotations

from extractor.schemas import SCHEMAS


SYSTEM_PROMPT = """You extract facts from documents. Return only JSON that conforms to the supplied schema.
Never invent a value: use null for unavailable optional fields and [] for unavailable lists.
Dates use ISO 8601 (YYYY-MM-DD); currency uses ISO 4217 uppercase codes.
"""

FEW_SHOT = """Example input: Invoice INV-7; total due: $12.50 USD
Example output: {\"document_type\":\"invoice\",\"invoice_number\":\"INV-7\",\"vendor_name\":null,\"customer_name\":null,\"issue_date\":null,\"due_date\":null,\"currency\":\"USD\",\"line_items\":[{\"description\":\"Unspecified charge\",\"quantity\":1,\"unit_price\":12.5,\"line_total\":12.5}],\"subtotal\":12.5,\"tax\":0,\"total\":12.5}
"""


def build_messages(document_type: str, text: str) -> list[dict[str, str]]:
    if document_type not in SCHEMAS:
        raise ValueError(f"Unsupported document type: {document_type}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": FEW_SHOT if document_type == "invoice" else "Extract only facts present in the resume."},
        {"role": "user", "content": f"Document type: {document_type}\\n\\nDocument:\\n{text}"},
    ]
