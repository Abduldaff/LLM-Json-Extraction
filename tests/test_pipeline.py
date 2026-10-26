from pathlib import Path

from extractor.providers import OfflineProvider


ROOT = Path(__file__).parents[1]


def test_invoice_sample_is_validated():
    result = OfflineProvider().extract("invoice", (ROOT / "samples/invoice.txt").read_text())
    assert result.invoice_number == "INV-1042"
    assert result.currency == "USD"
    assert str(result.total) == "2021.78"
    assert len(result.line_items) == 2


def test_resume_sample_is_validated():
    result = OfflineProvider().extract("resume", (ROOT / "samples/resume.txt").read_text())
    assert result.full_name == "Ada Lovelace"
    assert result.email == "ada@example.com"
    assert "Python" in result.skills
