from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from typing import Literal, Optional, Type, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator


class InvoiceLineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    line_total: Decimal = Field(ge=0)


class Invoice(BaseModel):
    document_type: Literal["invoice"] = "invoice"
    invoice_number: str = Field(min_length=1)
    vendor_name: Optional[str] = None
    customer_name: Optional[str] = None
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[InvoiceLineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)


class Experience(BaseModel):
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    highlights: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    document_type: Literal["resume"] = "resume"
    full_name: str = Field(min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    portfolio_url: Optional[HttpUrl] = None

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        return sorted({value.strip() for value in values if value.strip()}, key=str.lower)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: Optional[str]) -> Optional[str]:
        if value and not re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
            raise ValueError("must be a valid email address")
        return value


Document = Union[Invoice, Resume]
SCHEMAS: dict[str, Union[Type[Invoice], Type[Resume]]] = {"invoice": Invoice, "resume": Resume}
