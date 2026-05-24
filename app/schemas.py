from datetime import date as date_type, datetime
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class LineItemIn(BaseModel):
    description: str
    quantity: Decimal = Decimal("1")
    unit_price: Decimal | None = None
    total_price: Decimal | None = None


class ExtractedReceipt(BaseModel):
    """Raw output of the VLM extraction step (before categorization/persistence)."""
    merchant: str | None = None
    date: date_type | None = None
    line_items: list[LineItemIn] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    currency: str | None = None


class LineItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    description: str
    quantity: Decimal
    unit_price: Decimal | None
    total_price: Decimal | None
    category: str
    category_confidence: float | None


class ReceiptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    merchant: str | None
    receipt_date: date_type | None
    subtotal: Decimal | None
    tax: Decimal | None
    total: Decimal | None
    currency: str | None
    confidence: float | None
    extraction_notes: str | None
    created_at: datetime
    line_items: list[LineItemOut]


class UploadResponse(BaseModel):
    receipts: list[ReceiptOut]
    skipped_duplicates: list[str] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)


class QueryRequest(BaseModel):
    q: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sql: str | None = None
    rows: list[dict] | None = None
