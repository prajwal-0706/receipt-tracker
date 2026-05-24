import uuid
from datetime import date, datetime
from decimal import Decimal
from sqlalchemy import String, Date, Numeric, ForeignKey, DateTime, Float, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Receipt(Base):
    __tablename__ = "receipts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    merchant: Mapped[str | None] = mapped_column(String(256))
    receipt_date: Mapped[date | None] = mapped_column(Date)
    subtotal: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    tax: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    total: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    raw_extraction: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    extraction_notes: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    line_items: Mapped[list["LineItem"]] = relationship(
        "LineItem", back_populates="receipt", cascade="all, delete-orphan"
    )


class LineItem(Base):
    __tablename__ = "line_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    receipt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("receipts.id", ondelete="CASCADE"), nullable=False
    )
    description: Mapped[str] = mapped_column(String(512), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("1"))
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    total_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    category_confidence: Mapped[float | None] = mapped_column(Float)

    receipt: Mapped[Receipt] = relationship("Receipt", back_populates="line_items")


Index("idx_receipts_date", Receipt.receipt_date)
Index("idx_line_items_category", LineItem.category)
Index("idx_receipts_merchant", Receipt.merchant)
