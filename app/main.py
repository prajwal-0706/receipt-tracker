from __future__ import annotations
import hashlib
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, Depends, HTTPException, Query
from sqlalchemy import select, and_
from sqlalchemy.orm import Session, selectinload

from app.db import get_session
from app.models import Receipt, LineItem
from app.schemas import (
    ExtractedReceipt, ReceiptOut, UploadResponse, QueryRequest, QueryResponse,
)
from app.extraction import extract_receipt
from app.categorization import categorize_item
from app.preprocessing import preprocess_for_vlm
from app.query import answer_question
from app.model_clients import VLMClient, SLMClient, FakeVLM, FakeSLM

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Smart Receipt Tracker", version="0.1.0")


def get_vlm() -> VLMClient:
    return app.state.vlm if hasattr(app.state, "vlm") else FakeVLM()


def get_slm() -> SLMClient:
    return app.state.slm if hasattr(app.state, "slm") else FakeSLM()


@app.on_event("startup")
def _startup():
    backend = os.environ.get("VLM_BACKEND", "fake").lower()
    if backend == "fake" and not hasattr(app.state, "vlm"):
        app.state.vlm = FakeVLM()
    if os.environ.get("SLM_BACKEND", "fake").lower() == "fake" and not hasattr(app.state, "slm"):
        app.state.slm = FakeSLM()


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "vlm": type(get_vlm()).__name__,
        "slm": type(get_slm()).__name__,
    }


@app.post("/receipts", response_model=UploadResponse, status_code=201)
async def upload_receipts(
    files: list[UploadFile],
    db: Annotated[Session, Depends(get_session)],
    vlm: Annotated[VLMClient, Depends(get_vlm)],
    slm: Annotated[SLMClient, Depends(get_slm)],
):
    if not files:
        raise HTTPException(400, "no files supplied")

    receipts_out: list[Receipt] = []
    skipped: list[str] = []
    failed: list[dict] = []

    for upload in files:
        try:
            raw_bytes = await upload.read()
            image_hash = hashlib.sha256(raw_bytes).hexdigest()

            existing = db.scalar(select(Receipt).where(Receipt.image_hash == image_hash))
            if existing:
                skipped.append(image_hash)
                continue

            image_path = UPLOAD_DIR / f"{image_hash}_{upload.filename or 'receipt'}"
            image_path.write_bytes(raw_bytes)

            image = preprocess_for_vlm(raw_bytes)
            result = extract_receipt(image, vlm)

            extracted: ExtractedReceipt = result.receipt or ExtractedReceipt()

            has_money = (
                extracted.total is not None
                or extracted.subtotal is not None
                or any(
                    (li.total_price is not None or li.unit_price is not None)
                    for li in extracted.line_items
                )
            )
            if result.confidence < 0.4 or not has_money:
                reasons = []
                if not has_money:
                    reasons.append("no total or priced line items found")
                if result.confidence < 0.4:
                    reasons.append(f"low confidence ({result.confidence:.2f})")
                image_path.unlink(missing_ok=True)
                failed.append({
                    "filename": upload.filename,
                    "error": (
                        f"image does not appear to be a receipt: {'; '.join(reasons)}. "
                        f"extracted_merchant={extracted.merchant!r}, "
                        f"extracted_total={extracted.total!r}"
                    ),
                })
                continue

            receipt = Receipt(
                id=uuid.uuid4(),
                image_hash=image_hash,
                image_path=str(image_path),
                merchant=extracted.merchant,
                receipt_date=extracted.date,
                subtotal=extracted.subtotal,
                tax=extracted.tax,
                total=extracted.total,
                currency=extracted.currency,
                raw_extraction={"text": result.raw_text},
                confidence=result.confidence,
                extraction_notes=result.notes,
            )

            for li in extracted.line_items:
                cat = categorize_item(li.description, extracted.merchant, slm)
                receipt.line_items.append(LineItem(
                    description=li.description,
                    quantity=li.quantity or 1,
                    unit_price=li.unit_price,
                    total_price=li.total_price,
                    category=cat.category,
                    category_confidence=cat.confidence,
                ))

            db.add(receipt)
            db.commit()
            db.refresh(receipt)
            receipts_out.append(receipt)
        except Exception as e:
            db.rollback()
            failed.append({"filename": upload.filename, "error": f"{type(e).__name__}: {e}"})

    return UploadResponse(
        receipts=[ReceiptOut.model_validate(r) for r in receipts_out],
        skipped_duplicates=skipped,
        failed=failed,
    )


@app.get("/receipts/{receipt_id}", response_model=ReceiptOut)
def get_receipt(
    receipt_id: uuid.UUID,
    db: Annotated[Session, Depends(get_session)],
):
    stmt = select(Receipt).where(Receipt.id == receipt_id).options(selectinload(Receipt.line_items))
    receipt = db.scalar(stmt)
    if not receipt:
        raise HTTPException(404, "receipt not found")
    return receipt


@app.get("/receipts", response_model=list[ReceiptOut])
def list_receipts(
    db: Annotated[Session, Depends(get_session)],
    from_date: date | None = Query(None),
    to_date: date | None = Query(None),
    merchant: str | None = Query(None),
    category: str | None = Query(None),
    limit: int = Query(50, le=200),
):
    conditions = []
    if from_date:
        conditions.append(Receipt.receipt_date >= from_date)
    if to_date:
        conditions.append(Receipt.receipt_date <= to_date)
    if merchant:
        conditions.append(Receipt.merchant.ilike(f"%{merchant}%"))

    stmt = select(Receipt).options(selectinload(Receipt.line_items))
    if conditions:
        stmt = stmt.where(and_(*conditions))
    if category:
        stmt = stmt.where(Receipt.id.in_(
            select(LineItem.receipt_id).where(LineItem.category == category)
        ))
    stmt = stmt.order_by(Receipt.receipt_date.desc().nullslast()).limit(limit)
    return list(db.scalars(stmt).all())


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    db: Annotated[Session, Depends(get_session)],
    slm: Annotated[SLMClient, Depends(get_slm)],
):
    if not req.q.strip():
        raise HTTPException(400, "q must be non-empty")
    result = answer_question(req.q, db, slm)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sql=result.sql,
        rows=result.rows,
    )
