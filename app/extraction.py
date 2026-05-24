"""Vision-language extraction with a validation + repair loop.

Pipeline:
    image bytes  -> preprocess  -> VLM call  -> JSON parse
                                    ^               |
                                    |               v
                                    +-- repair --- validate (math, types)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from datetime import date as date_cls
from decimal import Decimal, InvalidOperation
from pathlib import Path
from PIL import Image
from pydantic import ValidationError

from app.model_clients import VLMClient
from app.schemas import ExtractedReceipt

PROMPTS_DIR = Path(__file__).parent / "prompts"
EXTRACT_PROMPT = (PROMPTS_DIR / "extract.txt").read_text(encoding="utf-8")

MAX_REPAIR_ATTEMPTS = 2
SUM_TOLERANCE = Decimal("0.05")  # 5% tolerance for sum-of-items vs subtotal


@dataclass
class ExtractionResult:
    receipt: ExtractedReceipt | None
    raw_text: str
    confidence: float
    notes: str


def extract_receipt(image: Image.Image, vlm: VLMClient) -> ExtractionResult:
    """Run the VLM and return a validated receipt (or as much as we could salvage)."""
    notes: list[str] = []
    raw = vlm.extract(image, EXTRACT_PROMPT)
    parsed, parse_err = _try_parse(raw)

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        if parsed is None:
            problem = f"Could not parse JSON: {parse_err}"
        else:
            problem = _validate(parsed)
            if problem is None:
                return ExtractionResult(parsed, raw, _score(parsed), "; ".join(notes) or "ok")
        notes.append(f"repair attempt {attempt + 1}: {problem}")
        repair_prompt = EXTRACT_PROMPT + (
            f"\n\nYour previous output had this problem: {problem}\n"
            f"Previous output:\n{raw}\n\nReturn corrected JSON only."
        )
        raw = vlm.extract(image, repair_prompt)
        parsed, parse_err = _try_parse(raw)

    if parsed is not None:
        return ExtractionResult(parsed, raw, _score(parsed) * 0.5, "; ".join(notes))
    return ExtractionResult(None, raw, 0.0, "; ".join(notes) or "extraction failed")


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse(raw: str) -> tuple[ExtractedReceipt | None, str | None]:
    if not raw:
        return None, "empty response"
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None, "no JSON object found in response"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e.msg}"
    try:
        return ExtractedReceipt.model_validate(data), None
    except ValidationError as e:
        return None, f"schema mismatch: {e.errors()[0].get('msg', 'unknown')}"


def _validate(r: ExtractedReceipt) -> str | None:
    """Return None if everything passes, otherwise an error description."""
    if r.currency and len(r.currency) != 3:
        return f"currency must be a 3-letter ISO code, got '{r.currency}'"

    if r.date and (r.date.year < 2000 or r.date > date_cls.today().replace(year=date_cls.today().year + 1)):
        return f"date {r.date} is outside plausible range"

    if r.line_items and r.subtotal is not None:
        item_sum = Decimal("0")
        for li in r.line_items:
            line_total = li.total_price if li.total_price is not None else (li.unit_price or Decimal("0")) * li.quantity
            item_sum += line_total
        if r.subtotal > 0 and abs(item_sum - r.subtotal) / r.subtotal > SUM_TOLERANCE:
            return f"line item sum {item_sum} does not match subtotal {r.subtotal}"

    if r.subtotal is not None and r.tax is not None and r.total is not None:
        expected = r.subtotal + r.tax
        if r.total > 0 and abs(expected - r.total) / r.total > SUM_TOLERANCE:
            return f"subtotal+tax ({expected}) does not match total ({r.total})"

    return None


def _score(r: ExtractedReceipt) -> float:
    """Cheap confidence proxy: fraction of expected fields present."""
    fields = [r.merchant, r.date, r.subtotal, r.total, r.currency]
    present = sum(1 for f in fields if f is not None)
    item_score = 0.5 if r.line_items else 0.0
    return min(1.0, (present / len(fields)) * 0.7 + item_score * 0.3)


def safe_decimal(x) -> Decimal | None:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return None
