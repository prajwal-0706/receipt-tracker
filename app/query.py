from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sqlglot
from sqlglot import exp
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.model_clients import SLMClient

PROMPTS_DIR = Path(__file__).parent / "prompts"
TEXT2SQL_PROMPT = (PROMPTS_DIR / "text2sql.txt").read_text(encoding="utf-8")
SUMMARIZE_PROMPT = (PROMPTS_DIR / "summarize.txt").read_text(encoding="utf-8")

ALLOWED_TABLES = {"receipts", "line_items"}


@dataclass
class QueryResult:
    question: str
    sql: str | None
    rows: list[dict] | None
    answer: str


def answer_question(question: str, db: Session, slm: SLMClient) -> QueryResult:
    today = date.today().isoformat()
    prompt = TEXT2SQL_PROMPT.replace("{TODAY}", today).replace("{QUESTION}", question)
    raw_sql = slm.generate(prompt, max_tokens=256).strip()
    sql = _strip_code_fences(raw_sql)

    safety_error = _is_safe_select(sql)
    if safety_error:
        return QueryResult(question, sql, None, f"I couldn't run that query safely ({safety_error}).")

    try:
        result = db.execute(text(sql))
        rows = [dict(r._mapping) for r in result]
    except Exception as e:
        return QueryResult(question, sql, None, f"The generated query failed against the database ({type(e).__name__}).")

    answer = _summarize(question, sql, rows, slm)
    return QueryResult(question, sql, _serialize_rows(rows), answer)


def _strip_code_fences(sql: str) -> str:
    # Models echo bits of the prompt back: "SQL:" labels, ``` fences with or without
    # a language tag. None of that should reach sqlglot.
    sql = sql.strip()
    sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    sql = sql.strip("`").strip()
    sql = re.sub(r"^(sql|postgres(?:ql)?)\s*\n", "", sql, flags=re.IGNORECASE)
    return sql.strip().rstrip(";").strip()


def _is_safe_select(sql: str) -> str | None:
    if not sql:
        return "empty SQL"
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        return f"parse error: {e}"
    if len(parsed) != 1:
        return "multiple statements not allowed"
    stmt = parsed[0]
    if not isinstance(stmt, exp.Select) and not (isinstance(stmt, exp.With) and isinstance(stmt.this, exp.Select)):
        return f"only SELECT statements allowed (got {type(stmt).__name__})"
    for table in stmt.find_all(exp.Table):
        if table.name.lower() not in ALLOWED_TABLES:
            return f"table '{table.name}' is not allowed"
    return None


def _summarize(question: str, sql: str, rows: list[dict], slm: SLMClient) -> str:
    snippet = _rows_preview(rows, max_rows=20)
    prompt = (
        SUMMARIZE_PROMPT
        .replace("{QUESTION}", question)
        .replace("{SQL}", sql)
        .replace("{ROWS}", snippet)
    )
    return slm.generate(prompt, max_tokens=200).strip() or "No results."


def _rows_preview(rows: list[dict], max_rows: int) -> str:
    if not rows:
        return "(no rows)"
    head = rows[:max_rows]
    return "\n".join(str(_stringify(r)) for r in head)


def _stringify(row: dict) -> dict:
    return {k: (str(v) if v is not None else None) for k, v in row.items()}


def _serialize_rows(rows: list[dict]) -> list[dict]:
    return [_stringify(r) for r in rows]
