"""Evaluation harness — hits a running API and scores extraction + query accuracy.

Suites:  --suite extraction | queries | all
Inputs:  samples/labels.json, samples/queries.json (see samples/README.md for shape)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent


FIELD_MAP = {"date": "receipt_date", "total": "total", "merchant": "merchant", "currency": "currency"}


def eval_extraction(api_url: str, labels_path: Path) -> dict:
    if not labels_path.exists():
        print(f"!! {labels_path} not found — skipping extraction eval")
        return {"skipped": True}
    labels = json.loads(labels_path.read_text())
    fields = ["merchant", "date", "total", "currency"]
    correct = {f: 0 for f in fields}
    total = len(labels)
    item_precision_sum = 0.0
    item_recall_sum = 0.0
    failures: list[dict] = []

    for entry in labels:
        image_path = ROOT / entry["image"]
        expected = entry["expected"]
        with image_path.open("rb") as fh:
            resp = httpx.post(f"{api_url}/receipts", files={"files": (image_path.name, fh)}, timeout=120)
        if resp.status_code != 201:
            failures.append({"image": entry["image"], "error": resp.text[:200]})
            continue
        body = resp.json()
        if not body.get("receipts") and body.get("skipped_duplicates"):
            failures.append({"image": entry["image"], "error": "duplicate — clear DB before extraction eval"})
            continue
        if not body.get("receipts"):
            failures.append({"image": entry["image"], "error": resp.text[:200]})
            continue
        got = body["receipts"][0]
        for f in fields:
            api_key = FIELD_MAP[f]
            if _eq(got.get(api_key), expected.get(f)):
                correct[f] += 1
        prec, rec = _item_overlap(got.get("line_items", []), expected.get("items", []))
        item_precision_sum += prec
        item_recall_sum += rec

    return {
        "total_receipts": total,
        "per_field_accuracy": {f: round(correct[f] / total, 3) if total else 0 for f in fields},
        "item_precision_avg": round(item_precision_sum / total, 3) if total else 0,
        "item_recall_avg": round(item_recall_sum / total, 3) if total else 0,
        "failures": failures,
    }


def _eq(api_val, expected_val) -> bool:
    if api_val is None or expected_val is None:
        return api_val == expected_val
    try:
        return abs(float(api_val) - float(expected_val)) < 0.01
    except (TypeError, ValueError):
        return str(api_val).strip().lower() == str(expected_val).strip().lower()


def eval_queries(api_url: str, queries_path: Path) -> dict:
    if not queries_path.exists():
        print(f"!! {queries_path} not found — skipping query eval")
        return {"skipped": True}
    queries = json.loads(queries_path.read_text())
    passed = 0
    details: list[dict] = []
    for q in queries:
        resp = httpx.post(f"{api_url}/query", json={"q": q["q"]}, timeout=60)
        if resp.status_code != 200:
            details.append({"q": q["q"], "ok": False, "error": resp.text[:200]})
            continue
        body = resp.json()
        answer = body.get("answer", "")
        sql = body.get("sql", "")
        # Strip commas so "$4,044.60" matches needle "4044".
        normalized = answer.lower().replace(",", "")
        needles = q.get("expected_contains", [])
        match_any = q.get("match") == "any"
        check = any if match_any else all
        ok = check(str(n).lower() in normalized for n in needles)
        if ok:
            passed += 1
        details.append({"q": q["q"], "answer": answer, "sql": sql, "ok": ok})

    return {
        "total_queries": len(queries),
        "passed": passed,
        "pass_rate": round(passed / len(queries), 3) if queries else 0,
        "details": details,
    }


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _item_overlap(got: list[dict], expected: list[dict]) -> tuple[float, float]:
    if not expected:
        return (1.0, 1.0) if not got else (0.0, 1.0)
    got_descs = {_norm(g.get("description")) for g in got}
    exp_descs = {_norm(e.get("description")) for e in expected}
    if not got_descs:
        return (0.0, 0.0)
    hits = len(got_descs & exp_descs)
    return (hits / len(got_descs), hits / len(exp_descs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--suite", choices=["extraction", "queries", "all"], default="all")
    ap.add_argument("--labels", default=str(ROOT / "samples" / "labels.json"))
    ap.add_argument("--queries", default=str(ROOT / "samples" / "queries.json"))
    args = ap.parse_args()

    output: dict[str, Any] = {}
    if args.suite in ("extraction", "all"):
        output["extraction"] = eval_extraction(args.api, Path(args.labels))
    if args.suite in ("queries", "all"):
        output["queries"] = eval_queries(args.api, Path(args.queries))

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
