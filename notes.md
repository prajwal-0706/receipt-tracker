# notes.md — running log of non-obvious decisions

> Append entries as I go. Format: date, what got me stuck, what I chose, why.

---

## 2026-05-22 — Initial design choices

**Decision: One VLM + one SLM, both Qwen2.5-7B AWQ, not a single model for everything.**
Why: Interviewer hinted at VLM + SLM split. Qwen2.5-VL-7B *can* do text-only tasks (it's built on Qwen2.5-7B), so a single-model setup is technically sufficient. Splitting buys ~3-5% on pure-text tasks (categorization, text-to-SQL) and matches the brief's "custom VLM/SLM" framing.
Alternative considered: single-VLM-for-everything. Saves ~5GB VRAM. Will document this as a "would simplify with another week" if eval shows the gap is small.

**Decision: AWQ 4-bit quants rather than FP16.**
Why: Free Colab T4 has 16GB VRAM. Two 7B models at FP16 is ~30GB → OOM. AWQ 4-bit gets both to ~10GB combined with room for KV cache.
Risk: ~1-2% accuracy hit per Qwen team benchmarks. Acceptable for a demo; document as a trade-off.

**Decision: Neon Postgres, not SQLite.**
Why: Already have a Neon URL. DB persists across Colab disconnects (SQLite would die with the runtime). Free tier covers all expected load.

**Decision: Validation loop with 2 repair retries, 5% tolerance on sum checks.**
Why: Receipts often have rounding artifacts (1¢ off due to per-item tax rounding). Strict equality would trigger repairs constantly.

**Decision: Day-first date parsing when ambiguous.**
Why: Most global receipts use DD/MM/YYYY. US receipts use MM/DD/YYYY but the brief mentions "varying quality...foreign currency" — suggests the sample set is international.

---

## TEMPLATE for future entries

## YYYY-MM-DD — short title

**Problem:**
**Options considered:**
**Chose:**
**Why:**
**Risk / what to watch:**

---
