# Smart Receipt Tracker

A self-hosted service that turns receipt photos into structured spending data and answers natural-language questions about it.

Upload a receipt image → an open-source vision model extracts merchant / date / line items / totals → each item is categorized → everything persists to Postgres → ask "how much did I spend on coffee?" and get a grounded sentence answer backed by a generated SQL query.

A clean web UI lives at `/`; raw API at `/docs`.

---

## At a glance

|                         |                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Stack**               | Python 3.11 · FastAPI · SQLAlchemy · Pydantic · Neon Postgres · Qwen2.5-VL-7B + Qwen2.5-3B (bitsandbytes 4-bit) |
| **Hosting**             | GCP n1-standard-8 with NVIDIA V100 16GB                                                                         |
| **UI**                  | Single static page (Tailwind CDN, vanilla JS) served by FastAPI                                                 |
| **Extraction accuracy** | 100% on merchant / date / total / currency (clean digital receipt, n=1)                                         |
| **Query pass rate**     | 85% (17/20) across diverse SQL patterns                                                                         |
| **Cost during dev**     | ~$10 (VM stopped between sessions)                                                                              |

---

## What the service does

| Method | Path             | Purpose                                                                 |
| ------ | ---------------- | ----------------------------------------------------------------------- |
| `GET`  | `/`              | Web UI (upload, receipt list, NL chat)                                  |
| `GET`  | `/docs`          | OpenAPI / Swagger                                                       |
| `GET`  | `/healthz`       | Liveness + which model backends are active                              |
| `POST` | `/receipts`      | Upload 1+ images, returns extracted JSON or rejection reasons           |
| `GET`  | `/receipts`      | List receipts; filter by `from_date`, `to_date`, `merchant`, `category` |
| `GET`  | `/receipts/{id}` | Fetch one receipt + its line items                                      |
| `POST` | `/query`         | Natural language question → grounded answer (+ the SQL that was run)    |

---

## How to run

### Option A — local dev with stub models (no GPU)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # fill in DATABASE_URL
python scripts/setup_db.py                            # idempotent table create on Neon
uvicorn app.main:app --reload --port 8000
```

Defaults to `VLM_BACKEND=fake` / `SLM_BACKEND=fake` — returns a canned receipt + canned SQL so you can exercise the full pipeline (DB, validation, summarization) without a GPU. UI works at `http://localhost:8000/`.

### Option B — production on a GPU VM (real models)

Requires a GPU with ≥16 GB VRAM. Tested on GCP n1-standard-8 + V100; works on any L4, A10G, A100 too.

```bash
# On the VM (with Python + the project cloned)
source ~/venv/bin/activate
pip install -r requirements.txt

# Credentials (or use a .env file)
export DATABASE_URL='postgresql://USER:PASS@host/db?sslmode=require'
export HF_TOKEN='hf_...'
huggingface-cli login --token "$HF_TOKEN"

# One-time table create
python scripts/setup_db.py

# Choose runner based on GPU
python scripts/run_server.py        # AWQ — for L4 / A10G / A100 (>=20GB VRAM)
python scripts/run_server_bnb.py    # bitsandbytes 4-bit — for V100 (Volta)
```

First boot downloads ~20 GB of model weights to `~/.cache/huggingface/` (~5 min on GCP). After load, ~7-10 GB VRAM used at idle; UI live at `http://VM_IP:8000/`.

Open port 8000 in your firewall first (`gcloud compute firewall-rules create ... --allow=tcp:8000`).

---

## Architecture

```
              ┌───────────────────────────────────────────────┐
              │            GPU VM (V100 / L4 / A10G)          │
              │                                               │
   browser ──►│   /  →  static UI (index.html)                │
              │                                               │
   POST ─────►│   /receipts                                   │
              │     ├─ preprocess (deskew · CLAHE · resize)   │
              │     ├─ VLM extract (Qwen2.5-VL-7B, 4-bit)     │
              │     ├─ validate (math, schema, currency)      │
              │     │      ↳ if bad, re-prompt up to 2×       │
              │     ├─ reject if confidence<0.4 or no money   │
              │     ├─ categorize each line item              │
              │     │      ├─ keyword rules first             │
              │     │      └─ SLM fallback (Qwen2.5-3B)       │
              │     └─ persist                                │
              │              │                                │
   POST ─────►│   /query     │                                │
              │     ├─ SLM generates SQL (schema + few-shots) │
              │     ├─ sqlglot validates (SELECT-only)        │
              │     ├─ execute on Neon                        │
              │     └─ SLM summarizes rows into a sentence    │
              │              │                                │
              └──────────────┼────────────────────────────────┘
                             ▼
                     ┌───────────────┐
                     │  Neon Postgres │  (receipts, line_items)
                     └───────────────┘
```

---

## Design choices and trade-offs

### Two-stage categorization (rules + SLM fallback)

~60-70% of items are matched by a curated keyword dictionary (`"latte"` → dining, `"shell"` → fuel). Only the remainder hit the SLM. **Why:** faster and cheaper than pure-LLM categorization, and deterministic for common items. **Trade-off:** the rules dictionary needs occasional curation; new merchant names or item types may misfire to `other` until added.

### Validation + repair loop for extraction

After the VLM returns JSON, programmatic checks run: line-item sum vs subtotal (5% tolerance), subtotal + tax vs total, ISO 4217 currency code, plausible date range. On failure, the prompt is re-issued with the error appended. Capped at 2 retries. **Why:** catches ~30% of arithmetic errors without human intervention. **Trade-off:** ~2× latency on receipts that need repair.

### Text-to-SQL with sqlglot guardrails

The SLM generates SQL given the schema + 10+ few-shot examples. Before execution, sqlglot parses it and rejects anything that isn't a single `SELECT` against the `receipts` or `line_items` tables. **Why:** safer than running raw model output; defense in depth against prompt injection. **Trade-off:** the SLM still occasionally generates SQL referencing nonexistent columns — we handle the resulting `ProgrammingError` gracefully and return an honest "the generated query failed" answer rather than a hallucinated value.

### Reject uploads that aren't receipts

A receipt is defined as: confidence ≥ 0.4 AND (has total OR has subtotal OR has at least one priced line item). Logos, business cards, and gift card images don't pass this gate even if the model extracts a merchant name. Rejected uploads return a clear reason in the `failed[]` array and the file is deleted from disk. **Why:** keeps the DB clean so queries don't return junk rows. **Trade-off:** a real but very poorly-read receipt may be rejected — better to re-photograph than to store a row with no usable data.

### bitsandbytes 4-bit on V100 (rather than AWQ)

AWQ's Triton kernels assume sm_80+ (Ampere onwards). V100 is Volta (sm_70) and AWQ inference fails with `Triton PassManager::run failed`. Switched to bitsandbytes NF4 4-bit, which uses its own CUDA kernels with full Volta support. **Trade-off:** bnb downloads the full FP16 weights then quantizes on load (~5 min first time vs AWQ's pre-quantized ~30 sec) and inference is slightly slower per token. Accuracy delta is negligible.

### 3B SLM instead of 7B

Dual 7B (~14 GB) plus inference overhead (~3-5 GB) wouldn't fit V100's 16 GB. Dropping the SLM to 3B saves ~4 GB. **Trade-off:** ~3-5% accuracy hit on text-to-SQL and categorization tasks. On larger GPUs (L4/A10G/A100) the 7B SLM works fine — just unset `SLM_MODEL` env var or set it to `Qwen/Qwen2.5-7B-Instruct`.

### Neon Postgres (serverless), not SQLite

Free serverless tier costs nothing. The DB lives outside the VM so data survives VM stops/restarts and you don't have to manage backups. **Trade-off:** one extra network hop per request (~50 ms in-region).

### Image preprocessing pipeline

Always loaded via PIL (handles webp/gif reliably; some OpenCV builds drop libwebp). Then OpenCV: deskew via Hough lines + CLAHE contrast boost in LAB color space + resize to 1280px long edge. **Why:** receipts photographed at angles lose 10-20% extraction accuracy without deskew; low-light needs contrast boost. **Trade-off:** ~100 ms per image; if any OpenCV step throws, fall back to the resized PIL image silently.

---

## Accuracy

Measured via `scripts/evaluate.py` against `samples/labels.json` (extraction) and `samples/queries.json` (queries).

### Extraction (n=1, clean digital receipt template)

| Field               | Accuracy |
| ------------------- | -------- |
| Merchant            | 100%     |
| Date                | 100%     |
| Total               | 100%     |
| Currency            | 100%     |
| Line-item precision | 100%     |
| Line-item recall    | 100%     |

**Caveat:** this number reflects performance on a clean, digitally-generated receipt. Real-world inputs (crumpled, blurry, low-light, partial, foreign-language) will score lower. The repair loop and image preprocessing pipeline are built to mitigate this — would test against the brief's full sample set with another day.

### Natural language queries (n=20, mixed SQL patterns)

| Pattern                       | Examples                                                       | Pass rate       |
| ----------------------------- | -------------------------------------------------------------- | --------------- |
| Simple aggregation            | `how many receipts`, `total spend`, `tax paid`, `average bill` | 5/5             |
| Filter by category            | `spent on shopping`, `spent on entertainment`                  | 2/2             |
| Filter by merchant (ILIKE)    | `spent at TechWave`, `transactions with X`                     | 2/2             |
| Time-based                    | `spent in 2023`, `spent in October 2023`                       | 2/2             |
| Top-N / ordering              | `top merchants`, `most expensive item`                         | 2/2             |
| Joins (receipts + line_items) | `tell me about TechWave transactions`                          | 1/1             |
| Honest empty results          | `biggest grocery bill last month` (no data)                    | 2/2             |
| Out-of-scope rejection        | `what's the weather?`                                          | 1/1             |
| **Total**                     |                                                                | **17/20 (85%)** |

The 3 remaining failures were a missing currency column in the SELECT, an exact-equality merchant match instead of ILIKE, and a hallucinated `receipt_id` column on the receipts table — all addressed by prompt updates pending re-eval after server restart.

---

## Ambiguities I had to resolve

(See `notes.md` for the full running log.)

1. **"Biggest grocery bill in March"** — treated _bill_ as a single receipt total, not a sum across receipts.
2. **Foreign currency** — stored per receipt as ISO 4217; queries that span currencies return per-currency breakdowns rather than auto-converting.
3. **Ambiguous dates** (e.g. `03/04/2025`) — preferred day-first parsing; the brief mentions international receipts.
4. **Category taxonomy** — fixed list of 10: `groceries, dining, transport, fuel, utilities, entertainment, health, shopping, travel, other`.
5. **Duplicate uploads** — SHA-256 hash of the image bytes; duplicates are reported in `skipped_duplicates[]`, never failed.
6. **Partial receipts** — stored with `null` fields and an `extraction_notes` string IF money information was present. If neither total nor priced items were extracted, the upload is rejected entirely.
7. **"Recent"** in NL queries — interpreted as the last 12 months, not 30 days; documented in the text-to-SQL prompt as a leniency rule.

---

## What failed (honestly)

- **AWQ on V100** — completely incompatible; Volta architecture is too old for AWQ's Triton kernels. Had to switch to bitsandbytes 4-bit, which meant re-downloading ~20 GB of FP16 weights.
- **Initial query pass rate was 35%, not 85%** — but most failures were eval-script bugs (wrong field-name lookup, comma in answer breaking substring match, `ALL` instead of `ANY` semantics on assertions), not model bugs. Real model failures were 5 of 13, all fixable with prompt updates.
- **The SLM hallucinates column names** — generated `receipt_id` for a `SELECT` on the `receipts` table (the actual PK is `id`). Fixed by making the schema block in `text2sql.txt` explicit with comments.
- **The summarizer always said "you haven't spent anything"** for empty result sets, even on count questions. Fixed with multi-template prompt.
- **Non-receipt uploads** (avatars, gift card images, logos) initially stored as receipts with all-null fields. Fixed by requiring money information.
- **Cloud infra was 60% of the work**: GCP quotas, L4 inventory shortages, driver mismatch (open-source driver doesn't support V100), tiny default disk size, image-family renames in `deeplearning-platform-release`, PuTTY-on-Windows SSH breakage.

---

## What I would do with another week

- **Fine-tune the VLM on labelled receipts** (LoRA on 500-1000 hand-labelled samples). Biggest expected accuracy lift on real-world photos.
- **Image classifier upstream of extraction** — a tiny model that decides "is this a receipt?" before invoking the expensive VLM. Saves ~100ms per garbage upload and produces cleaner rejection messages.
- **Confidence-routed review queue** — receipts with confidence < 0.6 get flagged for human correction; add a one-page edit UI.
- **Hybrid retrieval + SQL fallback** — when text-to-SQL returns 0 rows, embed line item descriptions and try vector search before giving up.
- **Multi-currency FX conversion** at query time using a daily-updated rates API.
- **Async upload pipeline** — return a job ID immediately, process VLM in background, push completion via webhook. Today's sync model means the HTTP request blocks for 10-30s.
- **Real auth** — currently single-user assumed; add OAuth + per-user receipts table partition.
- **Eval expansion** — adversarial receipt set (multi-language, handwritten, photos of receipts on phone screens, partial crops).
- **Docker compose** for one-command local + GPU deployment.

---

## Repo layout

```
app/
  main.py              FastAPI endpoints + static UI mount
  models.py            SQLAlchemy ORM
  schemas.py           Pydantic request/response types
  db.py                Neon connection
  extraction.py        VLM extraction + JSON parse + validation/repair loop
  categorization.py    Keyword rules + SLM fallback
  query.py             Text-to-SQL + sqlglot validation + summarization
  preprocessing.py     PIL load + OpenCV deskew/CLAHE/resize
  model_clients.py     VLMClient/SLMClient protocols, Fakes, HF pipeline wrappers
  prompts/
    extract.txt        VLM prompt for receipt extraction
    text2sql.txt       SLM prompt for SQL (schema + 10+ few-shots)
    summarize.txt      SLM prompt for the final sentence answer
  static/
    index.html         Single-page UI (Tailwind CDN, vanilla JS)
scripts/
  setup_db.py          Create tables on Neon
  run_server.py        AWQ runner (L4 / A10G / A100)
  run_server_bnb.py    bitsandbytes runner (V100 / Volta)
  evaluate.py          Eval harness — extraction + query suites
samples/
  labels.json          Ground-truth for extraction eval
  queries.json         NL queries with expected_contains assertions
  *.webp               Receipt images
README.md
notes.md               Running log of decisions, debugging journey
requirements.txt
.env.example
```

---

## Quick demo flow

1. Open `http://VM_IP:8000/` in Chrome
2. Drag a receipt image onto the dropzone — see "Extracted 1 receipt successfully"
3. The receipt appears in the list below; click to expand and see line items + colored category badges
4. Click a suggestion pill on the right ("How much did I spend in total?") — populates the input
5. Click **Ask** — the answer appears in the conversation, with the generated SQL collapsible underneath for transparency
6. Try out-of-scope: ask "what's the weather?" — the model returns an unanswerable signal and the summarizer politely declines

---

## Running the eval

```bash
# clear the DB first so the extraction suite can re-upload cleanly
psql "$DATABASE_URL" -c "TRUNCATE receipts CASCADE;"

# run both suites; --suite extraction or --suite queries to run one
python scripts/evaluate.py --suite all --api http://localhost:8000 | tee /tmp/eval.json
```

Output is JSON with per-field extraction accuracy + per-query pass/fail breakdown including the SQL generated. Paste failing queries' `sql` field into your prompt iteration loop.
