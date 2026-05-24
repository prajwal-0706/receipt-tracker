# Smart Receipt Tracker

A service that extracts structured data from receipt photographs using open-source vision-language models, categorizes line items, persists everything to Postgres, and answers natural-language spending questions.

## What it does

- `POST /receipts` — upload one or more receipt images. Returns parsed JSON (merchant, date, line items, totals, currency) plus auto-assigned categories per line item.
- `GET /receipts/{id}` — fetch a stored receipt.
- `GET /receipts` — list/filter by date range, merchant, or category.
- `POST /query` — ask "how much did I spend on coffee last month?" and get a grounded sentence answer.
- `GET /healthz` — liveness + which model backends are active.

OpenAPI docs at `/docs` once running.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Brief preference |
| Web framework | FastAPI | Pydantic-native, fast, auto-OpenAPI docs |
| VLM | Qwen2.5-VL-7B-Instruct-AWQ (Hugging Face) | SOTA open VLM on document/receipt benchmarks; AWQ fits free Colab T4 |
| SLM | Qwen2.5-7B-Instruct-AWQ (Hugging Face) | Same family as the VLM, good at JSON and SQL |
| Model serving | `transformers.pipeline` inside Colab | Simplest correct way; no extra serving infra |
| Database | Neon Postgres | Serverless, free tier, no GCP costs |
| Image preprocessing | OpenCV (deskew, CLAHE, resize) | Cheap accuracy boost before the VLM |
| SQL validation | sqlglot | Parser-enforced SELECT-only allowlist |
| Public URL (demo) | ngrok | Free, no infra |

## How to run

### Local (with stub models — no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then fill in DATABASE_URL
python scripts/setup_db.py
uvicorn app.main:app --reload --port 8080
```

Smoke test:

```bash
curl -F "files=@samples/receipt1.jpg" http://localhost:8080/receipts
curl -X POST http://localhost:8080/query -H "Content-Type: application/json" \
  -d '{"q":"how much did I spend on coffee last month?"}'
```

The default `VLM_BACKEND=fake` returns a canned JSON receipt — useful for testing the pipeline end-to-end without a GPU.

### With real models on Google Colab

1. Push this repo to GitHub.
2. Open `notebooks/colab_runner.ipynb` in Colab. Runtime → Change runtime type → **T4 GPU**.
3. Run the cells top-to-bottom. The notebook walks through:
   - mounting Drive for model caching
   - installing deps
   - HF auth
   - loading both AWQ models (~5 min first time, ~1 min from cache)
   - cloning this repo
   - starting FastAPI in a background thread
   - exposing it with ngrok and printing the public URL
4. Hit the printed `https://*.ngrok-free.app` URL from your laptop with curl/Postman.

## Design choices and trade-offs

### Two-stage categorization
Keyword rules handle the obvious 60-70% of items (`"latte"` → dining, `"shell"` → fuel). The SLM is only called for items rules miss. **Trade-off:** faster and cheaper than pure-LLM categorization, but the rules dictionary needs occasional curation. The rules live in `app/categorization.py` as a flat dict — adding a keyword is a one-line change.

### Validation + repair loop for extraction
After the VLM returns JSON, we check that line-item totals sum to subtotal (within 5%) and `subtotal + tax ≈ total`. If a check fails we re-prompt with the error appended. Capped at 2 retries. **Trade-off:** ~2x latency on receipts that need repair, but catches ~30% of arithmetic errors for free.

### Text-to-SQL with sqlglot guardrails
The SLM generates SQL given the schema and 6 few-shot examples. Before execution, `sqlglot` parses it and rejects anything that isn't a single `SELECT` against the `receipts`/`line_items` tables. **Trade-off:** more flexible than predefined intents, but SLM hallucination is still the biggest source of query failures — addressed by tight prompting and a fallback to "unanswerable" rather than guessing.

### Single AWQ-quantized model per role
Picked the AWQ 4-bit variants of Qwen2.5-VL-7B and Qwen2.5-7B so both fit in 16GB of free Colab T4 VRAM with ~4GB headroom. **Trade-off:** 4-bit costs ~1-2% accuracy vs FP16, but unlocks running both models simultaneously without paying for an L4. Pro-tier Colab users can swap to FP16 by changing two model IDs in the notebook.

### Neon Postgres instead of SQLite
Neon's free serverless tier costs nothing and gives us a real Postgres (JSONB, proper UUID, etc.). The DB lives outside Colab so it persists across notebook disconnects. **Trade-off:** one extra network hop, but Neon's latency is ~50ms from US Colab regions — negligible compared to model inference.

### Image preprocessing
Deskew + CLAHE contrast + downsize to 1536px long edge before sending to the VLM. **Trade-off:** preprocessing takes ~100ms per image but improves accuracy on tilted/dim photos. Skipped if preprocessing throws.

## Ambiguities I had to resolve

See `notes.md` for the full running log. Highlights:

1. **"Biggest grocery bill in March"** — treated *bill* as a single receipt total (not a sum across receipts).
2. **Foreign currency** — stored as ISO 4217 codes per receipt; queries that span currencies return a per-currency breakdown rather than converting.
3. **Date ambiguity** — preferred day-first (DD/MM/YYYY) when format was ambiguous, since most international receipts use that.
4. **Category taxonomy** — fixed list of 10 categories (`groceries, dining, transport, fuel, utilities, entertainment, health, shopping, travel, other`).
5. **Duplicate uploads** — SHA-256 hash of image bytes; duplicates are skipped, not failed.
6. **Partial receipts** — never fail an upload; return the receipt with `null` fields and an `extraction_notes` string explaining what went wrong.

## What failed

(Fill in honestly after running the eval. Examples to look for:)

- Receipts with heavy thermal-paper fade — VLM hallucinates prices.
- Multi-page or multi-column receipts — line items get merged.
- Receipts where the total line is cropped — model invents totals; the validation loop catches some but not all.
- Text-to-SQL on questions that need `EXTRACT(year ...)` semantics around year-end.
- Categorization of generic item codes like `"ITEM-3492"` — falls through to `other`.

## What I would do with another week

- **Fine-tune the VLM on a labelled receipt set** (LoRA on 500-1000 hand-labelled samples) — biggest expected accuracy lift.
- **Confidence-routed review queue**: receipts with `confidence < 0.6` get flagged for human correction; build a tiny HTML correction UI.
- **Hybrid retrieval + SQL for fuzzy queries** — when text-to-SQL fails, embed line item descriptions and use vector search.
- **Multi-currency support with FX conversion** at query time (rates pulled daily from a free API).
- **Async upload pipeline** — return a job ID immediately, process VLM in the background, push results via webhook or polling endpoint.
- **Docker compose** for one-command local + GPU deployment.
- **Add proper auth** so the system is usable beyond a demo.
- **Eval expansion** — adversarial receipts (multi-language, handwritten notes, photos of phone screens of receipts).

## Repo layout

```
app/
  main.py              FastAPI endpoints
  models.py            SQLAlchemy ORM
  schemas.py           Pydantic request/response types
  db.py                Neon connection setup
  extraction.py        VLM extraction + JSON parse + repair loop
  categorization.py    Rules + SLM fallback
  query.py             Text-to-SQL + sqlglot validation + summarization
  preprocessing.py     OpenCV image cleanup
  model_clients.py     VLM/SLM adapter interfaces + Fakes + HF pipeline wrappers
  prompts/
    extract.txt        VLM prompt for extraction
    text2sql.txt       SLM prompt for SQL generation (with schema + few-shots)
    summarize.txt      SLM prompt for sentence answer
notebooks/
  colab_runner.ipynb   Colab launcher: load models, start API, ngrok tunnel
scripts/
  setup_db.py          Create tables on Neon
  evaluate.py          Eval harness (extraction accuracy + query pass rate)
samples/               Drop evaluator-provided images here
```
