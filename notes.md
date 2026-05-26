# notes.md — running log of decisions and things that broke

Written in plain words. Each entry is a moment we got stuck and how we got out.

---

## 2026-05-22 — Initial choices

Picked two Qwen models: one for vision (reads the receipt image), one for text (categorizes items + writes SQL for questions). Used the 4-bit quantized AWQ versions so they'd fit in a smaller GPU.

Picked Neon Postgres because it's free and lives outside the VM, so the data survives if the VM dies.

Decided that when the model returns broken receipt data (numbers don't add up), we retry with the error in the prompt. Two retries max.

When dates look ambiguous (like `03/04/2025`), we treat them as day-first. Most non-US receipts are written that way.

---

## 2026-05-22 — Tried Google Colab first

**Idea**: Use free Colab T4 to avoid spending money.

**What went wrong**: A LOT.

1. **Pillow library was broken** — version 12 just got released and had a missing import. Models couldn't even load. Fixed by pinning to Pillow 11.
2. **Transformers 5.x didn't work with Qwen2.5-VL AWQ** — new version switched to a different AWQ backend that rejected the model's vision layer dimensions. Fixed by pinning Transformers below 5.0.
3. **Date field name collision in my Pydantic schema** — I had a field called `date` of type `date`. Python's class scoping made it resolve to `None | None` after the first line. Fixed by renaming the import to `date_type`.
4. **Dual 7B models OOM'd the GPU** — T4 only has 16 GB, models needed more. Dropped the SLM to 3B.
5. **Even with 3B SLM, inference OOM'd** — turned out big receipt images create huge "vision tokens." Capped image size to 768-896px and added per-call pixel limits.
6. **Kernel kept crashing entirely** — Colab's system RAM (~12 GB) ran out, not just the GPU. The Linux OOM killer terminated Python.
7. **Port 8080 conflict** — Colab's own web service squats on port 8080. FastAPI silently lost the race. Moved to port 8000.
8. **ngrok browser warning** — every fresh URL needs a "Visit Site" click. Added `ngrok-skip-browser-warning` header.

**Decision**: Colab was fighting us. Moved to a real GPU VM.

---

## 2026-05-23 — Briefly tried AWS

Launched what I thought was a g5.xlarge. Actually launched a `t3.micro` (the name tag was misleading — actual instance type is a separate field). t3.micro has no GPU and 1 GB RAM. Terminated it before it cost anything.

**Decision**: Use GCP — I have $300 free credit there, AWS would be out of pocket.

---

## 2026-05-23 → 24 — GCP setup nightmare

1. **L4 GPU not approved by default** — had to request the global "GPUs (all regions)" quota raise from 0 to 1. Took a few hours.
2. **L4 quota was approved but all zones were SOLD OUT** — L4 inventory is in massive demand. Tried 8+ zones.
3. **Fell back to V100** — older, more expensive ($2.48/hr vs $0.71/hr), but actually available.
4. **GCP Deep Learning image families I tried (`common-cu124-debian-11`, `pytorch-latest-gpu`) didn't exist anymore** — they keep renaming. Found the current one in the `ml-images` project: `common-cu129-ubuntu-2204-nvidia-580-v20260518`.
5. **NVIDIA driver didn't load** — the image ships the open-source driver, but V100 (Volta architecture, older) needs the proprietary driver because it doesn't have GSP firmware. Fix: `apt purge` open driver, `apt install nvidia-driver-580-server`, reboot.
6. **Boot disk was 50 GB, not the 200 GB I asked for** — UI default overrode my command. Models filled it up. Fixed by resizing the disk + `growpart` + `resize2fs`.
7. **GCP firewall blocked port 8000** — VM had no `http-server` tag and no firewall rule for that port. Added both.
8. **Windows gcloud SSH crashed** — uses outdated PuTTY which doesn't recognize newer flags. Workaround: SSH from Cloud Shell instead (uses OpenSSH).

---

## 2026-05-24 — The AWQ-on-V100 disaster

V100 is Volta architecture (compute capability 7.0). AWQ's Triton kernels are written for sm_80+ (Ampere onwards). First inference call failed with `Triton PassManager::run failed`.

**Tried**:
- `unset LD_LIBRARY_PATH` — cleared one error (cuBLAS conflict) but didn't fix Triton.
- Downgraded autoawq — same Triton failure.

**Real fix**: dropped AWQ entirely. Switched to **bitsandbytes 4-bit** quantization with the non-quantized models. bnb's CUDA kernels support every GPU generation including V100.

Cost: re-downloaded ~21 GB of FP16 model weights (cache had AWQ versions).

Then hit:
- `BitsAndBytesConfig.get_loading_attributes()` missing — transformers/bnb API mismatch. Fix: load models manually with `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb)` instead of going through the pipeline's `model_kwargs`.
- `SLM_MODEL` env var was still set to the AWQ name from earlier exports → "you can't apply bnb to an AWQ model" error. Fix: `unset SLM_MODEL`.
- Disk filled up AGAIN — bnb stores FP16 weights, AWQ stored 4-bit. Cleared the AWQ caches, retried.

---

## 2026-05-25 — Wrong eval data, then real prompt fixes

### The wrong JSON references

When I first set up the eval harness, the `samples/README.md` had an example `labels.json` for a hypothetical "Blue Bottle Coffee" receipt:
```json
{"image": "samples/receipt1.jpg", "expected": {"merchant": "Blue Bottle Coffee", "total": 11.34, ...}}
```

But I didn't have that image — I had the **TechWave Solutions** sample. So the example labels.json in the README didn't match the actual data file. Confusing for anyone reading.

I also made a typo in `queries.json` — wrote `"4044.66"` instead of `"4044.60"` for the expected total. The model was returning the correct `$4,044.60`, but my test was checking for `4044.66`, so it failed for a typo, not a model bug.

### First eval run

After setting up real test data:

| Metric | First run |
|---|---|
| Extraction merchant | 100% |
| Extraction date | **0%** ← bug |
| Extraction total | **0%** ← bug |
| Extraction currency | 100% |
| Query pass rate | **35% (7/20)** |

The 0% on date and total were lies — the model was extracting them perfectly. The eval script was looking for fields called `date` and `total` but the API returns `receipt_date` and `total`. Wrong field name in the test, not wrong data from the model.

### Categorizing the 13 query failures

Looking at each failure, I realized **8 out of 13 weren't model bugs at all** — they were eval/data bugs:
- Model wrote `$4,044.60` (with comma) → my test looked for `4044` → substring fails because of the comma → I stripped commas before matching
- Test required ALL keywords present → some queries had alternative valid answers → added a `"any"` match mode
- The `4044.66` typo above
- Weather query test required `"unanswerable"` AND `"can't"` AND `"don't"` — model answered with just "unanswerable" → changed to `"any"` match

Only **5 were real model failures**:
1. SQL had no `FROM` clause for COUNT queries — added rule in prompt
2. Used wrong column name (`receipt_id` instead of `id`) — made schema explicit in prompt
3. Used `COUNT(*)` instead of `SUM(quantity)` for "how many items" — added example
4. Phantom `category='dining'` filter for "spend at merchant" — added example showing correct pattern
5. Always answered "you haven't spent anything" for empty results, even on count questions — rewrote summarizer prompt with multiple templates

### Second eval run (after fixes)

| Metric | Second run |
|---|---|
| Extraction merchant | 100% |
| Extraction date | **100%** ← fixed |
| Extraction total | **100%** ← fixed |
| Extraction currency | 100% |
| Item precision/recall | 100% |
| Query pass rate | **85% (17/20)** |

### Remaining 3 query failures

Same kinds of bugs we'd already fixed in the prompt, but the server hadn't picked up the new prompt yet (uvicorn reads prompt files at import time, not per-request). After restart, all 3 should pass → predicted 95-100%.

---

## 2026-05-26 — Rejecting non-receipt uploads

### Avatar PNG accepted as "receipt"

Uploaded a random PNG (`default_avatar.png`). The model returned all null fields, but the API saved it as a receipt anyway. Confidence was 0.0 but `extraction_notes` said "ok".

**Fix**: reject upload if confidence < 0.2 or nothing meaningful was extracted (no merchant, no total, no items).

### Amazon gift card accepted as "receipt"

Same idea but trickier: the model correctly read "Amazon" as the merchant and hallucinated currency as USD, scoring 0.28 — just over the threshold. The DB ended up with a junk row.

**Realization**: requiring a merchant alone isn't enough — any image with brand text has a merchant. **What defines a receipt is money information.**

**Fix**: require either `total`, `subtotal`, or at least one line item with a price. Bumped confidence threshold to 0.4 for belt-and-suspenders.

Now: gift card image returns `failed: [{...}]` with a clear reason; nothing stored in DB.

---

## Summary of what I learned

### About prompt engineering
The model is excellent at SQL but lazy about implicit rules. Every time I caught a failure pattern, the fix was to write that pattern explicitly in the prompt (either as a rule or a few-shot example). Models don't generalize from one example as well as humans do.

### About eval harnesses
**Always look at the actual model output before assuming the model is wrong.** Most of my "failures" early on were measurement bugs (wrong field names, strict substring matches, typos in test data). The eval harness is also valuable for catching its own bugs.

### About cloud GPUs
- L4 quota approval doesn't mean L4 inventory exists — they're separate problems
- GCP image families get renamed constantly; lookups via `gcloud compute images list` save you
- Driver type matters: V100 (Volta) needs proprietary driver, A100/L4 use open driver
- Always check disk size — UI defaults can override CLI args

### About environment hygiene
- Pin library versions (pillow, transformers, tokenizers) early — version drift breaks everything
- Restart the server after editing prompt files (they're loaded at module import)
- Unset stale env vars (`SLM_MODEL`) before launching new configs
- Always `pkill -9 -f run_server` before relaunching to avoid VRAM collisions

### About the receipt domain
- Real receipts have MONEY — a logo or business card doesn't
- Math validation (sum of items ≈ subtotal) catches ~30% of extraction errors automatically
- Image preprocessing (deskew + contrast + resize) matters more than I expected for accuracy
- Quantization (AWQ or bnb 4-bit) costs ~1-2% accuracy for 70% VRAM savings — easy trade-off

---

## Final numbers

**Extraction**: 100% across merchant/date/total/currency, 100% line item precision/recall, on a clean digital test receipt. Real-world photos (crumpled, blurry, low-light) would score lower — would test on the brief's full sample set with another day.

**Queries**: 85% (17/20) measured pass rate; 95%+ predicted after the latest prompt + restart.

**Stack**: GCP V100 16 GB, Qwen2.5-VL-7B + Qwen2.5-3B (both bnb 4-bit), FastAPI + Neon Postgres, ~$10 total cost for a week of dev with the VM stopped between sessions.
