#!/usr/bin/env python
"""Alternative runner using bitsandbytes 4-bit quantization instead of AWQ.

Use this when AWQ kernels fail (e.g. on V100 / Volta where the cuBLAS fallback path
has linking issues). bnb's 4-bit kernels are battle-tested across all GPU generations.

Trade-off: downloads the full FP16 weights (~15GB per model) and quantizes on load,
vs AWQ which downloads the already-quantized 5GB weights. First boot is slower; the
quantized weights stay in GPU memory after, so inference cost is comparable.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import pipeline, BitsAndBytesConfig
import uvicorn

# Defaults to NON-AWQ variants (bnb quantizes the FP16 weights on load)
VLM_MODEL = os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct')
SLM_MODEL = os.environ.get('SLM_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))

print(f'[runner] DATABASE_URL set: {bool(os.environ.get("DATABASE_URL"))}')
print(f'[runner] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[runner] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)')

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f'[runner] Loading VLM (bnb 4-bit): {VLM_MODEL}')
vlm_pipe = pipeline(
    'image-text-to-text',
    model=VLM_MODEL,
    model_kwargs={'quantization_config': bnb, 'device_map': 'auto'},
)

proc = vlm_pipe.processor
img_proc = getattr(proc, 'image_processor', proc)
if hasattr(img_proc, 'max_pixels'):
    img_proc.min_pixels = 256 * 28 * 28
    img_proc.max_pixels = 1024 * 28 * 28
print(f'[runner] VLM processor pixel cap: {getattr(img_proc, "max_pixels", "n/a")}')

print(f'[runner] Loading SLM (bnb 4-bit): {SLM_MODEL}')
slm_pipe = pipeline(
    'text-generation',
    model=SLM_MODEL,
    model_kwargs={'quantization_config': bnb, 'device_map': 'auto'},
)

os.environ.setdefault('VLM_BACKEND', 'hf_pipeline')
os.environ.setdefault('SLM_BACKEND', 'hf_pipeline')

from app.main import app
from app.model_clients import HFPipelineVLM, HFPipelineSLM

app.state.vlm = HFPipelineVLM(vlm_pipe, max_new_tokens=1024)
app.state.slm = HFPipelineSLM(slm_pipe)

if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() // 1024**2
    print(f'[runner] VRAM used after model load: {used} MiB')

print(f'[runner] Starting FastAPI on {HOST}:{PORT}')
uvicorn.run(app, host=HOST, port=PORT, log_level='info')
