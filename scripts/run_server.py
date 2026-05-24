#!/usr/bin/env python
"""Standalone runner for EC2 / any GPU host with >=20GB VRAM.

Loads both AWQ models, wires them into the FastAPI app, starts uvicorn on
0.0.0.0:8000. No notebook, no ngrok required (open the port in your security
group). Designed to be run as `python scripts/run_server.py` from the repo root.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import pipeline
import uvicorn

VLM_MODEL = os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ')
SLM_MODEL = os.environ.get('SLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct-AWQ')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))

print(f'[runner] DATABASE_URL set: {bool(os.environ.get("DATABASE_URL"))}')
print(f'[runner] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[runner] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)')

print(f'[runner] Loading VLM: {VLM_MODEL}')
vlm_pipe = pipeline(
    'image-text-to-text',
    model=VLM_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
)

proc = vlm_pipe.processor
img_proc = getattr(proc, 'image_processor', proc)
if hasattr(img_proc, 'max_pixels'):
    img_proc.min_pixels = 256 * 28 * 28
    img_proc.max_pixels = 1280 * 28 * 28
print(f'[runner] VLM processor pixel cap: {getattr(img_proc, "max_pixels", "n/a")}')

print(f'[runner] Loading SLM: {SLM_MODEL}')
slm_pipe = pipeline(
    'text-generation',
    model=SLM_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
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
