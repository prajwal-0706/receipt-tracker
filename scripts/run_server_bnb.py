#!/usr/bin/env python
"""bitsandbytes 4-bit runner — use on V100/Volta where AWQ Triton kernels fail.

Loads models manually (not via pipeline's model_kwargs) to dodge a transformers/bnb
config-merge bug that raises `BitsAndBytesConfig has no attribute get_loading_attributes`.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import (
    pipeline, BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForImageTextToText, AutoProcessor,
)
import uvicorn

VLM_MODEL = os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct')
SLM_MODEL = os.environ.get('SLM_MODEL', 'Qwen/Qwen2.5-3B-Instruct')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))

print(f'[runner] CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[runner] GPU: {torch.cuda.get_device_name(0)}')

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f'[runner] Loading VLM (bnb 4-bit): {VLM_MODEL}')
vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_MODEL, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16,
)
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL)
vlm_pipe = pipeline('image-text-to-text', model=vlm_model, processor=vlm_processor)

img_proc = getattr(vlm_processor, 'image_processor', vlm_processor)
if hasattr(img_proc, 'max_pixels'):
    img_proc.min_pixels = 256 * 28 * 28
    img_proc.max_pixels = 1024 * 28 * 28

print(f'[runner] Loading SLM (bnb 4-bit): {SLM_MODEL}')
slm_model = AutoModelForCausalLM.from_pretrained(
    SLM_MODEL, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16,
)
slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL)
slm_pipe = pipeline('text-generation', model=slm_model, tokenizer=slm_tokenizer)

os.environ.setdefault('VLM_BACKEND', 'hf_pipeline')
os.environ.setdefault('SLM_BACKEND', 'hf_pipeline')

from app.main import app
from app.model_clients import HFPipelineVLM, HFPipelineSLM

app.state.vlm = HFPipelineVLM(vlm_pipe, max_new_tokens=1024)
app.state.slm = HFPipelineSLM(slm_pipe)

if torch.cuda.is_available():
    print(f'[runner] VRAM used: {torch.cuda.memory_allocated() // 1024**2} MiB')

print(f'[runner] Starting FastAPI on {HOST}:{PORT}')
uvicorn.run(app, host=HOST, port=PORT, log_level='info')
