"""Protocol + adapters so the app depends on an interface, not on torch."""
from __future__ import annotations
from typing import Any, Protocol
from PIL import Image


class VLMClient(Protocol):
    def extract(self, image: Image.Image, prompt: str) -> str: ...


class SLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str: ...


_FAKE_RECEIPT_JSON = """{
  "merchant": "Test Merchant",
  "date": "2026-05-01",
  "line_items": [
    {"description": "Latte", "quantity": 1, "unit_price": 4.50, "total_price": 4.50},
    {"description": "Croissant", "quantity": 2, "unit_price": 3.00, "total_price": 6.00}
  ],
  "subtotal": 10.50,
  "tax": 0.84,
  "total": 11.34,
  "currency": "USD"
}"""


class FakeVLM:
    def extract(self, image: Image.Image, prompt: str) -> str:
        return _FAKE_RECEIPT_JSON


class FakeSLM:
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        lowered = prompt.lower()
        if "category" in lowered:
            return "dining"
        if "sql" in lowered:
            return "SELECT COALESCE(SUM(total), 0) AS total_spend FROM receipts;"
        if "summarize" in lowered or "answer" in lowered:
            return "You spent $0 in the requested period (stub answer)."
        return ""


def _free_gpu():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class HFPipelineVLM:
    def __init__(self, pipe: Any, max_new_tokens: int = 768):
        self.pipe = pipe
        self.max_new_tokens = max_new_tokens
        # Qwen2.5-VL counts vision tokens as (H*W)/(28*28). Cap so a 4K receipt
        # doesn't blow VRAM on a constrained GPU (Colab T4, V100 16GB).
        proc = getattr(pipe, "image_processor", None) or getattr(pipe, "processor", None)
        if proc is not None:
            for attr in ("min_pixels", "max_pixels"):
                if hasattr(proc, attr):
                    setattr(proc, attr, 256 * 28 * 28 if attr == "min_pixels" else 1024 * 28 * 28)

    def extract(self, image: Image.Image, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": 256 * 256,
                            "max_pixels": 1024 * 1024,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            out = self.pipe(text=messages, max_new_tokens=self.max_new_tokens)
            return _extract_assistant_text(out)
        finally:
            _free_gpu()


class HFPipelineSLM:
    def __init__(self, pipe: Any):
        self.pipe = pipe

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                kwargs["temperature"] = temperature
            out = self.pipe(messages, **kwargs)
            return _extract_assistant_text(out)
        finally:
            _free_gpu()


def _extract_assistant_text(pipeline_output: Any) -> str:
    # transformers' chat pipelines have returned several different shapes across
    # 4.45+ — list of messages, string, list of content parts. Handle them all.
    if not pipeline_output:
        return ""
    first = pipeline_output[0]
    generated = first.get("generated_text", first)
    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return "".join(texts).strip()
    return str(generated).strip()
