import os
import time
import logging
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from contextlib import asynccontextmanager

LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("polish")

model = None
tokenizer = None
device = None

SYSTEM_PROMPT = (
    "你是一个语音助手文案润色专家。"
    "把用户输入的文本用不同的表达方式重写，使句子更自然、更口语化、更有亲和力。\n"
    "要求：\n"
    "1. 保持原意不变\n"
    "2. 字数与原文相差不多\n"
    "3. 每次输出的表达方式要有变化，不要重复\n"
    "4. 只输出润色后的文本，不要加任何解释、引号、前缀或后缀\n"
    "5. 语言风格要自然，像真人说话一样"
)


class PolishRequest(BaseModel):
    requestID: str = Field(..., description="请求唯一标识")
    txt: str = Field(..., min_length=1, description="需要润色的文本")
    temperature: float = Field(0.85, ge=0.1, le=2.0, description="采样温度，越高变化越大")


class PolishResponse(BaseModel):
    requestID: str
    txt: str


def load_model() -> None:
    global model, tokenizer, device

    model_path = "/models/Qwen3-0.6B"

    if os.path.isdir(model_path):
        log.info("PROC | Loading model from local path: %s", model_path)
    else:
        model_path = "Qwen/Qwen3-0.6B"
        log.info("PROC | Model path not found, trying: %s", model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("PROC | Using device: %s", device)

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        torch.cuda.empty_cache()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3 if device == "cuda" else 0
    log.info("PROC | Model loaded. GPU memory: %.2f GB", gpu_memory)


def warmup() -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "你好"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(
            **inputs, max_new_tokens=8, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    log.info("PROC | Warmup complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    warmup()
    yield


app = FastAPI(title="Qwen3 Text Polish", lifespan=lifespan)


@app.post("/polish", response_model=PolishResponse)
async def polish_text(request: PolishRequest, raw: Request):
    t0 = time.perf_counter()
    log.info(
        "RECV | id=%s | text=%s | temp=%.2f | ip=%s",
        request.requestID, request.txt, request.temperature, raw.client.host if raw.client else "-",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.txt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    max_new = min(max(round(len(request.txt) * 1.5), 20), 256)

    t1 = time.perf_counter()
    log.info("PROC | id=%s | input_len=%d | max_new=%d | prefill=%.0fms",
             request.requestID, input_len, max_new, (t1 - t0) * 1000)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=request.temperature,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    t2 = time.perf_counter()
    gen_tokens = outputs.shape[1] - input_len
    gen_time = t2 - t1
    log.info(
        "PROC | id=%s | gen_tokens=%d | infer=%.0fms | speed=%.1f tok/s",
        request.requestID, gen_tokens, gen_time * 1000, gen_tokens / gen_time if gen_time > 0 else 0,
    )

    result = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    total = (t2 - t0) * 1000
    log.info(
        "SEND | id=%s | input=[%s] | output=[%s] | total=%.0fms",
        request.requestID, request.txt, result, total,
    )

    return PolishResponse(requestID=request.requestID, txt=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="::", port=8001, log_level="info")
