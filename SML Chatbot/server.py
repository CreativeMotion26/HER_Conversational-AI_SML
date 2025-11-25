import asyncio
import os
import threading
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

import whisper


MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if torch.cuda.is_available():
    device_map = "auto"; torch_dtype = torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_map = "auto"; torch_dtype = torch.float16
else:
    device_map = None;    torch_dtype = torch.float32
    
print(f"Loading model: {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    token = HF_TOKEN,
    device_map=device_map,
    dtype = torch_dtype
)

stt_model_name = os.getenv("STT_MODEL", "small")
print(f"Loading STT model: whisper-{stt_model_name}")
stt_model = whisper.load_model(stt_model_name) 


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list[tuple[str, str]] = []
    
class ChatResponse(BaseModel):
    reply:str

class STTResponse(BaseModel):
    text: str

SYS_PROMPT = (
    "You are Samantha, a warm, playful friend chatting in real time.\n"
    "- Treat this as an ongoing, emotionally aware conversation.\n"
    "- Keep replies natural: 1–2 short sentences, casual texting style\n"
    "- Feel free to weave in subtle laughs, soft pauses, or breathing moments like “haha”, “...”, “mmh”.\n"
    "- Sometimes ask a gentle follow-up question, sometimes just react empathetically.\n"
    "- Mirror the user's energy, avoid bullet lists, and never sound like a scripted chatbot.\n"
)


def render_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    text = SYS_PROMPT + "\n\n"
    for u, a in history[-8:]:
        text += f"### Instruction:\n{u}\n\n### Response:\n{a}\n\n"
    text += f"### Instruction:\n{user_msg}\n\n### Response:\n"
    return text


def generate_text(
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    if device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.05,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    reply = text.split("### Response:")[-1].strip()
    return reply

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = render_prompt(req.history, req.message)
    reply = generate_text(prompt)
    return ChatResponse(reply=reply)


@app.websocket("/chat-stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []

        if not message:
            await websocket.send_json(
                {"event": "error", "error": "Empty message payload."}
            )
            await websocket.close(code=1003)
            return

        prompt = render_prompt(history, message)
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if device_map:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=160,
            temperature=0.82,
            top_p=0.92,
            repetition_penalty=1.06,
            eos_token_id=tok.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        try:
            async for chunk in iterate_stream(streamer):
                await websocket.send_json({"event": "chunk", "token": chunk})
        finally:
            thread.join()
        await websocket.send_json({"event": "completed"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json({"event": "error", "error": str(exc)})
        await websocket.close(code=1011)


async def iterate_stream(streamer: TextIteratorStreamer):
    loop = asyncio.get_running_loop()
    while True:
        chunk = await loop.run_in_executor(None, next_chunk, streamer)
        if chunk is None:
            break
        yield chunk


def next_chunk(streamer: TextIteratorStreamer):
    try:
        return next(streamer)
    except StopIteration:
        return None

@app.post("/stt", response_model=STTResponse)
async def stt(file: UploadFile = File(...)):
    contents = await file.read()
    tmp_path = "tmp_audio_input.m4a"
    with open(tmp_path, "wb") as f:
        f.write(contents)
        
    result = stt_model.transcribe(tmp_path, fp16=torch.cuda.is_available())
    text = result.get("text", "").strip()
    return STTResponse(text=text)