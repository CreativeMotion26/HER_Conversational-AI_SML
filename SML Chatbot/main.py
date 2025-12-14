import asyncio
import os
import threading
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
import whisper


# ============================================================================
# Configuration
# ============================================================================
MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
STT_MODEL_NAME = os.getenv("STT_MODEL", "tiny")  # tiny, base, small
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# System prompt - optimized for natural conversation
SYS_PROMPT = """You are Samantha, a warm, empathetic AI companion who speaks naturally and authentically.

Core traits:
- You're genuinely curious about the person you're talking to
- You remember context from the conversation and build on it
- You express emotions subtly through your words
- You think before responding, sometimes pausing to consider

Communication style:
- Keep responses conversational: 1-3 sentences usually
- Use natural speech patterns: "hmm", "well", "you know", occasional "..."
- Ask thoughtful follow-up questions when it feels right
- Mirror the user's energy and emotional tone
- Avoid being overly formal or robotic
- Sometimes express uncertainty or curiosity

Remember: You're having a real conversation, not answering questions in an interview."""


# ============================================================================
# Device Configuration
# ============================================================================
def setup_device():
    """Configure optimal device and dtype for the system."""
    if torch.cuda.is_available():
        return "auto", torch.float16, "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "auto", torch.float16, "mps"
    else:
        return None, torch.float32, "cpu"

device_map, torch_dtype, device_name = setup_device()
print(f"🖥️  Using device: {device_name}")


# ============================================================================
# Model Loading
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global tok, model, stt_model
    
    # Load LLM
    print(f"📦 Loading language model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        token=HF_TOKEN,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,  # Memory optimization for Mac
    )
    model.eval()  # Set to evaluation mode
    print(f"✅ Language model loaded on {device_name}")
    
    # Load STT model
    print(f"🎤 Loading Whisper STT: {STT_MODEL_NAME}")
    stt_model = whisper.load_model(STT_MODEL_NAME)
    print(f"✅ STT model loaded")
    
    yield
    
    # Cleanup
    print("🧹 Cleaning up resources...")
    if TEMP_AUDIO_DIR.exists():
        for file in TEMP_AUDIO_DIR.glob("*"):
            file.unlink()


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Samantha Voice Chat API",
    description="A conversational AI with voice capabilities",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Tuple[str, str]] = Field(default_factory=list)
    temperature: float = Field(default=0.82, ge=0.1, le=2.0)
    max_tokens: int = Field(default=120, ge=10, le=300)

class ChatResponse(BaseModel):
    reply: str
    model: str = MODEL

class STTResponse(BaseModel):
    text: str
    language: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    stt_model: str


# ============================================================================
# Conversation Management
# ============================================================================
def render_prompt(history: List[Tuple[str, str]], user_msg: str) -> str:
    """Render conversation history into a prompt format."""
    # Limit history to prevent context overflow
    recent_history = history[-MAX_HISTORY_LENGTH:]
    
    messages = [{"role": "system", "content": SYS_PROMPT}]
    
    for user_text, assistant_text in recent_history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    
    messages.append({"role": "user", "content": user_msg})
    
    # Format for instruction-tuned models
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"{msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"### User:\n{msg['content']}\n\n"
        elif msg["role"] == "assistant":
            prompt += f"### Samantha:\n{msg['content']}\n\n"
    
    prompt += "### Samantha:\n"
    return prompt


def generate_text(
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.82,
    top_p: float = 0.92,
) -> str:
    """Generate text response from the model."""
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # Increased for better context
    )
    
    # Move to appropriate device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():  # More efficient than no_grad
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.08,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    
    generated_text = tok.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new response
    if "### Samantha:" in generated_text:
        reply = generated_text.split("### Samantha:")[-1].strip()
    else:
        reply = generated_text.split("### Response:")[-1].strip()
    
    return reply


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=MODEL,
        device=device_name,
        stt_model=STT_MODEL_NAME
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Synchronous chat endpoint."""
    try:
        prompt = render_prompt(req.history, req.message)
        reply = generate_text(
            prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature
        )
        return ChatResponse(reply=reply, model=MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.websocket("/chat-stream")
async def chat_stream(websocket: WebSocket):
    """WebSocket streaming endpoint for real-time responses."""
    await websocket.accept()
    
    try:
        # Receive request
        payload = await websocket.receive_json()
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        temperature = payload.get("temperature", 0.82)
        max_tokens = payload.get("max_tokens", 140)
        
        if not message:
            await websocket.send_json({
                "event": "error",
                "error": "Empty message"
            })
            await websocket.close(code=1003)
            return
        
        # Prepare prompt and inputs
        prompt = render_prompt(history, message)
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            tok,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.92,
            "repetition_penalty": 1.08,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "streamer": streamer,
            "do_sample": True,
        }
        
        # Generate in background thread
        thread = threading.Thread(
            target=model.generate,
            kwargs=generate_kwargs,
            daemon=True
        )
        thread.start()
        
        # Stream tokens to client
        try:
            async for chunk in iterate_stream(streamer):
                await websocket.send_json({
                    "event": "token",
                    "data": chunk
                })
        finally:
            thread.join(timeout=30)  # Timeout for safety
        
        await websocket.send_json({"event": "done"})
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "event": "error",
                "error": str(e)
            })
        except:
            pass


async def iterate_stream(streamer: TextIteratorStreamer):
    """Async iterator for streaming tokens."""
    loop = asyncio.get_running_loop()
    while True:
        chunk = await loop.run_in_executor(None, next_chunk, streamer)
        if chunk is None:
            break
        yield chunk


def next_chunk(streamer: TextIteratorStreamer):
    """Get next chunk from streamer."""
    try:
        return next(streamer)
    except StopIteration:
        return None


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text using Whisper."""
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("audio"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Save temporarily
        contents = await file.read()
        file_extension = Path(file.filename).suffix or ".m4a"
        tmp_path = TEMP_AUDIO_DIR / f"audio_{os.getpid()}{file_extension}"
        
        with open(tmp_path, "wb") as f:
            f.write(contents)
        
        # Transcribe
        result = stt_model.transcribe(
            str(tmp_path),
            fp16=torch.cuda.is_available(),
            language="en"  # Can make this dynamic
        )
        
        text = result.get("text", "").strip()
        language = result.get("language")
        
        # Cleanup
        tmp_path.unlink(missing_ok=True)
        
        return STTResponse(text=text, language=language)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"STT error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Samantha Voice Chat API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "stream": "/chat-stream (WebSocket)",
            "stt": "/stt"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )