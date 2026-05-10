import asyncio
import json
import threading
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio as ta
import torch
import sys
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarlekha_model.tts import SwarlekhaTTS
from swarlekha_model.tts_nepali import SwarlekhaNepaliTTS
from swarlekha_model.models.t3.t3 import t3_step_callback, GenerationCancelled

app = FastAPI(title="Swarlekha TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

english_model = None
nepali_model = None

# Track model loading status for the frontend loading screen.
# Values: "pending" | "loading" | "loaded" | "error"
model_status: dict[str, str] = {"english": "pending", "nepali": "pending"}

# Per-session cancel events: session_id -> threading.Event
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def _get_cancel_event(session_id: str) -> threading.Event:
    with _cancel_lock:
        ev = threading.Event()
        _cancel_events[session_id] = ev
        return ev


def _remove_cancel_event(session_id: str) -> None:
    with _cancel_lock:
        _cancel_events.pop(session_id, None)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_model(lang: str):
    if lang == "nepali":
        if nepali_model is None:
            raise HTTPException(status_code=503, detail="Nepali model is not loaded")
        return nepali_model
    if english_model is None:
        raise HTTPException(status_code=503, detail="English model is not loaded")
    return english_model


@app.on_event("startup")
async def startup_event():
    global english_model, nepali_model

    loop = asyncio.get_event_loop()

    def load_english():
        global english_model
        model_status["english"] = "loading"
        try:
            english_model = SwarlekhaTTS.from_pretrained(device=device)
            model_status["english"] = "loaded"
            print("English model loaded.")
        except Exception as e:
            model_status["english"] = "error"
            print(f"Failed to load English model: {e}")

    def load_nepali():
        global nepali_model
        model_status["nepali"] = "loading"
        try:
            nepali_model = SwarlekhaNepaliTTS.from_pretrained(device=device)
            model_status["nepali"] = "loaded"
            print("Nepali model loaded.")
        except Exception as e:
            model_status["nepali"] = "error"
            print(f"Failed to load Nepali model: {e}")

    # Load sequentially (models share GPU memory)
    await loop.run_in_executor(None, load_english)
    await loop.run_in_executor(None, load_nepali)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Swarlekha TTS API",
        "version": "1.0.0",
        "device": device,
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_status": model_status,
        # Convenience booleans kept for compatibility
        "english_model_loaded": model_status["english"] == "loaded",
        "nepali_model_loaded": model_status["nepali"] == "loaded",
    }


# ---------------------------------------------------------------------------
# Blocking generate (original, unchanged behaviour)
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_voice(
    text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    voice_name: Optional[str] = Form("default"),
    language: Optional[str] = Form("english"),
):
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text is too long (max 5000 characters)")

    lang = (language or "english").lower().strip()
    active_model = _resolve_model(lang)

    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"examples/output/{voice_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"generated_{timestamp}_{session_id}.wav"
    output_path = output_dir / output_filename

    temp_audio_path = None
    try:
        if reference_audio:
            temp_audio_path = Path(tempfile.gettempdir()) / f"ref_audio_{session_id}.wav"
            with open(temp_audio_path, "wb") as buf:
                buf.write(await reference_audio.read())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _blocking_generate(
                active_model,
                text,
                str(temp_audio_path) if temp_audio_path else None,
                output_path,
            ),
        )

        if temp_audio_path and Path(temp_audio_path).exists():
            Path(temp_audio_path).unlink()

        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=output_filename,
            headers={"X-Session-ID": session_id, "X-Language": lang},
        )
    except Exception as e:
        if temp_audio_path and Path(temp_audio_path).exists():
            Path(temp_audio_path).unlink()
        raise HTTPException(status_code=500, detail=f"Error generating voice: {str(e)}")


def _blocking_generate(active_model, text, audio_prompt_path, output_path):
    if audio_prompt_path:
        wav = active_model.generate(text, audio_prompt_path=audio_prompt_path)
    else:
        wav = active_model.generate(text)
    ta.save(str(output_path), wav, active_model.sr)


# ---------------------------------------------------------------------------
# SSE streaming generate — real progress + cancellation
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/generate-stream")
async def generate_voice_stream(
    text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    voice_name: Optional[str] = Form("default"),
    language: Optional[str] = Form("english"),
):
    """
    SSE endpoint. Events emitted:

      {"phase": "session",      "session_id": "abc12345"}
      {"phase": "preparing"}
      {"phase": "generating",   "tokens": N}   — one per T3 token (real)
      {"phase": "synthesizing"}                — S3Gen starting
      {"phase": "complete",     "audio_url": "/api/result/abc12345"}
      {"phase": "cancelled"}
      {"phase": "error",        "detail": "..."}
    """
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text is too long (max 5000 characters)")

    lang = (language or "english").lower().strip()
    active_model = _resolve_model(lang)

    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"examples/output/{voice_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"generated_{timestamp}_{session_id}.wav"
    output_path = output_dir / output_filename

    cancel_event = _get_cancel_event(session_id)

    temp_audio_path = None
    if reference_audio:
        temp_audio_path = Path(tempfile.gettempdir()) / f"ref_audio_{session_id}.wav"
        with open(temp_audio_path, "wb") as buf:
            buf.write(await reference_audio.read())

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def step_cb(step: int, max_steps: int):
        if cancel_event.is_set():
            raise GenerationCancelled()
        if step == -1:
            loop.call_soon_threadsafe(queue.put_nowait, {"phase": "synthesizing"})
        else:
            loop.call_soon_threadsafe(queue.put_nowait, {"phase": "generating", "tokens": step})

    def worker():
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"phase": "preparing"})

            tok = t3_step_callback.set(step_cb)
            try:
                if temp_audio_path:
                    wav = active_model.generate(text, audio_prompt_path=str(temp_audio_path))
                else:
                    wav = active_model.generate(text)
            finally:
                t3_step_callback.reset(tok)

            ta.save(str(output_path), wav, active_model.sr)

            if temp_audio_path and Path(temp_audio_path).exists():
                Path(temp_audio_path).unlink()

            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"phase": "complete", "audio_url": f"/api/result/{session_id}"},
            )
        except GenerationCancelled:
            if temp_audio_path and Path(temp_audio_path).exists():
                Path(temp_audio_path).unlink()
            loop.call_soon_threadsafe(queue.put_nowait, {"phase": "cancelled"})
        except Exception as e:
            if temp_audio_path and Path(temp_audio_path).exists():
                Path(temp_audio_path).unlink()
            loop.call_soon_threadsafe(
                queue.put_nowait, {"phase": "error", "detail": str(e)}
            )
        finally:
            _remove_cancel_event(session_id)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        # First event: send session_id so frontend can cancel
        yield _sse({"phase": "session", "session_id": session_id})
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield _sse(msg)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Cancel endpoint
# ---------------------------------------------------------------------------

@app.post("/api/cancel/{session_id}")
async def cancel_generation(session_id: str):
    with _cancel_lock:
        ev = _cancel_events.get(session_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Session not found or already complete")
    ev.set()
    return {"cancelled": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Result retrieval (SSE flow)
# ---------------------------------------------------------------------------

@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    output_base = Path("examples/output")
    matches = list(output_base.rglob(f"*_{session_id}.wav"))
    if not matches:
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path=str(matches[0]), media_type="audio/wav", filename=matches[0].name)


@app.get("/api/voices")
async def list_voices():
    output_base = Path("examples/output")
    if not output_base.exists():
        return {"voices": []}
    voices = []
    for voice_dir in output_base.iterdir():
        if voice_dir.is_dir():
            files = list(voice_dir.glob("*.wav"))
            voices.append({"name": voice_dir.name, "count": len(files), "files": [f.name for f in files]})
    return {"voices": voices}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
