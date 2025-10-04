# main.py
import os
import json
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Levenshtein for efficient distance calculation
try:
    import Levenshtein
    LEV_AVAILABLE = True
except ImportError:
    LEV_AVAILABLE = False

# Try optional ASR modules
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Dataset location (default Render path)
DATA_PATH = os.environ.get("ATHISUDI_DATA_PATH", "/mnt/data/athisudi_dataset.json")

app = FastAPI(title="Pronunciation Checker API (Levenshtein Optimized)")

# Allow frontend (React) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset once
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

PHRASES = {item["id"]: item for item in data.get("athisudi", [])}
DEFAULT_THRESHOLD = 0.6


# ----------------- Helper Functions ----------------- #
def normalize_text(text: str) -> str:
    if not text:
        return ""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity_score(a: str, b: str) -> float:
    """Compute normalized similarity using Levenshtein distance."""
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    if LEV_AVAILABLE:
        distance = Levenshtein.distance(a, b)
    else:
        # fallback simple edit distance
        distance = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    max_len = max(len(a), len(b))
    score = 1 - (distance / max_len)
    return round(max(score, 0.0), 4)


async def transcribe_with_whisper(file_path: str) -> str:
    model = whisper.load_model("small")
    result = model.transcribe(file_path, language=None)
    return result.get("text", "").strip()


def transcribe_with_google(file_path: str, language="ta-IN") -> str:
    if not SR_AVAILABLE:
        raise RuntimeError("speech_recognition not installed.")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language).strip()
    except Exception:
        return ""


# ----------------- API Models ----------------- #
class CheckResult(BaseModel):
    success: bool
    phrase_id: int
    expected_tamil: Optional[str]
    expected_transliteration: Optional[str]
    meaning_en: Optional[str]
    audio_tts: Optional[str]
    transcribed_text: Optional[str]
    score: float
    threshold: float
    verdict: str
    details: dict


# ----------------- API Endpoints ----------------- #
@app.get("/phrases")
def list_phrases():
    return {"count": len(PHRASES), "phrases": list(PHRASES.values())}


@app.get("/phrases/{phrase_id}")
def get_phrase(phrase_id: int):
    phrase = PHRASES.get(phrase_id)
    if not phrase:
        raise HTTPException(status_code=404, detail="Phrase not found")
    return phrase


@app.post("/check_pronunciation", response_model=CheckResult)
async def check_pronunciation(
    phrase_id: int = Form(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    asr_engine: Optional[str] = Form("auto"),
    audio_file: UploadFile = File(...)
):
    if phrase_id not in PHRASES:
        raise HTTPException(status_code=404, detail="Invalid phrase_id")

    phrase = PHRASES[phrase_id]
    tmpdir = tempfile.mkdtemp()
    try:
        ext = os.path.splitext(audio_file.filename)[1] or ".wav"
        tmp_path = os.path.join(tmpdir, f"upload{ext}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # Choose ASR engine
        transcribed = ""
        used_engine = None
        if asr_engine == "whisper" and WHISPER_AVAILABLE:
            used_engine = "whisper"
            transcribed = await transcribe_with_whisper(tmp_path)
        elif asr_engine == "google":
            used_engine = "google"
            transcribed = transcribe_with_google(tmp_path)
        else:
            if WHISPER_AVAILABLE:
                used_engine = "whisper"
                transcribed = await transcribe_with_whisper(tmp_path)
            elif SR_AVAILABLE:
                used_engine = "google"
                transcribed = transcribe_with_google(tmp_path)

        if not transcribed:
            return CheckResult(
                success=False,
                phrase_id=phrase_id,
                expected_tamil=phrase["tamil"],
                expected_transliteration=phrase["transliteration"],
                meaning_en=phrase["meaning_en"],
                audio_tts=phrase["audio_tts"],
                transcribed_text="",
                score=0.0,
                threshold=threshold,
                verdict="try_again",
                details={"note": "ASR returned empty", "asr_engine": used_engine}
            )

        # Calculate similarity
        s1 = similarity_score(transcribed, phrase["tamil"])
        s2 = similarity_score(transcribed, phrase["transliteration"])
        score = max(s1, s2)
        verdict = "correct" if score >= threshold else "try_again"

        return CheckResult(
            success=True,
            phrase_id=phrase_id,
            expected_tamil=phrase["tamil"],
            expected_transliteration=phrase["transliteration"],
            meaning_en=phrase["meaning_en"],
            audio_tts=phrase["audio_tts"],
            transcribed_text=transcribed,
            score=score,
            threshold=threshold,
            verdict=verdict,
            details={"asr_engine": used_engine, "s_tamil": s1, "s_translit": s2}
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "whisper_available": WHISPER_AVAILABLE,
        "speech_recognition_available": SR_AVAILABLE,
        "levenshtein_available": LEV_AVAILABLE
    }
