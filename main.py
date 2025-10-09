import os, json, tempfile, shutil, re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
import Levenshtein

app = FastAPI(title="Athichudi Pronunciation Checker", version="1.0")

# Allow all origins for testing; restrict later in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load dataset ---
DATA_PATH = "athisudi_dataset.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
PHRASES = {item["id"]: item for item in data.get("athisudi", [])}

# --- Helpers ---
def normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text).strip()

def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return round(1 - (dist / max(len(a), len(b))), 4)

# --- Model ---
class Result(BaseModel):
    phrase_id: int
    transcribed: str
    score: float
    verdict: str
    expected_tamil: str
    expected_transliteration: str
    meaning_en: str
    audio_tts: str

# --- API Endpoints ---
@app.get("/phrases")
def get_phrases():
    return list(PHRASES.values())

from fastapi import UploadFile, Form
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

@app.post("/check_pronunciation")
async def check_pronunciation(audio_file: UploadFile, phrase_id: int = Form(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # Convert .webm → .wav using pydub
        wav_path = tmp_path.replace(".webm", ".wav")
        AudioSegment.from_file(tmp_path).export(wav_path, format="wav")

        # Now process with speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Cleanup temp files
        os.remove(tmp_path)
        os.remove(wav_path)

        # Return recognized text (or compare with your phrase)
        return {"transcribed_text": text, "verdict": "processed"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
