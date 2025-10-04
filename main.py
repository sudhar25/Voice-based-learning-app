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

@app.post("/check_pronunciation", response_model=Result)
async def check_pronunciation(
    phrase_id: int = Form(...),
    audio_file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    if phrase_id not in PHRASES:
        raise HTTPException(status_code=404, detail="Invalid phrase_id")
    phrase = PHRASES[phrase_id]

    # Save audio temp
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "user.wav")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    # Transcribe Tamil audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ta-IN")
    except Exception:
        text = ""

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Compare pronunciation
    s_tamil = similarity(text, phrase["tamil"])
    s_translit = similarity(text, phrase["transliteration"])
    score = max(s_tamil, s_translit)
    verdict = "correct" if score >= threshold else "try_again"

    return Result(
        phrase_id=phrase_id,
        transcribed=text,
        score=score,
        verdict=verdict,
        expected_tamil=phrase["tamil"],
        expected_transliteration=phrase["transliteration"],
        meaning_en=phrase["meaning_en"],
        audio_tts=phrase["audio_tts"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}
