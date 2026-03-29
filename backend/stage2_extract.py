import cv2
import numpy as np
import os
import re
from PIL import Image
import pytesseract
from faster_whisper import WhisperModel
from config import _call_ollama

_whisper = None

def get_whisper(force_cpu=False):
    global _whisper
    # Re-instantiate if forced
    if _whisper is None or force_cpu:
        if not force_cpu:
            print("[Stage 2] Attempting to load Whisper model on GPU (RTX 4060)...")
            _whisper = WhisperModel("base", device="cuda", compute_type="float16")
        else:
            print("[Stage 2] Loading Whisper model on CPU...")
            _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper

def classify_frame(image_path: str) -> str:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        return "talking_head"

    return "visual_content"


def extract_frame_features(image_path: str, content_type: str, use_vlm: bool = True) -> dict:
    if content_type == "talking_head":
        return {
            "ocr_text": "",
            "visual_description": "Talking head — instructor speaking, no slide content",
            "entities": [],
            "used_vlm": False
        }

    try:
        ocr_text = pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception:
        ocr_text = ""

    # Fast path: for text-heavy slides, OCR is usually enough and avoids slow VLM calls.
    if (not use_vlm) or len(ocr_text) >= 220:
        fallback_desc = f"Slide content: {ocr_text[:280]}" if ocr_text else "Visual slide content"
        return {
            "ocr_text": ocr_text,
            "visual_description": fallback_desc,
            "entities": [],
            "used_vlm": False
        }

    # Optional local VLM call on sampled frames only.
    try:
        desc = _call_ollama(
            prompt=(
                "You are parsing a frame from a lecture video. "
                "If it contains mostly text or code, transcribe the text accurately. "
                "If it contains a diagram, chart, or figure, describe what concept it illustrates. "
                "Don't add conversational filler, just the transcription or description."
            ),
            image_path=image_path
        )
    except Exception as e:
        print(f"[Stage 2] Local VLM error on frame: {e}")
        desc = ""
    
    if not desc:
        desc = f"Slide content: {ocr_text[:280]}" if ocr_text else "Visual slide content"

    return {
        "ocr_text": ocr_text,
        "visual_description": desc,
        "entities": [],
        "used_vlm": True
    }


def transcribe_audio(audio_path: str) -> list[dict]:
    model = get_whisper()
    try:
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
    except Exception as e:
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\\n[GPU Error] Missing Windows NVIDIA DLLs for Whisper. Automatically falling back to CPU... ({e})")
            model = get_whisper(force_cpu=True)
            segments, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )
        else:
            raise e
    print(f"[Stage 2] Detected language: {info.language}")

    result = []
    for seg in segments:
        result.append({
            "text": seg.text.strip(),
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
        })
    print(f"[Stage 2] Transcribed {len(result)} segments")
    return result

def compute_complexity_score(ocr_text: str, content_type: str) -> float:
    score = 0.0
    if content_type == "diagram":
        score += 0.3
    if content_type == "code":
        score += 0.2
    
    math_chars = sum(ocr_text.count(c) for c in ["∂", "∑", "∫", "→", "∇", "∈"])
    eq_count = ocr_text.count("=")
    score += min(math_chars * 0.08, 0.3)
    score += min(eq_count * 0.03, 0.2)
    return round(min(score, 1.0), 3)


def infer_visual_content_type(base_type: str, ocr_text: str, visual_description: str) -> str:
    if base_type == "talking_head":
        return "talking_head"

    text = (ocr_text or "").lower()
    desc = (visual_description or "").lower()
    combined = f"{text} {desc}"

    code_keywords = [
        "def ", "class ", "import ", "return ", "for ", "while ", "function", "console.log", "print(",
        "{", "}", "();", "#include", "public static"
    ]
    if any(k in combined for k in code_keywords):
        return "code"

    diagram_keywords = [
        "diagram", "flowchart", "chart", "graph", "plot", "figure", "arrow", "block", "tree", "network", "architecture"
    ]
    if any(k in combined for k in diagram_keywords):
        return "diagram"

    # If OCR is sparse and there are many symbols/arrows, it's likely a visual diagram.
    symbol_hits = len(re.findall(r"[<>\-_=|/\\]", combined))
    if len(text.strip()) < 60 and symbol_hits > 12:
        return "diagram"

    if len(text.strip()) > 120:
        return "text"

    return "visual_content"
