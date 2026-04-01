import os
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

# Set this to "ollama" for local-first, "gemini" for API-backed inference
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "gemini" | "ollama"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Using the fastest, most cost-efficient multimodal model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_EMBED_BATCH_SIZE = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "48"))
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "45"))
OLLAMA_REQUEST_RETRIES = int(os.getenv("OLLAMA_REQUEST_RETRIES", "3"))
OLLAMA_RETRY_BACKOFF_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_SEC", "1.0"))
VISUAL_LLM_BUDGET_MODE = os.getenv("VISUAL_LLM_BUDGET_MODE", "adaptive").lower()  # adaptive | fixed | off
VISUAL_LLM_MAX_FRAMES = int(os.getenv("VISUAL_LLM_MAX_FRAMES", "30"))
VISUAL_LLM_MIN_FRAMES = int(os.getenv("VISUAL_LLM_MIN_FRAMES", "8"))
VISUAL_LLM_FRAME_RATIO = float(os.getenv("VISUAL_LLM_FRAME_RATIO", "0.2"))
VISUAL_LLM_MAX_PER_MIN = int(os.getenv("VISUAL_LLM_MAX_PER_MIN", "12"))
OCR_EVERY_N_VISUAL_FRAMES = int(os.getenv("OCR_EVERY_N_VISUAL_FRAMES", "1"))

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")
GRAPHS_PATH = str(PROJECT_ROOT / "data" / "graphs")
FRAMES_PATH = str(PROJECT_ROOT / "data" / "frames")
UPLOADS_PATH = str(PROJECT_ROOT / "uploads")

SSIM_THRESHOLD = 0.85      # Lower = more keyframes. Tune between 0.75-0.92
KEYFRAME_SAMPLE_EVERY_N_FRAMES = int(os.getenv("KEYFRAME_SAMPLE_EVERY_N_FRAMES", "2"))
KEYFRAME_MIN_SCENE_GAP_SEC = float(os.getenv("KEYFRAME_MIN_SCENE_GAP_SEC", "0.7"))

# LLM caller — single function, works for both providers
def call_llm(prompt: str, image_path: str = None) -> str:
    provider = LLM_PROVIDER
    if provider == "gemini" and not GEMINI_API_KEY:
        print("[Config] GEMINI_API_KEY is missing. Falling back to Ollama.")
        provider = "ollama"

    if provider == "gemini":
        return _call_gemini(prompt, image_path)
    if provider == "ollama":
        return _call_ollama(prompt, image_path)

    print(f"[Config] Unknown LLM_PROVIDER='{LLM_PROVIDER}'. Falling back to Ollama.")
    return _call_ollama(prompt, image_path)

def _call_gemini(prompt: str, image_path: str = None) -> str:
    from google import genai
    from PIL import Image
    
    # Initialize the client. It automatically picks up GEMINI_API_KEY from environment
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    if image_path:
        img = Image.open(image_path)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, img]
        )
    else:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
    return response.text

def _call_ollama(prompt: str, image_path: str = None) -> str:
    import time

    client = Client(host=OLLAMA_BASE_URL)
    attempts = max(1, int(OLLAMA_REQUEST_RETRIES))
    backoff = max(0.0, float(OLLAMA_RETRY_BACKOFF_SEC))
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            if image_path:
                res = client.generate(
                    model=OLLAMA_VISION_MODEL,
                    prompt=prompt,
                    images=[image_path],
                )
            else:
                res = client.generate(
                    model=OLLAMA_CHAT_MODEL,
                    prompt=prompt,
                )
            return (res.get("response") or "").strip()
        except Exception as e:
            last_error = e
            if attempt < attempts:
                sleep_sec = backoff * attempt
                print(f"[Config] Ollama retry {attempt}/{attempts} after error: {e}")
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

    return f"Error from Ollama: {last_error}"
