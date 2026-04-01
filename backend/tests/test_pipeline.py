import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image

# Fix python path for pytest execution from root dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    FRAMES_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_VISION_MODEL,
    UPLOADS_PATH,
)


@pytest.fixture(scope="session")
def test_env_ready():
    Path(UPLOADS_PATH).mkdir(parents=True, exist_ok=True)
    Path(FRAMES_PATH).mkdir(parents=True, exist_ok=True)
    return True


@pytest.fixture(scope="session")
def ollama_session(test_env_ready):
    session = requests.Session()
    try:
        r = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    except Exception as e:
        pytest.fail(f"Ollama is not reachable at {OLLAMA_BASE_URL}: {e}")

    if r.status_code != 200:
        pytest.fail(f"Ollama /api/tags failed: HTTP {r.status_code} -> {r.text}")

    return session


def _normalize_model_name(name: str) -> str:
    return (name or "").strip().lower()


def _model_matches_config(installed_name: str, configured_name: str) -> bool:
    installed = _normalize_model_name(installed_name)
    configured = _normalize_model_name(configured_name)
    if installed == configured:
        return True

    # Allow "model" config to match installed "model:tag" variants.
    if ":" not in configured:
        return installed.startswith(f"{configured}:")

    return False


def _installed_model_names(session: requests.Session) -> list[str]:
    r = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    data = r.json()
    models = data.get("models", [])
    return [m.get("name", "") for m in models if m.get("name")]


def test_imports_core_modules(test_env_ready):
    """Ensure core backend modules import successfully in the current environment."""
    try:
        from stage1_ingest import extract_audio, extract_keyframes  # noqa: F401
        from stage2_extract import classify_frame, extract_frame_features, get_whisper  # noqa: F401
        from stage3_chunk import chunk_transcript, chunk_visual  # noqa: F401
        from stage4_index import get_embedder  # noqa: F401
        from stage6_retrieve import retrieve  # noqa: F401
    except Exception as e:
        pytest.fail(f"Backend import check failed: {e}")


def test_ollama_models_installed(ollama_session):
    """Verify configured chat/vision/embedding models are installed in Ollama."""
    installed = _installed_model_names(ollama_session)
    missing = []

    checks = {
        "OLLAMA_CHAT_MODEL": OLLAMA_CHAT_MODEL,
        "OLLAMA_VISION_MODEL": OLLAMA_VISION_MODEL,
        "OLLAMA_EMBED_MODEL": OLLAMA_EMBED_MODEL,
    }

    for key, configured in checks.items():
        if not any(_model_matches_config(name, configured) for name in installed):
            missing.append(f"{key}={configured}")

    if missing:
        pytest.fail(
            "Configured Ollama model(s) missing. Pull them first with 'ollama pull <model>'. "
            f"Missing: {', '.join(missing)}. Installed: {installed}"
        )


def test_ollama_chat_model_generate(ollama_session):
    """Smoke test chat generation endpoint with configured chat model."""
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "prompt": "Reply with exactly: OK",
        "stream": False,
    }
    r = ollama_session.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=max(15, int(OLLAMA_TIMEOUT_SEC)),
    )
    assert r.status_code == 200, f"Chat generate failed: {r.status_code} {r.text}"
    data = r.json()
    assert isinstance(data.get("response", ""), str) and data.get("response", "").strip(), "Empty chat response"


def test_ollama_vision_model_generate(ollama_session):
    """Smoke test vision generation endpoint with a tiny synthetic image."""
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": "Say one short sentence describing this image.",
        "images": [img_b64],
        "stream": False,
    }
    r = ollama_session.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=max(20, int(OLLAMA_TIMEOUT_SEC) + 10),
    )
    assert r.status_code == 200, f"Vision generate failed: {r.status_code} {r.text}"
    data = r.json()
    assert isinstance(data.get("response", ""), str) and data.get("response", "").strip(), "Empty vision response"


def test_ollama_embedding_model_embed(ollama_session):
    """Smoke test embedding endpoint with configured embedding model."""
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": "quick embedding health-check",
    }
    r = ollama_session.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json=payload,
        timeout=max(20, int(OLLAMA_TIMEOUT_SEC)),
    )

    if r.status_code == 404:
        # Compatibility fallback for older Ollama endpoint.
        r = ollama_session.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": "quick embedding health-check"},
            timeout=max(20, int(OLLAMA_TIMEOUT_SEC)),
        )

    assert r.status_code == 200, f"Embedding call failed: {r.status_code} {r.text}"
    data = r.json()

    if "embeddings" in data and data["embeddings"]:
        vec = data["embeddings"][0]
    else:
        vec = data.get("embedding", [])

    assert isinstance(vec, list) and len(vec) > 0, "Embedding vector missing or empty"


def test_whisper_model_initialization(test_env_ready):
    """Ensure faster-whisper model can initialize in this environment (GPU or CPU fallback)."""
    from stage2_extract import get_whisper

    model = get_whisper()
    assert model is not None
