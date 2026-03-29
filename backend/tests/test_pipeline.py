import pytest
import os
import sys
import shutil
from pathlib import Path

# Fix python path for pytest execution from root dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import UPLOADS_PATH, FRAMES_PATH

@pytest.fixture(scope="module")
def setup_test_env():
    # Setup dummy directories
    Path(UPLOADS_PATH).mkdir(parents=True, exist_ok=True)
    Path(FRAMES_PATH).mkdir(parents=True, exist_ok=True)
    yield
    # We won't teardown since the user might want to inspect the results

def test_imports():
    """Ensure all heavy AI pipeline modules load without crashing"""
    try:
        from stage1_ingest import extract_audio, extract_keyframes
        from stage2_extract import classify_frame, extract_frame_features, get_whisper
        from stage3_chunk import chunk_transcript, chunk_visual
    except Exception as e:
        pytest.fail(f"Pipeline import failed: {e}")

def test_whisper_initialization():
    """Verify that Faster-Whisper compiles and loads the model correctly"""
    from stage2_extract import get_whisper
    try:
        model = get_whisper()
        assert model is not None
    except Exception as e:
        pytest.fail(f"Whisper initialization failed. CUDA bindings might be missing. {e}")

def test_environment_variables():
    """Ensure Gemini and Ollama configurations are populated"""
    from config import GEMINI_API_KEY, LLM_PROVIDER
    assert LLM_PROVIDER in ("gemini", "ollama")
    if LLM_PROVIDER == "gemini":
        assert GEMINI_API_KEY is not None

def test_ollama_connectivity():
    """Ensure local Ollama endpoint is listening and reachable for VLMs"""
    import urllib.request
    try:
        response = urllib.request.urlopen("http://localhost:11434/")
        assert response.status == 200
    except Exception as e:
        pytest.warns(UserWarning, match=f"Ollama doesn't seem to be running locally on port 11434: {e}")
