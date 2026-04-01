import json
import time
import networkx as nx
from pathlib import Path
import numpy as np
import chromadb
import spacy
from ollama import Client
from config import (
    CHROMA_PATH,
    GRAPHS_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_BATCH_SIZE,
    OLLAMA_EMBED_MODEL,
    OLLAMA_REQUEST_RETRIES,
    OLLAMA_RETRY_BACKOFF_SEC,
    OLLAMA_TIMEOUT_SEC,
)

_embedder = None


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = Client(host=self.base_url)

    def _embed_with_retry(self, texts: list[str]):
        attempts = max(1, int(OLLAMA_REQUEST_RETRIES))
        backoff = max(0.0, float(OLLAMA_RETRY_BACKOFF_SEC))
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                return self._client.embed(model=self.model, input=texts)
            except Exception as e:
                last_error = e
                if attempt < attempts:
                    sleep_sec = backoff * attempt
                    print(f"[Stage 4] Ollama request retry {attempt}/{attempts} after error: {e}")
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)

        raise RuntimeError(f"Ollama embed failed after {attempts} attempts: {last_error}")

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        data = self._embed_with_retry(texts)
        if "embeddings" in data and data["embeddings"]:
            return [np.asarray(v, dtype=float).tolist() for v in data["embeddings"]]
        if "embedding" in data:
            return [np.asarray(data["embedding"], dtype=float).tolist()]

        raise RuntimeError("Ollama embed response missing embedding vector")

    def encode(self, text: str):
        vecs = self.encode_many([text])
        if not vecs:
            raise RuntimeError("Ollama embed response missing embedding vector")
        return np.asarray(vecs[0], dtype=float)


def ensure_embedding_backend_ready():
    """Fail fast if Ollama embedding backend/model is unavailable."""
    client = Client(host=OLLAMA_BASE_URL)
    try:
        tags = client.list()
    except Exception as e:
        raise RuntimeError(f"Ollama is not reachable at {OLLAMA_BASE_URL}: {e}")

    models = [m.get("name") or m.get("model") or "" for m in (tags.get("models", []) or [])]
    configured = (OLLAMA_EMBED_MODEL or "").strip().lower()
    found = any(
        (name.lower() == configured) or (
            ":" not in configured and name.lower().startswith(configured + ":")
        )
        for name in models
    )
    if not found:
        raise RuntimeError(
            f"Embedding model '{OLLAMA_EMBED_MODEL}' is not installed in Ollama. "
            f"Installed: {models}"
        )

    # Tiny embed call verifies endpoint + model are actually usable.
    get_embedder().encode_many(["embedding backend health-check"])


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBED_MODEL,
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        print(f"[Stage 4] Using Ollama embedding model: {OLLAMA_EMBED_MODEL}")
    return _embedder

_chroma_client = None
def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

nlp = spacy.load("en_core_web_sm")


def _upsert_in_batches(col, ids, embeddings, metadatas, documents):
    if not ids:
        return

    # Chroma has an internal max batch size limit; chunk large ingests to avoid 500s.
    max_batch = 5000
    try:
        client = getattr(col, "_client", None)
        if client is not None:
            if hasattr(client, "get_max_batch_size"):
                max_batch = int(client.get_max_batch_size())
            elif hasattr(client, "max_batch_size"):
                max_batch = int(client.max_batch_size)
    except Exception:
        max_batch = 5000

    max_batch = max(1, max_batch)
    total_batches = (len(ids) + max_batch - 1) // max_batch
    print(f"[Stage 4] Upserting {len(ids)} vectors in {total_batches} batch(es) (max_batch={max_batch})")
    for i in range(0, len(ids), max_batch):
        j = i + max_batch
        col.upsert(
            ids=ids[i:j],
            embeddings=embeddings[i:j],
            metadatas=metadatas[i:j],
            documents=documents[i:j],
        )


def _is_dimension_mismatch_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("dimension" in msg) and ("expecting embedding" in msg or "got" in msg)


def _recreate_collection(name: str):
    client = get_chroma()
    try:
        client.delete_collection(name=name)
        print(f"[Stage 4] Recreated collection '{name}' due to embedding dimension mismatch")
    except Exception:
        # If collection does not exist or delete fails silently, continue to create.
        pass

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _safe_upsert_with_recreate(collection_name: str, col, ids, embeddings, metadatas, documents):
    try:
        _upsert_in_batches(col, ids, embeddings, metadatas, documents)
        return col
    except Exception as e:
        if not _is_dimension_mismatch_error(e):
            raise

        print(f"[Stage 4] Dimension mismatch detected for '{collection_name}': {e}")
        col = _recreate_collection(collection_name)
        _upsert_in_batches(col, ids, embeddings, metadatas, documents)
        return col

def get_or_create_collections():
    client = get_chroma()
    transcript_col = client.get_or_create_collection(
        name="transcript_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    visual_col = client.get_or_create_collection(
        name="visual_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return transcript_col, visual_col


def index_transcript_chunks(chunks: list[dict]):
    col, _ = get_or_create_collections()
    embedder = get_embedder()

    ids, embeddings, metadatas, documents = [], [], [], []
    for c in chunks:
        ids.append(c["chunk_id"])
        documents.append(c["text"])
        metadatas.append({
            "chunk_type": c["chunk_type"],
            "course_id": c["course_id"],
            "lecture_id": c["lecture_id"],
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "rewatch_count": c["rewatch_count"],
            "query_hit_count": c["query_hit_count"],
        })

    batch_size = max(1, OLLAMA_EMBED_BATCH_SIZE)
    total = len(documents)
    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        embeddings.extend(embedder.encode_many(documents[i:j]))
        print(f"[Stage 4][Transcript] Embedded {j}/{total} chunks")

    _safe_upsert_with_recreate("transcript_chunks", col, ids, embeddings, metadatas, documents)
    print(f"[Stage 4] Indexed {len(chunks)} transcript chunks")


def index_visual_chunks(chunks: list[dict]):
    _, col = get_or_create_collections()
    embedder = get_embedder()

    ids, embeddings, metadatas, documents = [], [], [], []
    for c in chunks:
        text = f"{c.get('visual_description', '')} {c.get('ocr_text', '')}".strip()
        if not text:
            text = "visual content"
        ids.append(c["chunk_id"])
        documents.append(text)
        metadatas.append({
            "chunk_type": c["chunk_type"],
            "course_id": c["course_id"],
            "lecture_id": c["lecture_id"],
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "content_type": c.get("content_type", "unknown"),
            "complexity_score": c.get("complexity_score", 0.0),
            "rewatch_count": c["rewatch_count"],
            "query_hit_count": c["query_hit_count"],
        })

    batch_size = max(1, OLLAMA_EMBED_BATCH_SIZE)
    total = len(documents)
    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        embeddings.extend(embedder.encode_many(documents[i:j]))
        if j == total or j % max(batch_size * 5, 250) == 0:
            print(f"[Stage 4][Visual] Embedded {j}/{total} chunks")

    _safe_upsert_with_recreate("visual_chunks", col, ids, embeddings, metadatas, documents)
    print(f"[Stage 4] Indexed {len(chunks)} visual chunks")


def load_graph(course_id: str) -> nx.DiGraph:
    path = Path(GRAPHS_PATH) / f"{course_id}.graphml"
    if path.exists():
        if path.stat().st_size > 0:
            try:
                return nx.read_graphml(str(path))
            except Exception as e:
                print(f"[Stage 4] Warning: Existing graphml file is corrupted, creating a new graph. Error: {e}")
        else:
            print("[Stage 4] Found empty graphml file, starting fresh.")
            
    return nx.DiGraph()

def save_graph(G: nx.DiGraph, course_id: str):
    Path(GRAPHS_PATH).mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(Path(GRAPHS_PATH) / f"{course_id}.graphml"))

def extract_and_store_concepts(chunks: list[dict], course_id: str):
    G = load_graph(course_id)

    for chunk in chunks:
        if chunk["chunk_type"] != "fine":
            continue
        doc = nlp(chunk["text"])
        concepts = [nc.text.lower().strip() for nc in doc.noun_chunks if len(nc.text.strip()) > 3]

        for concept in concepts:
            if concept not in G:
                G.add_node(concept, course_id=course_id, appearances_json="[]")
            existing = json.loads(G.nodes[concept].get("appearances_json", "[]"))
            existing.append(chunk["t_start"])
            G.nodes[concept]["appearances_json"] = json.dumps(existing)

        for i in range(len(concepts) - 1):
            a, b = concepts[i], concepts[i + 1]
            if G.has_edge(a, b):
                G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
            else:
                G.add_edge(a, b, weight=1)

    save_graph(G, course_id)
    print(f"[Stage 4] Knowledge graph: {G.number_of_nodes()} concepts, {G.number_of_edges()} edges")
