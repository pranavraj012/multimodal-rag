import json
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import spacy
from config import CHROMA_PATH, GRAPHS_PATH, EMBEDDING_MODEL

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

_chroma_client = None
def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

nlp = spacy.load("en_core_web_sm")

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
        vec = embedder.encode(c["text"]).tolist()
        ids.append(c["chunk_id"])
        embeddings.append(vec)
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

    if ids:
        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"[Stage 4] Indexed {len(chunks)} transcript chunks")


def index_visual_chunks(chunks: list[dict]):
    _, col = get_or_create_collections()
    embedder = get_embedder()

    ids, embeddings, metadatas, documents = [], [], [], []
    for c in chunks:
        text = f"{c.get('visual_description', '')} {c.get('ocr_text', '')}".strip()
        if not text:
            text = "visual content"
        vec = embedder.encode(text).tolist()
        ids.append(c["chunk_id"])
        embeddings.append(vec)
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

    if ids:
        col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
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
