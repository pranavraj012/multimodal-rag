# Multimodal Video RAG System Implementation Plan

This plan analyzes the AI-generated `video_rag_guide_v2.md` against the original `mini proj approach.pdf` architecture, validates the technical choices, and proposes the final approach to proceed with implementation.

## Validation of the AI-Generated Code

The AI-generated code provides an excellent, pragmatic foundation for an MVP of the architecture outlined in the PDF. It successfully implements the 8-stage pipeline while making sensible simplifications for local development (e.g., WSL2, no Docker).

### Strengths & Accurate Architectural Alignments
1. **Cost-Reduction Routing (Stage 2):** Correctly implements the heuristic pre-classifier (Haar cascades, image thresholding) to avoid expensive Vision LLM calls on talking heads and plain text slides.
2. **Coarse-to-Fine Chunking (Stage 3):** Successfully implements the two-pass chunking strategy, mapping fine chunks to larger parent segments.
3. **Late Fusion (Stage 6):** The temporal clipping logic accurately merges visual and transcript hits that occur within the same time window into single high-confidence clips.
4. **Complexity Scoring:** Accurately implements a heuristic complexity score (counting math symbols, equations, and diagram presence) for the analytics dashboard.

### Simplifications & Deviations (For Consideration)
The AI guide makes a few technological substitutions to achieve a "One-Day, No Docker" implementation:

1. **Knowledge Graph: NetworkX vs. Neo4j**
   - **PDF:** Recommends Neo4j for hybrid graph+vector enrichment.
   - **Code:** Uses `NetworkX` (in-memory, persisted as `.graphml`).
   - *Impact:* Perfectly fine for a minor project and single courses, much easier to set up than a dedicated Neo4j JVM process.
2. **Vector DB: ChromaDB vs. Qdrant / Weaviate**
   - **PDF:** Suggests Qdrant/Weaviate (HNSW, dense/sparse).
   - **Code:** Uses `ChromaDB` (local embedded, pure Python).
   - *Impact:* Great for local development. We lose hybrid search (dense + BM25 sparse) out of the box, but dense embeddings alone are passable for V1.
3. **Intent Classification (Stage 5)**
   - **PDF:** Predicts a lightweight LLM for intent classification into 6 categories.
   - **Code:** Uses brittle keyword matching (e.g., `if "show me" in query:`).
   - *Impact:* This keyword matching might fail on nuanced questions. We should probably upgrade this to use an LLM function call or structured output for better accuracy.
4. **Keyword Extraction for Graph (Stage 4)**
   - **Code:** Uses `spaCy` noun chunks to identify concepts for the graph.
   - *Impact:* Quick and dirty, but might result in noisy concepts.

## Proposed Approach & Tools

I propose we stick to the AI-generated guide's technology stack for rapid development, but make a few crucial upgrades to ensure it meets the standard of the original design.

### Tech Stack Choices
- **Frontend:** React + Vite (Standard, fast)
- **Backend API:** FastAPI + Python (Excellent for ML pipelines)
- **Vector Database:** **ChromaDB** (Keep this choice for simplicity, we can upgrade to Qdrant if search isn't accurate enough)
- **Knowledge Graph:** **NetworkX** (Keep this to avoid Neo4j setup overhead)
- **Vision/LLM Provider:** **Gemini API** (Using Gemini 1.5 Flash is currently the fastest, cheapest, and most capable option for multimodal parsing over Ollama)
- **Audio:** `faster-whisper` (Local)
- **OCR:** `pytesseract` (Local)

> [!IMPORTANT]
> **User Review Required**
> 1. Are you comfortable using **ChromaDB and NetworkX** over Qdrant and Neo4j for this minor project to keep dependencies light and avoid Docker?
> 2. Do you have a Gemini API key available? The guide utilizes `google-generativeai`. (We can also use Groq or OpenAI if you prefer).
> 3. Does this tech stack run smoothly in your environment (WSL2 as mentioned in the guide, or native Windows)? We can install everything locally.

## Execution Steps

Once approved, we will execute the following steps:

1. **Bootstrap Project Workspace**
   - Create the directory structure (`backend`, `frontend`, `data`) and initialized Python environment.
2. **Implement Backend Core (Stages 1-4)**
   - Implement ingestion, Whisper audio extraction, OpenCV frame extraction, Tesseract OCR, and ChromaDB indexing.
3. **Implement Retrieval & AI (Stages 5-8)**
   - Wire up the Gemini API for Vision, Intent Classification (upgraded from keyword matching to LLM), and response generation.
4. **Frontend Implementation**
   - Build the React interface with video player syncing and chat elements.
5. **End-to-End Testing**
   - We will test with a sample video as detailed in the guide.
