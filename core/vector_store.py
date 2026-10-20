import os
import uuid
import logging
import torch
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Global embedding model singleton cache
_embedding_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Singleton loader for local multilingual sentence-transformers embeddings.
    Strictly local, no cloud calls. Prefers local cache to avoid external network requests.
    """
    global _embedding_model_cache
    if _embedding_model_cache is None:
        logger.info(f"Loading local multilingual embedding model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            # Try loading strictly from local cache first (offline / zero network)
            _embedding_model_cache = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cpu",
                local_files_only=True,
            )
            # Model loaded from local cache — now enforce offline mode to prevent future network checks
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("Local multilingual embedding model loaded from cache (offline mode).")
        except Exception:
            # Model not yet in local cache — download it once (then subsequent loads will use cache)
            logger.info(f"Embedding model not in local cache. Downloading '{EMBEDDING_MODEL_NAME}' once...")
            _embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
            logger.info("Local multilingual embedding model successfully downloaded and loaded.")
    return _embedding_model_cache


class FaissDocument:
    """Represents a retrieved transcript chunk with metadata."""

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"FaissDocument(text={self.page_content[:40]}..., metadata={self.metadata})"


class FaissVectorStore:
    """
    In-memory, session-scoped FAISS vector store.
    Guarantees zero cross-session pollution and total privacy.
    """

    def __init__(self, dimension: int, embedding_model: SentenceTransformer):
        self.dimension = dimension
        self.embedding_model = embedding_model
        # Inner Product with normalized vectors calculates Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and insert text chunks with associated metadata."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        with torch.inference_mode():
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        embeddings_np = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into session FAISS vector store (total: {self.index.ntotal}).")

    def similarity_search(self, query: str, k: int = 5) -> List[FaissDocument]:
        """
        Perform cross-lingual semantic similarity search using normalized FAISS Inner Product.
        """
        clean_query = query.strip() if query else ""
        if not clean_query or self.index.ntotal == 0:
            return []

        # Embed and normalize query vector (L2 unit length)
        with torch.inference_mode():
            query_embedding = self.embedding_model.encode(
                [clean_query],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        query_np = np.ascontiguousarray(query_embedding, dtype=np.float32)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_np, actual_k)

        results: List[FaissDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_data = self.chunks[idx]
            # Preserve source metadata (especially visual timestamps) when a
            # retrieved vector is converted back into a document.
            meta = {key: value for key, value in chunk_data.items() if key != "text"}
            meta.setdefault("chunk_id", str(idx))
            meta.setdefault("chunk_index", idx)
            meta.setdefault("start_time", 0.0)
            meta.setdefault("end_time", 0.0)
            meta["score"] = float(score)
            results.append(FaissDocument(page_content=chunk_data["text"], metadata=meta))

        # Temporary debugging output per specification
        logger.info(f"[QUERY]\n{clean_query}")
        for rank, doc in enumerate(results[:3]):
            logger.info(
                f"[RETRIEVAL {rank + 1}]\n"
                f"similarity: {doc.metadata.get('score', 0):.4f}\n"
                f"text: {doc.page_content[:150]}..."
            )

        return results

    def as_retriever(self, k: int = 5):
        """Return a callable retriever object compatible with LangChain / LCEL."""
        return FaissRetriever(self, k=k)


class FaissRetriever:
    """Retriever wrapper compatible with LCEL pipeline."""

    def __init__(self, vector_store: FaissVectorStore, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def invoke(self, query: str) -> List[FaissDocument]:
        return self.vector_store.similarity_search(query, k=self.k)

    def get_relevant_documents(self, query: str) -> List[FaissDocument]:
        return self.vector_store.similarity_search(query, k=self.k)



def _split_to_embedding_limit(texts: List[str], embedding_model: SentenceTransformer) -> List[str]:
    """Split character chunks further when they exceed the model token window.

    SentenceTransformer tokenizers truncate to ``max_seq_length``. Preserve the
    existing character chunking as an upper bound, then split only oversized
    chunks using the same tokenizer, reserving room for model special tokens.
    """
    tokenizer = getattr(embedding_model, "tokenizer", None)
    max_seq_length = getattr(embedding_model, "max_seq_length", None)
    if tokenizer is None or not isinstance(max_seq_length, int) or max_seq_length <= 8:
        return texts

    token_budget = max_seq_length - 8
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    if not any(token_count(text) > token_budget for text in texts):
        return texts

    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=token_budget,
        chunk_overlap=min(32, token_budget // 8),
        length_function=token_count,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return [piece for text in texts for piece in token_splitter.split_text(text)]

def build_vector_store(
    transcript: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    duration_seconds: float = 0.0,
) -> FaissVectorStore:
    """
    Split transcript into structured chunks, embed using local multilingual model,
    and build a fresh in-memory FAISS index.
    Uses 800-character chunks with 100 overlap for optimal cross-lingual semantic density.
    Stores both percentage offsets and absolute seconds for temporal multimodal matching.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build vector store: Transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    raw_texts = splitter.split_text(transcript.strip())
    if not raw_texts:
        raise ValueError("Cannot build vector store: No text chunks extracted from transcript.")

    embedding_model = get_embedding_model()
    raw_texts = _split_to_embedding_limit(raw_texts, embedding_model)
    get_dim_fn = getattr(embedding_model, "get_embedding_dimension", getattr(embedding_model, "get_sentence_embedding_dimension", None))
    dimension = get_dim_fn() if get_dim_fn else 384
    store = FaissVectorStore(dimension=dimension, embedding_model=embedding_model)

    chunks: List[Dict[str, Any]] = []
    search_from = 0
    previous_chunk_start = 0
    previous_chunk_end = 0
    total_chars = len(transcript)

    for i, text in enumerate(raw_texts):
        # Chunks overlap, so summing their lengths drifts timestamps past the
        # source duration. Locate each chunk in the original transcript instead.
        source_offset = transcript.find(text, search_from)
        if source_offset < 0:
            approximate_offset = max(0, previous_chunk_end - chunk_overlap)
            source_offset = transcript.find(text, approximate_offset)
        if source_offset < 0:
            source_offset = min(previous_chunk_end, total_chars)
        source_offset = min(max(source_offset, previous_chunk_start), total_chars)
        search_from = source_offset + 1
        previous_chunk_start = source_offset
        previous_chunk_end = source_offset + len(text)

        start_ratio = source_offset / max(1, total_chars)
        end_ratio = min(source_offset + len(text), total_chars) / max(1, total_chars)
        start_sec = round(start_ratio * duration_seconds, 1) if duration_seconds > 0 else 0.0
        end_sec = round(end_ratio * duration_seconds, 1) if duration_seconds > 0 else 0.0

        chunk_id = f"chunk_{i}_{uuid.uuid4().hex[:6]}"
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_index": i,
            "text": text,
            "start_time": round(start_ratio * 100, 1),
            "end_time": round(end_ratio * 100, 1),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
        })
    store.add_chunks(chunks)
    return store


def get_retriever(vector_store: FaissVectorStore, k: int = 5) -> FaissRetriever:
    """Obtain similarity retriever for cross-lingual FAISS search."""
    return vector_store.as_retriever(k=k)


def build_visual_vector_store(
    visual_evidence: List[Dict[str, Any]], video_id: Optional[str] = None,
) -> Optional[FaissVectorStore]:
    """
    Build a session-scoped in-memory FAISS vector store over timestamped visual evidence frames.
    Returns FaissVectorStore or None if visual_evidence is empty.
    """
    if not visual_evidence:
        return None

    embedding_model = get_embedding_model()
    get_dim_fn = getattr(embedding_model, "get_embedding_dimension", getattr(embedding_model, "get_sentence_embedding_dimension", None))
    dimension = get_dim_fn() if get_dim_fn else 384
    store = FaissVectorStore(dimension=dimension, embedding_model=embedding_model)

    chunks: List[Dict[str, Any]] = []
    for i, item in enumerate(visual_evidence):
        summary_text = (item.get("summary_text") or item.get("description") or "").strip()
        if not summary_text:
            continue
        chunk_id = f"visual_{i}_{uuid.uuid4().hex[:6]}"
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_index": i,
            "source": "visual",
            "video_id": video_id if video_id is not None else item.get("video_id"),
            "frame_index": item.get("frame_index", i),
            "source_path": item.get("source_path"),
            "source_file": item.get("source_file"),
            "text": summary_text,
            "timestamp": item.get("timestamp", "00:00:00"),
            "timestamp_display": item.get("timestamp_display", item.get("timestamp", "00:00:00")),
            "timestamp_seconds": item.get("timestamp_seconds", 0.0),
            "end_seconds": item.get("end_seconds", item.get("timestamp_seconds", 0.0)),
            "start_time": item.get("timestamp_seconds", 0.0),
            "end_time": item.get("end_seconds", item.get("timestamp_seconds", 0.0)),
            "people_count": item.get("people_count", 0),
            "scene": item.get("scene", ""),
            "screen_ui": item.get("screen_ui", ""),
            "slide_title": item.get("slide_title", ""),
            "charts": item.get("charts", ""),
            "diagrams": item.get("diagrams", ""),
            "visible_text": item.get("visible_text", ""),
            "actions": item.get("actions", ""),
            "objects": item.get("objects", ""),
            "important_details": item.get("important_details", ""),
        })

    if not chunks:
        return None

    store.add_chunks(chunks)
    logger.info(f"Indexed {len(chunks)} visual evidence frames into FAISS visual store.")
    return store


