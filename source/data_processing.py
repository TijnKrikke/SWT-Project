"""
Data processing module for document corpus management.
Handles story loading, chunking, tokenization, and QA dataset loading.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from config import PROJECT_ROOT, STOPWORDS


@dataclass
class Chunk:
    """One complete story and its identifying metadata."""

    text: str
    story_id: str
    title: str
    chunk_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_story_chunks(path: Path) -> List[Chunk]:
    """Load the subset, keeping each entire story as one chunk."""
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for chunk_id, line in enumerate(file):
            story = json.loads(line)
            metadata = {
                k: v for k, v in story.items()
                if k not in {"text", "story_id", "title"}
            }
            chunks.append(Chunk(
                text=story["text"],
                story_id=story["story_id"],
                title=story["title"],
                chunk_id=chunk_id,
                metadata=metadata,
            ))
    return chunks


def load_qa_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load the dataset with questions from a JSON file.

    Args:
        path (Path): The location where the dataset is stored.

    Raises:
        ValueError: If an item in the dataset does not contain a question or an answer.

    Returns:
        List[Dict[str, Any]]: I list of dictionaries containing the question and answer of every
            entry.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError("Each QA item must be a dict with 'question' and 'answer'.")
    return data


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25, removing stopwords.

    Args:
        text (str): The text that needs to be tokenized for BM25.

    Returns:
        List[str]: List of tokenized words with removed stopwords.
    """
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS]


class VectorStore:
    """ChromaDB storage for chunk texts, embeddings, and metadata."""

    def __init__(self, persist_dir: Path, collection_name: str):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "ip"}} # needed for cosine distance
        )

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], k: int = 5, where: dict | None = None) -> dict:
        return self.collection.query(query_embeddings=[embedding], n_results=k, where=where)

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "ip"}},
        )

    def count(self) -> int:
        return self.collection.count()


def buildVectorDB(
    chunks: List[Chunk],
    embedder,
    persist_dir: Path = PROJECT_ROOT / "chroma_db",
    collection_name: str = "stories",
) -> VectorStore | None:
    """Embed the current chunks and replace the collection with this corpus."""
    if not chunks:
        print("No chunks to index.")
        return

    store = VectorStore(persist_dir, collection_name)
    if store.count() == len(chunks):
        print(f"Collection '{collection_name}' already indexed with {store.count()} chunks. Skipping embedding.")
        return store

    texts = [chunk.text for chunk in chunks]
    vectors = embedder.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=True,normalize_embeddings=True).tolist()

    store.reset()

    chroma_metadatas = []
    for chunk in chunks:
        meta = {
            "story_id": chunk.story_id,
            "title": chunk.title,
            "chunk_id": chunk.chunk_id,
        }
        for k, v in chunk.metadata.items():
            if isinstance(v, list):
                meta[k] = ", ".join(str(x) for x in v)
            elif isinstance(v, (str, int, float, bool)):
                meta[k] = v
        chroma_metadatas.append(meta)

    store.add(
        ids=[str(i) for i in range(len(chunks))],
        texts=texts,
        embeddings=vectors,
        metadatas=chroma_metadatas,
    )
    print(f"Indexed {store.count()} chunks into '{collection_name}'.")
    return store
