"""
Data processing module for document corpus management.
Handles PDF loading, text cleaning, chunking, tokenization, and QA dataset loading.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import fitz
import numpy as np
from config import STOPWORDS


@dataclass
class Chunk:
    """A dataclass to store information about a single chunk."""

    text: str
    page: int
    chunk_id: int


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


def _clean_text(string: str) -> str:
    """Clean text by removing soft hyphens and normalizing whitespace.

    Args:
        string (str): The text that needs to be cleaned.

    Returns:
        str: The cleaned text.
    """
    string = (string or "").replace("\u00ad", "")
    string = re.sub(r"\s+", " ", string).strip()
    return string


def pdf_to_chunks(pdf_path: Path, chunk_words: int, overlap_words: int) -> List[Chunk]:
    """Split a PDF into overlapping text chunks based on word count.

    Args:
        pdf_path (Path): The location where the pdf is located.
        chunk_words (int): The number of tokens in each chunk.
        overlap_words (int): The number of over lap tokens.

    Raises:
        FileNotFoundError: If the pdf is not located at the specified path.

    Returns:
        List[Chunk]: A list of chunks containing all the text of the pdf.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    chunks = []
    chunk_id = 0

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        text = page.get_text("text") or ""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        current_words = []

        for para in paragraphs:
            para = _clean_text(para)
            if not para:
                continue

            words = para.split()
            if not words:
                continue

            # If adding this paragraph would exceed the chunk size,
            # save the current chunk and start a new one
            if len(current_words) + len(words) > chunk_words:
                if current_words:
                    chunks.append(Chunk(text=" ".join(current_words), page=page_index + 1, chunk_id=chunk_id))
                    chunk_id += 1

                # keep overlap words for the next chunk
                if overlap_words > 0:
                    current_words = current_words[-overlap_words:]
                else:
                    current_words = []

            current_words.extend(words)

        # add the last chunk of the page
        if current_words:
            chunks.append(Chunk(text=" ".join(current_words), page=page_index + 1, chunk_id=chunk_id))
            chunk_id += 1

    doc.close()
    return chunks


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25, removing stopwords.

    Args:
        text (str): The text that needs to be tokenized for BM25.

    Returns:
        List[str]: List of tokenized words with removed stopwords.
    """
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS]


class DenseIndex:
    """FAISS-based dense embedding index using inner product similarity."""

    def __init__(self, embeddings: np.ndarray) -> None:
        """
        Initialize dense index with embeddings.

        Args:
            embeddings: Numpy array of shape (n_chunks, embedding_dim) containing the embeddings
                of all tokens.
        """
        emb = embeddings.astype(np.float32)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def search(self, q_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of a query embedding.

        Args:
            q_emb (np.ndarray): The embedding of the query.
            k (int): The number of neighbors to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: (indices, scores) where indices are chunk
                indices and scores are similarity scores.
        """
        q = q_emb.astype(np.float32).reshape(1, -1)
        scores, idx = self.index.search(q, k)
        return idx[0], scores[0]
