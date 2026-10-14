"""
Main entry point for the RAG QA system.

Loads all models, initializes the retriever, and orchestrates either evaluation or interactive mode.
"""

import subprocess
import sys

import torch
from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    PDF_PATH,
    RERANK_STAGE1_MODEL,
    RERANK_STAGE2_MODEL,
    RUN_EVALUATION,
    USE_STAGE2_RERANK,
)
from data_processing import pdf_to_chunks
from evaluation import run_evaluation
from retrieval import Retriever
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_interactive() -> None:
    """Run interactive mode where user asks questions one at a time."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "source/app.py"])


def main():
    """Initialize system and run evaluation or interactive mode."""
    if not RUN_EVALUATION:
        run_interactive()
        return

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print("Loading models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load embedding and reranking models
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    rer1 = CrossEncoder(RERANK_STAGE1_MODEL)
    rer2 = CrossEncoder(RERANK_STAGE2_MODEL) if USE_STAGE2_RERANK else None

    # Load and chunk PDF
    print("Loading and processing PDF")
    chunks = pdf_to_chunks(PDF_PATH, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
    print(f"Loaded {len(chunks)} chunks from PDF")

    # Initialize retriever
    print("Initializing BM25 and faiss")
    retriever = Retriever(
        chunks=chunks,
        embedder=embedder,
        reranker_stage1=rer1,
        reranker_stage2=rer2,
    )

    # Load tokenizer and generation model
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL).to(device)
    gen_model.eval()

    print("\nRunning evaluation mode")
    run_evaluation(retriever, tokenizer, gen_model, device)


if __name__ == "__main__":
    main()
