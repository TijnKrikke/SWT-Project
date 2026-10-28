"""
Main entry point for the RAG QA system.

Loads all models, initializes the retriever, and orchestrates either evaluation or interactive mode.
"""

import torch
from config import (
    EMBEDDING_MODEL,
    RERANK_STAGE1_MODEL,
    RERANK_STAGE2_MODEL,
    STORIES_PATH,
    USE_RDF_AGENT,
    USE_STAGE2_RERANK,
)
from data_processing import load_story_chunks
from evaluation import run_evaluation
from retrieval import Retriever
from sentence_transformers import CrossEncoder, SentenceTransformer


def main():
    """Initialize system and run evaluation or interactive mode."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if USE_RDF_AGENT:
        from RDF_Agent import create_rdf_graph, create_tools, read_json_files

        graph = create_rdf_graph(read_json_files(str(STORIES_PATH)))
        run_evaluation(None, device, csv_out="rdf_results.csv", rdf_graph=graph, rdf_tools=create_tools())
        return

    chunks = load_story_chunks(STORIES_PATH)
    print(f"Loaded {len(chunks)} stories as {len(chunks)} chunks.")

    print("Loading models")
    print(f"Device: {device}")

    # Load embedding and reranking models
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    rer1 = CrossEncoder(RERANK_STAGE1_MODEL)
    rer2 = CrossEncoder(RERANK_STAGE2_MODEL) if USE_STAGE2_RERANK else None

    # Initialize retriever
    print("Initializing BM25 and ChromaDB")
    retriever = Retriever(
        chunks=chunks,
        embedder=embedder,
        reranker_stage1=rer1,
        reranker_stage2=rer2,
    )

    print("\nRunning evaluation mode")
    run_evaluation(retriever, device)


if __name__ == "__main__":
    main()
