"""
Configuration and hyperparameters for the RAG QA system.
Keeps all constants, model names, paths, and flags in one place for easy adjustment and reproducibility.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_PATH = PROJECT_ROOT / "data" / "dutch_subset.jsonl"
QA_DATASET_PATH = PROJECT_ROOT / "resources" / "test.json"

# Dense Retrieval Models
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Reranking Models
RERANK_STAGE1_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANK_STAGE2_MODEL = "cross-encoder/ms-marco-electra-base"
USE_STAGE2_RERANK = True  # accuracy-first (slower)

# Generation Model
GENERATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Base Retrieval Sizes
BM25_CANDIDATES = 30
DENSE_CANDIDATES = 30
RERANK_KEEP_STAGE1 = 30
FINAL_CONTEXT_CHUNKS = 5

# enhancement, standard true
USE_HYBRID = True
USE_RERANK = True
REFUSAL_THRESHOLD = 0.4

# Generation Settings
BEAM_SIZE = 1
MAX_NEW_TOKENS = 64

# Evaluation Metrics
ANSWER_CORRECTNESS_BERTSCORE_THRESHOLD = 0.6
RETRIEVAL_METRIC_KS = (1, 3, 5, 10)
MEASURE_LATENCY = True

# Run Mode
RUN_EVALUATION = True  # True: evaluate on QA_DATASET_PATH; False: interactive
BASELINE = False  # Generate answer with LLM only

# Deterministic Stopwords (BM25)
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "at",
    "from", "that", "this", "these", "those", "it", "its", "into", "than",
    "then", "but", "if", "so", "such", "their", "there", "here", "what",
    "why", "how", "when", "which", "who", "whom", "does", "do", "did",
    "can", "could", "should", "would", "may", "might", "will", "just",
}
