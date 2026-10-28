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
GENERATION_MODEL = "gemma4:12b"

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
MAX_NEW_TOKENS = 512

# Evaluation Metrics
BERTSCORE_MODEL = "bert-base-uncased"
BERTSCORE_NUM_LAYERS = None
RETRIEVAL_METRIC_KS = (1, 3, 5)
MEASURE_LATENCY = True

# Run Mode
USE_RDF_AGENT = True  # False: use the existing RAG pipeline
RUN_EVALUATION = True  # True: evaluate on QA_DATASET_PATH; False: interactive
BASELINE = False  # Generate answer with LLM only

# Deterministic Stopwords (BM25)
STOPWORDS = {
    "de", "het", "een", "en", "of", "te", "naar", "aan", "tot",
    "van", "in", "op", "voor", "met",
    "is", "zijn", "was", "waren", "wezen", "geweest", "zijnde",
    "als", "zoals", "door", "bij",
    "uit", "vanaf", "dat", "die", "dit", "deze", "het",
    "hij", "zij", "zijn", "haar", "binnen", "dan",
    "toen", "vervolgens", "maar", "indien", "dus", "zo",
    "zulk", "zulke", "hun", "daar", "er", "hier", "wat",
    "waarom", "hoe", "wanneer", "welke", "welk", "wie",
    "doet", "doen", "deed",
    "kan", "kunnen", "kon", "konden",
    "moet", "moeten", "zou", "zouden",
    "mag", "mogen", "misschien", "wellicht",
    "zal", "zullen", "slechts", "net", "pas", "gewoon",
}
