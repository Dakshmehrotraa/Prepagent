import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "retrieval" / "index_store"
CLASSIFIER_DIR = BASE_DIR / "models" / "lora_classifier"

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", 3))

BASE_CLASSIFIER_MODEL = os.environ.get("BASE_CLASSIFIER_MODEL", "distilbert-base-uncased")
PATTERN_LABELS = [
    "two_pointers",
    "sliding_window",
    "binary_search",
    "dfs_backtracking",
    "bfs_graph",
    "dynamic_programming",
    "greedy",
    "heap_priority_queue",
    "union_find",
    "prefix_sum_hashing",
]

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
