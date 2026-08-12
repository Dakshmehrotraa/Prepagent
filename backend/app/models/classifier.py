import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import CLASSIFIER_DIR, BASE_CLASSIFIER_MODEL, PATTERN_LABELS

# Simple keyword fallback so the API stays usable before you've trained
# and dropped in the LoRA adapter (see app/training/train_lora_classifier.py).
_KEYWORD_FALLBACK = {
    "two_pointers": ["two pointer", "left and right", "sorted array pair"],
    "sliding_window": ["substring", "subarray", "window", "contiguous"],
    "binary_search": ["rotated", "sorted array", "log n", "search"],
    "dfs_backtracking": ["backtrack", "permutation", "subset", "board", "combination"],
    "bfs_graph": ["graph", "island", "grid", "shortest path", "connected"],
    "dynamic_programming": ["dp", "maximum subarray", "minimum cost", "ways to"],
    "greedy": ["greedy", "interval", "jump", "gas station"],
    "heap_priority_queue": ["kth largest", "merge k", "top k", "heap"],
    "union_find": ["connected components", "redundant", "disjoint set"],
    "prefix_sum_hashing": ["two sum", "subarray sum", "hashmap", "prefix sum"],
}


class PatternClassifier:
    def __init__(self):
        self.available = CLASSIFIER_DIR.exists() and any(CLASSIFIER_DIR.iterdir())
        if self.available:
            from peft import PeftModel

            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_CLASSIFIER_MODEL, num_labels=len(PATTERN_LABELS)
            )
            self.model = PeftModel.from_pretrained(base, str(CLASSIFIER_DIR))
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
        else:
            self.model = None
            self.tokenizer = None

    def predict(self, text: str) -> dict:
        if self.available:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            top_idx = int(torch.argmax(probs))
            return {
                "pattern": PATTERN_LABELS[top_idx],
                "confidence": float(probs[top_idx]),
                "source": "lora_distilbert",
            }

        # Fallback: keyword scoring
        text_lower = text.lower()
        scores = {label: 0 for label in PATTERN_LABELS}
        for label, keywords in _KEYWORD_FALLBACK.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[label] += 1
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        return {
            "pattern": best if scores[best] > 0 else "prefix_sum_hashing",
            "confidence": scores[best] / total if scores[best] > 0 else 0.1,
            "source": "keyword_fallback",
        }


@lru_cache(maxsize=1)
def get_classifier() -> "PatternClassifier":
    return PatternClassifier()
