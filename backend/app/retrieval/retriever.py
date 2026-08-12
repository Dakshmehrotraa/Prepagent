import json
import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.config import INDEX_DIR, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL


class Retriever:
    def __init__(self):
        index_path = INDEX_DIR / "faiss.index"
        meta_path = INDEX_DIR / "metadata.json"
        if not index_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python -m app.retrieval.build_index` first."
            )
        self.index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            self.metadata = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query: str, k: int = TOP_K_RETRIEVAL):
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        vec = vec.astype(np.float32)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            problem = dict(self.metadata[idx])
            problem["similarity"] = float(score)
            results.append(problem)
        return results


@lru_cache(maxsize=1)
def get_retriever() -> "Retriever":
    # Cached singleton so the embedding model + index load only once per process.
    return Retriever()
