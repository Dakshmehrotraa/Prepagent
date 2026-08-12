"""
Builds a FAISS index over the problem statements + hints in problems.json.
Run this once (locally or in CI) whenever problems.json changes:

    python -m app.retrieval.build_index

Produces:
    app/retrieval/index_store/faiss.index
    app/retrieval/index_store/metadata.json
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from app.config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL_NAME


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "problems.json") as f:
        data = json.load(f)
    problems = data["problems"]

    # Embed statement + hint together so retrieval captures both surface
    # wording and the underlying technique.
    corpus = [f"{p['title']}. {p['statement']} Hint: {p['hint']}" for p in problems]

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Encoding {len(corpus)} problems...")
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Index built with {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {INDEX_DIR}")


if __name__ == "__main__":
    build()
