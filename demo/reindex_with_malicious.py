#!/usr/bin/env python3
"""Индексирует demo/malicious.txt в общий индекс (Задание 5, шаг 2)."""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
import faiss, numpy as np, pickle
from rag_bot.core import INDEX_DIR, EMBED_MODEL
from sentence_transformers import SentenceTransformer

text = Path("demo/malicious.txt").read_text(encoding="utf-8").strip()
enc = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
chunks = pickle.load(open(INDEX_DIR / "chunks_meta.pkl", "rb"))

v = enc.encode([text], convert_to_numpy=True,
               normalize_embeddings=True).astype("float32")
index.add(v)
chunks.append({"text": text, "source": "demo/malicious.txt",
               "title": "malicious", "chunk_index": 0, "n_chunks": 1})
faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
pickle.dump(chunks, open(INDEX_DIR / "chunks_meta.pkl", "wb"))
print(f"malicious.txt проиндексирован: в индексе {index.ntotal} векторов")
