#!/usr/bin/env python3
"""
build_index.py — Задание 3: векторный индекс базы знаний.

Пайплайн:
  1. Читаем knowledge_base/*.md (32 документа, мир «Квантарис»).
  2. Режем на чанки (RecursiveCharacterTextSplitter, ~180 слов, overlap 15%)
     с сохранением источника и позиции чанка в метаданных.
  3. Эмбеддинги: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
     (384 измерения, мультиязычная — база русскоязычная; локально, без API).
  3. Индекс: FAISS (IndexFlatIP + нормализация L2 => косинусное сходство),
     метаданные — рядом в pickle.

Артефакты: index/faiss.index, index/chunks_meta.pkl
Запуск:    python build_index.py
"""
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
KB = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIM = 384  # размерность эмбеддингов выбранной модели
CHUNK_SIZE = 900   # символов (~150-200 слов на русском)
CHUNK_OVERLAP = 130  # ~15% — перекрытие, чтобы не терять связки


def load_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for path in sorted(KB.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            chunks.append({
                "text": part,
                "source": f"knowledge_base/{path.name}",
                "title": path.stem.replace("_", " "),
                "chunk_index": i,
                "n_chunks": len(parts),
            })
    return chunks


def main() -> None:
    t0 = time.time()
    chunks = load_chunks()
    print(f"Чанков: {len(chunks)} из {len(list(KB.glob('*.md')))} документов")

    print(f"Загружаю эмбеддер {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    t1 = time.time()

    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype("float32")
    print(f"Эмбеддинги: {emb.shape}, время {time.time() - t1:.1f} c")

    # FAISS: внутреннее произведение на нормализованных векторах == косинус
    index = faiss.IndexFlatIP(MODEL_DIM)
    index.add(emb)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks_meta.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Сохранено: index/faiss.index ({index.ntotal} векторов), "
          f"index/chunks_meta.pkl")
    print(f"Полное время индексации: {time.time() - t0:.1f} c")

    # Самопроверка: 2-3 запроса к индексу (совет из ТЗ)
    print("\n--- Самопроверка поиска ---")
    for q in ["Кто такой Ксарн Велгор?", "Что такое Ядро Пустоты?",
              "Где обитают филлины?"]:
        qv = model.encode([q], convert_to_numpy=True,
                          normalize_embeddings=True).astype("float32")
        scores, ids = index.search(qv, 3)
        print(f"Q: {q}")
        for s, i in zip(scores[0], ids[0]):
            meta = chunks[i]
            print(f"   [{s:.3f}] {meta['source']}#{meta['chunk_index']}: "
                  f"{meta['text'][:90]}...")


if __name__ == "__main__":
    main()
