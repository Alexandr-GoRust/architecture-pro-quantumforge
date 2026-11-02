#!/usr/bin/env python3
"""update_index.py — Задание 6: ежедневное обновление базы знаний.

Источник: knowledge_base/ (+ demo/malicious.txt при наличии).
Сканирует новые/изменённые .md, режет на чанки, генерит эмбеддинги,
дописывает их в существующий FAISS-индекс (инкрементально), логирует.

Запуск вручную:   python update_index.py
Запуск по cron:   0 6 * * *  cd /path/to/rag-bot && .venv/bin/python update_index.py
Лог:              logs/update.log  (время старта/финиша, новые чанки,
                  размер индекса, ошибки)
"""
import hashlib
import pickle
import time
from datetime import datetime
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag_bot.core import EMBED_MODEL, INDEX_DIR

ROOT = Path(__file__).resolve().parent
KB = ROOT / "knowledge_base"
STATE = ROOT / "index" / ".file_state.json"   # отпечатки уже проиндексированных файлов
LOG = ROOT / "logs" / "update.log"

import json, sys


def logline(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def main() -> None:
    t0 = time.time()
    logline("=== update_index запущен ===")
    try:
        state = load_state()
        new_files, changed = [], []
        for p in sorted(KB.glob("*.md")):
            h = file_hash(p)
            if p.name not in state:
                new_files.append(p)
            elif state[p.name] != h:
                changed.append(p)
            state[p.name] = h

        if not new_files and not changed:
            index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
            logline(f"изменений нет; размер индекса: {index.ntotal} векторов")
            return

        from rag_bot.core import INDEX_DIR  # noqa
        import sys; sys.path.insert(0, str(ROOT))
        from build_index import load_chunks  # переиспользуем чанкер

        # пересобираем только затронутые документы
        enc = SentenceTransformer(EMBED_MODEL)
        index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        chunks = pickle.load(open(INDEX_DIR / "chunks_meta.pkl", "rb"))

        added = 0
        for p in new_files + changed:
            text = p.read_text(encoding="utf-8").strip()
            # чанкер из build_index работает по каталогу; для одного файла:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            sp = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=130,
                                                separators=["\n\n", "\n", ". ", " ", ""])
            parts = sp.split_text(text)
            vecs = enc.encode(parts, convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
            index.add(vecs)
            for i, part in enumerate(parts):
                chunks.append({"text": part, "source": f"knowledge_base/{p.name}",
                               "title": p.stem, "chunk_index": i, "n_chunks": len(parts)})
            added += len(parts)

        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        pickle.dump(chunks, open(INDEX_DIR / "chunks_meta.pkl", "wb"))
        STATE.write_text(json.dumps(state, indent=1))

        logline(f"новых файлов: {len(new_files)}, изменённых: {len(changed)}; "
                f"добавлено чанков: {added}; размер индекса: {index.ntotal} векторов; "
                f"ошибок: 0; время {time.time()-t0:.1f} c")
        logline(f"index updated at {datetime.now():%Y-%m-%d %H:%M}, "
                f"{added} chunks added, 0 errors")
    except Exception as e:
        logline(f"ОШИБКА: {e}")
        raise
    finally:
        logline("=== update_index завершён ===")


if __name__ == "__main__":
    main()
