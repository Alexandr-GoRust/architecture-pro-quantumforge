"""logger.py — логирование запросов в JSONL (Задание 7).
Каждый запрос: текст, timestamp, найдены ли чанки, длина ответа,
флаг успеха, найденные источники."""
import json, time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "queries.jsonl"

def log_query(question: str, result) -> None:
    LOG.parent.mkdir(exist_ok=True)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "chunks_found": len(result.chunks),
        "answer_len": len(result.answer),
        "success": ("не знаю" not in result.answer.lower()
                    and "заблокирован" not in result.answer.lower()
                    and len(result.chunks) > 0),
        "sources": [f"{c['source']}#{c['chunk_index']}" for c in result.chunks],
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
