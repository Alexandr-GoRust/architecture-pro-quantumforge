#!/usr/bin/env python3
"""evaluate.py — Задание 7: автоматическое тестирование качества RAG.

Читает golden_questions.txt (вопрос | есть_в_базе | ожидаемый_фрагмент),
прогоняет через RAG-бота, пишет лог logs/queries.jsonl и отчёт
logs/evaluation.txt: покрытие тем, где бот «слеп», рекомендации.

Запуск: python evaluate.py
"""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
from rag_bot.core import RAGBot
from rag_bot.logger import log_query

GOLDEN = Path("golden_questions.txt")
REPORT = Path("logs/evaluation.txt")


def load_golden():
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        q, in_kb = parts[0], parts[1] == "1"
        expected = parts[2] if len(parts) > 2 else ""
        rows.append((q, in_kb, expected))
    return rows


def main():
    bot = RAGBot()
    rows = load_golden()
    rep = []
    tp = fp = fn = tn = 0
    gaps = []
    for q, in_kb, expected in rows:
        r = bot.ask(q)
        log_query(q, r)
        ans = r.answer.lower()
        got_answer = "не знаю" not in ans and "заблокирован" not in ans
        ok = (got_answer == in_kb)
        if in_kb and got_answer:
            hit = expected.lower() in ans if expected else True
            tp += hit
            status = "OK" if hit else f"ОТВЕТ БЕЗ ОЖИДАЕМОГО ФРАГМЕНТА ({expected})"
        elif in_kb and not got_answer:
            fn += 1; status = "ПРОБЕЛ: должен был ответить, сказал «не знаю»"
            gaps.append(q)
        elif not in_kb and got_answer:
            fp += 1; status = "ЛОЖНЫЙ ОТВЕТ: базы нет, но бот ответил"
        else:
            tn += 1; status = "OK (честное «не знаю»)"
        rep.append(f"[{'PASS' if ok else 'FAIL'}] {q}\n   {status}\n   A: {r.answer[:160]}\n")
    summary = [
        "=== Отчёт оценки RAG (golden set) ===",
        f"Всего вопросов: {len(rows)}",
        f"Верных ответов на известных темах: {tp}",
        f"Пробелов (бот не ответил, хотя данные есть): {fn}",
        f"Ложных ответов (ответил без данных): {fp}",
        f"Честных «не знаю»: {tn}",
        "",
        "Темы-пробелы: " + ("; ".join(gaps) if gaps else "нет"),
        "Рекомендации:",
        "  - при пробелах: расширить knowledge_base/ по соответствующим темам;",
        "  - при ложных ответах: усилить порог min_score / System-инструкцию;",
        "  - после правок перезапустить build_index.py и evaluate.py.",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(summary) + "\n\n" + "\n".join(rep),
                      encoding="utf-8")
    print("\n".join(summary))
    print(f"\nПолный отчёт: {REPORT}")


if __name__ == "__main__":
    main()
