#!/usr/bin/env python3
"""chat.py — консольный REPL-интерфейс RAG-бота (Задание 4, вариант интерфейса 1).
Запуск: python chat.py"""
from rag_bot.core import RAGBot

def main():
    bot = RAGBot()
    print("RAG-бот по вселенной «Квантарис». Вопросы ('exit' — выход):")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        r = bot.ask(q)
        print(f"\nБот> {r.answer}")
        src = ", ".join(f"{c['source']}#{c['chunk_index']}" for c in r.chunks)
        print(f"     Источники: {src or '— (ничего не найдено)'}")

if __name__ == "__main__":
    main()
