#!/usr/bin/env python3
"""telegram_bot.py — Telegram-интерфейс RAG-бота (Задание 4; тариф Про).
Запуск: TELEGRAM_TOKEN=... python telegram_bot.py (токен в .env)"""
import os
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from rag_bot.core import RAGBot
from rag_bot.logger import log_query

load_dotenv()
bot = RAGBot()

async def start(update, ctx):
    await update.message.reply_text(
        "Привет! Я справочный бот по вселенной «Квантарис». "
        "Задай вопрос — отвечу по базе знаний со ссылками на источники. "
        "Если данных нет, честно скажу «Я не знаю».")

async def answer(update, ctx):
    q = update.message.text
    r = bot.ask(q)
    log_query(q, r)
    src = "\n".join(f"[{i+1}] {c['source']}#{c['chunk_index']}"
                    for i, c in enumerate(r.chunks)) or "—"
    await update.message.reply_text(f"{r.answer}\n\nИсточники:\n{src}")

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))
    print("Telegram-бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
