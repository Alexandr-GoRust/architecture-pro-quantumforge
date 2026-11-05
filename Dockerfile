FROM python:3.11-slim

WORKDIR /app

# системные зависимости для faiss/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код, база знаний и готовый индекс
COPY rag_bot/ ./rag_bot/
COPY chat.py telegram_bot.py build_index.py update_index.py evaluate.py ./
COPY knowledge_base/ ./knowledge_base/
COPY index/ ./index/

# индекс собираем на этапе запуска, если его нет (модель качается в кеш)
ENV PYTHONUNBUFFERED=1
CMD ["python", "chat.py"]
