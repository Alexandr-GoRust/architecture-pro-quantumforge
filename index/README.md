# Векторный индекс (Задание 3)

- **Модель эмбеддингов**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  — размерность **384**, мультиязычная (база знаний русскоязычная), работает локально на CPU.
- **База знаний**: `knowledge_base/` — 32 документа вселенной «Квантарис»
  (статьи Wookieepedia с полной подменой терминов, см. `terms_map.json`).
- **Чанкинг**: `RecursiveCharacterTextSplitter` (LangChain), chunk_size=900 символов
  (~150–200 слов), overlap=130 (~15%). Метаданные каждого чанка: `source`, `title`,
  `chunk_index`, `n_chunks` — источник и позиция для цитирования.
- **Векторная БД**: FAISS, `IndexFlatIP` + L2-нормализация = косинусное сходство,
  точный поиск (flat). Файлы: `faiss.index` + `chunks_meta.pkl`.
- **Объём**: 61 чанк / 32 документа; эмбеддинги — 1.0 c, полная индексация
  (включая загрузку модели) — ~54 c на MacBook (CPU).
- **Самопроверка** (вывод `build_index.py`): «Кто такой Ксарн Велгор?» →
  darth_vader.md#0 (0.620); «Что такое Ядро Пустоты?» → death_star.md#0 (0.554);
  «Где обитают филлины?» → endor.md#1. Поиск находит релевантные чанки.

Пересоздать индекс: `python build_index.py`
