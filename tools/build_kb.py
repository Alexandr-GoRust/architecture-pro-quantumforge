#!/usr/bin/env python3
"""
build_kb.py — Задание 2: построение «вымышленной» базы знаний.

Берёт статьи из raw_kb/ (скачанные саммари starwars.fandom.com — Wookieepedia),
заменяет все ключевые термины по словарю terms_map.json и складывает
готовые документы в knowledge_base/.

Так получается «локальный корпоративный мир, невидимый внешнему ИИ»:
LLM не узнаёт персонажей/планеты/события после подмены и вынуждена
опираться только на векторный индекс (что и проверяем в Задании 4/5).

Запуск:  python tools/build_kb.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_kb"
OUT = ROOT / "knowledge_base"
TERMS = ROOT / "terms_map.json"


def load_map() -> list[tuple[str, str]]:
    raw = json.loads(TERMS.read_text(encoding="utf-8"))
    pairs = [(k, v) for k, v in raw.items() if not k.startswith("_")]
    # длинные ключи применяем первыми, чтобы «Дарт Вейдер» не разбился
    # раньше «Вейдера» и составные термины не портились частичными заменами
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def replace_terms(text: str, pairs: list[tuple[str, str]]) -> tuple[str, int]:
    total = 0
    for src, dst in pairs:
        if not src:
            continue
        pat = re.compile(re.escape(src))
        text, n = pat.subn(dst, text)
        total += n
    return text, total


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pairs = load_map()
    grand = 0
    for src in sorted(RAW.glob("*.md")):
        text = src.read_text(encoding="utf-8")
        replaced, n = replace_terms(text, pairs)
        dst = OUT / src.name
        dst.write_text(replaced, encoding="utf-8")
        grand += n
        print(f"  {src.name:24} {n:4d} замен")
    files = list(OUT.glob('*.md'))
    print(f"\nИтого: {len(files)} документов в knowledge_base/, {grand} замен терминов")


if __name__ == "__main__":
    main()
