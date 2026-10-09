"""Оценка логов hw01.py: сравнивает упоминания `файл:строка` в финальных ответах с эталоном.

    python score.py logs/*.log
"""

import re
import sys
from pathlib import Path

H, R = "handler.go", "repo.go"
# эталонные нарушения для каждого вопроса (номер вопроса -> множество (файл, строка))
EXPECTED = {
    1: {(H, 14), (H, 15), (H, 22), (H, 24)},
    2: {(H, 15), (R, 45)},
    3: {(H, 24)},
    4: {(H, 22)},
    5: {(R, 15)},
    6: {(R, 19), (R, 47)},
    7: set(),
    8: {(H, 14), (H, 15), (H, 22), (H, 24), (R, 15), (R, 19), (R, 45), (R, 47)},
    9: set(),
    10: set(),
}
REF_RE = re.compile(r"\b(handler\.go|repo\.go):(\d+)")


def final_answer(block: str) -> str:
    steps = re.findall(r"--- ассистент \(шаг \d+\)[^\n]*\n(.*?)(?=\n--- )", block, re.S)
    return steps[-1] if steps else ""


def is_correct(q: int, answer: str, found: set) -> bool:
    exp = EXPECTED[q]
    if q == 9:
        return not found and re.search(r"не найден|нет такого файла|не существует", answer, re.I) is not None
    if q == 10:
        return re.search(r"\b8\b", answer) is not None
    if q == 7:
        return not found and re.search(r"нарушений нет|нет нарушений|соответствует", answer, re.I) is not None
    return found == exp


def score(path: Path):
    text = path.read_text(encoding="utf-8")
    runs = re.split(r"\n##### ", text)[1:]
    rows = []
    for run in runs:
        for block in re.split(r"\n=== ", run)[1:]:
            q = int(block.split(".", 1)[0])
            answer = final_answer(block)
            found = {(f, int(n)) for f, n in REF_RE.findall(answer)}
            tools = re.search(r"--- вызванные функции: (.*)", block)
            rows.append((q, found, is_correct(q, answer, found), tools.group(1) if tools else ""))
    n = len(rows)
    exp_total = sum(len(EXPECTED[q]) for q, *_ in rows)
    hits = sum(len(found & EXPECTED[q]) for q, found, *_ in rows)
    fps = sum(len(found - EXPECTED[q]) for q, found, *_ in rows)
    correct = sum(ok for _, _, ok, _ in rows)
    guide = sum("get_style_guide" in t for *_, t in rows)
    no_tools = sum(t in ("", "нет") for *_, t in rows)
    per_q = "".join("+" if ok else "." for _, _, ok, _ in rows)
    return (f"{path.stem:45} | верно {correct:2}/{n:<2} | найдено {hits:2}/{exp_total:<2} | "
            f"ложных {fps:3} | стайлгайд {guide:2}/{n:<2} | без вызовов {no_tools:2} | {per_q}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(score(Path(p)))
