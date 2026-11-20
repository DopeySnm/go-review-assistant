"""Скачивает внешние руководства по стилю Go в kb/ и приводит их к Markdown: python fetch_kb.py

Снимок кладётся в репозиторий, чтобы прогон ноутбука не зависел от сети и был воспроизводимым.
Командное руководство kb/team_style.md пишется вручную и этим скриптом не трогается.
"""

import json
import re
import urllib.request
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

KB = Path(__file__).resolve().parent / "kb"
SOURCES = {
    "code_review_comments.md": {
        "title": "Go Code Review Comments",
        "url": "https://raw.githubusercontent.com/golang/wiki/master/CodeReviewComments.md",
        "page": "https://go.dev/wiki/CodeReviewComments",
        "license": "CC BY 4.0 (текст), BSD (код) — The Go Authors",
    },
    "effective_go.md": {
        "title": "Effective Go",
        "url": "https://raw.githubusercontent.com/golang/website/master/_content/doc/effective_go.html",
        "page": "https://go.dev/doc/effective_go",
        "license": "CC BY 4.0 (текст), BSD (код) — The Go Authors",
    },
    "uber_style.md": {
        "title": "Uber Go Style Guide",
        "url": "https://raw.githubusercontent.com/uber-go/guide/master/style.md",
        "page": "https://github.com/uber-go/guide/blob/master/style.md",
        "license": "CC BY 4.0 — Uber Technologies",
    },
}


class HtmlToMarkdown(HTMLParser):
    """Effective Go — HTML: заголовки h2/h3 → ##/###, <pre> → блок кода, остальное — текст абзацами."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.pre = False

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3", "h4"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self.pre = True
            self.out.append("\n\n```go\n")
        elif tag in ("p", "li", "br"):
            self.out.append("\n\n" if tag == "p" else "\n- " if tag == "li" else "\n")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre = False
            self.out.append("\n```\n")
        elif tag in ("h2", "h3", "h4"):
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data if self.pre else re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def bad_good(m: re.Match) -> str:
    """HTML-таблица «Bad | Good» → код с подписями; ячейки идут по порядку заголовков."""
    table = m.group(0)
    heads = re.findall(r"<th>(.*?)</th>", table)
    cells = re.findall(r"<td>(.*?)</td>", table, flags=re.S)
    parts = []
    for i, cell in enumerate(cells):
        label = heads[i % len(heads)] if heads else ""
        parts.append((f"{label}:\n" if label else "") + re.sub(r"<[^>]+>", "", cell).strip())
    return "\n\n".join(parts)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def main():
    KB.mkdir(exist_ok=True)
    meta = {}
    for name, src in SOURCES.items():
        raw = fetch(src["url"])
        if name.endswith("effective_go.md"):
            raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)  # шапка шаблона сайта
            parser = HtmlToMarkdown()
            parser.feed(raw)
            text = parser.markdown()
        else:
            text = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)  # front matter вики
            text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
            text = re.sub(r"<table>.*?</table>", bad_good, text, flags=re.S)  # Uber: примеры Bad / Good
        text = unescape(text).strip()
        if not text.startswith("# "):
            text = f"# {src['title']}\n\n{text}"
        (KB / name).write_text(text + "\n", encoding="utf-8")
        meta[name] = {**src, "fetched": date.today().isoformat()}
        print(f"{name}: {len(text)} символов")
    meta["team_style.md"] = {"title": "Код-стайл команды", "page": "kb/team_style.md", "license": "свой текст"}
    (KB / "sources.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
