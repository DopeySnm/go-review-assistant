"""База знаний для RAG: руководства по стилю Go → чанки → эмбеддинги → векторное хранилище.

Источники (kb/*.md): командный код-стайл S1–S6 (по-русски), Go Code Review Comments, Effective Go,
Uber Go Style Guide (по-английски; ссылки и лицензии — kb/sources.json). Вопросы пользователя — по-русски,
поэтому модель эмбеддингов мультиязычная: bge-m3 в llama-server с --embeddings (OpenAI-совместимый API).
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 150


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=os.environ.get("EMB_BASE_URL", "http://127.0.0.1:8081/v1"),
        api_key=os.environ.get("EMB_API_KEY", "none"),
        model=os.environ.get("EMB_MODEL", "bge-m3"),
        check_embedding_ctx_length=False,  # иначе режет текст токенизатором tiktoken под модели OpenAI
        chunk_size=32,  # текстов в одном запросе
    )


def load_chunks(kb_dir: str | Path = "kb") -> list[Document]:
    """Markdown → разделы по заголовкам → куски до CHUNK_SIZE символов.

    В текст чанка добавляется «Источник. Раздел»: заголовок раздела часто и есть ответ на вопрос
    («Receiver Names»), а сплиттер убирает его из тела.
    """
    kb_dir = Path(kb_dir)
    sources = json.loads((kb_dir / "sources.json").read_text(encoding="utf-8"))
    by_headers = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for name, src in sources.items():
        text = (kb_dir / name).read_text(encoding="utf-8")
        for part in by_size.split_documents(by_headers.split_text(text)):
            lines = [l for l in part.page_content.splitlines() if l.strip()]
            # оглавления (списки ссылок) и обрывки без смысла в индекс не берём
            if len(part.page_content) < 100 or sum(l.lstrip().startswith("- [") for l in lines) > len(lines) / 2:
                continue
            section = " > ".join(part.metadata[h] for h in ("h2", "h3", "h4") if h in part.metadata)
            chunks.append(Document(
                page_content=f"{src['title']}. {section}\n{part.page_content}",
                metadata={"source": name, "title": src["title"], "section": section, "url": src.get("page", "")},
            ))
    return chunks


class KnowledgeBase:
    """Векторное хранилище чанков. Индекс сохраняется в файл и пересчитывается, только если поменялись чанки."""

    def __init__(self, store: InMemoryVectorStore, threshold: float = 0.5, meta: Path | None = None):
        self.store, self.threshold, self.meta = store, threshold, meta

    @classmethod
    def build_or_load(cls, kb_dir: str | Path = "kb", index: str | Path = "kb_index.json", log=None, **kwargs):
        log = log or print
        chunks, embeddings, index = load_chunks(kb_dir), make_embeddings(), Path(index)
        digest = hashlib.sha256(json.dumps([c.page_content for c in chunks]).encode()).hexdigest()[:16]
        meta = index.with_suffix(".meta.json")
        saved = json.loads(meta.read_text()) if meta.exists() else {}
        if index.exists() and saved.get("digest") == digest:
            store = InMemoryVectorStore.load(str(index), embeddings)
            log(f"индекс прочитан из `{index}`: {len(store.store)} чанков, эмбеддинги не пересчитывались"
                + (f"; порог {saved['threshold']}" if "threshold" in saved else ""))
            if "threshold" in saved:
                kwargs.setdefault("threshold", saved["threshold"])
        else:
            t = time.time()
            store = InMemoryVectorStore(embeddings)
            store.add_documents(chunks)
            store.dump(str(index))
            meta.write_text(json.dumps({"digest": digest, "chunks": len(chunks)}))
            log(f"индекс построен: {len(chunks)} чанков за {time.time() - t:.0f} с, сохранён в `{index}`")
        return cls(store, meta=meta, **kwargs)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Косинусная близость запроса и чанков, по убыванию."""
        return self.store.similarity_search_with_score(query, k=k)

    def calibrate(self) -> dict:
        """Порог «вопрос по Go / не по теме»: середина между худшим вопросом из EVAL и лучшим из OFF_TOPIC.

        Сохраняется рядом с индексом — после перезапуска ядра не нужно калибровать снова.
        """
        inside = {q: self.search(q, 1)[0][1] for q, _ in EVAL}
        outside = {q: self.search(q, 1)[0][1] for q in OFF_TOPIC}
        self.threshold = round((min(inside.values()) + max(outside.values())) / 2, 3)
        if self.meta:
            self.meta.write_text(json.dumps({**json.loads(self.meta.read_text()), "threshold": self.threshold}))
        return {"inside": inside, "outside": outside, "threshold": self.threshold}


def cite(doc: Document) -> str:
    return f"{doc.metadata['title']} — {doc.metadata['section'] or 'введение'}"


def group_hits(hits: list[tuple[Document, float]]) -> list[list[Document]]:
    """Чанки одного раздела — один фрагмент: иначе в источниках раздел повторяется («Error Wrapping» дважды)."""
    groups: dict[tuple, list[Document]] = {}
    for doc, _ in hits:
        groups.setdefault((doc.metadata["title"], doc.metadata["section"]), []).append(doc)
    return list(groups.values())


def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[{i}] {cite(docs[0])}\n" + "\n".join(d.page_content.split("\n", 1)[-1] for d in docs)
                       for i, docs in enumerate(group_hits(hits), 1))


def format_sources(hits: list[tuple[Document, float]]) -> str:
    return "Источники:\n" + "\n".join(f"[{i}] {cite(docs[0])} ({docs[0].metadata['url']})"
                                      for i, docs in enumerate(group_hits(hits), 1))


def strip_sources(answer: str) -> str:
    """Свой список источников модель дописывает по образцу прошлых ответов и путает — оставляем только код."""
    return re.split(r"\n\s*\**Источники\**:?", answer)[0].strip()


# --- классическая RAG-цепочка LCEL: для сравнения «модель без базы / с базой» ---

PLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Ты — эксперт по стилю кода на Go. Отвечай по-русски, кратко: 3–5 предложений."),
    ("human", "{question}"),
])
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — эксперт по стилю кода на Go. Отвечай по-русски, кратко: 3–5 предложений. Отвечай только по "
     "фрагментам руководств ниже (они могут быть на английском), ссылайся на них номерами [1], [2]. "
     "Если в фрагментах ответа нет — так и скажи. Список источников не пиши — его добавят.\n\nФрагменты:\n{context}"),
    ("human", "{question}"),
])


def make_plain_chain(llm):
    return PLAIN_PROMPT | llm | StrOutputParser()


def make_rag_chain(kb: KnowledgeBase, llm, k: int = 4):
    """question → поиск → контекст в промпт → ответ модели + список источников (его пишет код, не модель)."""
    retrieve = RunnableLambda(lambda q: kb.search(q, k))
    return (
        RunnablePassthrough.assign(hits=lambda x: retrieve.invoke(x["question"]))
        | RunnablePassthrough.assign(answer=(lambda x: {"question": x["question"],
                                                         "context": format_context(x["hits"])})
                                     | RAG_PROMPT | llm | StrOutputParser())
        | RunnableLambda(lambda x: f"{strip_sources(x['answer'])}\n\n{format_sources(x['hits'])}")
    )


# --- оценка поиска: вопрос → какие разделы считаются верным ответом (источник, кусок названия раздела) ---

EVAL = [
    ("Как называть ресивер метода?", [("code_review_comments.md", "Receiver Names")]),
    ("Когда метод должен принимать указатель, а когда значение?",
     [("code_review_comments.md", "Receiver Type"), ("effective_go.md", "Pointers vs. Values"),
      ("uber_style.md", "Receivers and Interfaces")]),
    ("С какой буквы начинать текст ошибки и ставить ли точку в конце?", [("code_review_comments.md", "Error Strings")]),
    ("Как добавить контекст к ошибке, чтобы errors.Is продолжал работать?",
     [("uber_style.md", "Error Wrapping"), ("team_style.md", "S5")]),
    ("Можно ли использовать panic для обработки обычных ошибок?",
     [("code_review_comments.md", "Don't Panic"), ("uber_style.md", "Don't Panic"), ("team_style.md", "S3")]),
    ("Где объявлять интерфейс: рядом с реализацией или там, где он используется?",
     [("code_review_comments.md", "Interfaces")]),
    ("Как писать аббревиатуры вроде URL и ID в именах?",
     [("code_review_comments.md", "Initialisms"), ("team_style.md", "S1")]),
    ("Можно ли хранить context.Context в поле структуры?",
     [("code_review_comments.md", "Contexts"), ("team_style.md", "S4")]),
    ("С чего должен начинаться комментарий к экспортируемой функции?",
     [("code_review_comments.md", "Doc Comments"), ("effective_go.md", "Commentary"), ("team_style.md", "S6")]),
    ("Как правильно называть пакеты?",
     [("effective_go.md", "Package names"), ("code_review_comments.md", "Package Names"),
      ("uber_style.md", "Package Names")]),
    ("Как объявить пустой срез: через var или через литерал?",
     [("code_review_comments.md", "Declaring Empty Slices"), ("uber_style.md", "nil is a valid slice")]),
    ("Стоит ли давать имена возвращаемым значениям функции?", [("code_review_comments.md", "Named Result Parameters")]),
    ("Что вернуть клиенту, если в HTTP-обработчике упал запрос к базе?", [("team_style.md", "S3")]),
    ("Можно ли не проверять ошибку, которую вернул Close?", [("team_style.md", "S2"), ("uber_style.md", "Defer")]),
]
OFF_TOPIC = [
    "Какая сегодня погода?",
    "Посоветуй рецепт борща.",
    "Кто выиграл чемпионат мира по футболу?",
    "Сколько стоит биткоин?",
]


def evaluate(kb: KnowledgeBase, k: int = 4) -> list[dict]:
    rows = []
    for question, relevant in EVAL:
        hits = kb.search(question, k)
        ranks = [i for i, (doc, _) in enumerate(hits, 1)
                 if any(doc.metadata["source"] == src and part.lower() in doc.metadata["section"].lower()
                        for src, part in relevant)]
        rows.append({"question": question, "rank": ranks[0] if ranks else None, "top1": cite(hits[0][0]),
                     "score": round(hits[0][1], 3)})
    return rows
