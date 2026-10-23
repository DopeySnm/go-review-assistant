"""Чат-наставник по Go-код-стайлу команды с долгосрочной памятью фактов в файле.

Память — не история сообщений, а набор фактов о пользователе (имя, уровень, стек, цели...).
Факты извлекаются отдельным LLM-вызовом со structured output и хранятся в memory.json по user_id.
История сообщений живёт только в RAM текущего сеанса и после перезапуска пропадает.
"""

import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from IPython.display import Markdown, display
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core._api import LangChainDeprecationWarning
from pydantic import BaseModel, Field, create_model

# предупреждения об устаревших классах мешаются с выводом чата
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

STYLE_GUIDE = """\
S1. Имена — MixedCaps/camelCase, без подчёркиваний. Аббревиатуры целиком: userID, не userId.
S2. Ошибки не игнорировать: ни `_ = f()`, ни вызов без проверки возвращаемой ошибки. `defer x.Close()` допускается.
S3. В HTTP-обработчиках не использовать panic — вернуть 500 через http.Error.
S4. Функции, которые ходят в БД или сеть, принимают ctx context.Context первым параметром.
    HTTP-обработчики берут контекст из r.Context() — отдельный параметр ctx им не нужен.
S5. Ошибки, возвращаемые наверх, оборачиваются: fmt.Errorf("контекст: %w", err).
S6. У экспортируемых функций и методов есть doc-комментарий, начинающийся с имени."""


class UserFacts(BaseModel):
    """Факты о разработчике. Поле заполняется, только если пользователь сам это сказал."""

    name: Optional[str] = Field(None, description="Имя пользователя")
    # Literal превращается в enum схемы: грамматика llama-server не даст записать сюда «год»
    go_level: Optional[Literal["junior", "middle", "senior"]] = Field(None, description="Уровень в Go")
    go_experience: Optional[str] = Field(None, description="Сколько времени пишет на Go: «1 год», «полгода»")
    stack: Optional[str] = Field(None, description="Технологии кроме Go: базы данных, инструменты, ОС")
    project: Optional[str] = Field(None, description="Над каким проектом или сервисом работает")
    weak_spots: Optional[str] = Field(None, description="Что даётся трудно в Go или в код-стайле")
    goal: Optional[str] = Field(None, description="Чего хочет добиться")


# порядок опроса и дословные вопросы: 3B-модель, получив «узнай уровень», спрашивает что-то своё
QUESTIONS = {
    "name": "Как тебя зовут?",
    "go_level": "Какой у тебя уровень в Go: junior, middle или senior?",
    "go_experience": "Сколько ты уже пишешь на Go?",
    "stack": "С какими технологиями ты работаешь кроме Go: базы, инструменты, ОС?",
    "project": "Над каким проектом ты сейчас работаешь?",
    "weak_spots": "Что тебе сейчас даётся труднее всего в Go или в код-стайле?",
    "goal": "Чего ты хочешь добиться?",
}
LABELS = {
    "name": "Имя", "go_level": "Уровень", "go_experience": "Опыт в Go", "stack": "Стек",
    "project": "Проект", "weak_spots": "Слабые места", "goal": "Цель",
}
INCIDENTAL = ("name", "go_level", "go_experience")  # можно назвать, отвечая на другой вопрос
EMPTY = {"", "null", "none", "нет", "неизвестно", "не указано", "n/a"}


def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Любой OpenAI-совместимый сервер: llama-server локально/на Kaggle или OpenRouter."""
    return ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("LLM_API_KEY", "none"),
        model=os.environ.get("LLM_MODEL", "local"),
        temperature=temperature,
        max_tokens=400,
    )


class LogCell:
    """Служебный вывод в собственной ячейке ноутбука, а не в чате.

    Ячейка, в которой создан LogCell, держит display_id; все записи обновляют её вывод.
    """

    def __init__(self, title: str = "Служебный лог"):
        self.title = title
        self.lines: list[str] = []
        self.handle = display(Markdown(self._render()), display_id=True)

    def __call__(self, text: str):
        self.lines.append(f"- `{datetime.now():%H:%M:%S}` {text}")
        self.handle.update(Markdown(self._render()))

    def _render(self) -> str:
        return f"**{self.title}**\n\n" + ("\n".join(self.lines) or "_пока пусто_")


def format_facts(facts: dict) -> str:
    if not facts:
        return "пока ничего не известно"
    return "\n".join(f"- {LABELS[k]}: {v}" for k, v in facts.items() if k in LABELS)


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты извлекаешь факты о пользователе из его последнего сообщения и возвращаешь JSON по схеме.\n"
     "Заполняй поле, только если пользователь сам прямо сказал это в своём сообщении. "
     "Всё, о чём он не сказал, — null. Не придумывай, не копируй слова из вопроса ассистента. "
     "Пиши значения коротко, по-русски."),
    ("human", "Ассистент спрашивал про поле «{field}».\nВопрос ассистента: {question}\n"
              "Сообщение пользователя: {message}"),
])


LEVEL_WORDS = {"junior": ("junior", "джун"), "middle": ("middle", "мидл"), "senior": ("senior", "сеньор", "синьор")}


def accept_fact(key: str, value: str, message: str, asked: str) -> bool:
    """Проверка, что факт действительно есть в реплике: маленькая модель заполняет поля «на всякий случай»."""
    text = message.lower()
    if key == "name":
        return value.lower() in text and (asked == "name" or "зовут" in text or re.search(r"\bя\s", text))
    if key == "go_level":
        return any(word in text for word in LEVEL_WORDS[value])
    if key == "go_experience":
        unit = r"\d|год|лет|месяц|полгода|недел"
        return bool(re.search(unit, value.lower())) and bool(re.search(unit, text))
    # стек, проект, слабые места, цель — только в ответ на свой вопрос
    return key == asked


class FileFactMemory(BaseChatMemory):
    """Долгосрочная память фактов в JSON-файле: {user_id: {"facts": {...}, "updated_at": ...}}.

    load_memory_variables() — отдаёт известные факты и список недостающих;
    save_context() — извлекает факты из реплики пользователя отдельным LLM-вызовом и пишет файл.
    """

    path: Path
    user_id: str
    llm: Any = None  # модель для извлечения фактов
    log: Any = None
    facts: dict = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        profile = self._read_all().get(self.user_id)
        self.facts = dict(profile["facts"]) if profile else {}
        if self.log:
            if not self.path.exists():
                self.log(f"`{self.path}` нет — память пустая")
            elif profile:
                self.log(f"прочитан `{self.path}`, профиль `{self.user_id}`: "
                         f"`{json.dumps(self.facts, ensure_ascii=False)}`")
            else:
                self.log(f"в `{self.path}` нет профиля `{self.user_id}` — память пустая")

    @property
    def memory_variables(self) -> list[str]:
        return ["facts", "missing"]

    def missing(self) -> list[str]:
        return [k for k in QUESTIONS if k not in self.facts]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"facts": format_facts(self.facts), "missing": self.missing()}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """inputs: {"input": реплика, "question": вопрос ассистента, "field": о каком факте он был}.

        Историю сообщений не сохраняем — только факты, прошедшие проверку.
        """
        if not self.missing():
            if self.log:
                self.log("все факты известны — извлечение не нужно")
            return
        message, field = inputs["input"], inputs.get("field") or "—"
        # В схеме — поле, о котором спросили, и недостающие короткие факты, которые называют попутно.
        # Иначе 3B-модель раскладывает перечисление по полям: stack="PostgreSQL", project="Docker"...
        fields = [k for k in self.missing() if k == field or k in INCIDENTAL]
        spec = {k: (UserFacts.model_fields[k].annotation, UserFacts.model_fields[k]) for k in fields}
        if field in fields and field not in INCIDENTAL:
            # Необязательное поле модель охотно оставляет null («Биллинг: считаем платежи...» → project=null),
            # поэтому спрошенное поле — обязательная строка с выходом «нет».
            spec[field] = (str, Field(description=UserFacts.model_fields[field].description
                                      + "; если пользователь об этом не сказал — «нет»"))
        schema = create_model("UserFacts", **spec)
        extractor = EXTRACT_PROMPT | self.llm.with_structured_output(schema, method="json_schema")
        result = extractor.invoke({"message": message, "question": inputs.get("question") or "—", "field": field})
        raw = {k: v.strip() for k, v in result.model_dump().items()
               if isinstance(v, str) and v.strip().lower() not in EMPTY}
        accepted, rejected = {}, {}
        for key, value in raw.items():
            if key in self.facts:
                rejected[key] = "уже известно"
            elif not accept_fact(key, value, message, field):
                rejected[key] = value
            else:
                accepted[key] = value
        if self.log:
            self.log(f"модель извлекла: `{json.dumps(raw, ensure_ascii=False)}`")
            if rejected:
                self.log(f"отброшено проверкой: `{json.dumps(rejected, ensure_ascii=False)}`")
        if accepted:
            self.facts.update(accepted)
            self._write()
            if self.log:
                self.log(f"записано в `{self.path}`: {', '.join(accepted)}")

    def clear(self) -> None:
        self.facts = {}
        self._write()

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _write(self) -> None:
        data = self._read_all()
        data[self.user_id] = {"facts": self.facts, "updated_at": datetime.now().isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — наставник по код-стайлу Go в команде разработки. Отвечай только по-русски, без английских "
     "и китайских слов (кроме кода), на «ты», кратко: 2–4 предложения.\n\n"
     "Что ты знаешь о пользователе (долгосрочная память):\n{facts}\n\n"
     "Задача на этот ход: {task}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])


class FactAgent:
    def __init__(self, user_id: str, path: str | Path = "memory.json", log: LogCell | None = None):
        self.log = log or (lambda text: None)
        self.memory = FileFactMemory(
            path=Path(path), user_id=user_id, log=log, llm=make_llm(0))
        self.chain = CHAT_PROMPT | make_llm(0.3)
        self._histories: dict[str, InMemoryChatMessageHistory] = {}
        self.chat_chain = RunnableWithMessageHistory(
            self.chain, self._history, input_messages_key="input", history_messages_key="history")
        self.last_question = ""
        self.asked = ""  # о каком факте был последний вопрос ассистента

    def _history(self, session_id: str) -> InMemoryChatMessageHistory:
        return self._histories.setdefault(session_id, InMemoryChatMessageHistory())

    def _task(self, start: bool) -> str:
        missing = self.memory.missing()
        if missing:
            ask = f"затем задай дословно этот вопрос и больше ничего не спрашивай: «{QUESTIONS[missing[0]]}»"
            if start and not self.memory.facts:
                return "поздоровайся и одной фразой представься; " + ask
            if start:
                return "пользователь вернулся: поприветствуй его по имени и одной фразой напомни, что уже знаешь; " + ask
            return "одной фразой отреагируй на ответ пользователя (если он что-то спросил — коротко ответь); " + ask
        guide = ("помогай с код-стайлом Go по правилам команды, учитывай уровень, проект и слабые места "
                 "пользователя. Если просят пример — сам напиши короткий пример кода на Go (до 10 строк) "
                 f"из его проекта: как плохо и как по правилам. Правила команды:\n{STYLE_GUIDE}")
        if start:
            return "пользователь вернулся, о нём известно всё нужное: поприветствуй по имени и предложи разобрать " \
                   "его слабое место. Дальше " + guide
        return guide

    def start(self) -> str:
        """Первая реплика сеанса — без ввода пользователя."""
        self.log(f"сеанс начат: user_id=`{self.memory.user_id}`, pid процесса `{os.getpid()}`")
        self.log(f"не хватает фактов: {', '.join(self.memory.missing()) or 'нет'}")
        text = self.chain.invoke({"facts": format_facts(self.memory.facts), "task": self._task(start=True),
                                  "history": [], "input": "(пользователь открыл чат)"}).content.strip()
        self._history(self.memory.user_id).add_ai_message(text)
        self.last_question = text
        self.asked = next(iter(self.memory.missing()), "")
        return text

    def turn(self, user_text: str) -> str:
        # сначала факты: следующий вопрос должен учитывать только что сказанное
        self.memory.save_context(
            {"input": user_text, "question": self.last_question, "field": self.asked}, {})
        missing = self.memory.missing()
        self.log(f"дальше: {'спросить ' + missing[0] if missing else 'режим наставника — все факты собраны'}")
        variables = self.memory.load_memory_variables({})
        text = self.chat_chain.invoke(
            {"input": user_text, "facts": variables["facts"], "task": self._task(start=False)},
            config={"configurable": {"session_id": self.memory.user_id}},
        ).content.strip()
        self.last_question = text
        self.asked = next(iter(missing), "")
        return text

    def chat(self, script: list[str] | None = None):
        """REPL. script — заранее заданные реплики (для демонстрации); без него читает input()."""
        print(f"ассистент> {self.start()}\n")
        replies = iter(script) if script is not None else None
        while True:
            if replies is None:
                user_text = input("вы> ").strip()
            else:
                user_text = next(replies, "/exit")
                print(f"вы> {user_text}")
            if user_text in ("/exit", "/quit", "выход"):
                self.log("сеанс завершён командой выхода")
                break
            if user_text:
                print(f"ассистент> {self.turn(user_text)}\n")
