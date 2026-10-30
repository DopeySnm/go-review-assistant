"""Чат-наставник по Go-код-стайлу команды с долгосрочной памятью фактов в файле.

Память — не история сообщений, а набор фактов о пользователе (имя, уровень, стек, цели...).
Факты извлекаются отдельным LLM-вызовом со structured output и хранятся в memory.json по user_id.
История сообщений живёт только в RAM текущего сеанса и после перезапуска пропадает; в промпт она
попадает не целиком: окно последних сообщений (buffer) или окно плюс резюме вытесненных (summary).

Когда факты собраны — режим наставника: классификатор реплики → TurnParser (свой Runnable: разбирает
параметры и решает, куда идти) → RunnableBranch: объяснить правило, проверить код инструментом check_code,
квиз по уровню знаний (прогресс — в memory.json), упростить объяснение, если пользователь недоволен.
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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
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
RULES = {rule: " ".join(text.split())
         for rule, text in re.findall(r"^(S\d)\. (.+?)(?=^S\d\.|\Z)", STYLE_GUIDE, re.S | re.M)}
RULE_WORDS = {  # о каком правиле речь, если номер не назван
    "S1": ("имен", "camel", "подчёрк", "подчерк", "аббревиатур"),
    "S2": ("игнор", "не проверя"),
    "S3": ("panic", "паник"),
    "S4": ("context", "ctx", "контекст"),
    "S5": ("оборач", "errorf", "%w", "ошибк"),
    "S6": ("коммент",),
}
RULE_NUMBER = r"(?<!\w)[SsСс]\s?([1-6])(?!\d)"  # S3, s3, С3 (кириллица)


def find_rule(text: str) -> Optional[str]:
    if m := re.search(RULE_NUMBER, text):
        return f"S{m.group(1)}"
    low = text.lower()
    return next((rule for rule, words in RULE_WORDS.items() if any(w in low for w in words)), None)


def lint_go(code: str) -> list[tuple[int, str, str]]:
    """Простой линтер по правилам S1–S6 на регулярках: (строка, правило, что не так)."""
    found = []
    lines = code.splitlines()
    handler = "http.ResponseWriter" in code
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("//"):
            continue
        names = re.findall(r"\b(?:func\s+(?:\([^)]*\)\s*)?|var\s+|const\s+)([A-Za-z_]\w*)", s)
        if m := re.match(r"(?:if\s+|for\s+)?([\w\s,]+?)\s*:=", s):
            names += [x.strip() for x in m.group(1).split(",")]
        for name in names:
            if "_" in name.strip("_"):
                found.append((n, "S1", f"имя `{name}` с подчёркиванием"))
            elif re.search(r"(Id|Url|Http|Json|Sql)(?=[A-Z]|$)", name):
                found.append((n, "S1", f"аббревиатура в `{name}` — целиком заглавными"))
        if re.search(r"(?:^|[\s(])_\s*(?:,\s*_\s*)?=", s) or re.search(r",\s*_\s*:?=", s):
            found.append((n, "S2", "ошибка присвоена `_`"))
        elif re.fullmatch(r"[\w.]+(?:\([^)]*\))?\.(?:Encode|Decode|Write|Exec|Scan)\(.*\)", s):
            found.append((n, "S2", "результат-ошибка вызова не проверяется"))
        if handler and re.search(r"\bpanic\(", s):
            found.append((n, "S3", "panic в HTTP-обработчике"))
        if m := re.search(r"\.(Query|QueryRow|Exec)\((?!\))", s):  # у url.Query() аргументов нет
            found.append((n, "S4", f"запрос к БД без контекста: нужен {m.group(1)}Context(ctx, ...)"))
        if re.fullmatch(r"return\s+(?:[^,]+,\s*)*err", s):
            found.append((n, "S5", "ошибка возвращается без обёртки fmt.Errorf(\"...: %w\", err)"))
        elif "fmt.Errorf(" in s and "err" in s and "%w" not in s:
            found.append((n, "S5", "в fmt.Errorf ошибка через %v, а не %w"))
        if m := re.match(r"func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)", s):
            prev = lines[n - 2].strip() if n > 1 else ""
            if not prev.startswith(f"// {m.group(1)} "):
                found.append((n, "S6", f"у экспортируемой `{m.group(1)}` нет doc-комментария, начинающегося с имени"))
    return found


@tool
def check_code(code: str) -> str:
    """Проверяет Go-код по правилам команды S1–S6 и возвращает нарушения с номерами строк."""
    return "\n".join(f"строка {n}: {rule} — {what}" for n, rule, what in lint_go(code)) or "нарушений не найдено"


# Квиз: в каждом фрагменте одно нарушение, ответ — номер правила. Проверяет ответ код, объяснение — why:
# пересказывая ответ, 3B-модель противоречила вердикту («Неверно: S5. Верно: S2…»)
QUIZ = [
    {"id": "j1", "level": "junior", "rule": "S1",
     "code": "func load_config(path string) (*Config, error) {",
     "why": "`load_config` — имя с подчёркиванием, по правилам `loadConfig`."},
    {"id": "j2", "level": "junior", "rule": "S2",
     "code": "data, _ := os.ReadFile(path)",
     "why": "ошибка из `os.ReadFile` присвоена `_`: нужно `data, err := os.ReadFile(path)` и проверить `err`."},
    {"id": "j3", "level": "junior", "rule": "S3",
     "code": "func ordersHandler(w http.ResponseWriter, r *http.Request) {\n"
             "    orders, err := repo.List(r.Context())\n    if err != nil {\n        panic(err)\n    }",
     "why": "panic в HTTP-обработчике роняет обработку запроса; нужно `http.Error(w, \"internal error\", http.StatusInternalServerError)` и `return`."},
    {"id": "m1", "level": "middle", "rule": "S4",
     "code": "// FindUser ищет пользователя по id.\nfunc (r *Repo) FindUser(id int64) (*User, error) {\n"
             "    row := r.db.QueryRow(\"SELECT name FROM users WHERE id = $1\", id)",
     "why": "`FindUser` ходит в БД, но не принимает контекст: нужно `FindUser(ctx context.Context, id int64)` и `QueryRowContext(ctx, ...)`."},
    {"id": "m2", "level": "middle", "rule": "S5",
     "code": "if err := row.Scan(&u.Name); err != nil {\n    return nil, err\n}",
     "why": "ошибка проверена, но уходит наверх без контекста: нужно `return nil, fmt.Errorf(\"scan user: %w\", err)`. S2 — это когда ошибку не проверяют вовсе, а здесь её проверяют."},
    {"id": "m3", "level": "middle", "rule": "S6",
     "code": "func ParseOrder(b []byte) (*Order, error) {",
     "why": "у экспортируемой `ParseOrder` нет doc-комментария; нужен `// ParseOrder разбирает заказ из JSON.`"},
    {"id": "s1", "level": "senior", "rule": "S1",
     "code": "// GetUserId возвращает id текущего пользователя.\n"
             "func (s *Service) GetUserId(ctx context.Context) (int64, error) {",
     "why": "аббревиатура пишется целиком: `GetUserID`, а не `GetUserId`."},
    {"id": "s2", "level": "senior", "rule": "S5",
     "code": "return nil, fmt.Errorf(\"load order %d: %v\", id, err)",
     "why": "с `%v` ошибка превращается в текст и `errors.Is`/`errors.As` перестают её видеть; нужен `%w`."},
    {"id": "s3", "level": "senior", "rule": "S6",
     "code": "// возвращает заказы клиента\nfunc ListOrders(ctx context.Context, clientID int64) ([]Order, error) {",
     "why": "комментарий есть, но не начинается с имени функции: нужно `// ListOrders возвращает заказы клиента.`"},
    {"id": "s4", "level": "senior", "rule": "S2",
     "code": "func statsHandler(w http.ResponseWriter, r *http.Request) {\n"
             "    json.NewEncoder(w).Encode(collectStats())\n}",
     "why": "`Encode` возвращает ошибку, а её не проверяют: `if err := json.NewEncoder(w).Encode(...); err != nil { http.Error(...) }`."},
]
LEVELS = ["junior", "middle", "senior"]
DEPTH = {
    "junior": "объясняй простыми словами, без лишних терминов; обязательно короткий пример кода «плохо / хорошо»",
    "middle": "объясни правило и зачем оно нужно, дай короткий пример «плохо / хорошо»",
    "senior": "коротко: суть правила, неочевидные случаи и исключения; пример — только если он показывает тонкость",
}


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
    """Долгосрочная память фактов в JSON-файле: {user_id: {"facts": {...}, "progress": {...}, "updated_at": ...}}.

    load_memory_variables() — отдаёт известные факты и список недостающих;
    save_context() — извлекает факты из реплики пользователя отдельным LLM-вызовом и пишет файл;
    progress — уровень знаний по квизу и ошибки по правилам (меняется в режиме наставника, save_progress()).
    """

    path: Path
    user_id: str
    llm: Any = None  # модель для извлечения фактов
    log: Any = None
    facts: dict = Field(default_factory=dict)
    progress: dict = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        profile = self._read_all().get(self.user_id)
        self.facts = dict(profile["facts"]) if profile else {}
        self.progress = dict(profile.get("progress", {})) if profile else {}
        if self.log:
            if not self.path.exists():
                self.log(f"`{self.path}` нет — память пустая")
            elif profile:
                self.log(f"прочитан `{self.path}`, профиль `{self.user_id}`: "
                         f"`{json.dumps(self.facts, ensure_ascii=False)}`")
                if self.progress:
                    self.log(f"прогресс: `{json.dumps(self.progress, ensure_ascii=False)}`")
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

    def level(self) -> str:
        """Уровень для сложности тем и вопросов: из квиза, а до первого квиза — названный пользователем."""
        return self.progress.get("level") or self.facts.get("go_level") or "junior"

    def save_progress(self, **changes) -> None:
        self.progress.update(changes)
        self._write()
        if self.log:
            self.log(f"прогресс записан в `{self.path}`: `{json.dumps(self.progress, ensure_ascii=False)}`")

    def clear(self) -> None:
        self.facts, self.progress = {}, {}
        self._write()

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _write(self) -> None:
        data = self._read_all()
        data[self.user_id] = {"facts": self.facts, "updated_at": datetime.now().isoformat(timespec="seconds")}
        if self.progress:
            data[self.user_id]["progress"] = self.progress
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")




SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Перескажи реплики разговора наставника по Go с разработчиком одним предложением, которое начинается "
     "со слов «Пользователь»: что он сообщил или о чём спросил. Сохрани имена людей и названия. "
     "Объяснения наставника и код не пересказывай. Только то, что есть в репликах, по-русски."),
    ("human", "{lines}"),
])


class DialogHistory(BaseChatMessageHistory):
    """История сообщений сеанса (только RAM); в промпт попадает не вся.

    buffer  — последние window сообщений, остальное просто отбрасывается;
    summary — те же window сообщений плюс резюме вытесненных: выпавшие из окна реплики отдельный LLM-вызов
              пересказывает одним-двумя предложениями, пункт добавляется в резюме (оно идёт в системный промпт).
              Пункты копятся, а не переписываются: переписывая резюме целиком, 3B-модель теряла старое.
    """

    def __init__(self, mode: str = "summary", window: int = 6, llm: Any = None, log: Any = None):
        assert mode in ("buffer", "summary")
        self.mode, self.window, self.llm = mode, window, llm
        self.log = log or (lambda text: None)
        self.full: list[BaseMessage] = []
        self.points: list[str] = []  # резюме — по пункту на свёрнутый кусок
        self.summarized = 0  # сколько первых сообщений уже свёрнуто в резюме

    @property
    def messages(self) -> list[BaseMessage]:
        # token_counter=len — считаем сообщения, а не токены
        return trim_messages(self.full, strategy="last", token_counter=len, max_tokens=self.window,
                             start_on="human")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.full.extend(messages)
        dropped = len(self.full) - len(self.messages)
        if self.mode == "summary" and dropped > self.summarized:
            chunk = self.full[self.summarized:dropped]
            # одно приветствие не сворачиваем: из него модель сочиняла целый диалог
            if any(m.type == "human" for m in chunk):
                # ответы наставника обрезаем: иначе 3B-модель копирует их в резюме вместо пересказа
                lines = "\n".join(f"Пользователь: {m.content}" if m.type == "human" else
                                   f"Наставник: {m.content[:150]}…" for m in chunk)
                lines = re.sub(r"```.*?(```|$)", "[код]", lines, flags=re.S)
                point = (SUMMARY_PROMPT | self.llm).invoke({"lines": lines}).content.strip()
                self.points.append(re.sub(r"^(новое )?резюме:\s*", "", point, flags=re.I))
                self.log(f"резюме: свёрнуто сообщений {len(chunk)} → «{self.points[-1]}»")
            self.summarized = dropped

    def prompt_summary(self) -> str:
        if self.mode == "summary" and self.points:
            return "Краткое содержание более раннего разговора:\n" + "\n".join(f"- {p}" for p in self.points) + "\n\n"
        return ""

    def clear(self) -> None:
        self.full, self.points, self.summarized = [], [], 0


CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — наставник по код-стайлу Go в команде разработки. Отвечай только по-русски, без английских "
     "и китайских слов (кроме кода), на «ты», кратко: 2–4 предложения.\n\n"
     "Что ты знаешь о пользователе (долгосрочная память):\n{facts}\n\n"
     "{summary}Задача на этот ход: {task}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])


class Route(BaseModel):
    """Класс реплики пользователя в режиме наставника."""

    intent: Literal["explain_rule", "review_code", "quiz", "other"] = Field(
        description="explain_rule — просит объяснить правило или тему код-стайла, показать пример; "
                    "review_code — прислал код на Go или просит проверить код; "
                    "quiz — просит проверить его знания, задать вопрос, устроить тест; "
                    "other — всё остальное")
    rule: Literal["S1", "S2", "S3", "S4", "S5", "S6", "нет"] = Field(
        description="О каком правиле речь; «нет», если ни о каком")
    # поле mood («недоволен») убрано: в первом прогоне модель ставила его почти на каждую реплику


ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Определи, чего хочет пользователь наставника по код-стайлу Go, и верни JSON по схеме.\n"
               f"Правила команды:\n{STYLE_GUIDE}"),
    ("human", "{input}"),
])

NEGATIVE = re.compile(r"не понял|не понимаю|непонятн|не ясно|запутал|слишком сложно|бред|ерунд|\bне то\b"
                      r"|вс[её] равно|опять", re.I)
QUIZ_WORDS = re.compile(r"квиз|тест|проверь меня|проверь мои знания|задай.{0,10}вопрос|ещё вопрос|еще вопрос", re.I)
MORE_WORDS = re.compile(r"^(ещё|еще|дальше|давай|следующий)\b", re.I)
EXPLAIN_WORDS = re.compile(r"объясн|расскаж|покажи|пример|что такое|зачем", re.I)
CODE_BLOCK = re.compile(r"```(?:go)?\s*\n(.*?)```", re.S)


def similar(a: str, b: str) -> bool:
    """Повтор той же просьбы: больше половины слов совпадает."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa) >= 3 and len(wa & wb) / len(wa | wb) >= 0.5


class TurnParser(Runnable[dict, dict]):
    """Свой компонент цепочки: разбирает реплику и ответ классификатора в параметры хода.

    На входе {"input", "route": Route, ...}, на выходе тот же словарь плюс intent, rule, code, reason.
    3B-модель ошибается в классе («Объясни правило S3» → review_code), поэтому явные признаки важнее её
    ответа: блок кода → проверка кода (без кода проверять нечего), ожидается ответ на вопрос квиза → проверка
    ответа, «не понял» или повтор просьбы → упростить, номер правила → объяснить.
    """

    def __init__(self, agent: "FactAgent"):
        self.agent = agent

    def invoke(self, input: dict, config=None, **kwargs) -> dict:
        agent, text, route = self.agent, input["input"], input["route"]
        intent, reasons = route.intent, []
        code = m.group(1) if (m := CODE_BLOCK.search(text)) else (
            text if "\n" in text and re.search(r"\bfunc\b|:=|\bif err\b", text) else "")
        if code:
            intent = "review_code"
        elif agent.pending_quiz and (re.search(r"(?<!\w)[SsСс]?\s?[1-6](?!\d)", text)
                                     or re.search(r"не знаю", text, re.I)):
            intent = "quiz_answer"
        elif QUIZ_WORDS.search(text) or (agent.last_intent in ("quiz", "quiz_answer") and MORE_WORDS.search(text)):
            intent = "quiz"
        elif intent == "review_code" or (intent == "other" and re.search(RULE_NUMBER, text)):
            intent = "explain_rule"
        if intent == "explain_rule" and not (find_rule(text) or EXPLAIN_WORDS.search(text)):
            intent = "other"  # «Кто у нас делает ревью?» — не про правило, хоть модель и назвала S1

        negative = bool(NEGATIVE.search(text))
        repeated = similar(text, agent.last_user)
        if negative:
            reasons.append("в реплике недовольство или непонимание")
        if repeated:
            reasons.append("повторяет прошлую просьбу")
        if reasons and intent in ("explain_rule", "other"):
            intent = "simplify"

        weak = find_rule(agent.memory.facts.get("weak_spots", ""))
        if intent == "simplify":  # упрощаем то, что объясняли, а не правило, названное в жалобе
            rule = agent.last_rule or find_rule(text) or weak
        else:
            rule = find_rule(text) or (route.rule if route.rule != "нет" else None)
            if intent == "other" and rule and EXPLAIN_WORDS.search(text):
                intent = "explain_rule"
            if intent == "explain_rule" and not rule:
                rule = weak or agent.last_rule or "S1"
        agent.frustration = agent.frustration + 1 if intent == "simplify" else 0
        if intent != "simplify":
            agent.quiz_context = None
        agent.last_intent = intent
        if intent in ("explain_rule", "simplify") and rule:
            agent.last_rule = rule
        agent.log(f"классификатор: `{route.model_dump_json()}` → парсер: intent=`{intent}`, rule=`{rule}`"
                  + (f", причина: {'; '.join(reasons)}" if reasons else ""))
        return {**input, "intent": intent, "rule": rule, "code": code, "reason": "; ".join(reasons)}


class FactAgent:
    def __init__(self, user_id: str, path: str | Path = "memory.json", log: LogCell | None = None,
                 dialog: str = "summary", window: int = 6):
        self.log = log or (lambda text: None)
        self.memory = FileFactMemory(
            path=Path(path), user_id=user_id, log=log, llm=make_llm(0))
        self.history = DialogHistory(dialog, window, llm=make_llm(0), log=self.log)
        self.chain = CHAT_PROMPT | make_llm(0.3)
        self.chat_chain = RunnableWithMessageHistory(
            self.chain, self._history, input_messages_key="input", history_messages_key="history")
        self.classifier = ROUTE_PROMPT | make_llm(0).with_structured_output(Route, method="json_schema")
        mentor = (
            RunnablePassthrough.assign(route=self.classifier)
            | TurnParser(self)
            | RunnableBranch(
                (lambda x: x["intent"] == "review_code", RunnableLambda(self._review)),
                (lambda x: x["intent"] == "quiz_answer", RunnableLambda(self._grade)),
                (lambda x: x["intent"] == "quiz", RunnableLambda(self._quiz)),
                (lambda x: x["intent"] == "simplify", RunnableLambda(self._simplify)),
                (lambda x: x["intent"] == "explain_rule", RunnableLambda(self._explain)),
                RunnableLambda(self._other),
            )
        )
        self.mentor_chain = RunnableWithMessageHistory(
            mentor, self._history, input_messages_key="input", history_messages_key="history")
        self.last_question = ""
        self.asked = ""  # о каком факте был последний вопрос ассистента
        # состояние режима наставника (RAM сеанса)
        self.pending_quiz: dict | None = None
        self.last_intent = self.last_rule = self.last_user = ""
        self.frustration = 0  # сколько ходов подряд пользователь недоволен
        self.quiz_context: dict | None = None  # неверный ответ на квиз — о чём упрощать
        self.log(f"память диалога: `{dialog}`, окно {window} сообщений")

    def _history(self, session_id: str) -> DialogHistory:
        return self.history

    def _task(self, start: bool) -> str:
        missing = self.memory.missing()
        if missing:
            ask = f"затем задай дословно этот вопрос и больше ничего не спрашивай: «{QUESTIONS[missing[0]]}»"
            if start and not self.memory.facts:
                return "поздоровайся и одной фразой представься; " + ask
            if start:
                return "пользователь вернулся: поприветствуй его по имени и одной фразой напомни, что уже знаешь; " + ask
            return "одной фразой отреагируй на ответ пользователя (если он что-то спросил — коротко ответь); " + ask
        offer = ("предложи на выбор: объяснить любое правило команды S1–S6, проверить его код "
                 "(пусть пришлёт фрагмент) или задать вопросы на знание правил")
        if start:
            weak = find_rule(self.memory.facts.get("weak_spots", ""))
            about = f" (к нему относится правило {weak}: «{RULES[weak]}»)" if weak else ""
            return ("пользователь вернулся, о нём известно всё нужное: поприветствуй по имени, "
                    f"напомни про его слабое место{about} и " + offer)
        return "поблагодари за ответы, одной фразой подытожь, что узнал о пользователе, и " + offer

    def _say(self, x: dict, task: str) -> AIMessage:
        return self.chain.invoke({**x, "task": task})

    def _note_mistakes(self, rules: list[str]) -> None:
        mistakes = dict(self.memory.progress.get("mistakes", {}))
        for rule in rules:
            mistakes[rule] = mistakes.get(rule, 0) + 1
        self.memory.save_progress(mistakes=mistakes)

    # --- ветки режима наставника ---

    def _explain(self, x: dict) -> AIMessage:
        level, rule = self.memory.level(), x["rule"]
        self.log(f"ветка «объяснить»: {rule}, глубина по уровню `{level}`")
        return self._say(x, f"объясни правило команды {rule}: «{RULES[rule]}». Уровень пользователя — {level}: "
                            f"{DEPTH[level]}. Пример кода (до 10 строк) — из его проекта.")

    def _review(self, x: dict) -> AIMessage:
        report = check_code.invoke({"code": x["code"]})  # инструмент: точный список нарушений
        rules = sorted({rule for _, rule, _ in lint_go(x["code"])})
        self.log(f"ветка «проверить код»: инструмент `check_code` → {', '.join(rules) or 'чисто'}")
        if rules:
            self._note_mistakes(rules)
            x["rule"] = rules[0]
        if not rules:
            return AIMessage(content="Инструмент check_code нарушений не нашёл — по правилам S1–S6 код чистый.")
        level = self.memory.level()
        reply = self._say(x, "пользователь прислал код. Инструмент check_code нашёл нарушения:\n"
                             f"{report}\nОбъясни каждое из них по одному предложению ({DEPTH[level]}) и покажи "
                             "исправленный код, где исправлены все они. Не называй нарушений, которых нет в списке.")
        # 3B-модель пересказывала отчёт не полностью (из S1, S4, S5 упомянула только S4) — список даём дословно
        return AIMessage(content=f"Инструмент check_code:\n{report}\n\n{reply.content.strip()}")

    def _quiz(self, x: dict) -> AIMessage:
        level, progress = self.memory.level(), self.memory.progress
        asked = progress.get("asked", [])
        candidates = [q for q in QUIZ if q["level"] == level and q["id"] not in asked]
        if not candidates:  # все вопросы уровня заданы — начинаем круг заново
            asked = [a for a in asked if not a.startswith(level[0])]
            candidates = [q for q in QUIZ if q["level"] == level]
        # сначала правила, на которых пользователь ошибался, и его слабое место
        weak, mistakes = find_rule(self.memory.facts.get("weak_spots", "")), progress.get("mistakes", {})
        q = max(candidates, key=lambda q: mistakes.get(q["rule"], 0) + (q["rule"] == weak))
        self.pending_quiz = q
        self.memory.save_progress(asked=asked + [q["id"]])
        self.log(f"ветка «квиз»: уровень `{level}`, вопрос `{q['id']}` (ответ {q['rule']})")
        return AIMessage(content=f"Вопрос уровня {level}. Какое правило (S1–S6) нарушено?\n```go\n{q['code']}\n```")

    def _grade(self, x: dict) -> AIMessage:
        q, self.pending_quiz = self.pending_quiz, None
        m = re.search(r"(?<!\w)[SsСс]?\s?([1-6])(?!\d)", x["input"])
        answer = f"S{m.group(1)}" if m else "не знаю"
        ok = answer == q["rule"]
        p = self.memory.progress
        level = self.memory.level()
        streak = max(p.get("streak", 0), 0) + 1 if ok else min(p.get("streak", 0), 0) - 1
        new_level = level
        if streak >= 2 and level != LEVELS[-1]:  # два верных подряд — вопросы сложнее
            new_level, streak = LEVELS[LEVELS.index(level) + 1], 0
        elif streak <= -2 and level != LEVELS[0]:  # две ошибки подряд — проще
            new_level, streak = LEVELS[LEVELS.index(level) - 1], 0
        mistakes = dict(p.get("mistakes", {}))
        if not ok:
            mistakes[q["rule"]] = mistakes.get(q["rule"], 0) + 1
        self.memory.save_progress(level=new_level, streak=streak, correct=p.get("correct", 0) + ok,
                                  wrong=p.get("wrong", 0) + (not ok), mistakes=mistakes)
        self.log(f"ветка «ответ на квиз»: ответ {answer}, верно {q['rule']} → {'верно' if ok else 'неверно'}"
                 + (f"; уровень `{level}` → `{new_level}`" if new_level != level else ""))
        self.last_rule = q["rule"]
        if not ok:
            self.quiz_context = {"q": q, "answer": answer}
        text = f"{'Верно' if ok else 'Неверно'}: здесь нарушено {q['rule']} — {q['why']}"
        if new_level != level:
            text += f" Дальше вопросы уровня {new_level}."
        return AIMessage(content=text)

    def _simplify(self, x: dict) -> AIMessage:
        rule = x["rule"]
        self.log(f"ветка «упростить»: {rule}, недоволен {self.frustration} ход(а) подряд")
        task = f"пользователь недоволен ({x['reason']}). Не оправдывайся. "
        if qc := self.quiz_context:
            task += (f"Он ответил на вопрос квиза {qc['answer']}, а верно {rule}. Код:\n{qc['q']['code']}\n"
                     f"Почему {rule}: {qc['q']['why']}\nПравило {rule}: «{RULES[rule]}». Объясни это ещё проще, "
                     f"своими словами, с бытовой аналогией; называй правило только как {rule}.")
        else:
            task += (f"Объясни правило {rule}: «{RULES[rule]}» совсем просто: одна бытовая аналогия и пример кода "
                     "3–6 строк «плохо / хорошо».")
        reply = self._say(x, task).content.strip()
        if self.frustration >= 2:  # просьбу в промпте 3B-модель пропускала — добавляем сами
            reply += "\n\nЕсли так и не стало понятнее — пришли свой фрагмент кода, разберём правило на нём."
        return AIMessage(content=reply)

    def _other(self, x: dict) -> AIMessage:
        self.log("ветка «другое»")
        return self._say(x, "если пользователь спрашивает о том, что говорил раньше, — ответь по истории и "
                            "краткому содержанию разговора, назови людей и детали оттуда. Если вопрос не про Go и не "
                            "про этот разговор — ты на него не отвечаешь и уточнений не просишь: одной фразой скажи, "
                            "что ты наставник только по код-стайлу Go, и предложи: объяснить правило, проверить код "
                            "или вопросы на знание правил.")

    # --- сеанс ---

    def start(self) -> str:
        """Первая реплика сеанса — без ввода пользователя."""
        self.log(f"сеанс начат: user_id=`{self.memory.user_id}`, pid процесса `{os.getpid()}`")
        self.log(f"не хватает фактов: {', '.join(self.memory.missing()) or 'нет'}")
        text = self.chain.invoke({"facts": format_facts(self.memory.facts), "task": self._task(start=True),
                                  "summary": "", "history": [], "input": "(пользователь открыл чат)"}
                                 ).content.strip()
        self.history.add_ai_message(text)
        self.last_question = text
        self.asked = next(iter(self.memory.missing()), "")
        return text

    def turn(self, user_text: str) -> str:
        config = {"configurable": {"session_id": self.memory.user_id}}
        if self.memory.missing() or self.asked:
            # знакомство: сначала факты — следующий вопрос должен учитывать только что сказанное
            self.memory.save_context(
                {"input": user_text, "question": self.last_question, "field": self.asked}, {})
            missing = self.memory.missing()
            self.log(f"дальше: {'спросить ' + missing[0] if missing else 'режим наставника — все факты собраны'}")
            variables = self.memory.load_memory_variables({})
            text = self.chat_chain.invoke(
                {"input": user_text, "facts": variables["facts"], "summary": self.history.prompt_summary(),
                 "task": self._task(start=False)}, config=config).content.strip()
            self.asked = next(iter(missing), "")
        else:
            self.log(f"история: в промпт {len(self.history.messages)} из {len(self.history.full)} сообщений"
                     + (", плюс резюме" if self.history.prompt_summary() else ""))
            reply = self.mentor_chain.invoke(
                {"input": user_text, "facts": format_facts(self.memory.facts),
                 "summary": self.history.prompt_summary()}, config=config)
            text = reply.content.strip()
            self.last_user = user_text
        self.last_question = text
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
