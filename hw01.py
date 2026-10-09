"""ДЗ 01: консольный чат и ассистент для проверки Go-кода по код-стайлу команды.

Вызовы функций парсятся прямо из текста ответа модели (без function calling / MCP).
Клиент работает с любым OpenAI-совместимым API: llama-server локально или OpenRouter.

Примеры:
    python hw01.py --prompt chat                      # REPL-чат с локальной моделью
    python hw01.py --prompt v1                        # REPL-агент
    python hw01.py --prompt v4 --batch --runs 3       # прогон 10 тестовых вопросов, лог в logs/
    python hw01.py --provider openrouter --model nvidia/nemotron-3-super-120b-a12b:free --prompt v4
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

PROVIDERS = {
    "local": "http://127.0.0.1:8080/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

MAX_STEPS = 6  # сколько раз подряд агент может вызвать функции в одном ответе
RESULT_MARKER = "РЕЗУЛЬТАТ"


# ---------------------------------------------------------------------------
# Данные: захардкоженные «инструменты». Ошибки в коде заложены специально.
# ---------------------------------------------------------------------------

STYLE_GUIDE = """\
S1. Имена — MixedCaps/camelCase, без подчёркиваний. Аббревиатуры целиком: userID, не userId.
S2. Ошибки не игнорировать: ни `_ = f()`, ни вызов без проверки возвращаемой ошибки. `defer x.Close()` допускается.
S3. В HTTP-обработчиках не использовать panic — вернуть 500 через http.Error.
S4. Функции, которые ходят в БД или сеть, принимают ctx context.Context первым параметром.
    HTTP-обработчики берут контекст из r.Context() — отдельный параметр ctx им не нужен.
S5. Ошибки, возвращаемые наверх, оборачиваются: fmt.Errorf("контекст: %w", err).
S6. У экспортируемых функций и методов есть doc-комментарий, начинающийся с имени."""

CODE = {
    "handler.go": """\
package api

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// Handler обслуживает HTTP-запросы.
type Handler struct {
	repo *Repo
}

func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
	user_id, err := strconv.Atoi(r.URL.Query().Get("id"))
	if err != nil {
		http.Error(w, "bad id", http.StatusBadRequest)
		return
	}
	user, err := h.repo.GetUser(user_id)
	if err != nil {
		panic(err)
	}
	json.NewEncoder(w).Encode(user)
}

// ListOrders возвращает список заказов.
func (h *Handler) ListOrders(w http.ResponseWriter, r *http.Request) {
	orders, err := h.repo.ListOrders(r.Context())
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(orders); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
	}
}""",
    "repo.go": """\
package api

import (
	"context"
	"database/sql"
	"fmt"
)

// Repo даёт доступ к базе данных.
type Repo struct {
	db *sql.DB
}

// GetUser загружает пользователя по id.
func (r *Repo) GetUser(id int) (*User, error) {
	var u User
	err := r.db.QueryRow("SELECT id, name FROM users WHERE id = $1", id).Scan(&u.ID, &u.Name)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// ListOrders возвращает все заказы.
func (r *Repo) ListOrders(ctx context.Context) ([]Order, error) {
	rows, err := r.db.QueryContext(ctx, "SELECT id, total FROM orders")
	if err != nil {
		return nil, fmt.Errorf("list orders: %w", err)
	}
	defer rows.Close()
	var orders []Order
	for rows.Next() {
		var o Order
		if err := rows.Scan(&o.ID, &o.Total); err != nil {
			return nil, fmt.Errorf("scan order: %w", err)
		}
		orders = append(orders, o)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate orders: %w", err)
	}
	return orders, nil
}

func (r *Repo) delete_order(ctx context.Context, id int) error {
	_, err := r.db.ExecContext(ctx, "DELETE FROM orders WHERE id = $1", id)
	return err
}""",
}

# Отчёт специально неполный: 3 нарушения из 8.
LINTER_REPORT = """\
handler.go:15:2: var-naming: don't use underscores in Go names; var user_id should be userID (revive)
handler.go:24:22: Error return value of `(*json.Encoder).Encode` is not checked (errcheck)
repo.go:45:16: var-naming: don't use underscores in Go names; method delete_order should be deleteOrder (revive)"""


def get_style_guide() -> str:
    return STYLE_GUIDE


def get_code(file: str = "") -> str:
    name = os.path.basename(str(file).strip().strip("\"'`"))
    if name not in CODE:
        return f"ОШИБКА: файл {name or '(не указан)'} не найден. Доступные файлы: {', '.join(CODE)}"
    lines = CODE[name].splitlines()
    return "\n".join(f"{i:2}| {line}" for i, line in enumerate(lines, 1))


def get_linter_report() -> str:
    return LINTER_REPORT


TOOLS = {
    "get_style_guide": get_style_guide,
    "get_code": get_code,
    "get_linter_report": get_linter_report,
}


# ---------------------------------------------------------------------------
# Тестовые вопросы для п. 4.1 / 4.2. expected в модель не уходит — только в лог.
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("Проверь handler.go.",
     "4 нарушения: h:14 нет doc-комментария (S6), h:15 user_id (S1), h:22 panic (S3), h:24 ошибка Encode не проверена (S2)"),
    ("Есть ли в коде имена не в camelCase?",
     "да: user_id (handler.go:15), delete_order (repo.go:45)"),
    ("Где проигнорированы ошибки?",
     "только handler.go:24 (Encode). `_, err :=` в repo.go:46 — НЕ игнорирование"),
    ("Можно ли вызывать panic в обработчике? Есть ли такое у нас?",
     "нельзя (S3); есть в handler.go:22"),
    ("Все ли функции работы с БД принимают context?",
     "нет: Repo.GetUser (repo.go:15)"),
    ("Правильно ли оборачиваются ошибки в repo.go?",
     "нет: repo.go:19 и repo.go:47; в ListOrders всё обёрнуто"),
    ("Проверь метод ListOrders в handler.go.",
     "нарушений нет"),
    ("Что нашёл линтер? Он нашёл всё?",
     "3 из 8; пропустил panic h:22, doc h:14, ctx r:15, обёртку r:19 и r:47"),
    ("Проверь файл main.go.",
     "файла нет — ничего не выдумывать"),
    ("Сколько всего нарушений и какое правило нарушено чаще всего?",
     "8; чаще всего S1 и S5 — по 2 раза"),
]


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

PROMPT_CHAT = "Ты полезный ассистент. Отвечай по-русски, кратко и по делу."

PROMPT_V1 = """\
Ты — ассистент для ревью Go-кода. Ты проверяешь, соответствует ли код команды её код-стайлу.

Доступные функции:
- get_style_guide() — правила код-стайла команды.
- get_code(file) — исходный код файла с номерами строк. Файлы: handler.go, repo.go.
- get_linter_report() — отчёт линтера.

Чтобы вызвать функцию, напиши строку:
CALL имя_функции {"аргумент": "значение"}

Результат придёт следующим сообщением. Затем ответь пользователю."""

PROMPT_V2 = """\
Ты — ассистент для ревью Go-кода. Ты проверяешь код команды на соответствие её код-стайлу.

Функции:
- get_style_guide() — правила S1–S6 код-стайла команды.
- get_code(file) — код файла с номерами строк. Файлы: handler.go, repo.go.
- get_linter_report() — отчёт линтера.

Формат вызова — одна строка, без другого текста:
CALL get_code {"file": "handler.go"}

Шаги:
1. Вызови get_style_guide.
2. Вызови get_code для нужных файлов.
3. Сравни код с правилами и ответь.

Правила:
- Отвечай только по данным из функций. Не выдумывай правила и код.
- Отвечай по-русски.

Пример:
Пользователь: Есть ли panic в handler.go?
Ассистент: CALL get_style_guide {}
Ассистент: CALL get_code {"file": "handler.go"}
Ассистент: Да, handler.go:22 — panic(err) в обработчике, нарушено правило S3."""

PROMPT_V3 = """\
Ты — ассистент для ревью Go-кода. Ты проверяешь код команды на соответствие её код-стайлу.

Функции:
- get_style_guide() — правила S1–S6.
- get_code(file) — код файла с номерами строк. Есть ТОЛЬКО handler.go и repo.go.
- get_linter_report() — отчёт линтера. Линтер находит НЕ все нарушения.

Как вызвать функцию: напиши строку CALL и закончи сообщение.
CALL get_code {"file": "repo.go"}
НЕ пиши результат функции сам — его пришлёт система в сообщении РЕЗУЛЬТАТ.

Порядок работы:
1. Сначала CALL get_style_guide {}.
2. Потом CALL get_code для каждого файла, о котором вопрос. Если файл не указан — для обоих.
3. Читай код сам. Отчёт линтера — только подсказка.
4. Ответ — список, по одной строке на нарушение:
   файл:строка | правило | что не так
   Если нарушений нет — напиши «Нарушений нет».

Запрещено: выдумывать строки кода, файлы и правила, которых нет в результатах функций."""

PROMPT_V4 = """\
Ты — ревьюер Go-кода. Ты сверяешь код с правилами команды и находишь нарушения.

ФУНКЦИИ
get_style_guide() — правила S1–S6.
get_code(file) — код с номерами строк. Существуют ТОЛЬКО два файла: handler.go, repo.go.
get_linter_report() — отчёт линтера. Линтер пропускает многие нарушения, ему нельзя доверять.

ВЫЗОВ — одна строка, после неё сразу закончи сообщение:
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
Результат придёт в сообщении «РЕЗУЛЬТАТ». Никогда не пиши РЕЗУЛЬТАТ сам.

ПОРЯДОК
1. CALL get_style_guide {}
2. CALL get_code для файла из вопроса. Если файл не назван — для handler.go, потом для repo.go.
3. Проверь КАЖДОЕ правило S1–S6 по коду. Как искать:
   S1 имена: ищи «_» в именах переменных и функций (user_id, delete_order). `_,` слева от := — это не имя.
   S2 ошибки: вызов, который возвращает error, но результат не проверен (`json.NewEncoder(w).Encode(x)` без if).
      `_, err :=` с последующей проверкой или `return err` — НЕ игнорирование.
   S3 panic: слово panic внутри функции с (w http.ResponseWriter, r *http.Request).
   S4 context: функция вызывает r.db.*, а среди параметров нет ctx context.Context.
   S5 обёртка: строка `return ..., err` или `return err` без fmt.Errorf.
   S6 doc: у функции с большой буквы нет строки // перед func.
4. Ответь списком, по одной строке на нарушение, номер строки бери из кода:
   файл:строка | правило | что не так
   Нарушений нет — «Нарушений нет». Файла нет — «Файл не найден».

ПРИМЕРЫ (код в примерах выдуманный, номера строк бери только из РЕЗУЛЬТАТ)
Вопрос: Проверь функцию Save в store.go.
Ответ 1: CALL get_style_guide {}
Ответ 2: CALL get_code {"file": "store.go"}
Ответ 3:
store.go:7 | S1 | имя переменной item_id с подчёркиванием
store.go:12 | S5 | return err без fmt.Errorf

Вопрос: Проверь функцию Load в store.go.
Ответ 1: CALL get_style_guide {}
Ответ 2: CALL get_code {"file": "store.go"}
Ответ 3: Нарушений нет.

Отвечай по-русски. Не выдумывай код, строки и правила."""

PROMPT_V5 = """\
Ты — ревьюер Go-кода. Отвечай по-русски, кратко.

ФУНКЦИИ
get_style_guide() — правила S1–S6.
get_code(file) — код с номерами строк. Есть только два файла: handler.go и repo.go.
get_linter_report() — отчёт линтера. Он находит не все нарушения.

ШАГ 1. Первым сообщением напиши только вызовы, каждый на своей строке:
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
- В вопросе назван handler.go или repo.go — вызови get_code только для него.
- Вопрос про линтер — добавь строку CALL get_linter_report {}.
- Назван другой файл — вызови get_code для него, система ответит, что файла нет.

ШАГ 2. Найди в вопросе слово и проверяй ТОЛЬКО его правило:
имена, camelCase, подчёркивание → S1
проигнорированы ошибки, не проверена ошибка → S2
panic → S3
context, ctx → S4
оборачивание ошибок, fmt.Errorf → S5
комментарий, документация → S6
«проверь файл», «проверь метод», «сколько нарушений», линтер → все правила S1–S6

Как найти нарушение:
S1: в имени переменной или функции есть «_», например user_id. Одиночный `_,` — не имя.
S2: вызов возвращает ошибку, но результат никуда не присвоен и не проверен. `_, err :=`, `defer x.Close()` и http.Error — НЕ нарушения.
S3: в строке есть panic(.
S4: функция обращается к r.db, а в её параметрах нет ctx context.Context.
S5: строка `return nil, err` или `return err` без fmt.Errorf.
S6: строка `func` с заглавной буквой в имени, а строкой выше нет комментария //.

ШАГ 3. Ответ — одна строка на нарушение:
файл:строка | правило | `код этой строки, скопированный из РЕЗУЛЬТАТ`
- Номер и код строки копируй из РЕЗУЛЬТАТ. Не можешь процитировать строку — нарушения нет.
- Большинство правил в коде НЕ нарушено. Не пиши строку для каждого правила.
- Нарушений нет — напиши «Нарушений нет».
- Файла нет — напиши «Файл не найден».
- Вопрос про линтер — сначала перечисли строки из отчёта линтера, потом «Линтер пропустил:» и нарушения, которых нет в отчёте.
- Вопрос «сколько» — в конце напиши «Всего: N»."""

PROMPT_V6 = """\
Ты — ревьюер Go-кода. Отвечай по-русски.

Функции: get_style_guide(), get_code(file), get_linter_report(). Файлы: handler.go и repo.go.

Первое сообщение ВСЕГДА одно и то же — эти четыре строки и ничего больше:
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
CALL get_linter_report {}
Если в вопросе назван другой файл .go — допиши пятую строку: CALL get_code {"file": "<этот файл>"}

Система пришлёт РЕЗУЛЬТАТ. После этого ответь.

Проверяй ТОЛЬКО правило, о котором вопрос:
имена, camelCase → S1: «_» внутри имени (user_id)
проигнорированы ошибки → S2: результат вызова с ошибкой никто не проверяет
panic → S3: строка с panic(
context → S4: функция с r.db, у которой нет ctx в параметрах
оборачивание ошибок → S5: строка return err или return nil, err
комментарии → S6: func с заглавной буквы, а строкой выше нет //
проверь файл, проверь метод, сколько нарушений, линтер → все правила, но только в этом файле или методе

Формат — одна строка на нарушение:
файл:строка | правило | `код этой строки из РЕЗУЛЬТАТ`
Пиши только строки, которые есть в РЕЗУЛЬТАТ. Обычно нарушений 0–3, а не по одному на каждое правило.
Если после проверки нужных строк нарушений нет — «Нарушений нет».
Если в РЕЗУЛЬТАТ для файла ОШИБКА — «Файл не найден».
Вопрос про линтер — две части: «Линтер нашёл:» (строки отчёта) и «Линтер пропустил:» (нарушения из кода, которых нет в отчёте).
Вопрос «сколько» — в конце «Всего: N».

Пример
Вопрос: Есть ли panic в repo.go?
Ответ после РЕЗУЛЬТАТ: правило S3, ищу panic( в repo.go — такой строки нет. Нарушений нет."""

PROMPT_V7 = """\
Ты — ревьюер Go-кода. Ты сверяешь код с правилами команды и находишь нарушения.

ФУНКЦИИ
get_style_guide() — правила S1–S6.
get_code(file) — код с номерами строк. Существуют только handler.go и repo.go.
get_linter_report() — отчёт линтера. Линтер пропускает часть нарушений.

ВЫЗОВ — одна строка, после неё закончи сообщение:
CALL get_code {"file": "handler.go"}
Результат придёт в сообщении «РЕЗУЛЬТАТ». Никогда не пиши РЕЗУЛЬТАТ сам.

ПОРЯДОК
1. CALL get_style_guide {}
2. CALL get_code для файла из вопроса. Файл не назван — для handler.go и для repo.go.
3. Вопрос про линтер — ещё CALL get_linter_report {}.
4. Выбери правило по словам вопроса и проверяй только его:
   имена, camelCase → S1: «_» внутри имени (user_id). `_,` — не имя.
   ошибки проигнорированы → S2: вызов возвращает error, а результат не проверен. `_, err :=` и `defer x.Close()` — не нарушение.
   panic → S3: строка с panic(
   context → S4: функция обращается к r.db, а ctx в параметрах нет.
   оборачивание → S5: return err или return nil, err без fmt.Errorf.
   комментарии → S6: func с заглавной буквы, над ней нет //.
   «проверь файл», «проверь метод», «сколько», линтер → все правила, но только в этом файле или методе.
5. Ответ — по строке на нарушение, код строки копируй из РЕЗУЛЬТАТ:
   файл:строка | правило | `код строки`
   Обычно нарушений 0–3. Не пиши строку для правила, если не нашёл в коде такую строку.

ПРИМЕРЫ
Вопрос: Есть ли panic в repo.go?
Ответ 1: CALL get_style_guide {}
Ответ 2: CALL get_code {"file": "repo.go"}
Ответ 3: Правило S3. В repo.go нет строки с panic( — нарушений нет.

Вопрос: Проверь имена в example.go. (пример, такого файла нет)
Ответ 1: CALL get_style_guide {}
Ответ 2: CALL get_code {"file": "example.go"}
Ответ 3: Правило S1.
example.go:7 | S1 | `item_id := req.ID`

Отвечай по-русски. Не выдумывай код, строки и правила."""

PROMPT_V8 = """\
Ты — ревьюер Go-кода. Ты сверяешь код с правилами команды и находишь нарушения.

ФУНКЦИИ
get_style_guide() — правила S1–S6.
get_code(file) — код с номерами строк. Существуют только handler.go и repo.go.
get_linter_report() — отчёт линтера. Линтер пропускает часть нарушений.

Первое сообщение — только строки CALL, без другого текста. Результат придёт в сообщении «РЕЗУЛЬТАТ».

КАКОЕ ПРАВИЛО ПРОВЕРЯТЬ
имена, camelCase → S1: «_» внутри имени (user_id). `_,` — не имя.
ошибки проигнорированы → S2: вызов возвращает error, а результат не проверен. `_, err :=` и `defer x.Close()` — не нарушение.
panic → S3: строка с panic(
context → S4: функция обращается к r.db, а ctx в параметрах нет.
оборачивание → S5: return err или return nil, err без fmt.Errorf.
комментарии → S6: func с заглавной буквы, над ней нет //.
проверь файл, проверь метод, сколько, линтер → ВСЕ правила S1–S6 по очереди.

ОТВЕТ — по строке на нарушение, код строки копируй из РЕЗУЛЬТАТ:
файл:строка | правило | `код строки`

ПРИМЕР 1
Пользователь: Есть ли panic в коде?
Ассистент:
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
Ассистент после РЕЗУЛЬТАТ:
Ищу panic( в обоих файлах. В repo.go такой строки нет.
handler.go:22 | S3 | `panic(err)`

ПРИМЕР 2 (файл example.go выдуман только для примера)
Пользователь: Проверь example.go.
Ассистент:
CALL get_style_guide {}
CALL get_code {"file": "example.go"}
Ассистент после РЕЗУЛЬТАТ:
Проверяю S1–S6 по очереди. S2, S3, S4, S6 — нарушений нет.
example.go:7 | S1 | `item_id := req.ID`
example.go:12 | S5 | `return err`

ПРИМЕР 3
Пользователь: Что нашёл линтер?
Ассистент:
CALL get_linter_report {}
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
Ассистент после РЕЗУЛЬТАТ:
Линтер нашёл: (строки отчёта)
Линтер пропустил: (нарушения из кода, которых нет в отчёте)

Если файла нет — так и скажи. Отвечай по-русски. Не выдумывай код, строки и правила."""

PROMPT_V9 = """\
Ты — ревьюер Go-кода. Ты сверяешь код с правилами команды и находишь нарушения.

ФУНКЦИИ
get_style_guide() — правила S1–S6.
get_code(file) — код с номерами строк. Существуют только handler.go и repo.go.
get_linter_report() — отчёт линтера. Линтер пропускает часть нарушений.

Первое сообщение — только строки CALL, без другого текста. Потом система пришлёт данные.

КАКОЕ ПРАВИЛО ПРОВЕРЯТЬ
имена, camelCase → S1: «_» внутри имени (user_id). `_,` — не имя.
ошибки проигнорированы → S2: вызов возвращает error, а результат не проверен. `_, err :=` и `defer x.Close()` — не нарушение.
panic → S3: строка с panic(
context → S4: функция обращается к r.db, а ctx в параметрах нет.
оборачивание → S5: return err или return nil, err без fmt.Errorf.
комментарии → S6: func с заглавной буквы, над ней нет //.
проверь файл, проверь метод, сколько, линтер → ВСЕ правила S1–S6 по очереди.

ОТВЕТ — по строке на нарушение, код строки копируй из данных:
файл:строка | правило | `код строки`

ПРИМЕР 1
Пользователь: Есть ли в коде нарушения правила S3?
Ассистент:
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
Ассистент, получив данные:
Ищу panic( в обоих файлах. В одном файле такой строки нет, в другом есть.
<файл>:<номер строки> | S3 | `<код этой строки>`

ПРИМЕР 2 (файл example.go выдуман только для примера)
Пользователь: Проверь example.go.
Ассистент:
CALL get_style_guide {}
CALL get_code {"file": "example.go"}
Ассистент, получив данные:
Проверяю S1–S6 по очереди. S2, S3, S4, S6 — нарушений нет.
example.go:7 | S1 | `item_id := req.ID`
example.go:12 | S5 | `return err`

ПРИМЕР 3
Пользователь: Что нашёл линтер?
Ассистент:
CALL get_linter_report {}
CALL get_style_guide {}
CALL get_code {"file": "handler.go"}
CALL get_code {"file": "repo.go"}
Ассистент, получив данные:
Линтер нашёл: (строки отчёта)
Линтер пропустил: (нарушения из кода, которых нет в отчёте)

Если файла нет — так и скажи. Отвечай по-русски. Не выдумывай код, строки и правила."""

PROMPTS = {"chat": PROMPT_CHAT, "v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4,
           "v5": PROMPT_V5, "v6": PROMPT_V6, "v7": PROMPT_V7, "v8": PROMPT_V8, "v9": PROMPT_V9}


# ---------------------------------------------------------------------------
# Клиент к OpenAI-совместимому API со стримингом (SSE)
# ---------------------------------------------------------------------------

class Client:
    def __init__(self, base_url: str, model: str | None, api_key: str | None, temperature: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.model = model or self._detect_model()
        self.last_tps: float | None = None

    def _request(self, path: str, body: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"нет соединения с {self.base_url}: {e.reason}") from None

    def _detect_model(self) -> str:
        # llama-server отдаёт имя загруженного gguf в /v1/models
        try:
            with self._request("/models") as resp:
                return json.load(resp)["data"][0]["id"]
        except Exception:
            return "local"

    def stream(self, messages: list[dict], stop: list[str] | None = None, echo: bool = True) -> str:
        """Стримит ответ в stdout и возвращает его целиком. Ctrl+C обрывает генерацию."""
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": self.temperature,
                "max_tokens": 800}  # защита от зацикливания маленьких моделей
        if stop:
            body["stop"] = stop
        parts: list[str] = []
        self.last_tps = None
        resp = self._request("/chat/completions", body)
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue  # пустые строки и комментарии SSE (": OPENROUTER PROCESSING")
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise RuntimeError(f"ошибка API: {chunk['error']}")
                if timings := chunk.get("timings"):  # только у llama-server
                    self.last_tps = timings.get("predicted_per_second")
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        if echo:
                            print(piece, end="", flush=True)
        except KeyboardInterrupt:
            if echo:
                print(" [прервано]", end="")
        finally:
            resp.close()
            if echo:
                print()
        return "".join(parts)


# ---------------------------------------------------------------------------
# Парсер вызовов из текста ответа
# ---------------------------------------------------------------------------

# Основной формат: CALL get_code {"file": "handler.go"}
# Толерантно к: регистру CALL, обратным кавычкам, `get_code("handler.go")`, `get_code(file="handler.go")`.
CALL_RE = re.compile(
    r"(?:\bCALL\s*:?\s*`*\s*)?\b(" + "|".join(TOOLS) + r")\b`*\s*(\{[^{}]*\}|\([^()\n]*\))?",
    re.IGNORECASE,
)


def parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            args = json.loads(raw)
            return args if isinstance(args, dict) else {}
        except json.JSONDecodeError:
            raw = raw[1:-1]
    else:
        raw = raw[1:-1]
    # ("handler.go") / (file="handler.go") / {file: handler.go}
    m = re.search(r"([\w./-]+\.go)", raw)
    return {"file": m.group(1)} if m else {}


def find_calls(text: str) -> list[re.Match]:
    calls = []
    for m in CALL_RE.finditer(text):
        explicit = m.group(0).lstrip("`").upper().startswith("CALL")
        if explicit or m.group(2):  # просто упоминание функции в тексте — не вызов
            calls.append(m)
    return calls


def parse_calls(text: str) -> list[tuple[str, dict]]:
    calls = []
    for m in find_calls(text):
        name = m.group(1).lower()
        call = (name, normalize_args(name, parse_args(m.group(2))))
        if call not in calls:
            calls.append(call)
    return calls


def cut_after_calls(text: str) -> str:
    """Отрезает всё после строки с последним вызовом: туда модели часто дописывают выдуманный результат."""
    calls = find_calls(text)
    if not calls:
        return text
    last = calls[-1]
    end = text.find("\n", last.end(2) if last.group(2) else last.end(1))
    return text if end == -1 else text[:end]


def normalize_args(name: str, args: dict) -> dict:
    """Приводит аргументы к одному виду, чтобы одинаковые вызовы не выполнялись дважды."""
    if name != "get_code":
        return {}
    # ключ модели пишут как угодно ("file", "filename", "файл"), поэтому берём значение, похожее на имя файла
    values = [str(v) for v in args.values() if isinstance(v, (str, int, float))]
    file = next((v for v in values if v.strip().endswith(".go")), values[0] if values else "")
    return {"file": os.path.basename(file.strip().strip("\"'`"))}


def run_tool(name: str, args: dict) -> str:
    if name == "get_code":
        return get_code(args["file"])
    return TOOLS[name]()


def call_key(name: str, args: dict) -> str:
    return f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"


# ---------------------------------------------------------------------------
# Логи
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path: Path | None):
        self.f = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = path.open("a", encoding="utf-8")

    def write(self, text: str = ""):
        if self.f:
            self.f.write(text + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------------------
# Агент
# ---------------------------------------------------------------------------

def agent_turn(client: Client, messages: list[dict], log: Log, echo: bool = True) -> str:
    """Один ход пользователя: модель -> вызовы -> модель ... пока не ответит без вызовов."""
    done: set[str] = set()
    tools_used: list[str] = []
    for step in range(1, MAX_STEPS + 1):
        if echo:
            print("\033[36mассистент>\033[0m ", end="", flush=True)
        # stop не даёт модели дописать выдуманный результат функции после CALL
        text = client.stream(messages, stop=[RESULT_MARKER], echo=echo)
        tps = f" [{client.last_tps:.1f} ток/с]" if client.last_tps else ""
        log.write(f"--- ассистент (шаг {step}){tps}\n{text.strip()}")

        new_calls = [(n, a) for n, a in parse_calls(text) if call_key(n, a) not in done]
        if not new_calls:
            messages.append({"role": "assistant", "content": text})
            log.write(f"--- вызванные функции: {', '.join(tools_used) or 'нет'}")
            return text
        messages.append({"role": "assistant", "content": cut_after_calls(text)})

        results = []
        for name, args in new_calls:
            key = call_key(name, args)
            done.add(key)
            tools_used.append(key)
            result = run_tool(name, args)
            results.append(f"{RESULT_MARKER} {key}:\n{result}")
            if echo:
                print(f"\033[33m[вызов {key}]\033[0m")
        tool_msg = "\n\n".join(results)
        messages.append({"role": "user", "content": tool_msg})
        log.write(f"--- функции\n{tool_msg}")

    note = "[лимит шагов исчерпан]"
    log.write(f"--- {note}; вызванные функции: {', '.join(tools_used)}")
    if echo:
        print(note)
    return messages[-2]["content"]


def chat_turn(client: Client, messages: list[dict], log: Log) -> str:
    print("\033[36mассистент>\033[0m ", end="", flush=True)
    text = client.stream(messages)
    messages.append({"role": "assistant", "content": text})
    tps = f" [{client.last_tps:.1f} ток/с]" if client.last_tps else ""
    log.write(f"--- ассистент{tps}\n{text.strip()}")
    return text


# ---------------------------------------------------------------------------
# Режимы
# ---------------------------------------------------------------------------

def repl(client: Client, prompt_name: str, log: Log):
    system = {"role": "system", "content": PROMPTS[prompt_name]}
    messages = [system]
    turn = agent_turn if prompt_name != "chat" else chat_turn
    print(f"модель: {client.model} | промпт: {prompt_name} | /reset — очистить историю, /exit или Ctrl+D — выход")
    while True:
        try:
            user = input("\033[32mвы>\033[0m ").strip()
            if not sys.stdin.isatty():
                print(user)  # при вводе из файла/пайпа показываем вопрос в выводе
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            messages = [system]
            print("история очищена")
            continue
        before = len(messages)
        messages.append({"role": "user", "content": user})
        log.write(f"=== вопрос: {user}")
        try:
            turn(client, messages, log)
        except RuntimeError as e:
            print(f"\033[31m{e}\033[0m")
            del messages[before:]  # не оставляем в истории недоделанный ход


def batch(client: Client, prompt_name: str, log: Log, runs: int):
    """Прогон тестовых вопросов. Каждый вопрос — с чистой историей."""
    for run in range(1, runs + 1):
        log.write(f"\n##### прогон {run}/{runs} | {datetime.now():%Y-%m-%d %H:%M:%S} "
                  f"| модель {client.model} | промпт {prompt_name} | t={client.temperature}")
        for i, (question, expected) in enumerate(QUESTIONS, 1):
            print(f"\n\033[32m[{run}/{runs}] {i}. {question}\033[0m")
            log.write(f"\n=== {i}. {question}\n--- ожидается: {expected}")
            messages = [{"role": "system", "content": PROMPTS[prompt_name]},
                        {"role": "user", "content": question}]
            started = time.monotonic()
            try:
                if prompt_name == "chat":
                    chat_turn(client, messages, log)
                else:
                    agent_turn(client, messages, log)
            except RuntimeError as e:
                print(f"\033[31m{e}\033[0m")
                log.write(f"--- ОШИБКА: {e}")
            log.write(f"--- время: {time.monotonic() - started:.1f} с")


def slug(s: str) -> str:
    s = Path(s).stem if s.endswith(".gguf") else s
    return re.sub(r"[^\w.-]+", "_", s).strip("_")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", choices=PROMPTS, default="chat")
    p.add_argument("--provider", choices=PROVIDERS, default="local")
    p.add_argument("--base-url", help="переопределяет адрес провайдера")
    p.add_argument("--model", help="имя модели (для llama-server определяется само)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--batch", action="store_true", help="прогнать тестовые вопросы вместо REPL")
    p.add_argument("--runs", type=int, default=1, help="сколько раз прогнать вопросы в --batch")
    p.add_argument("--log", help="путь к логу (по умолчанию logs/<модель>_<промпт>.log)")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("LLM_API_KEY") if args.provider == "openrouter" else None
    if args.provider == "openrouter":
        if not api_key:
            sys.exit("задайте ключ: export LLM_API_KEY=...")
        if not args.model:
            sys.exit("для OpenRouter нужен --model, например nvidia/nemotron-3-super-120b-a12b:free")

    client = Client(args.base_url or PROVIDERS[args.provider], args.model, api_key, args.temperature)

    log_path = None
    if not args.no_log:
        log_path = Path(args.log) if args.log else Path("logs") / f"{slug(client.model)}_{args.prompt}.log"
    log = Log(log_path)
    try:
        if args.batch:
            batch(client, args.prompt, log, args.runs)
        else:
            repl(client, args.prompt, log)
    finally:
        log.close()
        if log_path:
            print(f"лог: {log_path}")


if __name__ == "__main__":
    main()
