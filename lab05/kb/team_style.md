# Код-стайл команды

Правила S1–S6 обязательны для всего Go-кода сервиса. Ревьюер ссылается на номер правила в комментарии.
Остальное — по Effective Go, Go Code Review Comments и Uber Go Style Guide.

## S1. Имена

Имена — MixedCaps/camelCase, без подчёркиваний: `loadConfig`, а не `load_config`; экспортируемые — с заглавной:
`LoadConfig`. Аббревиатуры пишутся целиком в одном регистре: `userID`, `serveHTTP`, `parseURL`, а не `userId`,
`serveHttp`, `parseUrl`. В начале неэкспортируемого имени аббревиатура пишется строчными: `urlPattern`, `idx`.

Почему: так принято в стандартной библиотеке и так проверяет линтер revive (var-naming); единый стиль имён
избавляет ревью от споров о вкусе.

Исключения: подчёркивание допустимо в именах тестов (`TestParse_EmptyInput`) и в сгенерированном коде.

## S2. Ошибки не игнорировать

Каждую возвращённую ошибку нужно проверить: нельзя `_ = f()`, `data, _ := os.ReadFile(path)` или вызов
`json.NewEncoder(w).Encode(v)` без проверки результата. Ошибку либо обрабатывают на месте, либо возвращают
наверх (по правилу S5).

```go
// плохо
data, _ := os.ReadFile(path)

// хорошо
data, err := os.ReadFile(path)
if err != nil {
	return fmt.Errorf("read config %s: %w", path, err)
}
```

Исключение: `defer x.Close()` для ресурсов, открытых только на чтение, допускается без проверки.
Для файлов, открытых на запись, ошибку Close нужно проверять — при ней данные могут не сохраниться.

Разница с S5: S2 нарушено, когда ошибку не проверяют вовсе; если ошибку проверили, но вернули наверх
без контекста, — это S5.

## S3. Без panic в HTTP-обработчиках

В HTTP-обработчике ошибка превращается в ответ клиенту: `http.Error(w, "internal error",
http.StatusInternalServerError)` и `return`. panic в обработчике обрывает запрос, клиент получает
оборванное соединение вместо кода 500, а в логах остаётся стек без контекста запроса.

```go
// плохо
if err != nil {
	panic(err)
}

// хорошо
if err != nil {
	log.Printf("get user %d: %v", id, err)
	http.Error(w, "internal error", http.StatusInternalServerError)
	return
}
```

panic допустим только для ошибок программиста при инициализации (например, `regexp.MustCompile` с
константным шаблоном в переменной пакета).

## S4. context.Context первым параметром

Функции, которые ходят в базу данных или в сеть, принимают `ctx context.Context` первым параметром и
передают его дальше: `QueryContext`, `QueryRowContext`, `ExecContext`, `http.NewRequestWithContext`.
Так отмена запроса клиентом и таймауты доходят до базы.

```go
// плохо
func (r *Repo) GetUser(id int64) (*User, error) {
	row := r.db.QueryRow("SELECT name FROM users WHERE id = $1", id)

// хорошо
func (r *Repo) GetUser(ctx context.Context, id int64) (*User, error) {
	row := r.db.QueryRowContext(ctx, "SELECT name FROM users WHERE id = $1", id)
```

HTTP-обработчики берут контекст из `r.Context()` — отдельный параметр ctx им не нужен. Контекст не хранят
в полях структур.

## S5. Оборачивать ошибки

Ошибка, которую функция возвращает наверх, оборачивается с контекстом: `fmt.Errorf("get user %d: %w", id, err)`.
Глагол — `%w`, а не `%v`: с `%w` вызывающий код может проверить причину через `errors.Is` и `errors.As`.
Контекст — короткий, без «failed to» и «ошибка при»: по цепочке получится
`get order 42: scan row: sql: no rows in result set`.

```go
// плохо
if err := row.Scan(&u.Name); err != nil {
	return nil, err
}

// хорошо
if err := row.Scan(&u.Name); err != nil {
	return nil, fmt.Errorf("scan user %d: %w", id, err)
}
```

Исключение: ошибку можно вернуть как есть, если функция — тонкая обёртка и контекст уже добавлен уровнем ниже.

## S6. Doc-комментарии

У экспортируемых функций, методов и типов есть doc-комментарий, который начинается с имени и является
законченным предложением: `// GetUser возвращает пользователя по id.` Комментарий без имени в начале
(`// возвращает пользователя`) не годится: godoc и линтеры ищут имя первым словом.

```go
// GetUser загружает пользователя по id. Если пользователя нет, возвращает ErrNotFound.
func (r *Repo) GetUser(ctx context.Context, id int64) (*User, error) {
```

У неэкспортируемых функций комментарий необязателен, но приветствуется для неочевидной логики.
