# Serialization Mastery

## 1. Two Serialization Modes

Pydantic has two modes for `model_dump()`:

| Mode | Output | Use Case |
|------|--------|----------|
| `mode="python"` (default) | Native Python objects | Internal use, further processing |
| `mode="json"` | JSON-compatible types | API responses, JSON serialization |

```python
from datetime import datetime, date
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    ts: datetime
    day: date

e = Event(name="release", ts=datetime(2026, 7, 1, 12, 0), day=date(2026, 7, 1))

e.model_dump(mode="python")
# {"name": "release", "ts": datetime(2026, 7, 1, 12, 0), "day": date(2026, 7, 1)}

e.model_dump(mode="json")
# {"name": "release", "ts": "2026-07-01T12:00:00", "day": "2026-07-01"}
```

## 2. model_dump vs model_dump_json

```python
d = e.model_dump()              # dict[str, Any]
j = e.model_dump_json()         # str (JSON)
j2 = e.model_dump_json(indent=2) # pretty-printed
```

`model_dump_json` is faster for network serialization (avoids intermediate dict-to-JSON step).

## 3. Include / Exclude

```python
e.model_dump(include={"name", "ts"})       # only these fields
e.model_dump(exclude={"day"})              # everything except day
e.model_dump(exclude_none=True)            # skip None fields
e.model_dump(exclude_unset=True)           # skip fields not explicitly set
e.model_dump(exclude_defaults=True)        # skip fields still at default
```

Nested paths with `__` separator:

```python
e.model_dump(include={"name", "address": {"city"}})  # only name + address.city
```

## 4. Alias Serialization

```python
class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    first_name: str = Field(serialization_alias="firstName")
    last_name: str = Field(validation_alias="lastName", serialization_alias="last_name")

u = User(first_name="Alice", lastName="Smith")  # validation alias for input
u.model_dump(by_alias=True)
# {"firstName": "Alice", "last_name": "Smith"}
```

`by_alias=True` uses serialization aliases. `validation_alias` controls input parsing.

## 5. Custom Serializers

### @field_serializer

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class Log(BaseModel):
    event: str
    timestamp: datetime

    @field_serializer("timestamp", mode="plain")
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat(timespec="seconds")
```

### @model_serializer

```python
from pydantic import BaseModel, model_serializer

class Response(BaseModel):
    status: str
    data: dict

    @model_serializer(mode="wrap")
    def wrap_response(self, handler):
        result = handler(self)
        return {"api_version": "2.0", "result": result}
```

### mode="plain" vs mode="wrap"

- `plain`: replace the serializer entirely (receive the Python value, return serialized value)
- `wrap`: wrap the default serializer (receive the model/handler, can modify before/after)

## 6. Recursive Serialization Control

```python
class Node(BaseModel):
    name: str
    children: list[Node] | None = None

    @field_serializer("children", mode="wrap")
    def serialize_children(self, handler):
        if not handler:
            return []
        return handler
```

## 7. Custom JSON Encoders

```python
from pydantic import ConfigDict
from decimal import Decimal

class Invoice(BaseModel):
    model_config = ConfigDict(json_encoders={
        Decimal: lambda v: str(v),
        datetime: lambda v: v.isoformat(),
    })
    amount: Decimal
    issued: datetime
```

## 8. Round-Trip

```python
data = e.model_dump(mode="json")
e2 = Event.model_validate(data)  # fails — expects datetime/date, got strings
e3 = Event.model_validate_json(json.dumps(data))  # works — JSON parser re-parses dates
```

`model_validate` expects Python types; `model_validate_json` expects JSON strings and handles type re-parsing.

## Key Takeaways

- `model_dump(mode="json")` produces JSON-safe types (str for datetime, etc.)
- `model_dump_json` is the fastest path to a JSON string
- `include`/`exclude` filters fields at any nesting level
- Serialization aliases decouple API format from internal names
- `@field_serializer` and `@model_serializer` for custom serialization logic
- Use `model_validate_json` for JSON string input, `model_validate` for Python dicts
