# Validators Deep Dive

## 1. Validator Architecture

Pydantic v2 runs validation in four phases:

1. **Type coercion** — `pydantic-core` converts input to the target Python type
2. **Field constraints** — `Field(ge=0)`, `min_length`, `pattern`, etc.
3. **`field_validator`** — Per-field logic (before/after/wrap/plain modes)
4. **`model_validator`** — Cross-field logic (before/after/wrap modes)

Validators run after coercion. Use `mode='before'` to run before coercion.

## 2. field_validator

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name cannot be blank")
        return v.strip()

    @field_validator("email")
    @classmethod
    def email_must_have_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email")
        return v.lower()
```

Multiple fields:

```python
    @field_validator("name", "email")
    @classmethod
    def no_whitespace(cls, v: str) -> str:
        return v.strip()
```

## 3. Field Validator Modes

### mode='after' (default)

Runs after coercion and constraint checks. Receives the Python-native value.

```python
    @field_validator("score", mode="after")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 2)
```

### mode='before'

Runs before coercion. Receives raw input (could be a string, dict, etc.).

```python
    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | int) -> int:
        if isinstance(v, str):
            return int(datetime.fromisoformat(v).timestamp())
        return v
```

### mode='wrap'

Receives a validator function to call. Enables try/except around validation.

```python
    @field_validator("value", mode="wrap")
    @classmethod
    def clamp_to_range(cls, v: int, handler) -> int:
        try:
            return handler(v)
        except ValueError:
            return 0
```

### mode='plain'

Replaces all built-in validation for the field entirely.

```python
    @field_validator("raw_data", mode="plain")
    @classmethod
    def parse_raw(cls, v: str | bytes) -> bytes:
        return v if isinstance(v, bytes) else v.encode()
```

## 4. Annotated-Based Validators

Prefer these for reusable, composable validators:

```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator, AfterValidator, WrapValidator, PlainValidator

def trim_string(v: str) -> str:
    return v.strip()

def ensure_lower(v: str) -> str:
    return v.lower()

TrimmedLower = Annotated[str, BeforeValidator(trim_string), AfterValidator(ensure_lower)]

class User(BaseModel):
    email: TrimmedLower
    name: TrimmedLower
```

### Built-in Annotated Validators

```python
from pydantic import StringConstraints, Field

ShortStr = Annotated[str, StringConstraints(min_length=1, max_length=50, to_lower=True, strip_whitespace=True)]
```

`StringConstraints` supports: `min_length`, `max_length`, `to_lower`, `to_upper`, `strip_whitespace`, `pattern`.

## 5. model_validator

Cross-field validation:

```python
from pydantic import BaseModel, model_validator

class Order(BaseModel):
    items: list[str]
    discount_code: str | None = None
    discount_pct: float = 0.0

    @model_validator(mode="after")
    def check_discount(self):
        if self.discount_code and not self.discount_pct:
            raise ValueError("discount_code requires discount_pct")
        return self
```

### mode='before' model_validator

Receives raw dict before any field validation:

```python
    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: dict) -> dict:
        return {k.replace("-", "_"): v for k, v in data.items()}
```

### mode='wrap' model_validator

Wraps the entire model validation:

```python
    @model_validator(mode="wrap")
    @classmethod
    def log_validation(cls, data: dict, handler):
        result = handler(data)
        print(f"Validated: {result}")
        return result
```

## 6. ValidationInfo Context

Validators receive a `ValidationInfo` with field name, data, and config:

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class Signup(BaseModel):
    password: str
    confirm_password: str

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info: ValidationInfo):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("passwords do not match")
        return v
```

`ValidationInfo` fields: `data` (all values so far), `field_name`, `config`.

## 7. ValidationError Structure

```python
from pydantic import ValidationError

try:
    User(name="", email="bad")
except ValidationError as e:
    for error in e.errors():
        print(error["loc"], error["type"], error["msg"])
```

Error dict structure:
- `loc`: tuple of field path (e.g., `("address", "street")`)
- `type`: error type (e.g., `"string_too_short"`, `"value_error"`)
- `msg`: human-readable message
- `input`: the value that failed
- `ctx`: extra context (e.g., `{"min_length": 1}`)

## Key Takeaways

- `field_validator`: per-field validation, 4 modes (after/before/wrap/plain)
- `model_validator`: cross-field validation, 3 modes (after/before/wrap)
- `Annotated` validators are composable and reusable
- `ValidationInfo` provides access to other field values
- `ValidationError.errors()` gives structured error details for API responses
