# model_config Systematic Tour

## 1. ConfigDict

All model settings live in `model_config`:

```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
```

## 2. All ConfigDict Options

### Validation Behavior

| Option | Values | Effect |
|--------|--------|--------|
| `strict` | `bool` | Disable all type coercion globally |
| `frozen` | `bool` | Make model immutable after init |
| `validate_assignment` | `bool` | Re-validate on attribute set |
| `validate_default` | `bool` | Validate fields with default values |
| `extra` | `"ignore"` / `"forbid"` / `"allow"` | Behavior for unknown fields |
| `populate_by_name` | `bool` | Allow field population by name (not just alias) |
| `coerce_numbers_to_str` | `bool` | Auto-convert numbers to strings |

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    name: str
    age: int

m = StrictModel(name="Alice", age=30)
m.name = "Bob"  # TypeError: model is frozen
StrictModel(name="Bob", age=30, extra_field=1)  # ValidationError: extra fields not permitted
```

### String Handling

| Option | Effect |
|--------|--------|
| `str_to_lower` | Auto-lowercase all strings |
| `str_to_upper` | Auto-uppercase all strings |
| `str_strip_whitespace` | Strip whitespace from all strings |

### Type Handling

| Option | Effect |
|--------|--------|
| `arbitrary_types_allowed` | Allow non-Pydantic types (e.g., `np.ndarray`) |
| `from_attributes` | Allow `model_validate(obj)` from arbitrary objects with matching attributes |

```python
class MLModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    embedding: np.ndarray
    features: dict[str, float]
```

### Serialization

| Option | Effect |
|--------|--------|
| `ser_json_timedelta` | `"iso8601"` or `"float"` |
| `ser_json_bytes` | `"utf8"`, `"base64"`, or `"hex"` |
| `ser_json_inf_nan` | `"null"`, `"constants"`, or `"strings"` |
| `json_schema_extra` | Extra JSON Schema properties |
| `json_encoders` | Custom JSON encoders (v1 compatibility) |

### Error Messages

| Option | Effect |
|--------|--------|
| `hide_input_in_errors` | Omit input values from error output |
| `protected_namespaces` | Tuple of forbidden name prefixes (default `("model_", "settings_")`) |
| `use_attribute_docstrings` | Use class docstrings for field descriptions |

```python
class SafeModel(BaseModel):
    model_config = ConfigDict(
        hide_input_in_errors=True,
        protected_namespaces=("model_", "settings_", "internal_"),
    )
```

## 3. Per-Model vs Inheritance

```python
class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class UserConfig(BaseConfig):
    model_config = ConfigDict(extra="ignore")  # override for this subclass

    key: str
```

Subclasses inherit `model_config` but can override individual options.

## 4. Common Config Patterns

### ML Inference Schema

```python
class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    features: list[float]
    model_version: str | None = None
```

### Settings from Env

```python
class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    api_key: str
    database_url: str
```

### Immutable Domain Model

```python
class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: int
    created_at: datetime
```

## Key Takeaways

- `ConfigDict` centralizes all model behavior in one place
- `extra="forbid"` is the safest default for API schemas
- `frozen=True` enables immutability (hashable models)
- `arbitrary_types_allowed` for numpy/tensor integration
- `from_attributes=True` for DB mapping (SQLAlchemy, SQLModel)
- Subclasses inherit but can override specific options
