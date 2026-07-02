# BaseModel, Field and Type System

## 1. BaseModel Fundamentals

Every Pydantic model inherits from `BaseModel`:

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None
```

`BaseModel` auto-generates `__init__`, `__repr__`, `__str__`, `__eq__`, and `__hash__`. Fields with no default are required; fields with a default are optional.

## 2. Core Methods

| Method | Purpose |
|--------|---------|
| `model_dump()` | Serialize to dict |
| `model_dump_json()` | Serialize to JSON string |
| `model_validate(dict)` | Parse from dict (strict) |
| `model_validate_json(str)` | Parse from JSON string |
| `model_copy(update={})` | Shallow copy with field overrides |

```python
u = User(id=1, name="Alice", email="a@b.com")
d = u.model_dump()                     # {"id": 1, "name": "Alice", ...}
j = u.model_dump_json()                # '{"id":1,"name":"Alice",...}'
u2 = User.model_validate(d)            # round-trip
u3 = u.model_copy(update={"age": 30})  # copy with changes
```

## 3. Field() Constraints

`Field()` adds validation, metadata, and serialization control:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    sku: str = Field(min_length=8, max_length=16, pattern=r"^[A-Z]{3}-\d{5}$")
    price: float = Field(ge=0.0, le=10000.0, multiple_of=0.01)
    quantity: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list, max_length=10)
    metadata: dict[str, str] = Field(default_factory=dict)
```

Key `Field()` parameters:

| Parameter | Purpose |
|-----------|---------|
| `default` | Static default value |
| `default_factory` | Callable returning default (use for mutable types) |
| `ge`, `le`, `gt`, `lt` | Numeric bounds |
| `min_length`, `max_length` | String/collection length |
| `pattern` | Regex validation (str only) |
| `multiple_of` | Numeric divisibility |
| `frozen` | Field is immutable after init |
| `alias` | Alternative name for serialization/parsing |
| `validation_alias` | Alias only for parsing |
| `serialization_alias` | Alias only for serialization |
| `repr` | Include in `__repr__` (default True) |
| `init` | Include in `__init__` (default True) |
| `json_schema_extra` | Extra JSON Schema metadata |

## 4. Type System

Pydantic v2 supports Python's full type system:

```python
from pydantic import BaseModel
from typing import Literal, Optional
from uuid import UUID, uuid4
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: Status = Status.ACTIVE
    items: list[str]
    metadata: dict[str, str] | None = None
    created_at: datetime
    ship_by: date | None = None
    discount: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    priority: Literal["low", "medium", "high"] = "medium"
    metadata: Optional[dict[str, str]] = None
```

Pydantic coerces compatible types by default (lax mode). A string `"123"` is accepted for `int`, a datetime string is parsed for `datetime`. Use strict mode to disable coercion.

## 5. Strict vs Lax Mode

```python
class Point(BaseModel):
    model_config = {"strict": True}
    x: float
    y: float

Point(x=1.0, y=2.0)  # OK
Point(x="1.0", y="2.0")  # ValidationError: float expected
```

Strict mode on individual fields via `Field(strict=True)`.

## 6. Union and Optional

```python
class Result(BaseModel):
    value: int | str  # Union[int, str]

Result(value=42)
Result(value="forty-two")

class OptionalFields(BaseModel):
    a: int | None = None  # Optional, defaults to None
    b: int = 0            # Required but has default
    c: int                # Required, no default
```

## 7. Annotated Type Metadata

`typing.Annotated` adds per-field metadata without `Field()`:

```python
from typing import Annotated
from pydantic import Field

PositiveInt = Annotated[int, Field(ge=0)]
ShortStr = Annotated[str, Field(max_length=50)]

class Config(BaseModel):
    timeout: PositiveInt = 30
    name: ShortStr
```

This pattern enables reusable field type aliases.

## 8. Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Employee(BaseModel):
    name: str
    address: Address  # nested
    emergency_contacts: list[Address]  # list of nested
```

Nested models validate recursively. `model_dump()` produces nested dicts.

## 9. RootModel

`RootModel` wraps a single type (useful for list endpoints):

```python
from pydantic import RootModel

class ItemList(RootModel):
    root: list[Item]

items = ItemList(root=[Item(name="a"), Item(name="b")])
# items.root is the list; items.model_dump() returns the list directly
```

## Key Takeaways

- `BaseModel` auto-generates init, dump, validate, copy
- `Field()` adds constraints, defaults, aliases, metadata
- Pydantic supports the full Python type system + coercions
- `Annotated` enables reusable typed field aliases
- `RootModel` for single-type wrappers (lists, dicts)
- Strict mode disables coercion per-field or per-model
