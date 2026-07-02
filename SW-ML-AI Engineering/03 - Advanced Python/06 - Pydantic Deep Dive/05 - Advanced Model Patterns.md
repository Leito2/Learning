# Advanced Model Patterns

## 1. RootModel

Wraps a single type for validation. Useful for list endpoints, dict wrappers, and type aliases with validation.

```python
from pydantic import RootModel

class ItemList(RootModel):
    root: list[str]

items = ItemList(root=["a", "b", "c"])
items.model_dump()  # ["a", "b", "c"]
items.model_dump_json()  # '["a","b","c"]'
```

`RootModel` subclasses inherit `model_validate`, `model_validate_json`, `model_dump`, etc. The `root` attribute holds the inner value.

## 2. TypeAdapter

Validates any type without a `BaseModel`:

```python
from pydantic import TypeAdapter
from typing import Literal

PositiveInt = TypeAdapter(int, config={"ge": 0})
PositiveInt.validate_python(42)  # 42
PositiveInt.validate_python(-1)  # ValidationError

Color = TypeAdapter(Literal["red", "green", "blue"])
Color.validate_python("red")    # "red"
Color.validate_python("yellow") # ValidationError
```

`TypeAdapter` supports `validate_python`, `validate_json`, `dump_python`, `dump_json`. Reuse the adapter instance for performance.

## 3. create_model (Dynamic Models)

```python
from pydantic import create_model

DynamicUser = create_model("DynamicUser", name=(str, ...), age=(int, 0))
u = DynamicUser(name="Alice")  # name="Alice", age=0
```

Field definitions: `(type, default)` where `...` means required.

Dynamic models from config:

```python
field_defs = {f"feature_{i}": (float, 0.0) for i in range(10)}
FeatureVector = create_model("FeatureVector", **field_defs)
```

## 4. Generics with TypeVar

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int

class User(BaseModel):
    id: int
    name: str

UserPage = PaginatedResponse[User]
resp = UserPage(items=[User(id=1, name="Alice")], total=1, page=1)
```

## 5. @computed_field

Fields computed at runtime that appear in serialization:

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

r = Rectangle(width=3, height=4)
r.model_dump()  # {"width": 3.0, "height": 4.0, "area": 12.0}
r.model_dump(exclude={"area"})  # exclude computed fields
```

`@computed_field` supports `alias`, `repr`, `json_schema_extra`, and `return_type` parameters.

## 6. model_post_init

Post-initialization hook (runs after `__init__`):

```python
from pydantic import BaseModel

class Config(BaseModel):
    raw: str
    parsed: dict | None = None

    def model_post_init(self, __context):
        if self.parsed is None and self.raw:
            self.parsed = json.loads(self.raw)
```

## 7. cached_property with Pydantic

```python
from functools import cached_property
from pydantic import BaseModel, computed_field

class DataSet(BaseModel):
    values: list[float]

    @cached_property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @computed_field
    @property
    def stats(self) -> dict:
        return {"mean": self.mean, "count": len(self.values)}
```

`cached_property` persists on the instance; `@computed_field` appears in serialization.

## 8. Private Attributes

```python
from pydantic import BaseModel, PrivateAttr

class Service(BaseModel):
    name: str
    _client: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._client = create_client()
```

Private attrs start with `_`, are excluded from serialization, and are not treated as fields.

## Key Takeaways

- `RootModel`: single-type wrapper, useful for list endpoints
- `TypeAdapter`: validates any Python type without a model
- `create_model`: runtime model definition from dynamic schemas
- Generics: reusable paginated/list response types
- `@computed_field`: serialized computed properties
- `model_post_init`: post-`__init__` hook for derived fields
- `PrivateAttr`: internal state excluded from serialization
