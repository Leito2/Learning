# JSON Schema and OpenAPI

## 1. model_json_schema

Every Pydantic model generates a JSON Schema:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    sku: str = Field(..., min_length=8, max_length=16, description="Stock keeping unit")
    price: float = Field(..., ge=0, description="Price in USD")
    in_stock: bool = True

schema = Product.model_json_schema()
```

Output:

```json
{
  "title": "Product",
  "type": "object",
  "properties": {
    "sku": {"type": "string", "minLength": 8, "maxLength": 16, "description": "Stock keeping unit"},
    "price": {"type": "number", "minimum": 0, "description": "Price in USD"},
    "in_stock": {"type": "boolean", "default": true}
  },
  "required": ["sku", "price"]
}
```

## 2. Schema Customization with json_schema_extra

### On Field

```python
class Product(BaseModel):
    sku: str = Field(
        json_schema_extra={
            "examples": ["ABC-12345"],
            "pattern_description": "3 uppercase letters, dash, 5 digits"
        }
    )
```

### On ConfigDict

```python
class APIResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": "Standard API response wrapper"
    })
    status: str
    data: dict
```

## 3. Schema Resolution: by_alias and ref_template

```python
class User(BaseModel):
    first_name: str = Field(serialization_alias="firstName")

Product.model_json_schema(by_alias=True)  # uses "firstName" in schema
Product.model_json_schema(ref_template="#/components/schemas/{model}")  # custom $ref
```

## 4. Multiple Schema Modes

```python
# Default: includes all fields
Product.model_json_schema()

# Serialization mode: applies aliases and serializers
Product.model_json_schema(mode="serialization")

# Validation mode: uses validation aliases
Product.model_json_schema(mode="validation")
```

## 5. OpenAPI Integration (FastAPI)

FastAPI automatically uses Pydantic's JSON Schema for OpenAPI generation:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class ItemRequest(BaseModel):
    name: str = Field(..., description="Item name", max_length=100)
    price: float = Field(..., ge=0, description="Price in USD")

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float

@app.post("/items")
def create_item(item: ItemRequest) -> ItemResponse:
    return ItemResponse(id=1, name=item.name, price=item.price)

# OpenAPI /docs → auto-generates schemas for ItemRequest and ItemResponse
```

## 6. Custom JSON Schema with __get_pydantic_core_schema__

For custom types that need schema representation:

```python
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class HexColor(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.with_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.to_string_ser_schema(),
            json_schema={"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}
        )

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, str) and v.startswith("#") and len(v) == 7:
            return cls(v)
        raise ValueError("invalid hex color")
```

## 7. Schema Filtering

```python
# Exclude private fields, include only certain fields
Product.model_json_schema(include={"sku", "price"})
Product.model_json_schema(exclude={"internal_notes"})
```

## 8. JSON Schema from TypeAdapter

```python
from pydantic import TypeAdapter
from typing import Literal

Color = TypeAdapter(Literal["red", "green", "blue"])
Color.json_schema()  # {"enum": ["red", "green", "blue"], "type": "string"}
```

## Key Takeaways

- `model_json_schema()` generates a JSON Schema from any `BaseModel`
- `json_schema_extra` on `Field()` or `ConfigDict` for custom metadata
- FastAPI auto-generates OpenAPI from Pydantic schemas
- `mode="serialization"` vs `mode="validation"` for alias differences
- `TypeAdapter.json_schema()` for non-model types
- Custom types need `__get_pydantic_core_schema__` for schema support
