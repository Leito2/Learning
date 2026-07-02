# SQLModel Bridge

## 1. What is SQLModel?

SQLModel is a library that bridges **Pydantic** and **SQLAlchemy**. A single class definition creates both a Pydantic validation model and a SQLAlchemy ORM table.

```python
from sqlmodel import SQLModel, Field as SQLField

class Hero(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str
    secret_name: str
    age: int | None = None
```

`table=True` makes it a database table. Without `table=True`, it's a plain Pydantic model.

## 2. Table vs Model

```python
class Hero(SQLModel, table=True):       # Database table + Pydantic model
    id: int | None = SQLField(default=None, primary_key=True)
    name: str

class HeroRead(SQLModel):                # Pydantic-only (response schema)
    id: int
    name: str

class HeroCreate(SQLModel):              # Pydantic-only (request schema)
    name: str
    secret_name: str
    age: int | None = None
```

The table model is the single source of truth. Read/Create schemas are plain `SQLModel` (no `table=True`).

## 3. Session CRUD

```python
from sqlmodel import Session, create_engine, select

engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    hero = Hero(name="Alice", secret_name="A1")
    session.add(hero)
    session.commit()
    session.refresh(hero)
    print(hero.id)  # auto-generated

    statement = select(Hero).where(Hero.name == "Alice")
    result = session.exec(statement)
    alice = result.one()
```

## 4. Relationship Patterns

```python
class Team(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str
    heroes: list["Hero"] = SQLField(back_populates="team", default=[])

class Hero(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str
    team_id: int | None = SQLField(default=None, foreign_key="team.id")
    team: Team | None = SQLField(back_populates="heroes", default=None)
```

SQLModel relationships mirror SQLAlchemy. `back_populates` creates bidirectional links.

## 5. Alembic Migrations

SQLModel uses Alembic under the hood:

```bash
pip install alembic
alembic init alembic
```

In `alembic/env.py`:

```python
from sqlmodel import SQLModel
from models import Hero, Team  # import all table models

target_metadata = SQLModel.metadata
```

```bash
alembic revision --autogenerate -m "add hero table"
alembic upgrade head
```

## 6. Inheritance and Mixins

```python
class TimestampMixin(SQLModel):
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime | None = SQLField(default=None, sa_column_kwargs={"onupdate": datetime.utcnow})

class Item(TimestampMixin, SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str
```

## 7. Pydantic-SQLModel Interop

Table models are Pydantic models. They support `model_dump()`, `model_validate()`, validators, and `ConfigDict`:

```python
class Hero(SQLModel, table=True):
    model_config = ConfigDict(extra="forbid")

    id: int | None = SQLField(default=None, primary_key=True)
    name: str = SQLField(min_length=1, max_length=100)
    age: int | None = SQLField(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        return v.strip()
```

## 8. Performance: SelectInLoad vs Join

```python
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

# Eager load relationships in one query
statement = select(Hero).options(selectinload(Hero.team))
result = session.exec(statement)
```

## Key Takeaways

- SQLModel = Pydantic + SQLAlchemy: one class defines both schema and table
- `table=True` for DB models, plain `SQLModel` for request/response schemas
- All Pydantic features work (validators, config, serialization)
- Alembic migrations work via `SQLModel.metadata`
- `selectinload` for efficient relationship loading
- Pattern: one table model + multiple read/create/update schemas
