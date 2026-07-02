# 🧬 Models, Relationships, and Eager Loading

## 🎯 Learning Objectives

- Build SQLAlchemy 2.0 declarative models using the typed `Mapped[...]` API
- Model one-to-many, many-to-one, and many-to-many relationships correctly
- Diagnose and eliminate the N+1 query problem using `selectinload` and `joinedload`
- Choose the right cascade strategy to avoid orphan rows and unexpected deletes
- Use table inheritance, mixins, and abstract base classes for DRY models

## Introduction

A SQLAlchemy model is a Python class that maps to a database table. The 2.0 release made the mapping **fully typed**: instead of `Column(Integer, primary_key=True)`, you write `id: Mapped[int] = mapped_column(primary_key=True)`. The type annotation feeds the schema, the ORM, and your IDE. The trade-off is verbosity on simple models, but the gain is enormous on complex ones: every relationship, every index, every constraint is statically visible.

Most production performance issues in database access are not slow queries but **too many queries**. The N+1 problem is the canonical example: a list of 100 users with one query, then 100 follow-up queries to fetch each user's profile. SQLAlchemy 2.0 has explicit, type-safe APIs for fixing this. The two main strategies — `joinedload` (one big JOIN) and `selectinload` (one extra IN-query) — solve the same problem in different ways; the right choice depends on the cardinality of the relationship.

This note covers the model layer in depth, from a single table to a multi-table schema with relationships, cascading, and inheritance.

---

## 1. The Declarative Base and the `Mapped[...]` API

### 1.1 The base class

Every model inherits from a `DeclarativeBase` subclass. The base holds the metadata and the registry that SQLAlchemy uses to track classes.

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

`Base.metadata` is the container for all tables. Alembic reads it to autogenerate migrations (covered in [[03 - Alembic Migrations Workflow|note 03]]). One application = one base. Multiple bases are a code smell.

### 1.2 The new typed mapping

The 2.0 typed API replaced the older `Column(...)` style. The two are functionally equivalent; mixing them in one app is a code smell.

```python
from datetime import datetime
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    bio: Mapped[str | None] = mapped_column(String(500), default=None)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
```

Every line teaches a concept:

| Annotation | Meaning |
|------------|---------|
| `Mapped[int]` | Python type hint, used by IDEs and `mypy` |
| `mapped_column(...)` | Tells SQLAlchemy this is a column (not a regular Python attribute) |
| `String(255)` | VARCHAR(255). Always specify length; `String` alone is a TEXT field |
| `String \| None` | Nullable column. The type hint is the source of truth for nullability |
| `server_default=func.now()` | Database-side default. Applied by PostgreSQL, not Python |
| `default=True` | Python-side default. Applied by SQLAlchemy before INSERT |

💡 **Tip**: prefer `server_default` for `created_at`, `updated_at`, and similar. The DB clock is the source of truth across multiple app servers; Python clocks drift.

### 1.3 Common field types

| Python type | SQLAlchemy / SQL | Notes |
|-------------|------------------|-------|
| `int` | `Integer` / `BigInteger` | Use `BigInteger` for IDs that may exceed 2.1B |
| `str` | `String(N)` | Always specify length; bare `String` is TEXT |
| `bool` | `Boolean` | Uses native `BOOLEAN` in PostgreSQL |
| `datetime` | `DateTime(timezone=True)` | Always `timezone=True` to avoid naive/aware bugs |
| `date` | `Date` | Date only, no time |
| `float` | `Float` / `Numeric(precision, scale)` | Use `Numeric` for money |
| `Decimal` | `Numeric(precision, scale)` | Money, financial calcs |
| `UUID` | `UUID(as_uuid=True)` | PostgreSQL native `uuid` type |
| `dict` / `list` | `JSON` | PostgreSQL `jsonb` underneath; queryable |
| `bytes` | `LargeBinary` | BLOBs, hashes, encrypted data |
| `Enum` | `Enum(MyEnum)` | Use Python `StrEnum` (3.11+) or `enum.Enum` |

---

## 2. Relationships: The Heart of the ORM

### 2.1 One-to-many

The most common relationship. One `User` has many `Post`s; each `Post` has one `User` (its author).

```python
from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    # "One user has many posts" → list on the parent side
    posts: Mapped[List["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",
        lazy="selectin",  # default eager loading strategy (see §3)
    )


class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    # "Each post has one author" → single object on the child side
    author: Mapped["User"] = relationship(back_populates="posts")
```

Three key decisions:

1. **`back_populates`** wires the two sides. Changing `user.posts.append(post)` keeps `post.author` in sync, and vice versa.
2. **`ondelete="CASCADE"`** at the database level: deleting a user removes all their posts. The `cascade` argument on the relationship tells SQLAlchemy to do the same at the ORM level.
3. **`lazy="selectin"`** sets the default loading strategy (covered in §3).

### 2.2 Many-to-one and one-to-one

```python
# Many-to-one: many posts → one category
class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    posts: Mapped[List["Post"]] = relationship(back_populates="category")

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    category: Mapped["Category"] = relationship(back_populates="posts")
```

```python
# One-to-one: a user has one profile
class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,  # the unique constraint enforces 1:1
    )
    bio: Mapped[str]
    user: Mapped["User"] = relationship(back_populates="profile", single_parent=True)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    profile: Mapped["Profile"] = relationship(
        back_populates="user",
        uselist=False,        # the relationship returns a scalar, not a list
        cascade="all, delete-orphan",
    )
```

### 2.3 Many-to-many

A many-to-many relationship requires an **association table** that connects the two entities. The classic example: posts and tags.

```python
from sqlalchemy import Table, Column, ForeignKey

post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    posts: Mapped[List["Post"]] = relationship(
        secondary=post_tags,
        back_populates="tags",
    )

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    tags: Mapped[List["Tag"]] = relationship(
        secondary=post_tags,
        back_populates="posts",
    )
```

If the association needs extra columns (e.g., `added_at`, `added_by`), use the **association object** pattern:

```python
class PostTag(Base):
    __tablename__ = "post_tags"
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Post(Base):
    tags: Mapped[List["Tag"]] = relationship(secondary="post_tags")
```

---

## 3. The N+1 Problem and Eager Loading

### 3.1 What N+1 looks like

Imagine a blog homepage that lists 50 users with their post count. The naïve code:

```python
# ❌ This is the N+1 problem
users = (await session.execute(select(User))).scalars().all()
for user in users:
    print(user.name, len(user.posts))  # each access triggers a SELECT
```

The first query loads 50 users. Each `user.posts` access fires a separate `SELECT * FROM posts WHERE user_id = ?`. Total queries: **51** (1 + 50).

In a synchronous code path this is hidden. In a FastAPI handler serving thousands of requests, 51 queries per request is the difference between 100 RPS and 2 RPS. The fix is **eager loading**: tell SQLAlchemy to fetch the related data in the same round trip (or a small fixed number of additional queries).

### 3.2 `selectinload`: the default choice

```python
from sqlalchemy.orm import selectinload

users = (
    await session.execute(select(User).options(selectinload(User.posts)))
).scalars().all()
```

This produces exactly **2 queries**:

```sql
SELECT * FROM users;
SELECT * FROM posts WHERE user_id IN (1, 2, 3, ..., 50);
```

`selectinload` is the right default for **to-many** relationships (one-to-many, many-to-many). The second query is a single `IN`-clause fetch; PostgreSQL handles this efficiently even for thousands of IDs.

### 3.3 `joinedload`: the alternative

```python
from sqlalchemy.orm import joinedload

users = (
    await session.execute(select(User).options(joinedload(User.posts)))
).scalars().all()
```

This produces **1 query** with a `LEFT OUTER JOIN`:

```sql
SELECT users.*, posts.* FROM users LEFT OUTER JOIN posts ON posts.user_id = users.id;
```

`joinedload` is the right choice for **to-one** relationships (many-to-one, one-to-one). The cost of the join grows with the cardinality of the loaded side; loading 50 users with all their posts via `joinedload` returns 50 × N rows in a single result, which can balloon memory.

### 3.4 Decision matrix

| Relationship | Cardinality | Recommended | Why |
|--------------|-------------|-------------|-----|
| `User.posts` (one-to-many) | to-many | `selectinload` | Avoids row explosion; one IN-query is cheap |
| `Post.author` (many-to-one) | to-one | `joinedload` | Adds one column to the main query |
| `Post.tags` (many-to-many) | to-many | `selectinload` | One IN-query for tags |
| Nested (`user.posts.author.profile`) | mixed | Combined: `selectinload(Posts).joinedload(Author.profile)` | Mix strategies per level |

### 3.5 The `lazy=` relationship argument

The `lazy` argument on `relationship()` sets the **default** loading strategy. You can override it per query with `.options(...)`.

```python
# Relationship default
posts: Mapped[List["Post"]] = relationship(lazy="selectin")  # eager by default

# Override in a query
stmt = select(User).options(lazyload(User.posts))  # force lazy for this query
```

Options:

| Value | Behavior |
|-------|----------|
| `"select"` | Lazy: load on first access (the N+1 risk) |
| `"selectin"` | Eager via IN-query (the safe default for to-many) |
| `"joined"` | Eager via JOIN (good for to-one) |
| `"raise"` | Lazy, but raise on access (best in tests to catch N+1) |
| `"noload"` | Never load; raise on access (use only for "I know I don't need this") |
| `"write_only"` | Load via a separate query on commit (rare, advanced) |

💡 **Tip**: set `lazy="raise"` in your test suite. A single test that asserts "loading this page does not trigger N queries" can catch a regression that would otherwise hit production at 3 AM.

### 3.6 Chained eager loading

For nested relationships, chain the options:

```python
# Load users, their posts, and each post's tags
stmt = (
    select(User)
    .options(
        selectinload(User.posts).selectinload(Post.tags),
        selectinload(User.posts).joinedload(Post.author),
    )
)
```

The order matters: each chained call adds a sub-option. SQLAlchemy figures out the right queries to fire.

---

## 4. Cascades: The ORM's Garbage Collector

### 4.1 The `cascade` argument

`cascade` tells SQLAlchemy what to do with related objects when you mutate the parent. The values:

| Value | Effect |
|-------|--------|
| `"save-update"` (default) | Adding a parent adds the children to the session |
| `"merge"` | Merging a parent merges the children |
| `"refresh"` | Refreshing the parent refreshes the children |
| `"expunge"` | Expunging the parent expunges the children |
| `"delete"` | Deleting the parent deletes the children (per-row DELETE) |
| `"delete-orphan"` | Removing a child from the parent's collection deletes the child |
| `"all"` | All of the above except `delete-orphan` |
| `"all, delete-orphan"` | All of the above including `delete-orphan` |

### 4.2 The classic `delete-orphan` pattern

```python
class User(Base):
    posts: Mapped[List["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",  # remove a post from the collection → delete it
    )

# Now this works as expected
user.posts.remove(some_post)  # marks some_post for deletion on next flush
await session.commit()        # the post is gone
```

Without `delete-orphan`, removing `some_post` from the collection would do nothing — the post would still exist as an orphan row with a dangling foreign key. The `cascade` argument fixes this.

### 4.3 Cautions with `delete` cascade

ORM-level `delete` cascade issues a `DELETE` per child. For a parent with 10,000 children, that's 10,000 statements. For bulk operations, use **bulk delete** instead:

```python
# ❌ Slow: 10,000 DELETE statements
await session.execute(delete(Post).where(Post.user_id == some_user_id))

# ✅ Fast: one DELETE
from sqlalchemy import delete
await session.execute(delete(Post).where(Post.user_id == some_user_id))
```

For deletion performance, the **database-level** `ON DELETE CASCADE` on the foreign key is the right tool for permanent invariants. ORM cascade is for ensuring ORM-level consistency in code that mutates collections.

---

## 5. Indexes, Constraints, and Other Column Options

### 5.1 Indexes

```python
from sqlalchemy import Index

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    last_login_at: Mapped[datetime | None] = mapped_column(index=True)

    # Composite index
    __table_args__ = (
        Index("ix_users_name_email", "name", "email"),
        Index("ix_users_active_login", "is_active", "last_login_at"),
    )
```

Indexes are not free. Every index slows down writes. Index columns you actually filter, sort, or join on. Don't index `bio` or `description` (you almost never search by exact match).

### 5.2 Constraints

```python
from sqlalchemy import CheckConstraint, UniqueConstraint

class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    balance: Mapped[float] = mapped_column(default=0.0)
    email: Mapped[str]

    __table_args__ = (
        CheckConstraint("balance >= 0", name="ck_balance_non_negative"),
        UniqueConstraint("email", name="uq_accounts_email"),
    )
```

Database-level constraints are the **last line of defense**. Application-level validation is convenient; database constraints are guaranteed.

---

## 6. Mixins and Inheritance

### 6.1 Mixins for shared columns

```python
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
```

Mixins are not registered as tables (no `__tablename__`). They contribute columns and properties to the inheriting class.

### 6.2 Table inheritance (advanced, use sparingly)

SQLAlchemy supports single-table, joined-table, and concrete-table inheritance. Each has trade-offs:

| Strategy | Tables | Pros | Cons |
|----------|--------|------|------|
| Single-table | 1 | No joins, fast reads | Many nullable columns, discriminator column |
| Joined-table | N | Clean schema, no nullable columns | Joins for polymorphic queries |
| Concrete-table | N | No joins, no shared table | Duplicate columns, complex inserts |

For most applications, prefer **composition** (one entity holds a reference to another) over inheritance. The classic mistake is `User` inheriting from `Account` because "users have accounts"; the right design is `User` having a `One-to-One` `account` relationship.

---

## 7. The Mapped[...] Type System

The `Mapped[T]` annotation is the heart of the 2.0 typed API. It does three things at once:

1. **Tells the IDE the column type** for autocomplete and refactor safety.
2. **Tells SQLAlchemy the SQL type** when the model is registered.
3. **Tells `mypy` the attribute type** when reading code.

```python
# Mapped[int] + primary_key=True → INTEGER PRIMARY KEY
id: Mapped[int] = mapped_column(primary_key=True)

# Mapped[str | None] → nullable VARCHAR
nickname: Mapped[str | None] = mapped_column(String(50), default=None)

# Mapped[List["Post"]] → one-to-many relationship
posts: Mapped[List["Post"]] = relationship(...)
```

The `| None` syntax (Python 3.10+) is the modern way to express nullability. The 2.0 API accepts `Optional[str]` too, but `str | None` is preferred.

---

## 8. The Query: 2.0 Style

The 2.0 style uses `select()`, not the legacy `Query` API. The patterns:

```python
from sqlalchemy import select, func

# All
stmt = select(User)
result = await session.execute(stmt)
users = result.scalars().all()

# Filter
stmt = select(User).where(User.is_active == True)
stmt = select(User).where(User.email.like("%@example.com"))
stmt = select(User).where(User.id.in_([1, 2, 3]))
stmt = select(User).where(User.created_at >= some_date)

# Order
stmt = select(User).order_by(User.created_at.desc())

# Limit / offset
stmt = select(User).limit(20).offset(0)

# Aggregate
stmt = select(func.count(User.id), func.avg(User.age))
total, avg_age = (await session.execute(stmt)).one()

# Join
stmt = select(User, Post).join(Post, Post.user_id == User.id)
for user, post in (await session.execute(stmt)).all():
    ...

# Group by
stmt = (
    select(User.country, func.count(User.id))
    .group_by(User.country)
    .having(func.count(User.id) > 10)
)
```

💡 **Tip**: when you need both the parent and the relationship in one query, use `selectinload` instead of `join`. The join gives you a flat row per match; you have to deduplicate in Python.

---

## 9. Common Pitfalls

### 9.1 Using `relationship` as a column

```python
# ❌ This looks like a column but is a relationship
class User(Base):
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"))
    profile: Mapped["Profile"] = relationship()  # accessed via .profile, not .profile_id
```

`profile` is a relationship (returns a `Profile` object). `profile_id` is the foreign key column. They are different things; don't confuse them.

### 9.2 Forgetting to commit

```python
async with SessionLocal() as session:
    user = User(name="alice")
    session.add(user)
    # forgot await session.commit()
# nothing was saved
```

### 9.3 Mutating a list outside a session

```python
async def get_user_posts(user_id: int):
    async with SessionLocal() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalar_one()
    # session closed; modifying user.posts here does nothing
    user.posts.append(Post(title="new"))  # silently dropped
```

---

## 10. Código de Compresión

```python
"""
Compresión: Models, Relationships, Eager Loading
Cubre: typed Mapped API, relationships, N+1 fix with selectinload,
       cascade delete-orphan, mixins, indexes.
"""
from datetime import datetime
from typing import List
from sqlalchemy import String, ForeignKey, Index, select
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    selectinload, joinedload,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        server_default="now()", nullable=False,
    )


# 1) Models
class Author(Base, TimestampMixin):
    __tablename__ = "authors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    books: Mapped[List["Book"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",
        lazy="selectin",  # safe default
    )


class Book(Base, TimestampMixin):
    __tablename__ = "books"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id", ondelete="CASCADE"),
        index=True,
    )
    author: Mapped["Author"] = relationship(back_populates="books")


# 2) Anti-N+1: selectinload for to-many
async def get_authors_with_books(session: AsyncSession) -> list[Author]:
    stmt = select(Author).options(selectinload(Author.books))
    return list((await session.execute(stmt)).scalars().all())
    # fires exactly 2 queries, regardless of author count


# 3) Joinedload for to-one
async def get_books_with_authors(session: AsyncSession) -> list[Book]:
    stmt = select(Book).options(joinedload(Book.author))
    return list((await session.execute(stmt)).scalars().unique().all())
    # 1 query with a single JOIN; .unique() de-duplicates the joined rows


# 4) Cascade demo
async def delete_first_book(author: Author, session: AsyncSession):
    if author.books:
        victim = author.books[0]
        author.books.remove(victim)  # cascade="delete-orphan" deletes victim
    await session.commit()


# 5) Index demo
class IndexedAuthor(Base, TimestampMixin):
    __tablename__ = "indexed_authors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)

    __table_args__ = (
        Index("ix_indexed_authors_name_email", "name", "email"),
    )
```

---

## Key Takeaways

- The 2.0 typed `Mapped[T]` API is the canonical way to declare models. It feeds the IDE, the type checker, and SQLAlchemy from one source.
- The N+1 problem is the most common performance bug in production. The fix is eager loading via `selectinload` (to-many) or `joinedload` (to-one).
- Set `lazy="selectin"` or `lazy="joined"` on relationship definitions to make eager loading the default. Combine with `.options(...)` for query-specific overrides.
- `cascade="all, delete-orphan"` is the right default for owned relationships. It keeps the DB and the ORM in sync when you mutate collections.
- Use `Mapped[str | None]` (not `Mapped[Optional[str]]`) for nullable columns. The type hint is the source of truth for nullability.
- Avoid table inheritance for "is-a" relationships that are really "has-a". Use composition (a foreign key + relationship) instead.

## References

- [SQLAlchemy 2.0 — ORM Quickstart](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [SQLAlchemy 2.0 — Relationship Loading Techniques](https://docs.sqlalchemy.org/en/20/orm/loading_relationships.html)
- [SQLAlchemy 2.0 — Cascades](https://docs.sqlalchemy.org/en/20/orm/cascades.html)
- [Hibernate's "fetch join" — same N+1 problem, same solutions](https://docs.jboss.org/hibernate/orm/current/userguide/html_single/Hibernate_User_Guide.html#hql-explicit-join)
