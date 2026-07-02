# 📜 Pagination, Filtering, and Sorting

## 🎯 Learning Objectives

- Choose the right pagination strategy: offset vs cursor
- Implement cursor-based pagination correctly with stable sort keys
- Validate query parameters (filter, sort, page size) at the FastAPI layer
- Avoid the five most common pagination and query mistakes at scale

## Introduction

Every API that returns a list of resources needs pagination. The choice between **offset** (`?page=5&size=20`) and **cursor** (`?after_id=12345`) determines the user experience, the database cost, and the ability to handle concurrent inserts. Offset is simple but breaks; cursor is robust but more code. The industry has converged on cursor for any API serving a list that grows or changes over time.

Filtering and sorting are equally important. A list endpoint that accepts `?status=active&sort=-created_at` is much more useful than one that returns everything. The challenge is validating the query parameters: clients can send any string, and the server must reject invalid input cleanly (RFC 7807, not a stack trace).

This note covers the patterns for each, with FastAPI examples and the production trade-offs.

---

## 1. The Two Pagination Strategies

### 1.1 Offset pagination

```http
GET /users?page=5&size=20
```

The server skips `page * size` rows and returns the next `size` rows. Simple to implement, simple to consume, but breaks under concurrent inserts and large offsets.

### 1.2 Cursor pagination

```http
GET /users?limit=20&after_id=12345
```

The server returns the next `limit` rows after the row with id `= 12345`. Stable under concurrent inserts; no skip cost.

### 1.3 The trade-offs

| Concern | Offset | Cursor |
|---------|--------|--------|
| Implementation | Simple | More code (cursor encoding) |
| Total count | Trivial | Hard or impossible |
| Jump to page 50 | Trivial | Not possible |
| Concurrent inserts | Breaks (duplicates or skips) | Stable |
| Performance at depth | Poor (skip 100K rows) | Constant |
| Industry adoption | Legacy APIs | Stripe, GitHub, Twitter, Slack |

For a new FastAPI service that returns lists that grow over time, **cursor pagination is the right choice**. Offset is fine for small, stable lists (e.g., the 50 states of the US).

---

## 2. Cursor Pagination in FastAPI

### 2.1 The basic pattern

```python
from fastapi import Query
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False


@router.get("/users", response_model=Page[UserOut])
async def list_users(
    limit: int = Query(20, ge=1, le=100),
    after_id: int | None = Query(None, ge=1),
    uow: UnitOfWork = Depends(get_uow),
):
    items = await uow.users.list_cursor(after_id=after_id, limit=limit + 1)
    has_more = len(items) > limit
    items = items[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return Page(items=items, next_cursor=next_cursor, has_more=has_more)
```

The `limit + 1` trick: ask for one more row than the limit; if you got it, there's a next page. The `next_cursor` is the last item's id; the client passes it as `after_id` for the next request.

### 2.2 Stable sort keys

The cursor must be **stable** under concurrent inserts and updates. The primary key is the natural choice, but consider:

- **`id` (autoincrement)**: stable, monotonic, simple. The default.
- **`created_at` + `id`**: handles tie-breaks when two rows have the same timestamp.
- **A dedicated `sort_key` column**: explicit; you control the ordering.

```python
# Compound cursor for stable ordering
@router.get("/events", response_model=Page[EventOut])
async def list_events(
    limit: int = Query(20, ge=1, le=100),
    after_ts: datetime | None = Query(None),
    after_id: int | None = Query(None),
    uow: UnitOfWork = Depends(get_uow),
):
    stmt = select(Event).order_by(Event.created_at.asc(), Event.id.asc())
    if after_ts and after_id:
        # Compound cursor: rows with (created_at, id) > (after_ts, after_id)
        stmt = stmt.where(
            tuple_(Event.created_at, Event.id) > tuple_(after_ts, after_id)
        )
    items = (await uow.session.execute(stmt.limit(limit + 1))).scalars().all()
    has_more = len(items) > limit
    items = items[:limit]
    if has_more and items:
        next_cursor = f"{items[-1].created_at.isoformat()},{items[-1].id}"
    else:
        next_cursor = None
    return Page(items=items, next_cursor=next_cursor, has_more=has_more)
```

The compound cursor `(created_at, id)` handles tie-breaks. The cursor is encoded as a string for the client.

### 2.3 Encoding and decoding cursors

A base64-encoded JSON cursor is the standard:

```python
import base64
import json


def encode_cursor(data: dict) -> str:
    """Encode a cursor as base64 JSON."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode a base64 JSON cursor."""
    try:
        return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (ValueError, json.JSONDecodeError):
        raise BadRequestProblem("Invalid cursor")
```

The client never sees the cursor's structure; the encoding is opaque. The server can change the internal representation (e.g., switch from `id` to `(created_at, id)`) without breaking clients.

### 2.4 The bidirectional cursor

Some APIs support both forward and backward navigation:

```python
@router.get("/users", response_model=Page[UserOut])
async def list_users(
    limit: int = Query(20, ge=1, le=100),
    after_id: int | None = Query(None),
    before_id: int | None = Query(None),
    uow: UnitOfWork = Depends(get_uow),
):
    if after_id and before_id:
        raise BadRequestProblem("Cannot use both after_id and before_id")
    if before_id:
        # Reverse: fetch limit + 1 in reverse, then reverse
        items = await uow.users.list_cursor_desc(before_id=before_id, limit=limit + 1)
        items = list(reversed(items))
        has_more_previous = len(items) > limit
        items = items[-limit:] if has_more_previous else items
        # Cursor for "next" (going back to before before_id) is more complex
    else:
        items = await uow.users.list_cursor(after_id=after_id, limit=limit + 1)
        has_more = len(items) > limit
        items = items[:limit]
    return Page(items=items, next_cursor=..., has_more=...)
```

Bidirectional cursors are more code and rarely needed. The standard pattern is **forward-only**: the client tracks the "next" cursor and walks the list forward.

---

## 3. The Total Count Problem

### 3.1 The challenge

Cursor pagination does not provide a total count. Clients that want "Page 1 of 50" need a separate count.

### 3.2 The `total_count` field

```python
@router.get("/users", response_model=Page[UserOut])
async def list_users(
    limit: int = Query(20, ge=1, le=100),
    after_id: int | None = Query(None),
    include_total: bool = Query(False),
    uow: UnitOfWork = Depends(get_uow),
):
    items = await uow.users.list_cursor(after_id=after_id, limit=limit + 1)
    has_more = len(items) > limit
    items = items[:limit]
    next_cursor = items[-1].id if has_more and items else None
    total_count = None
    if include_total:
        # Expensive on large tables!
        total_count = await uow.users.count()
    return Page(
        items=items, next_cursor=next_cursor, has_more=has_more,
        total_count=total_count,
    )
```

The `include_total` parameter is opt-in. On a table with millions of rows, `COUNT(*)` is slow (full table scan unless covered by an index). The default is `False`; clients that need the count opt in and pay the cost.

### 3.3 The estimated count

For large tables, use `pg_class.reltuples` (PostgreSQL's table statistics) for an approximate count:

```sql
SELECT reltuples FROM pg_class WHERE relname = 'users';
```

The result is the last `ANALYZE`'s estimate. It's not exact, but it's instant. Use it when clients want a "approximately X items" indicator.

---

## 4. Filtering

### 4.1 The basic pattern

```python
@router.get("/users", response_model=Page[UserOut])
async def list_users(
    status: str | None = Query(None, description="Filter by status (active, inactive)"),
    role: str | None = Query(None, description="Filter by role (admin, member, viewer)"),
    search: str | None = Query(None, description="Search by name or email"),
    uow: UnitOfWork = Depends(get_uow),
):
    filters = {}
    if status:
        filters["status"] = status
    if role:
        filters["role"] = role
    items = await uow.users.list(**filters)
    if search:
        items = [u for u in items if search.lower() in u.name.lower() or search.lower() in u.email.lower()]
    return Page(items=items, ...)
```

### 4.2 Using an enum for validation

```python
from enum import Enum


class UserStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"


class UserRole(str, Enum):
    admin = "admin"
    member = "member"
    viewer = "viewer"


@router.get("/users", response_model=Page[UserOut])
async def list_users(
    status: UserStatus | None = Query(None),
    role: UserRole | None = Query(None),
    uow: UnitOfWork = Depends(get_uow),
):
    filters = {}
    if status:
        filters["status"] = status.value
    if role:
        filters["role"] = role.value
    items = await uow.users.list(**filters)
    return Page(items=items, ...)
```

An invalid value (`?status=foobar`) returns a 422 with a Pydantic validation error, not a 500. The OpenAPI schema shows the allowed values.

### 4.3 Multiple values

```python
# ?status=active,inactive (comma-separated)
async def list_users(
    status: list[UserStatus] | None = Query(None),
    uow: UnitOfWork = Depends(get_uow),
):
    if status:
        filters = {"status__in": [s.value for s in status]}
    items = await uow.users.list(**filters)
    ...
```

Or use the `?status=active&status=inactive` pattern (FastAPI handles both).

### 4.4 The SQL injection risk

The Pydantic + FastAPI validation prevents SQL injection at the parameter level. But if you build a raw SQL query from user input, you must use parameterized queries:

```python
# ❌ Vulnerable
stmt = text(f"SELECT * FROM users WHERE status = '{status}'")

# ✅ Safe
stmt = text("SELECT * FROM users WHERE status = :status")
```

SQLAlchemy 2.0's `select()` with column attributes is automatically parameterized.

### 4.5 Filter at the database, not in Python

```python
# ❌ Fetch everything, filter in Python
all_users = await uow.users.list()  # 1M rows
matching = [u for u in all_users if u.status == "active"]

# ✅ Filter at the database
active_users = await uow.users.list(status="active")
```

The database has indexes; Python does not. Always push filtering to the database.

---

## 5. Sorting

### 5.1 The basic pattern

```python
@router.get("/users", response_model=Page[UserOut])
async def list_users(
    sort: str = Query("created_at", description="Field to sort by"),
    order: str = Query("desc", description="asc or desc"),
    uow: UnitOfWork = Depends(get_uow),
):
    # Whitelist allowed sort fields
    allowed_sort_fields = {"id", "name", "email", "created_at"}
    if sort not in allowed_sort_fields:
        raise BadRequestProblem(
            detail=f"Invalid sort field. Allowed: {allowed_sort_fields}",
        )
    sort_column = getattr(User, sort)
    order_by = sort_column.desc() if order == "desc" else sort_column.asc()
    items = await uow.users.list(order_by=order_by)
    return Page(items=items, ...)
```

**Always whitelist** the sort fields. A `?sort=password` would expose password hashes (or worse, an SQL injection vector).

### 5.2 The combined sort+order syntax

```python
# ?sort=-created_at (desc), ?sort=+name (asc)
async def list_users(
    sort: str = Query("-created_at"),
    uow: UnitOfWork = Depends(get_uow),
):
    if sort.startswith("-"):
        field = sort[1:]
        direction = "desc"
    elif sort.startswith("+"):
        field = sort[1:]
        direction = "asc"
    else:
        field = sort
        direction = "asc"
    # ... validate field, apply
```

The `-` prefix is the de facto standard (GitHub, Stripe, etc.). The `+` is explicit ascending.

### 5.3 Multi-field sort

```python
# ?sort=-status,name (primary: status desc, secondary: name asc)
async def list_users(
    sort: str = Query("-status,name"),
    uow: UnitOfWork = Depends(get_uow),
):
    order_by_clauses = []
    for s in sort.split(","):
        if s.startswith("-"):
            field, direction = s[1:], "desc"
        elif s.startswith("+"):
            field, direction = s[1:], "asc"
        else:
            field, direction = s, "asc"
        order_by_clauses.append(
            getattr(User, field).desc() if direction == "desc" else getattr(User, field).asc()
        )
    items = await uow.users.list(order_by=order_by_clauses)
```

The query string is the standard "comma-separated, `-` for desc" syntax. The first field is the primary sort; subsequent fields are tiebreakers.

---

## 6. The Five Common Mistakes

### 6.1 Returning everything

```python
# ❌ Returns all users; the database falls over at 1M rows
@router.get("/users")
async def list_users(uow: UnitOfWork = Depends(get_uow)):
    return await uow.users.list()  # no limit!
```

Always have a default `limit`. Never return more than 1000 rows.

### 6.2 Not indexing the sort column

```python
# ❌ Sort by `email` but no index
stmt = select(User).order_by(User.email)
# PostgreSQL does a full table scan + sort
```

Index the columns you sort by. For compound sorts, use a compound index.

### 6.3 Whitelisting sort fields but not validating direction

```python
# ❌ The user can request an invalid direction
sort: str = Query("name")
# What if the user sends ?sort=-name? The sign is parsed; -name means desc
```

Validate the sign: `if sort[0] not in ("", "-", "+"): raise BadRequestProblem(...)`.

### 6.4 Cursors that leak

```python
# ❌ The cursor contains sensitive data
cursor_data = {"user_id": user.id, "is_admin": user.is_admin, "salary": user.salary}
return encode_cursor(cursor_data)  # client sees the salary!
```

A cursor is exposed to the client. It should contain only the minimum data needed to resume the query: `{"after_id": 123}`. Never include sensitive fields.

### 6.5 Unbounded filter combinations

```python
# ❌ The filter combination is so specific that the query returns 0 rows
# (or the database times out)
filters = {"status": "active", "role": "admin", "department": "engineering",
           "team": "platform", "manager": "alice", ...}
```

A list endpoint with 10 filters is hard to use and hard to maintain. Cap the number of filters (e.g., 5) or use a structured filter syntax (e.g., `?filter[status]=active&filter[role]=admin`).

---

## 7. The API Contract for Lists

### 7.1 The minimum contract

```python
class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False
    total_count: int | None = None  # optional, expensive
```

The `items` are the current page. `next_cursor` is the cursor for the next page. `has_more` is a quick boolean. `total_count` is the optional full count.

### 7.2 Client patterns

```python
# Iterate all pages
async def get_all_users(base_url: str) -> list[User]:
    all_users = []
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["after_id"] = cursor
        response = await client.get(f"{base_url}/users", params=params)
        data = response.json()
        all_users.extend(data["items"])
        if not data["has_more"]:
            break
        cursor = data["next_cursor"]
    return all_users
```

A standard client pattern. The `has_more` boolean is the loop terminator; the `next_cursor` is the next request's parameter.

### 7.3 The cap

```python
# Cap the iteration to prevent runaway clients
MAX_PAGES = 100  # 100 * 100 = 10,000 users; if you have more, use a different API

for _ in range(MAX_PAGES):
    ...
```

Or, on the server side:

```python
@router.get("/users", response_model=Page[UserOut])
async def list_users(
    limit: int = Query(20, ge=1, le=100),  # hard cap
    ...
):
    ...
```

The `le=100` is the server's hard cap. Clients can request up to 100; no more.

---

## 8. Sorting and Cursor Together

The cursor and the sort must be consistent. If the sort changes between requests, the cursor is invalid:

```python
# ❌ The cursor is for "created_at desc" but the second request sorts by "name asc"
# The "next page" is meaningless
GET /users?limit=20&after_id=12345&sort=-created_at
GET /users?limit=20&after_id=12345&sort=name
```

The fix: include the sort in the cursor, or require clients to keep the sort consistent across pages.

```python
# Option 1: include sort in the cursor
cursor = encode_cursor({"after_id": 12345, "sort": "created_at", "order": "desc"})

# Option 2: include sort in the response and document
class Page(BaseModel):
    items: list[T]
    next_cursor: str | None
    has_more: bool
    # The sort used for this page; the client must use the same for the next page
    sort: str
    order: str
```

The second option is more explicit but more API surface. Most APIs go with option 1: the sort is part of the cursor's opaque value.

---

## 9. Filtering Best Practices

### 9.1 Index the filter columns

```python
# Index every column you filter by
class User(Base):
    status: Mapped[str] = mapped_column(String(20), index=True)
    role: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
```

A filter that doesn't use an index is a full table scan. PostgreSQL will refuse large tables without indexes.

### 9.2 The index for compound filters

```python
# For frequent filter combinations, use a compound index
Index("ix_users_status_role", User.status, User.role)
Index("ix_users_tenant_status_created", User.tenant_id, User.status, User.created_at)
```

A compound index covers queries that filter on the leading columns. PostgreSQL can use `ix_users_status_role` for `WHERE status = ?` or `WHERE status = ? AND role = ?`, but not for `WHERE role = ?` alone.

### 9.3 Full-text search

```python
# For search across multiple columns, use PostgreSQL's full-text search
@router.get("/users/search", response_model=Page[UserOut])
async def search_users(
    q: str = Query(..., min_length=1, max_length=200),
    uow: UnitOfWork = Depends(get_uow),
):
    # Use tsvector + tsquery
    stmt = select(User).where(
        User.search_vector.match(q)
    ).order_by(
        # Order by relevance
        User.search_vector.op("ts_rank")(q).desc()
    )
    items = (await uow.session.execute(stmt.limit(100))).scalars().all()
    return Page(items=items, has_more=len(items) > 100, next_cursor=None)
```

A naive `LIKE '%query%'` is a full table scan. For real search, use PostgreSQL's `tsvector` or an external search engine (Elasticsearch, Meilisearch, Typesense).

---

## 10. Código de Compresión

```python
"""
Compresión: Pagination, Filtering, and Sorting
Covers: cursor pagination, filter validation, sort whitelisting,
       query param parsing, compound cursors.
"""
import base64
import json
from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


T = TypeVar("T")


# 1) Page response model
class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False
    total_count: int | None = None


# 2) Cursor encoding
def encode_cursor(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> dict:
    try:
        return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (ValueError, json.JSONDecodeError):
        raise BadRequestProblem("Invalid cursor")


# 3) The list endpoint
class UserStatus(str, Enum):
    active = "active"
    inactive = "inactive"


class UserRole(str, Enum):
    admin = "admin"
    member = "member"


router = APIRouter(prefix="/users")


@router.get("", response_model=Page[UserOut])
async def list_users(
    # Pagination
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, description="Opaque cursor for the next page"),
    # Filtering
    status: UserStatus | None = Query(None),
    role: UserRole | None = Query(None),
    # Sorting
    sort: str = Query("-created_at", description="Field to sort by, prefix - for desc"),
    uow: UnitOfWork = Depends(get_uow),
):
    # Parse cursor
    cursor_data = decode_cursor(cursor) if cursor else {}
    after_id = cursor_data.get("after_id")
    cursor_sort = cursor_data.get("sort", sort)
    if cursor_sort != sort:
        raise BadRequestProblem("Cannot change sort between paginated requests")

    # Validate sort
    allowed_sort = {"id", "name", "created_at"}
    if sort.lstrip("-") not in allowed_sort:
        raise BadRequestProblem(f"Invalid sort. Allowed: {allowed_sort}")

    # Build query
    filters = {}
    if status:
        filters["status"] = status.value
    if role:
        filters["role"] = role.value
    order_column = getattr(User, sort.lstrip("-"))
    order_by = order_column.desc() if sort.startswith("-") else order_column.asc()

    # Fetch limit + 1
    items = await uow.users.list_cursor(
        after_id=after_id, limit=limit + 1, order_by=order_by, **filters
    )
    has_more = len(items) > limit
    items = items[:limit]
    next_cursor = None
    if has_more and items:
        next_cursor = encode_cursor({"after_id": items[-1].id, "sort": sort})
    return Page(items=items, next_cursor=next_cursor, has_more=has_more)


# 4) Multi-field sort parser
def parse_sort(sort: str) -> list:
    clauses = []
    for s in sort.split(","):
        if s.startswith("-"):
            field, direction = s[1:], "desc"
        elif s.startswith("+"):
            field, direction = s[1:], "asc"
        else:
            field, direction = s, "asc"
        clauses.append((field, direction))
    return clauses
```

---

## Key Takeaways

- **Cursor pagination** is the right default for any list that grows or changes. It's stable under concurrent inserts and constant-time at any depth.
- **Offset pagination** is fine for small, stable lists (states, categories). It's simpler but breaks at scale.
- **The cursor is opaque**. Encode it as base64 JSON; the client never sees the structure. The server can change the internal representation without breaking clients.
- **The `limit + 1` trick** detects whether there's a next page. Don't try to count the total unless asked.
- **Whitelist sort fields**. A `?sort=password` is a security incident.
- **Validate filter values** with Pydantic enums. An invalid value returns 422, not 500.
- **Filter at the database, not in Python**. The database has indexes; Python does not.
- **Index the sort and filter columns**. A query without an index is a full table scan.
- **Cap the limit** (`le=100`). Never let a client request an unbounded page.
- **The sort must be consistent between pages**. Include it in the cursor, or document that clients must not change it.
- **`total_count` is optional and expensive**. On large tables, use an estimate (PostgreSQL's `reltuples`) or make it opt-in.

## References

- [Stripe API Pagination](https://stripe.com/docs/api/pagination)
- [GitHub REST API Pagination](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)
- [Microsoft REST API Guidelines — Pagination](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#121-pagination)
- [Use the Index, Luke — A Guide to Database Performance for Developers](https://use-the-index-luke.com/)
- [PostgreSQL Documentation — Cursor-based Pagination](https://www.postgresql.org/docs/current/queries-limit.html)
- [FastAPI — Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params/)
- [Pydantic — Field Constraints](https://docs.pydantic.dev/latest/concepts/fields/)
