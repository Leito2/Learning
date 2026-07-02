# 🔄 Alembic Migrations Workflow

## 🎯 Learning Objectives

- Initialize Alembic in a SQLAlchemy 2.0 project and configure it for async engines
- Generate, review, and apply migrations without breaking production
- Handle schema evolution: add columns, rename, change types, split tables
- Use branching and merging for parallel feature development
- Deploy migrations safely: zero-downtime strategies, lock timeouts, online schema changes

## Introduction

Schema changes are the most failure-prone part of a database-backed service. A bad migration can corrupt data, lock tables for hours, or take the service down. Alembic is the de facto migration tool for SQLAlchemy, and its design — a chain of versioned revision files, each representing a single atomic schema change — is what production engineers rely on to evolve databases without downtime.

The two core workflows are **autogenerate** (Alembic diffs your models against the live DB and writes the migration) and **manual revisions** (you hand-write the upgrade/downgrade when the diff is wrong or when the change is too complex for autogenerate to capture). Production teams use a mix: autogenerate for 80% of changes, manual for the rest.

This note covers the full workflow: from `alembic init` to deploying migrations to a live PostgreSQL cluster. Async support is included — Alembic works fine with `create_async_engine` once you understand a small configuration wrinkle.

---

## 1. Initializing Alembic

### 1.1 The `alembic init` command

```bash
alembic init alembic
```

This creates an `alembic/` directory:

```
alembic/
├── env.py              # ← the migration runner, you'll edit this
├── script.py.mako      # template for new migration files
├── README
└── versions/           # ← your migration files live here
alembic.ini            # ← the configuration file
```

### 1.2 Configuring `alembic.ini`

The default `alembic.ini` has a placeholder `sqlalchemy.url` that should never be used in production. The right approach is to inject the URL from your app config.

```ini
# alembic.ini — leave the URL empty
sqlalchemy.url =
```

Then override it in `env.py`.

### 1.3 Configuring `env.py` for async

The default `env.py` uses a sync engine. For SQLAlchemy 2.0 async, replace it with the async template. The cleanest way is to copy `alembic init -t async alembic`.

```bash
rm -rf alembic
alembic init -t async alembic
```

This generates the async-aware `env.py`. The key section:

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import your models' metadata
from app.db.base import Base  # your DeclarativeBase
from app.models import user, post, comment  # noqa: F401 ensure models are imported

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override sqlalchemy.url from your app settings
config.set_main_option(
    "sqlalchemy.url",
    str(settings.DATABASE_URL),  # pydantic Settings, env vars, etc.
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode, emitting SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # migrations should not use a pool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

The three subtle points:

1. **All models must be imported** before `target_metadata` is referenced. If a model isn't imported, its table isn't in `Base.metadata`, and Alembic will suggest dropping it.
2. **`poolclass=pool.NullPool`**: migrations should not reuse connections from a pool. Open, run, close.
3. **The URL is read from `config.set_main_option`**, so `alembic.ini` can stay generic across environments.

---

## 2. The First Migration: Autogenerate

### 2.1 `alembic revision --autogenerate`

```bash
alembic revision --autogenerate -m "initial schema"
```

Alembic connects to the DB, reads the current schema from `Base.metadata`, diffs the two, and writes a new revision file in `alembic/versions/`. A typical file looks like:

```python
# alembic/versions/abc123_initial_schema.py
"""initial schema

Revision ID: abc123def456
Revises:
Create Date: 2026-07-01 12:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "abc123def456"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_users_email", "users", ["email"])


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
```

### 2.2 Reviewing the autogenerated file

**Always read the autogenerated file before applying it.** Common autogenerate mistakes:

- Missing `unique` constraints on a column you marked `unique=True` in the model (Alembic sometimes doesn't detect them).
- Default values applied as Python defaults instead of `server_default` (means the DB doesn't enforce them).
- Missing `onupdate` callbacks for `onupdate=func.now()`.
- Wrong column type for `String | None` (sometimes generates as `VARCHAR` without length).
- Adding a `NOT NULL` column without a default — fails on populated tables.
- Renames detected as drop + add. Alembic cannot detect renames; it sees "old column gone, new column present" and writes a `drop_column` + `add_column`. You must manually rewrite this as a `alter_column` with `new_column_name=`.

### 2.3 Applying the migration

```bash
alembic upgrade head
```

This runs all migrations from the current revision to `head` (the latest). The database schema now matches `Base.metadata`.

### 2.4 Verifying the state

```bash
alembic current     # show current revision
alembic history     # show the full chain
alembic history --verbose  # show file paths and metadata
```

---

## 3. The Day-to-Day Workflow

### 3.1 Adding a column

Suppose you add a new field to the `User` model:

```python
class User(Base):
    # ... existing fields ...
    avatar_url: Mapped[str | None] = mapped_column(String(500), default=None)
```

```bash
alembic revision --autogenerate -m "add user avatar_url"
```

The generated migration:

```python
def upgrade() -> None:
    op.add_column("users", sa.Column("avatar_url", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "avatar_url")
```

### 3.2 Adding a NOT NULL column

This is where things get tricky. If the table has rows, adding `nullable=False` without a default fails. Two solutions:

**Solution 1: Add as nullable, backfill, then alter**:

```python
def upgrade() -> None:
    # Step 1: add as nullable
    op.add_column("users", sa.Column("phone", sa.String(20), nullable=True))
    # Step 2: backfill (manual or with raw SQL)
    op.execute("UPDATE users SET phone = '' WHERE phone IS NULL")
    # Step 3: alter to NOT NULL
    op.alter_column("users", "phone", nullable=False)
```

**Solution 2: Add with a server default** (the database fills the column for existing rows):

```python
def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
    # After the column is added, drop the server default if you want only
    # new rows to have a value:
    op.alter_column("users", "is_verified", server_default=None)
```

### 3.3 Renaming a column

Alembic cannot detect renames from model diffs. Hand-write the migration:

```python
def upgrade() -> None:
    op.alter_column("users", "name", new_column_name="display_name")


def downgrade() -> None:
    op.alter_column("users", "display_name", new_column_name="name")
```

`ALTER TABLE ... RENAME COLUMN` is fast in PostgreSQL (metadata-only operation, no table rewrite). On a large table, this is safe to do online.

### 3.4 Changing a column type

```python
def upgrade() -> None:
    op.alter_column(
        "users",
        "bio",
        type_=sa.Text(),
        existing_type=sa.String(500),
    )
```

**Warning**: `ALTER TABLE ... ALTER COLUMN ... TYPE` may rewrite the entire table. For a large table in production, this can lock writes for hours. Use PostgreSQL's tricks:

- `ALTER TABLE ... ALTER COLUMN ... TYPE USING expression` to coerce.
- `pg_repack` for online table rewrite.
- Create the new column, dual-write, backfill, then drop the old one (a multi-migration rollout).

### 3.5 Adding an index concurrently

PostgreSQL's default `CREATE INDEX` locks the table against writes for the duration of the build. On a large table, this is a multi-minute outage. Use `CONCURRENTLY`:

```python
def upgrade() -> None:
    op.execute("CREATE INDEX CONCURRENTLY ix_users_email_active ON users (email, is_active)")


def downgrade() -> None:
    op.execute("DROP INDEX CONCURRENTLY ix_users_email_active")
```

Alembic's `op.create_index` does not support `CONCURRENTLY` directly. The workaround is `op.execute("...")` with raw SQL. The migration must also run outside a transaction (PostgreSQL does not allow `CONCURRENTLY` inside a transaction block):

```python
def upgrade() -> None:
    # Tell Alembic: don't wrap this in a transaction
    op.execute("COMMIT")  # close any implicit transaction
    op.create_index(
        "ix_users_email_active", "users", ["email", "is_active"],
        postgresql_concurrently=True,
    )
```

Or in the `alembic.ini`:

```ini
[alembic]
# Per-migration: set transaction_per_migration = false
```

Then in the migration:

```python
# Mark this migration as not requiring a transaction
revision = "..."
transactional_ddl = False
```

### 3.6 Dropping a column

```python
def upgrade() -> None:
    op.drop_column("users", "legacy_field")
```

`ALTER TABLE ... DROP COLUMN` is fast in modern PostgreSQL (no rewrite when the column is not the only one). On a busy production table, consider:

1. Mark the column as unused in code (keep it in the DB).
2. After a few deploys, drop the column in a separate migration.

This two-step approach lets you roll back the code change without losing data.

---

## 4. Branching and Merging

### 4.1 When branches happen

Two developers working on parallel features may each create a migration. The revision graph then has a fork:

```
   abc123 (head) ──┐
                   ├── def456 (mine)
   ghi789 (yours) ──┘
```

Without intervention, `alembic upgrade head` fails because there are two heads.

### 4.2 The merge migration

```bash
alembic merge -m "merge feature branches" def456 ghi789
```

This creates a new revision that has both as dependencies. After applying it:

```
   abc123 ──┐
            ├── def456 ──┐
   ghi789 ──┘             ├── jkl012 (new head, empty upgrade)
                          └── ...
```

The new revision's `upgrade()` and `downgrade()` are typically empty (the migration is a graph operation, not a schema operation). If both branches added a column with the same name, you'll have a conflict to resolve in the merge migration.

### 4.3 Best practice: short-lived branches

Long-lived branches cause merge conflicts in the migration graph. Best practice:

- Migrations are **committed per PR** (no local-only revisions).
- Branches are **short-lived** (a day or two).
- The CI pipeline runs `alembic upgrade head` against an empty DB on every PR; if the chain has a fork, CI fails.

---

## 5. Data Migrations vs Schema Migrations

### 5.1 The difference

A **schema migration** changes the structure: add a column, drop a table, change a type. Alembic's `op.*` operations handle this.

A **data migration** changes the data: backfill a new column from old data, anonymize PII, delete stale rows. Use `op.execute("...")` with raw SQL or a `Session.execute(...)` call.

### 5.2 Pattern: schema + data in the same migration

```python
def upgrade() -> None:
    # 1) Add the new column (nullable for now)
    op.add_column("users", sa.Column("display_name", sa.String(100), nullable=True))

    # 2) Backfill from the old column
    op.execute("UPDATE users SET display_name = name WHERE display_name IS NULL")

    # 3) Make it NOT NULL
    op.alter_column("users", "display_name", nullable=False)


def downgrade() -> None:
    op.drop_column("users", "display_name")
```

This pattern is safe for tables of any size, as long as the backfill can complete within the migration's expected runtime. For multi-million-row tables, split the data migration into a separate "backfill" script that runs offline.

### 5.3 The async session inside a data migration

If your data migration needs ORM access (e.g., to use models with relationships), you can use an async session:

```python
from sqlalchemy.ext.asyncio import async_engine_from_config
from app.db.session import SessionLocal
from app.models.user import User

def upgrade() -> None:
    # Schema part
    op.add_column("users", sa.Column("slug", sa.String(100), nullable=True))

    # Data part using ORM
    conn = op.get_bind()
    # conn is a sync connection inside the migration; convert to async if needed
    # ... or just use raw SQL for simplicity
    op.execute("UPDATE users SET slug = lower(replace(email, '@', '-'))")
```

In practice, raw SQL is faster and easier to reason about for data migrations. Reach for the ORM only when the logic is complex (e.g., "for each user, generate a unique slug based on a hash of their email + creation timestamp").

---

## 6. The `alembic_version` Table

### 6.1 What it is

Every Alembic-managed database has a table called `alembic_version`:

```sql
CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL
);
```

The single row contains the current revision ID. Alembic checks this table on every `upgrade` and `downgrade`.

### 6.2 Manual intervention

If `alembic_version` is out of sync with the actual schema (e.g., after a manual SQL change), you can fix it:

```bash
# Stamp the DB as being at a specific revision
alembic stamp head
alembic stamp abc123def456

# View the current revision
alembic current
```

`alembic stamp` is the right tool when the schema was changed outside Alembic (a manual `psql` command, a 1Password DBA runbook, a Terraform-managed change). It records the new state without applying anything.

---

## 7. Zero-Downtime Production Deploys

### 7.1 The 5-stage pattern for adding a NOT NULL column

A safe production rollout of "add a NOT NULL column with no default" requires five migration stages, each shipped in a separate deploy:

**Stage 1**: Add the column as nullable, with no default.

```python
op.add_column("users", sa.Column("phone", sa.String(20), nullable=True))
```

**Stage 2**: Deploy code that **writes** to both old and new column (dual-write).

```python
# In your model logic
user.phone = new_value
user.legacy_phone = new_value
```

**Stage 3**: Backfill historical data.

```python
op.execute("UPDATE users SET phone = legacy_phone WHERE phone IS NULL")
```

**Stage 4**: Deploy code that **reads** from the new column.

**Stage 5**: Add the NOT NULL constraint, drop the old column.

```python
op.alter_column("users", "phone", nullable=False)
op.drop_column("users", "legacy_phone")
```

Each stage is its own migration file. Each requires a code deploy before the next migration runs. The process is slow but safe.

### 7.2 Online DDL in PostgreSQL

Operations that are safe to do online without locking writes:

| Operation | Online in PostgreSQL? | Notes |
|-----------|----------------------|-------|
| `ADD COLUMN` (nullable, no default) | ✅ | Metadata-only |
| `ADD COLUMN ... DEFAULT NULL` | ✅ | Metadata-only |
| `ADD COLUMN ... DEFAULT constant` | ✅ (11+) | Stores the default in metadata, no rewrite |
| `ADD COLUMN ... DEFAULT volatile` | ❌ | Rewrites the table |
| `DROP COLUMN` | ✅ | Quick metadata operation |
| `RENAME COLUMN` / `RENAME TABLE` | ✅ | Metadata-only |
| `CREATE INDEX` (default) | ❌ | Locks writes |
| `CREATE INDEX CONCURRENTLY` | ✅ | Allowed outside transaction |
| `ALTER COLUMN ... TYPE` | ❌ | Rewrites the table |
| `ADD CONSTRAINT` | ❌ (with validation) | Locks table |
| `ADD CONSTRAINT ... NOT VALID` | ✅ | Adds without validating existing rows; validate later |
| `SET NOT NULL` | ❌ | Validates all rows |
| `SET NOT NULL ... NOT VALID` (12+) | ✅ | Validates later |

For any operation that's not online, use the multi-stage pattern.

---

## 8. Testing Migrations

### 8.1 The "apply all, then apply all in reverse" test

```python
import pytest
from alembic import command
from alembic.config import Config


def test_migrations_apply_and_reverse():
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    # Schema should be in the latest state
    command.downgrade(cfg, "base")
    # Schema should be empty
```

### 8.2 The "schema matches models" test

```python
from sqlalchemy import MetaData
from app.db.base import Base


def test_schema_matches_models():
    # Connect to a DB that has been migrated to head
    engine = create_engine(test_db_url)
    metadata = MetaData()
    metadata.reflect(engine)
    assert set(metadata.tables) == set(Base.metadata.tables)
```

### 8.3 The "applied on empty DB" test

```bash
# CI step
dropdb test_db
createdb test_db
alembic upgrade head
pytest
```

A test DB started from scratch, migrated to head, then exercised by the test suite catches the bug where a model was changed but the migration was forgotten.

---

## 9. Common Pitfalls

### 9.1 Forgetting to import a model

Alembic only sees the tables of the models it knows about. If you add a new model but forget to import it in `env.py`, Alembic will suggest dropping its table (because it doesn't know it exists in the model layer).

### 9.2 Detecting renames as drop + add

Alembic compares column names by exact match. If you rename `name` to `display_name` in the model, the diff says "drop `name`, add `display_name`". You must hand-rewrite the migration as `op.alter_column(... new_column_name=...)`.

### 9.3 Using `op.execute("DROP TABLE ...")` in a downgrade

```python
# ❌ Data loss
def downgrade() -> None:
    op.execute("DROP TABLE users")

# ✅ Use op.drop_table
def downgrade() -> None:
    op.drop_table("users")
```

The `op.drop_table` form is reversible within Alembic's bookkeeping.

### 9.4 Running migrations in a transaction with `CONCURRENTLY`

```python
# ❌ Fails: CONCURRENTLY cannot be inside a transaction
def upgrade() -> None:
    op.create_index(..., postgresql_concurrently=True)
    # alembic wraps this in a transaction by default

# ✅ Set transactional_ddl = False on the migration
def upgrade() -> None:
    op.execute("COMMIT")
    op.create_index(..., postgresql_concurrently=True)
```

---

## 10. Código de Compresión

```python
"""
Compresión: Alembic Migrations Workflow
Cubre: env.py config, autogenerate, NOT NULL rollout, branching,
       data migrations, online DDL.
"""
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.db.base import Base
from app.models import user, post, comment  # noqa: F401
from app.core.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


# Sample multi-stage migration
# alembic/versions/abc_add_phone_column.py
def upgrade() -> None:
    """Stage 1: add the column as nullable, no default."""
    op.add_column("users", sa.Column("phone", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "phone")


# alembic/versions/def_backfill_phone.py
def upgrade() -> None:
    """Stage 2: backfill from the legacy column."""
    op.execute("UPDATE users SET phone = legacy_phone WHERE phone IS NULL")


# alembic/versions/ghi_phone_not_null.py
def upgrade() -> None:
    """Stage 3: enforce NOT NULL after backfill."""
    op.alter_column("users", "phone", nullable=False)
```

---

## Key Takeaways

- `alembic init -t async` gives you the async-aware `env.py` template. Always import every model so Alembic sees its full schema.
- **Always review autogenerated files.** Alembic cannot detect renames, defaults, or several subtle constraints; the human reviewer is the safety net.
- Adding a NOT NULL column to a populated table is a 3-stage process: nullable + default → backfill → alter to NOT NULL. In production, this is 5 stages with dual-write and dual-read code deploys.
- `CREATE INDEX CONCURRENTLY` is the only safe way to add an index to a busy table. It must run outside a transaction.
- Use `alembic merge` to reconcile parallel branches. Long-lived feature branches cause merge conflicts in the migration graph.
- `alembic stamp` records a version without applying it. Use it when the schema was changed outside Alembic.
- Data migrations and schema migrations can live in the same revision file, but data migrations on large tables should run offline (background job, not within the migration).

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/en/latest/)
- [Alembic — Auto Generating Migrations](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [Alembic — Branches](https://alembic.sqlalchemy.org/en/latest/branches.html)
- [PostgreSQL — Concurrent Index Creation](https://www.postgresql.org/docs/current/sql-createindex.html#SQL-CREATEINDEX-CONCURRENTLY)
- [Shopify's Online Schema Migrations for MySQL (analogous techniques)](https://shopify.engineering/online-mysql-schema-migrations)
- [GitHub's online schema migrations at scale](https://github.blog/engineering/enhancements-to-github-actions-mysql/)
