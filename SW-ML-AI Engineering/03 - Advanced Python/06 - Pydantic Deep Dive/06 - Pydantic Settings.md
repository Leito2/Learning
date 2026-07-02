# Pydantic Settings

## 1. BaseSettings

`BaseSettings` extends `BaseModel` with automatic env var loading:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    app_name: str = "my-app"
    database_url: str
    api_key: str
    debug: bool = False
```

Fields without defaults are **required from the environment**. Fields with defaults are optional.

```bash
# .env
DATABASE_URL=postgresql://localhost:5432/mydb
API_KEY=sk-...
```

Env var names match field names (case-insensitive by default). `database_url` reads from `DATABASE_URL` or `database_url`.

## 2. SettingsConfigDict

| Option | Effect |
|--------|--------|
| `env_file` | Path(s) to `.env` file (str or list[str]) |
| `env_prefix` | Prefix for env vars (e.g., `"MYAPP_"`) |
| `env_nested_delimiter` | Delimiter for nested model env vars (e.g., `"__"`) |
| `env_file_encoding` | `.env` encoding (default `"utf-8"`) |
| `secrets_dir` | Directory with secret files (Docker/K8s secrets) |
| `extra` | Behavior for unknown env vars |
| `frozen` | Immutable settings |

### Prefix and Nested Delimiter

```python
class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_nested_delimiter="__")

    database: DatabaseSettings
    name: str
```

```bash
APP_NAME=my-service
APP_DATABASE__HOST=db.internal
APP_DATABASE__PORT=6432
```

## 3. .env File Hierarchy

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env.shared", ".env.local"],  # shared defaults, then local overrides
        extra="forbid",
    )
```

Typical hierarchy: `.env.shared` (committed) → `.env.local` (gitignored) → env vars (runtime).

## 4. Secrets Directory

For Docker/K8s mounted secrets:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(secrets_dir="/run/secrets")

    db_password: str
    api_key: str
```

Each file in the secrets directory is a field. File name = field name (case-insensitive).

## 5. Validation and Transformation

Settings support all Pydantic validators:

```python
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str | None = None

    @field_validator("database_url", mode="before")
    @classmethod
    def expand_env_syntax(cls, v: str) -> str:
        return os.path.expandvars(v) if isinstance(v, str) else v
```

## 6. CLI Integration (argparse-style)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    host: str = "0.0.0.0"
    port: int = 8000
```

```bash
python app.py --host 127.0.0.1 --port 9000
```

## 7. Nested Settings

```python
class Database(BaseModel):
    host: str
    port: int = 5432
    name: str
    user: str
    password: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_nested_delimiter="__")

    db: Database
    app_name: str
```

```bash
APP_APP_NAME=api
APP_DB__HOST=localhost
APP_DB__NAME=mydb
APP_DB__USER=admin
APP_DB__PASSWORD=secret
```

## 8. Production Pattern: Singleton + Validation on Startup

```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()  # validates on first call

# In app startup:
settings = get_settings()  # crashes fast if env is misconfigured
```

## Key Takeaways

- `BaseSettings` reads from env vars, `.env` files, and secrets dirs
- `SettingsConfigDict` controls prefix, delimiter, file paths
- `.env` hierarchy: shared → local → env vars (each level overrides)
- All Pydantic validators work on settings fields
- Singleton pattern with `lru_cache` for fast startup validation
- `cli_parse_args` for CLI integration without argparse
