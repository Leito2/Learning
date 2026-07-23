# 🔍 Mypy and Pyright Strict Mode in CI

Static type checkers turn Python's type hints into actual correctness guarantees — but only when configured to be strict. By default, mypy ignores untyped code and silently accepts implicit `Any`. Pyright is stricter out of the box but still has gradual-typing escape hatches. The difference between "type hints that look good" and "type hints that catch bugs" is the difference between default settings and `--strict` (mypy) or `strict: true` (pyright).

This note covers the strict mode configuration, the most useful strict flags, and the CI integration patterns that make type errors fail builds. By the end you will know exactly which flags to enable, which to disable, and how to integrate static analysis into a portfolio project or a production codebase.

## 🎯 Learning Objectives

- Configure `mypy --strict` and `pyright strict` correctly.
- Distinguish the most useful strict flags (`disallow_untyped_defs`, `disallow_any_*`, `warn_return_any`).
- Integrate mypy and pyright into a CI pipeline (GitHub Actions, pre-commit).
- Generate per-package strict settings (gradual adoption).
- Decide when to use mypy vs pyright.

## 1. The Problem: Type Hints Are Not Type Checks

A type hint is documentation:

```python
def add(a: int, b: int) -> int:
    return a + b

add("hello", "world")  # no error at runtime; mypy without --strict accepts this too
```

The annotation says `int`, but the function is called with `str`. Without strict mode, mypy accepts the call because the function has typed parameters but the **caller** is untyped (or implicitly `Any`). Strict mode propagates the type and rejects the call.

```bash
# Default mypy
$ mypy script.py
Success: no issues found in 1 source file

# mypy --strict
$ mypy --strict script.py
error: Argument 1 to "add" has incompatible type "str"; expected "int"  [arg-type]
```

The strict flag turns documentation into a check.

## 2. Mypy `--strict`

```bash
mypy --strict src/
```

`--strict` is an alias for **seven individual flags**. Each catches a specific class of bug:

| Flag | Catches |
|------|---------|
| `--disallow-untyped-defs` | Functions without annotations are errors |
| `--disallow-incomplete-defs` | Functions with partial annotations are errors |
| `--no-implicit-optional` | `def f(x: int = None)` is an error (must be `Optional[int]`) |
| `--warn-return-any` | Returning `Any` from a typed function is a warning |
| `--warn-unused-ignores` | `# type: ignore` comments that don't suppress anything are flagged |
| `--strict-equality` | `1 == "1"` is an error (comparing unrelated types) |
| `--disallow-any-generics` | `list[Any]` is an error (must specify the type parameter) |

### Recommended Strict Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true           # Flag code that can never execute
warn_redundant_casts = true        # Catch useless `cast()` calls
warn_unused_configs = true         # Catches dead config in pyproject
disallow_subclassing_any = true    # Don't allow subclassing `Any`

# Per-module overrides (gradual adoption)
[[tool.mypy.overrides]]
module = "third_party.*"
ignore_missing_imports = true      # Skip type stubs we don't have

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false      # Tests can be looser
```

## 3. Pyright `strict`

```json
// pyrightconfig.json
{
  "include": ["src/"],
  "strict": ["src"],
  "pythonVersion": "3.12",
  "reportMissingTypeStubs": false,
  "reportUnusedImport": "error"
}
```

Pyright's `strict: ["src"]` is directory-scoped: code in `src/` is strict, code in `tests/` can be looser.

### Pyright-Specific Flags

```json
{
  "strict": ["src"],
  "reportCallInDefaultInitializer": "error",  // Catch `def f(x: int = g())` issues
  "reportInvalidTypeForm": "error",            // Catch malformed annotations
  "reportMissingTypeArgument": "error",         // Force generic args (matches mypy strict)
  "reportSelfClsParameterName": "warning",      // Convention: `self` not `this`
  "reportUnnecessaryIsInstance": "warning",     // Flag `isinstance(x, type(x))` and the like
  "reportUnnecessaryTypeIgnoreComment": "warning"
}
```

## 4. Mypy vs Pyright

| Aspect | mypy | pyright |
|--------|------|---------|
| Speed | Slower (JIT/CPython) | Faster (TypeScript-compiled) |
| PEP support | Some lag (PEP 695 support maturing) | Earlier (Microsoft Pylance team) |
| Strictness | `--strict` is opt-in | More strict by default |
| Editor integration | VS Code, PyCharm | VS Code (Pylance) |
| Library stubs | Mypy-based stubs ecosystem | Compatible |
| Best for | Library maintainers, CI | IDE-first, monorepos |

**Recommendation:** Use **pyright** for editor-driven development (Pylance in VS Code) and **mypy** for CI (broader library stub ecosystem, predictable). Both can run side-by-side: pyright in editor, mypy in CI.

## 5. CI Integration

### GitHub Actions

```yaml
# .github/workflows/type-check.yml
name: Type Check

on: [push, pull_request]

jobs:
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install mypy==1.10.0
      - run: pip install -r requirements.txt
      - run: mypy --strict src/
      # Optional: pyright as a second check
      - run: pip install pyright==1.1.350
      - run: pyright src/
```

### Pre-Commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        files: ^src/
        args: [--strict, --ignore-missing-imports]
        additional_dependencies: [types-requests, types-redis]
```

A failing type check blocks the commit. This is the cheapest place to catch type errors — before they enter CI.

### Diff-Only Type Check

For large codebases, run mypy only on changed files:

```bash
# mypy --follow-imports=silent on the diff
git diff --name-only origin/main | grep '\.py$' | xargs mypy --strict
```

Pyright has built-in diff mode:

```bash
pyright --threads $(nproc) src/
```

## 6. Type Stubs (`*.pyi`)

When a third-party library doesn't ship types, use **stub files** (`.pyi`):

```python
# stubs/fake_lib.pyi
def search(query: str, top_k: int = 5) -> list[dict[str, str]]: ...
class Client:
    def __init__(self, host: str, port: int) -> None: ...
    def query(self, sql: str) -> list[tuple]: ...
```

Many popular libraries ship `types-*` packages:

```bash
pip install types-requests types-redis types-pyyaml
```

mypy finds them automatically; pyright finds them via its own resolver.

### When to Write Your Own Stubs

If a library you depend on has 0 type support and is widely used, write a `.pyi` and submit a PR upstream. Until then, stub it locally:

```bash
# In your repo:
stub_dir/
  └── third_party_lib/
      └── __init__.pyi
```

Tell mypy about the stub dir in `pyproject.toml`:

```toml
[tool.mypy]
mypy_path = "stub_dir"
```

## 7. ❌/✅ Antipatterns

### ❌ `Any` to silence the checker

```python
from typing import Any

def process(data: Any) -> Any:
    return data["key"]  # no static check; runtime KeyError if missing
```

### ✅ Precise types, then narrow on uncertainty

```python
def process(data: dict[str, str]) -> str | None:
    return data.get("key")  # statically checked; safe at runtime
```

### ❌ `cast()` everywhere

```python
from typing import cast

def handler(req: object) -> dict:
    body = cast(dict, req)  # unchecked
    return body
```

### ✅ Validate first, narrow after

```python
def handler(req: object) -> dict:
    if not isinstance(req, dict):
        raise TypeError
    return req  # narrowed to dict
```

### ❌ `# type: ignore` everywhere

```python
x = something_weird()  # type: ignore
```

### ✅ Targeted, documented `# type: ignore`

```python
x = something_weird()  # type: ignore[arg-type]  # third-party lib has wrong stubs
```

### ❌ Mypy without `--strict`

```python
# With default mypy:
def add(a: int, b: int) -> int: ...
add("hi", "world")  # ⚠️ no error
```

### ✅ `--strict` from day one

```python
# With mypy --strict:
def add(a: int, b: int) -> int: ...
add("hi", "world")  # ✅ error: incompatible type
```

## 8. Gradual Adoption in Legacy Codebases

A 100k-line legacy codebase can't be strict all at once. The pattern:

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
# Start: not strict

[[tool.mypy.overrides]]
module = "src.legacy_module"
ignore_errors = true

[[tool.mypy.overrides]]
module = "src.new_module"
strict = true

# ...over time, remove legacy_module from the override list
```

Adopt strict in new code; migrate legacy modules one PR at a time. Pyright's directory-scoped strict mode is cleaner for this.

## 9. Performance: Speeding Up mypy in CI

```toml
# pyproject.toml
[tool.mypy]
# Cache results between runs
cache_dir = ".mypy_cache"

# Use mypy daemon (faster incremental checks)
# daemon = true  # requires mypy[dmypy] install

# Skip library code (focus on your changes)
follow_imports = "silent"
```

```bash
# Use dmypy for incremental checks (10x faster)
dmypy run -- src/
```

## 10. Production Reality

**Caso real — Multi-Agent Research System:** Adopted mypy `--strict` early. Initial wave of errors: 47 (mostly `Any` returns from LangChain integrations). Resolved with `TypeGuard` predicates and Protocol definitions for LangChain `Runnable`, `ChatModel`, etc. Now any new node that doesn't pass strict typing fails CI.

**Caso real — LLM Edge Gateway:** Pyright in editor (Pylance) + mypy in CI. Catches ~12 type errors per week before commit. Most common: missing return type annotations, `Optional` vs `| None` mixing, dict-key access on `Any`. Each one would have been a runtime bug in production.

## 📦 Compression Code

```toml
# 📦 Compression: mypy + pyright strict configuration in two files

# === pyproject.toml ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
warn_redundant_casts = true
disallow_subclassing_any = true
show_error_codes = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "third_party.*"
ignore_missing_imports = true

# === pyrightconfig.json ===
{
  "include": ["src/"],
  "strict": ["src"],
  "pythonVersion": "3.12",
  "reportCallInDefaultInitializer": "error",
  "reportInvalidTypeForm": "error",
  "reportMissingTypeArgument": "error",
  "reportSelfClsParameterName": "warning",
  "reportUnnecessaryIsInstance": "warning",
  "reportUnnecessaryTypeIgnoreComment": "warning"
}

# === .github/workflows/type-check.yml ===
# (See CI integration section above)
```

## 🎯 Key Takeaways

1. **`mypy --strict` and `pyright strict`** turn documentation into checks. Without strict mode, type hints are aspirational.
2. **Mypy strict is 7 flags.** The most valuable: `disallow_untyped_defs`, `warn_return_any`, `no_implicit_optional`.
3. **Pyright is faster and stricter by default**; mypy has the broader stub ecosystem.
4. **CI integration is one GitHub Action** — a failing type check fails the build.
5. **Pre-commit hooks** catch type errors before they enter CI.
6. **Gradual adoption** uses per-module overrides; strict in new code, lenient in legacy.
7. **`Any` and `cast()` are escape hatches** — use sparingly, document why.

## References

- [[00 - Welcome to Modern Python Typing|Welcome]] — the 2026 typing landscape.
- [[02 - Type Narrowing - TypeGuard, isinstance and Asserts|Type Narrowing]] — narrowing is the most impactful feature strict mode unlocks.
- [[06 - Runtime Checkers - Beartype, Pydantic, Cattrs|Runtime Checkers]] — runtime checks complement static checks.
- Mypy docs: https://mypy.readthedocs.io/en/stable/
- Pyright docs: https://microsoft.github.io/pyright/