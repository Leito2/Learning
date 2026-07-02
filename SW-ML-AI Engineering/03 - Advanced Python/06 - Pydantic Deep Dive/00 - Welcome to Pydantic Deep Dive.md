# Welcome to Pydantic Deep Dive

Pydantic is the most widely used data validation library in the Python ML ecosystem. It powers FastAPI request/response validation, LLM structured outputs, agent tool schemas, configuration management, and ML inference pipelines. Yet it is rarely studied as a standalone subject.

This course fixes that: 10 modules covering the entire Pydantic v2 surface area — from `BaseModel` fundamentals to `WrapValidator`, `model_config`, serialization, settings, SQLModel, JSON Schema, and production optimization.

## Why Pydantic v2

- **Rust core** (`pydantic-core`): 5-50x faster validation than v1
- **`Annotated`-based validators**: composable, reusable validation via type hints
- **`model_config` unification**: all settings in a single `ConfigDict`
- **`model_validate` / `model_validate_json`**: explicit parsing with clear failure modes
- **`TypeAdapter`**: validate any type, not just `BaseModel`
- **`@computed_field`**: computed properties that appear in serialization

## Course Map

| # | Module | Lines |
|---|--------|:-----:|
| 00 | Welcome — Why Pydantic v2 | 80 |
| 01 | BaseModel, Field and Type System | 120 |
| 02 | Validators Deep Dive | 120 |
| 03 | model_config Systematic Tour | 100 |
| 04 | Serialization Mastery | 100 |
| 05 | Advanced Model Patterns | 120 |
| 06 | Pydantic Settings | 100 |
| 07 | SQLModel Bridge | 100 |
| 08 | JSON Schema and OpenAPI | 80 |
| 09 | Performance & Production + Capstone | 180 |
| **Total** | | **~1,100** |

## Pre-requisites

- Python 3.10+ typing – [[../03 - Python Avanzado/00 - Bienvenida|Python Avanzado]]
- Basic FastAPI – [[../../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/00 - Welcome|FastAPI for ML]]

## References

- [Pydantic v2 docs](https://docs.pydantic.dev/latest/)
- [pydantic-core source](https://github.com/pydantic/pydantic-core)
- [SQLModel docs](https://sqlmodel.tiangolo.com/)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## Cross-links in the Vault

- PydanticAI agent framework: [[../../../07 - AI Agents y Agentic Systems/17 - Production Agent Frameworks/02 - PydanticAI - Type-Safe Agents|07/17/02 - PydanticAI]]
- Pydantic for ML schemas: [[../../../10 - Cloud, Infra y Backend/31 - FastAPI for ML/02 - Pydantic v2 for ML Input Output Schemas|10/31/02 - Pydantic v2 for ML]]
- FastAPI caching with Pydantic: [[../../../10 - Cloud, Infra y Backend/42 - Caching Strategies for FastAPI/03 - Response Caching Decorators|10/42/03 - Response Caching]]
- Structured LLM outputs: [[../../../06 - Large Language Models/08 - Generacion de Texto y Decodificacion/02 - Control de Generacion|06/08/02 - Control de Generacion]]
- LLM Gateway: [[../../../06 - Large Language Models/19 - LLM Gateway Patterns and LiteLLM/06 - Capstone - Multi-Provider RAG Gateway with LiteLLM|06/19/06 - LiteLLM Capstone]]
