# Python Packaging for ML Teams

## 1. Poetry Dependency Groups for ML

ML projects have three distinct dependency sets:

```toml
[tool.poetry.dependencies]
python = "^3.11"
pydantic = "^2.9"
fastapi = "^0.115"
numpy = "^1.26"

[tool.poetry.group.train.dependencies]
torch = "^2.4"
transformers = "^4.44"
wandb = "^0.18"

[tool.poetry.group.serve.dependencies]
ray = {version = "^2.30", extras = ["serve"]}
py-spy = "^0.4"
```

```bash
poetry install                     # base deps only
poetry install --with train        # training env
poetry install --with serve        # serving env
poetry install --with train,serve  # full dev env
```

## 2. Packaging a Model as a pip Package

```bash
mkdir my-ml-package/
├── pyproject.toml
├── src/
│   └── my_model/
│       ├── __init__.py
│       ├── model.py
│       └── artifacts/           # model weights (git LFS or s3 download)
│           └── model.pt
├── tests/
└── README.md
```

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ml-package"
version = "0.1.0"
dependencies = ["torch>=2.4", "numpy>=1.26"]

[project.optional-dependencies]
gpu = ["cuda-toolkit"]
serve = ["fastapi>=0.115", "uvicorn[standard]"]
```

```python
# src/my_model/model.py
import torch

class InferenceModel:
    def __init__(self, model_path: str | None = None):
        path = model_path or Path(__file__).parent / "artifacts" / "model.pt"
        self.model = torch.jit.load(path)
        self.model.eval()

    def predict(self, features: list[float]) -> list[float]:
        with torch.no_grad():
            return self.model(torch.tensor(features)).tolist()
```

## 3. Monorepo Patterns for ML Teams

```
ml-monorepo/
├── packages/
│   ├── feature-pipeline/
│   │   ├── pyproject.toml
│   │   └── src/...
│   ├── training-pipeline/
│   │   ├── pyproject.toml
│   │   └── src/...
│   └── inference-service/
│       ├── pyproject.toml
│       └── src/...
├── libs/
│   ├── ml-core/
│   │   ├── pyproject.toml        # shared model logic
│   │   └── src/...
│   └── ml-utils/
│       ├── pyproject.toml
│       └── src/...
├── pyproject.toml                # workspace root
└── uv.lock
```

uv workspace:

```toml
# pyproject.toml (root)
[tool.uv]
workspace = ["packages/*", "libs/*"]
```

```bash
uv pip install -e packages/inference-service
uv run --with packages/training-pipeline python train.py
```

## 4. Editable Installs for ML Development

```bash
# Install package in development mode
poetry install                     # includes `pip install -e .`
# or
pip install -e ".[train,serve]"
```

Changes to Python source are reflected immediately without reinstall. Model artifacts still need explicit download.

## 5. GPU Dependency Management

```toml
[project.optional-dependencies]
cpu = ["torch>=2.4"]
gpu = ["torch>=2.4"]  # torch automatically detects CUDA
cuda118 = ["torch>=2.4,<2.5", "nvidia-cublas-cu11"]
cuda124 = ["torch>=2.4", "nvidia-cublas-cu12"]
```

Use environment markers:

```toml
[tool.poetry.dependencies]
torch = {version = "^2.4", markers = "sys_platform == 'linux'"}
```

## 6. Docker Multi-Stage for ML

```dockerfile
FROM python:3.11-slim AS base
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

FROM base AS train
RUN poetry install --with train

FROM base AS serve
COPY --from=base /app /app
COPY src/ src/
COPY artifacts/ artifacts/
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Key Takeaways

- Poetry dep groups: `train`, `serve`, `dev` — install only what each env needs
- Package models as pip-installable packages with `src/` layout
- Monorepo with `uv workspace`: shared libs + deployable packages
- Editable installs: `pip install -e .` for development
- GPU deps: optional groups for cuda versions
- Docker multi-stage: `base` → `train` → `serve` for minimal serving images
