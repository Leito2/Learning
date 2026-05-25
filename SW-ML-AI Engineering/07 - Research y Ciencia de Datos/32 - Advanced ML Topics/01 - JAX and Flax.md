# ⚛️ JAX and Flax for High-Performance ML

## Introduction

JAX is Google's high-performance numerical computing framework that combines NumPy's familiar API with automatic differentiation and XLA (Accelerated Linear Algebra) compilation. Unlike PyTorch's eager execution model, JAX compiles your Python functions into optimized GPU/TPU kernels — giving you performance closer to handwritten CUDA without leaving Python. Flax is the neural network library built on top of JAX, providing a PyTorch-like Module API with functional purity underneath.

For ML engineers, JAX represents the frontier of model performance. DeepMind uses it for AlphaFold, Gemini training, and cutting-edge research. Understanding JAX means understanding the compute model that powers Google's most demanding ML workloads.

---

## 2. Key Concepts

| Concept | What It Is | PyTorch Equivalent |
|---|---|---|
| **`jax.numpy`** | NumPy-compatible API that runs on GPU/TPU | `torch.Tensor` |
| **`jax.grad`** | Automatic differentiation (take gradient of ANY function) | `loss.backward()` |
| **`jax.jit`** | Just-in-time compile Python → XLA → GPU kernels | `torch.compile()` |
| **`jax.vmap`** | Auto-vectorize functions over batch dimensions | Manual batch loops |
| **`jax.pmap`** | Parallelize across multiple devices (GPUs/TPUs) | `torch.nn.DataParallel` |
| **Immutable arrays** | JAX arrays cannot be modified in-place | Mutable tensors |
| **PRNG keys** | Explicit random state (no global seed) | `torch.manual_seed()` |

### Functional vs Object-Oriented

```
PyTorch:                          JAX + Flax:
class MyModel(nn.Module):         class MyModel(nn.Module):
    def __init__(self):               ...
        self.linear = nn.Linear(10,2)

model = MyModel()                 model = MyModel()
optim = Adam(model.parameters())  params = model.init(key, x)    # Separate params
                                  optim = optax.adam(1e-3)       # Separate optimizer state

# Implicit state in model         # Explicit state management
loss = criterion(model(x), y)     def loss_fn(params, x, y):
loss.backward()                       return -jnp.mean(...)
optim.step()                       grads = jax.grad(loss_fn)(params, x, y)
                                  updates, opt_state = optim.update(grads, opt_state)
                                  params = optax.apply_updates(params, updates)
```

JAX's functional purity forces you to separate model parameters from the model definition — more boilerplate, but the compiler can optimize across the entire loss function because there are no hidden side effects.

---

## 3. JAX + ML Ecosystem

| Integration | Use |
|---|---|
| **JAX + TensorFlow** | TF's data pipeline (tf.data) feeds JAX models |
| **JAX + MLflow** | `mlflow.log_params()` + `jax.jit` compiled models exported via ONNX |
| **JAX + TPU** | Native — JAX was designed for TPUs from day one |
| **JAX + Haiku** | DeepMind's alternative NN library (more PyTorch-like than Flax) |
| **JAX + Equinox** | PyTorch-style NN library with JAX backend |

---

## ⚠️ Key Differences from PyTorch

- **No in-place mutations:** `x[0] = 1.0` raises an error. Use `x.at[0].set(1.0)`.
- **Compilation warm-up cost:** First `jit` call compiles — subsequent calls are fast. Profile with `with jax.profiler.trace(...)`.
- **Explicit PRNG:** Every random operation needs a key: `key, subkey = jax.random.split(key)`.

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [DeepMind Haiku](https://github.com/deepmind/dm-haiku)
