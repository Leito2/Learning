# 🏷️ Automatic Differentiation — grad, vjp, jvp, and Hessians

## 🎯 Learning Objectives
- Master `jax.grad` and `jax.value_and_grad` for **first-order reverse-mode differentiation**
- Understand the **forward vs reverse mode** distinction: `jvp` (Jacobian-vector product) vs `vjp` (Vector-Jacobian product)
- Compute **higher-order derivatives**: Hessians, second-order gradients, and the meta-learning connection
- Define **custom VJPs** for operations outside JAX's autodiff registry
- Apply autodiff through **control flow** and understand how JAX traces it

## Introduction

Automatic differentiation is the engine of modern deep learning, and JAX's implementation is arguably the most powerful in any framework. While PyTorch's `autograd` builds implicit computation graphs during eager execution, JAX takes a fundamentally different approach: **explicit, functional differentiation**. You call `jax.grad(f)` and get back a *new function* that computes gradients — no `.backward()`, no `.grad` attribute, no graph to retain or clear. This explicitness is not just aesthetic; it enables features that are clunky or impossible in PyTorch: seamless higher-order derivatives (`grad(grad(f))`), differentiation through compiled functions (`grad(jit(f))`), and differentiation through vectorized functions (`grad(vmap(f))`), all composable in any order.

The practical implications are profound. Meta-learning algorithms like MAML require Hessian-vector products — in JAX, they're one-liners. Neural ODEs require differentiation through an ODE solver — JAX's autodiff traces through the solver's internal iteration without approximations. Implicit layers compute gradients via the implicit function theorem — JAX's custom VJP system lets you define the backward pass analytically while the forward pass uses any solver. This note builds the complete theory from `jax.grad` through to custom VJPs, with rigorous LaTeX for the mathematics underpinning each concept and practical code for every pattern.

We'll also cover a critical distinction that confuses even experienced ML practitioners: forward-mode vs reverse-mode autodiff. JAX provides both (`jax.jvp` and `jax.vjp`), and choosing the right one can make a 1000× difference in runtime. Understanding *when* to use each mode is a hallmark of JAX expertise.

---

## 1. jax.grad — First-Order Reverse-Mode Differentiation

### 1.1 The Basics

`jax.grad` computes the gradient of a scalar-output function with respect to its first argument:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)              # f(x) = Σ x_i²

grad_f = jax.grad(f)                    # ∇f(x) = 2x
x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))                        # [2.0, 4.0, 6.0]
```

Mathematical definition: for $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is:

$$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}^T \in \mathbb{R}^n$$

`jax.grad` uses **reverse-mode autodiff** (backpropagation). For a scalar output, reverse-mode computes the full gradient in **one backward pass** — the complexity is proportional to evaluating $f$ once, regardless of input dimension $n$.

### 1.2 The Chain Rule Inside JAX

JAX decomposes every computation into primitive operations with known derivatives. For $f = g \circ h$, the chain rule gives:

$$\nabla f(x) = (\nabla h(x))^T \cdot \nabla g(h(x))$$

In more precise autodiff notation, JAX computes the **VJP (Vector-Jacobian Product)** of each primitive operation. Let $\mathbf{v}$ be the upstream gradient (vector). For a primitive $y = \text{op}(x)$ with Jacobian $J_{\text{op}}(x) = \frac{\partial y}{\partial x}$, the VJP is:

$$\text{vjp}_{\text{op}}(x, \mathbf{v}) = \mathbf{v}^T \cdot J_{\text{op}}(x) = \mathbf{v}^T \cdot \frac{\partial y}{\partial x}$$

JAX concatenates these VJPs from output to input (reverse mode) to compute the full gradient.

### 1.3 Differentiating with Respect to Multiple Arguments

```python
def loss(params, x, y):
    w, b = params
    return jnp.sum((jnp.dot(x, w) + b - y) ** 2)

# Default: gradient w.r.t. first argument (params)
grad_wrt_params = jax.grad(loss)((w, b), x_batch, y_batch)

# Specific argument via argnums
grad_wrt_x = jax.grad(loss, argnums=1)((w, b), x_batch, y_batch)

# Multiple arguments
grad_params, grad_x = jax.grad(loss, argnums=(0, 1))((w, b), x_batch, y_batch)

# value_and_grad: get loss AND gradients in one pass
loss_value, grads = jax.value_and_grad(loss)((w, b), x_batch, y_batch)
```

> **💡 Tip:** Always use `jax.value_and_grad` instead of `jax.grad` when you need both the loss and its gradient — it avoids redundant computation. For complex loss functions, the forward pass can be expensive; computing it once for both value and gradient is more efficient.

### 1.4 The Scalar-Output Requirement

```python
# ✅ Works: scalar-output function
def scalar_loss(w, x, y):
    return jnp.sum((jnp.dot(x, w) - y) ** 2)  # scalar output

grads = jax.grad(scalar_loss)(w, x, y)  # ✅ Returns (d,) gradient

# ❌ Fails: vector-output function
def vector_output(w, x):
    return jnp.dot(x, w)                     # shape (batch_size,) — NOT scalar

try:
    jax.grad(vector_output)(w, x)            # ❌ TypeError!
except TypeError as e:
    print(f"Error: {e}")                     # "Gradient only defined for scalar-output functions"
```

> **¡Sorpresa!** `jax.grad` only accepts scalar-output functions. For vector-valued $f: \mathbb{R}^n \to \mathbb{R}^m$, use `jax.jacobian` (returns the full $m \times n$ Jacobian) or `jax.vmap(jax.grad(f))` (returns per-output-dimension gradients). We'll treat both patterns in Section 2.

---

## 2. Forward vs Reverse Mode — jvp and vjp

### 2.1 When Does Mode Choice Matter?

| Scenario | $\text{dim}(\text{input}) = n$ | $\text{dim}(\text{output}) = m$ | Best Mode | Complexity |
|----------|------|------|-----------|------------|
| Neural net loss | Large ($10^6$+) | 1 (scalar) | **Reverse** | $O(1) \times \text{cost}(f)$ |
| Parameter sensitivity | 1 | Large ($10^6$+) | **Forward** | $O(1) \times \text{cost}(f)$ |
| ODE derivatives | Medium | Medium | Depends | Requires analysis |
| Hessian-vector products | Large | 1 | Reverse-over-forward | $O(n) \times \text{cost}(f)$ |

The rule of thumb:
- **Reverse mode (vjp)**: efficient when output dimension ≪ input dimension (standard for neural network training)
- **Forward mode (jvp)**: efficient when input dimension ≪ output dimension (sensitivity analysis, uncertainty quantification)

### 2.2 jax.jvp — Forward-Mode (Jacobian-Vector Product)

Forward-mode computes the directional derivative $J_f(x) \cdot \mathbf{v}$ — how the output changes when you perturb the input in direction $\mathbf{v}$:

$$J_f(x) \cdot \mathbf{v} = \frac{\partial f}{\partial x} \cdot \mathbf{v} = \lim_{\epsilon \to 0} \frac{f(x + \epsilon \mathbf{v}) - f(x)}{\epsilon}$$

```python
def f(x):
    return jnp.array([jnp.sin(x[0]), jnp.cos(x[1]), jnp.exp(x[0] * x[1])])

x = jnp.array([1.0, 2.0])
v = jnp.array([1.0, 0.0])  # Perturbation direction: along x[0]

# Compute f(x) AND J_f(x) · v in one forward pass
primals, tangents = jax.jvp(f, (x,), (v,))
print(f"f(x) = {primals}")
print(f"J_f(x) · v = {tangents}")
# tangent = [cos(x[0]), 0, x[1] * exp(x[0]*x[1])]
```

Forward-mode propagates derivatives **from inputs to outputs** simultaneously with the function evaluation. For $f: \mathbb{R}^n \to \mathbb{R}^m$, one forward pass computes $m$ directional derivatives. Computing the full Jacobian requires $n$ forward passes (one per input dimension).

### 2.3 jax.vjp — Reverse-Mode (Vector-Jacobian Product)

Reverse-mode computes $\mathbf{v}^T \cdot J_f(x)$ — how a linear combination of the outputs depends on the inputs:

$$\mathbf{v}^T \cdot J_f(x) = \mathbf{v}^T \cdot \frac{\partial f}{\partial x}$$

```python
def f(x):
    return jnp.array([jnp.sin(x[0]), jnp.cos(x[1]), jnp.exp(x[0] * x[1])])

x = jnp.array([1.0, 2.0])

# Get VJP function
primals_out, vjp_fn = jax.vjp(f, x)
print(f"f(x) = {primals_out}")

# Now compute v^T · J_f(x) for any vector v
v = jnp.array([1.0, 0.0, 0.0])  # Only care about first output component
gradient = vjp_fn(v)             # Returns gradient as tuple
print(f"v^T · J_f(x) = {gradient}")
# gradient = (array([cos(x_0), 0]),) — only first component contributes
```

```mermaid
graph LR
    subgraph "Forward Mode (jvp)"
        A1[Input x] --> B1[Primitive ops]
        B1 --> C1[Output f(x)]
        D1[Tangent v] --> E1[Forward propagation]
        E1 --> F1[J·v]
    end
    subgraph "Reverse Mode (vjp)"
        A2[Input x] --> B2[Primitive ops]
        B2 --> C2[Output f(x)]
        C2 --> D2[Backward propagation]
        F2[vᵀ·J] --> D2
    end
```

### 2.4 Computing the Full Jacobian

```python
# Method 1: jax.jacfwd (forward-mode) — n passes
J_fwd = jax.jacfwd(f)(x)      # Uses jvp internally, good for tall Jacobians

# Method 2: jax.jacrev (reverse-mode) — m passes
J_rev = jax.jacrev(f)(x)      # Uses vjp internally, good for wide Jacobians

# Method 3: jax.jacobian (auto-selects based on input/output sizes)
J_auto = jax.jacobian(f)(x)

# For a function f: R^3 → R^2
# jacfwd: 3 forward passes (good: 3 < 2? No, bad choice, but still works)
# jacrev: 2 backward passes (better: 2 < 3)
```

> **💡 Tip:** `jax.jacrev` maps `jax.grad` over each output dimension. For $f: \mathbb{R}^3 \to \mathbb{R}^2$, `jacrev` effectively calls `grad` twice (once per output). This is why it's efficient for $m \ll n$.

> **¡Sorpresa!** `jax.jacfwd(jax.jacrev(f))` computes the full Hessian in $O(n \cdot m)$ time through a forward-over-reverse strategy. This is how `jax.hessian` is implemented internally.

---

## 3. Higher-Order Derivatives

### 3.1 Hessians — Second-Order Gradients

The Hessian matrix of $f: \mathbb{R}^n \to \mathbb{R}$ contains all second-order partial derivatives:

$$H_f(x) = \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix} \in \mathbb{R}^{n \times n}$$

```python
def f(x):
    return jnp.sum(x ** 3 + 2 * x ** 2)  # f(x) = Σ(x³ + 2x²)

# Method 1: jax.hessian — computes the full n×n Hessian
hess = jax.hessian(f)(x)  # For x ∈ R^n, returns (n, n) matrix

# Method 2: jax.grad(jax.grad(f)) — same thing, explicit composition
hess_via_grad = jax.grad(jax.grad(f))(x)  # Identical result

# Method 3: Hessian-vector product (HVP) — way more efficient for large n
def hvp(f, x, v):
    return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
# Computes H_f(x) · v in O(n) time without materializing the n×n Hessian!
```

> **⚠️ Warning:** Computing the full Hessian of a neural network loss with $10^6$ parameters produces a $10^6 \times 10^6$ matrix — that's 4 TB at float32. **Always use Hessian-vector products (HVPs) for neural networks**, never the full Hessian. HVPs are used in conjugate gradient optimizers, influence functions, and neural tangent kernel computations.

### 3.2 Meta-Learning — Gradients Through Gradients

MAML (Model-Agnostic Meta-Learning) requires differentiating through the inner gradient update:

```python
# MAML inner loop: θ' = θ - α · ∇L_task(θ)
# Outer loop: compute ∂L_meta(θ') / ∂θ — differentiate through the gradient step!

def maml_gradient(meta_params, task_data, alpha):
    def inner_loss(params):
        return cross_entropy(params, task_data['support'])

    # Inner gradient step
    grads = jax.grad(inner_loss)(meta_params)
    adapted_params = jax.tree_map(lambda p, g: p - alpha * g, meta_params, grads)

    # Outer loss on query set — with adapted_params
    def outer_loss(_):
        return cross_entropy(adapted_params, task_data['query'])

    # This is the magic: grad of a function that itself calls grad
    return jax.grad(outer_loss)(None)  # Differentiates through inner grad!

# In JAX, this is a one-liner conceptually.
# In PyTorch, you need create_graph=True and complex bookkeeping.
```

### 3.3 Arbitrary Order Differentiation

```python
# Third-order gradient
third_order = jax.grad(jax.grad(jax.grad(f)))(x)

# Arbitrarily compose
@jax.jit
@jax.grad
@jax.grad
@jax.grad
def cubic_norm(x):
    return jnp.sum(x ** 3) / 3
# jit(grad(grad(grad(f)))) — works perfectly
```

> **Caso real: Meta-learning research at DeepMind** uses third-order gradients for optimization-based meta-learning (iMAML, Reptile variants). These are effectively impossible to implement cleanly in PyTorch because each `create_graph=True` call adds overhead and the graph becomes unwieldy. In JAX, `grad(grad(grad(f)))` is a one-liner.

---

## 4. Custom VJPs — Defining Your Own Backward Pass

### 4.1 Why Custom VJPs?

JAX knows how to differentiate through most operations automatically. But for some computations, the automatic backward pass is inefficient or incorrect:

- **Fixed-point iteration**: solved to convergence — the autodiff'd backward would trace through all iterations
- **ODE solvers**: the solver's internal steps create a massive computation graph
- **Implicit layers**: defined by root-finding — no explicit forward computation to differentiate through
- **External library calls**: custom CUDA kernels or C/C++ code

For these cases, you define a **custom VJP**: provide JAX with an analytical backward pass while letting the forward pass use whatever implementation you want.

### 4.2 The custom_vjp Decorator

```python
import jax
from jax import custom_vjp

@custom_vjp
def safe_log(x):
    """Custom log that handles x ≤ 0 gracefully in forward mode"""
    return jnp.where(x > 0, jnp.log(x), 0.0)

def safe_log_fwd(x):
    return safe_log(x), (x,)  # Return output and residuals for backward

def safe_log_bwd(res, g):
    (x,) = res
    # g is the upstream gradient (∂loss/∂safe_log(x))
    # We need ∂loss/∂x = g * d(safe_log)/dx
    return (g / jnp.where(x > 0, x, 1.0),)  # Avoid division by zero

safe_log.defvjp(safe_log_fwd, safe_log_bwd)
```

Mathematically, the custom VJP defines:

$$\frac{\partial \text{safe\_log}}{\partial x} = \begin{cases} \frac{1}{x} & x > 0 \\ 0 & x \leq 0 \end{cases}$$

This is the **implicit function theorem** in practice — you tell JAX the local derivative without JAX needing to auto-differentiate through your forward implementation.

### 4.3 Complete Custom VJP Example: Fixed-Point Layer

An implicit layer solves $z = g(z, x)$ for $z$ given input $x$. The forward pass uses fixed-point iteration:

```python
from jax import custom_vjp, lax

@custom_vjp
def implicit_layer(x, g_fn, max_iter=100, tol=1e-6):
    """Solve z = g(z, x) for z via fixed-point iteration"""
    def cond_fn(state):
        z, _, iter_count = state
        return (jnp.max(jnp.abs(z - g_fn(z, x))) > tol) & (iter_count < max_iter)

    def body_fn(state):
        z, _, iter_count = state
        z_new = g_fn(z, x)
        return (z_new, z, iter_count + 1)

    z0 = jnp.zeros_like(x)
    z_final, _, _ = lax.while_loop(cond_fn, body_fn, (z0, z0, 0))
    return z_final
```

The backward pass uses the implicit function theorem. At equilibrium $z^* = g(z^*, x)$:

$$\frac{\partial z^*}{\partial x} = \left( I - \frac{\partial g}{\partial z} \right)^{-1} \cdot \frac{\partial g}{\partial x}$$

This avoids re-solving the fixed-point iteration during backprop — a 10–100× speedup.

> **¡Sorpresa!** JAX's `jax.lax.custom_linear_solve` and `jax.lax.custom_root` provide built-in infrastructure for implicit differentiation. For most use cases, you don't need raw `custom_vjp` — use these higher-level APIs instead.

---

## 5. Differentiating Through Control Flow

### 5.1 How JAX Handles If/Else, Loops, and While

JAX compiles functions, and compilation happens **before** execution. When you `jit` a function with `if` statements on traced values, JAX can't know which branch to take. The solution: JAX traces the function once, recording **all possible execution paths**, and inserts a `select` operation.

```python
# ✅ Python control flow on STATIC values (not from arrays)
@jax.jit
def fn(x, use_bias):
    if use_bias:            # Static — compile-time branch
        return x + 1.0
    else:
        return x * 2.0

# ❌ Python control flow on TRACED values
@jax.jit
def bad_fn(x):
    if x[0] > 0:            # Traced value — ERROR!
        return x * 2
    else:
        return x / 2
# → ConcretizationTypeError!

# ✅ Use JAX-native conditionals
from jax import lax
@jax.jit
def good_fn(x):
    return lax.cond(x[0] > 0, lambda x: x * 2, lambda x: x / 2, x)
```

```python
# Loops: lax.scan, lax.fori_loop
@jax.jit
def loop_fn(x, n):
    def body_fn(i, val):
        return val + jnp.sin(val) * 0.1
    return lax.fori_loop(0, n, body_fn, x)
```

### 5.2 grad Through Control Flow

The key insight: because JAX traces the function, `jax.grad` can differentiate through `lax.cond`, `lax.scan`, and `lax.while_loop` **automatically**. JAX traces the computation, records the primitive operations for the path actually taken during tracing, and constructs the backward pass from those primitives.

```python
@jax.jit
@jax.grad
def loop_grad_fn(x):
    def body_fn(i, val):
        return val * (1.0 + 0.1 * jnp.sin(val))
    final = lax.fori_loop(0, 10, body_fn, x)
    return final

print(loop_grad_fn(1.0))  # Computes d/dx of 10 iterations — works!
```

> **⚠️ Warning:** In dynamic control flow where the trace might change between invocations (different branch taken for different inputs), the gradient is **correct for the traced path**. If the function's control flow depends on input values, the gradient might be wrong for inputs that follow a different path. For such cases, use `jax.checkpoint` or manually ensure the control flow is deterministic given inputs.

---

## 🎯 Key Takeaways
- **`jax.grad` uses reverse-mode autodiff** optimized for scalar-output functions (standard neural network training). Use `jax.value_and_grad` to get both loss and gradient in one pass.
- **Forward mode (jvp)** computes directional derivatives efficiently when input dimension ≪ output dimension. Reverse mode (vjp) is efficient in the opposite case.
- **Higher-order derivatives are native**: `jax.grad(jax.grad(f))` for Hessians, `jax.hessian` for convenience, and Hessian-vector products via `grad(grad(f)(x) · v)` for large-scale applications.
- **Custom VJPs** enable differentiating through fixed-point iterations, ODE solvers, implicit layers, and external code — define the analytical backward pass and JAX handles the rest.
- **JAX traces through all control flow** (`lax.cond`, `lax.scan`, `lax.while_loop`), making differentiation through arbitrary logic possible without special treatment.
- **Composability is key**: `grad(jit(vmap(f)))` differentiates a compiled, vectorized function. `jit(grad(pmap(f)))` compiles a distributed gradient computation. Any order works.
- **Start with `jax.grad`** for standard training, reach for `jax.vjp`/`jax.jvp` when you understand the efficiency tradeoffs, and learn `custom_vjp` when you need implicit differentiation.

## 📦 Código de Compresión

```python
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jvp, vjp, hessian, value_and_grad

# Define a scalar-output function
def f(x):
    return jnp.sum(x ** 3 + 2 * jnp.sin(x))

x = jnp.array([1.0, 2.0, 3.0])

# 1. First-order gradients
g = grad(f)(x)                    # ∇f = [3x² + 2cos(x)]
loss, g2 = value_and_grad(f)(x)   # Both value and gradient

# 2. Forward mode: J_f(x) · v (directional derivative)
v = jnp.array([1.0, 0.0, 0.0])
_, directional = jvp(f, (x,), (v,))

# 3. Reverse mode: v^T · J_f(x)
out, vjp_fn = vjp(f, x)
gradient_via_vjp = vjp_fn(1.0)    # v=1.0 → standard gradient

# 4. Full Jacobian (f: R³ → R)
J_fwd = jacfwd(lambda x: f(x) * jnp.ones(2))(x)  # shape (2, 3)
J_rev = jacrev(lambda x: f(x) * jnp.ones(2))(x)  # shape (2, 3)

# 5. Hessian and HVP
H = hessian(f)(x)                 # Full 3×3 Hessian: H_ij = ∂²f/∂x_i∂x_j
hvp_fn = lambda v: grad(lambda x: jnp.dot(grad(f)(x), v))(x)
hvp = hvp_fn(jnp.ones(3))         # H·[1,1,1] without materializing H!

print(f"Gradient: {g}")
print(f"Hessian trace: {jnp.trace(H):.4f}")
print("¡Sorpresa! grad → jacobian → hessian → hvp — all composable, all pure functions.")
```

## References
- Griewank & Walther (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.* SIAM.
- Baydin, Pearlmutter, Radul, Siskind (2018). "Automatic Differentiation in Machine Learning: a Survey." *JMLR*.
- JAX Autodiff Cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- DeepMind (2022). "The DeepMind JAX Ecosystem."
- [[05/03 - Deep Learning con PyTorch]]
- [[04/01 - Matemáticas para ML]]
- [[07/32 - Advanced ML Topics]]
