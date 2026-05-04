# 06 - Caso Práctico: Implementando PCA desde Cero

Este caso práctico integra álgebra lineal, cálculo y estadística para construir un sistema de reducción de dimensionalidad completo. No desarrollaremos el código completo (eso es tu ejercicio), pero documentamos el diseño matemático, la arquitectura y los fragmentos clave.

---

## 🎯 Contexto del negocio

Eres Data Scientist en una empresa de genómica. Cada muestra de ADN tiene 20,000 features (expresión génica). Necesitas:

1. Visualizar las muestras en 2D/3D para identificar clusters (tipos de cáncer).
2. Reducir ruido eliminando componentes de baja varianza.
3. Comprimir los datos para entrenar modelos más rápido.
4. Interpretar qué genes contribuyen a cada componente principal.

---

## 🏗️ Arquitectura matemática del sistema

```mermaid
flowchart TD
    A[Datos brutos<br/>X ∈ R^{n×d}] --> B[Centrar datos<br/>X̃ = X - μ]
    B --> C[Matriz de covarianza<br/>Σ = X̃^T·X̃ / (n-1)]
    C --> D[Eigenvalores y eigenvectors<br/>Σv = λv]
    D --> E[Seleccionar top k<br/>V_k = [v₁...v_k]]
    E --> F[Proyección<br/>Z = X̃·V_k]
    F --> G[Reconstrucción<br/>X̂ = Z·V_k^T + μ]
```

### Fase 1: Preparación de datos

**Entrada:** Matriz `X ∈ R^(n×d)` donde `n` = muestras, `d` = genes.

**Preprocesamiento:**
- Centrar: `X̃ = X - μ` donde `μ` es el vector de medias por gen.
- (Opcional) Escalar: dividir por desviación estándar si las escalas difieren mucho.

> 💡 **Por qué centrar:** PCA encuentra direcciones de máxima varianza respecto al origen. Si los datos no están centrados, la primera componente puede apuntar al centro de masa en lugar de a la dirección de mayor dispersión.

### Fase 2: Matriz de covarianza

$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X}$$

La matriz de covarianza `Σ ∈ R^(d×d)` mide cómo varían los genes entre sí.
- `Σ_ii` = varianza del gen `i`.
- `Σ_ij` = covarianza entre gen `i` y gen `j`.

### Fase 3: Eigenvalores y eigenvectors

Resolver:

$$\Sigma v = \lambda v$$

Obtienes `d` eigenvalues `λ₁ ≥ λ₂ ≥ ... ≥ λd` y `d` eigenvectors `v₁, v₂, ..., vd`.

**Interpretación:**
- `v₁` (eigenvector del mayor eigenvalue) es la dirección de máxima varianza.
- `λ_i / Σλ_j` es la proporción de varianza explicada por la componente `i`.

### Fase 4: Proyección

Seleccionar los `k` eigenvectors principales y proyectar:

$$Z = \tilde{X} V_k$$

Donde `V_k = [v₁, v₂, ..., vk]`.

`Z ∈ R^(n×k)` son los datos en el nuevo espacio de menor dimensión.

### Fase 5: Reconstrucción

$$\hat{X} = Z V_k^T + \mu$$

El error de reconstrucción (MSE) está relacionado con los eigenvalues descartados:

$$MSE = \frac{1}{n} \sum_{i=k+1}^{d} \lambda_i$$

---

## 📐 Componentes del sistema

### 1. Cargador de datos con generadores

```python
def cargar_genomic_data(ruta, batch_size=1000):
    """Carga datos de expresión génica en batches para manejar 20k features."""
    for chunk in pd.read_csv(ruta, chunksize=batch_size):
        # Cada chunk: n_muestras × 20000
        yield chunk.values
```

### 2. Calculadora de covarianza incremental

Para datasets que no caben en memoria, calculamos la covarianza de forma incremental:

```python
class CovarianzaIncremental:
    """Welford's online algorithm para media y covarianza."""
    def __init__(self, dim):
        self.n = 0
        self.media = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))  # Suma de productos cruzados

    def update(self, x):
        self.n += 1
        delta = x - self.media
        self.media += delta / self.n
        delta2 = x - self.media
        self.M2 += np.outer(delta, delta2)

    def covarianza(self):
        return self.M2 / (self.n - 1)
```

> 💡 **Welford's algorithm** es numéricamente estable y requiere una sola pasada sobre los datos.

### 3. Solver de eigenvalores con SVD alternativo

Para matrices grandes (20k × 20k), calcular eigenvalues directamente es lento. Usamos SVD sobre `X̃`:

```python
# Si n < d, es más rápido hacer SVD de X̃ que eigen de X̃^T·X̃
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# Los eigenvalues de covarianza son S^2 / (n-1)
# Los eigenvectors son las filas de Vt
```

**Complejidad:**
- Eigen de covarianza: `O(d³)`.
- SVD de datos: `O(n·d²)` o `O(n²·d)` (elige el menor).

### 4. Selector de componentes

```python
def seleccionar_k(eigenvalues, umbral_varianza=0.95):
    """Elige k tal que se explique el % de varianza deseado."""
    varianza_acumulada = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(varianza_acumulada >= umbral_varianza) + 1
    return k, varianza_acumulada
```

### 5. Visualizador con interpretación

```python
def interpretar_componente(v, nombres_genes, top_n=10):
    """Muestra qué genes contribuyen más a una componente."""
    indices_ordenados = np.argsort(np.abs(v))[::-1]
    print(f"Top {top_n} genes en esta componente:")
    for idx in indices_ordenados[:top_n]:
        direccion = "positiva" if v[idx] > 0 else "negativa"
        print(f"  {nombres_genes[idx]}: contribución {direccion}")
```

---

## 📊 Métricas de calidad

| Métrica | Cálculo | Objetivo |
|---------|---------|----------|
| Varianza explicada | `λ_i / Σλ_j` | > 90% con las primeras k componentes |
| Error de reconstrucción | `‖X - X̂‖²_F / n` | Mínimo para k dado |
| Silhouette score (clustering) | Cohesión vs separación | > 0.5 en proyección 2D |
| Tiempo de computación | Wall-clock | < 1 minuto para 10k × 20k |

![Red neuronal](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/640px-Colored_neural_network.svg.png)

---

## 🧪 Plan de validación

1. **Test con datos sintéticos:** generar datos en 3D con varianza conocida, verificar que PCA recupera las direcciones correctas.
2. **Test con MNIST:** proyectar imágenes de 784D a 2D, visualizar clusters de dígitos.
3. **Test con genómica real:** dataset GTEx o TCGA, identificar tipos de tejido.
4. **Comparación con t-SNE/UMAP:** PCA es lineal y rápido; t-SNE/UMAP son no lineales pero más lentos.

---

## 🚀 Extensión: PCA Kernelizado

PCA lineal solo encuentra direcciones rectas. Para datos con estructura no lineal (ej. una espiral), usamos **Kernel PCA**:

1. Mapear datos a un espacio de mayor dimensión: `φ(x)`.
2. Aplicar PCA en ese espacio (sin calcular `φ` explícitamente, usando el kernel trick).

$$K_{ij} = k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Kernels comunes:
- **RBF/Gaussiano:** `k(x, y) = exp(-γ‖x-y‖²)`.
- **Polinomial:** `k(x, y) = (x^T y + c)^d`.

> 💡 **Caso real:** Kernel PCA se usa en detección de anomalías (outliers son puntos que no se reconstruyen bien en el espacio de componentes).

---

## 📚 Referencias para la implementación

- scikit-learn `PCA` y `KernelPCA`
- Eigen library (C++ para PCA de alta performance)
- Welford's online algorithm (estadística computacional)
- GTEx Portal (datos de expresión génica)
