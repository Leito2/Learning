# 03 - Probabilidad y Estadística

El machine learning es, en esencia, estadística computacional. Cada predicción es una estimación probabilística, y cada modelo es una aproximación de una distribución subyacente.

---

## 1. Fundamentos de probabilidad

### Espacio muestral y eventos

- **Espacio muestral (Ω)**: conjunto de todos los resultados posibles.
- **Evento**: subconjunto del espacio muestral.
- **Probabilidad**: función `P(A)` que asigna un número entre 0 y 1 a cada evento.

**Axiomas de Kolmogorov:**
1. `P(A) ≥ 0` para todo evento A.
2. `P(Ω) = 1`.
3. Si `A₁, A₂, ...` son mutuamente excluyentes: `P(∪ Aᵢ) = Σ P(Aᵢ)`.

### Probabilidad condicional

La probabilidad de A dado que B ocurrió:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Regla de Bayes

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

```mermaid
flowchart TD
    A[Evento A] --> C[P(A|B)]
    B[Evento B] --> C
    A --> D[P(B|A)]
    B --> E[P(B)]
    D --> C
    E --> C
```

> 💡 **Caso real:** Los filtros de spam usan Bayes. `P(spam|palabras) = P(palabras|spam) · P(spam) / P(palabras)`.

### Independencia

Dos eventos son independientes si:

$$P(A \cap B) = P(A) \cdot P(B)$$

En ML, asumir independencia simplifica cálculos masivamente (ej. Naive Bayes asume que las features son independientes dada la clase).

---

## 2. Variables aleatorias y distribuciones

### Variable aleatoria

Una variable aleatoria es una función que asigna un número a cada resultado del espacio muestral.

- **Discreta**: toma valores contables (ej. lanzar un dado).
- **Continua**: toma valores en un rango continuo (ej. altura de personas).

### Distribuciones discretas

| Distribución | PMF | Uso en ML |
|--------------|-----|-----------|
| **Bernoulli** | `P(X=1) = p` | Clasificación binaria |
| **Binomial** | `P(X=k) = C(n,k) p^k (1-p)^(n-k)` | Número de éxitos en n trials |
| **Categórica** | `P(X=i) = p_i` | Clasificación multiclase |
| **Poisson** | `P(X=k) = λ^k e^(-λ) / k!` | Conteo de eventos (ej. clicks) |

### Distribuciones continuas

| Distribución | PDF | Uso en ML |
|--------------|-----|-----------|
| **Uniforme** | `f(x) = 1/(b-a)` | Inicialización de pesos, sampling |
| **Normal (Gaussiana)** | `f(x) = (1/√(2πσ²)) e^(-(x-μ)²/(2σ²))` | Ruido, features, priors bayesianos |
| **Exponencial** | `f(x) = λe^(-λx)` | Tiempo entre eventos |
| **Beta** | Flexible en [0,1] | Priors para proporciones |

![Distribución Normal](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/640px-Normal_Distribution_PDF.svg.png)

```python
import numpy as np
import matplotlib.pyplot as plt

# Muestrear de distribuciones
normal = np.random.normal(loc=0, scale=1, size=1000)
uniform = np.random.uniform(low=-2, high=2, size=1000)

print(f"Normal: media={normal.mean():.2f}, std={normal.std():.2f}")
print(f"Uniforme: media={uniform.mean():.2f}, std={uniform.std():.2f}")
```

> 💡 **Caso real:** Las capas BatchNorm en redes neuronales asumen que las activaciones siguen aproximadamente una distribución normal y las estandarizan.

---

## 3. Estadística descriptiva

### Medidas de tendencia central

| Medida | Fórmula | Robustez a outliers |
|--------|---------|---------------------|
| Media | $\bar{x} = \frac{1}{n}\sum x_i$ | ❌ No robusta |
| Mediana | Valor central ordenado | ✅ Robusta |
| Moda | Valor más frecuente | ✅ Robusta |

### Medidas de dispersión

| Medida | Fórmula | Interpretación |
|--------|---------|----------------|
| Varianza | $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$ | Dispersión promedio al cuadrado |
| Desviación estándar | $\sigma = \sqrt{\sigma^2}$ | Dispersión en unidades originales |
| Rango intercuartílico | $Q_3 - Q_1$ | Dispersión del 50% central |

```python
datos = np.array([1, 2, 3, 4, 100])  # 100 es outlier

print(f"Media: {np.mean(datos):.2f}")      # 22.00 (afectada por outlier)
print(f"Mediana: {np.median(datos):.2f}")  # 3.00 (robusta)
print(f"Std: {np.std(datos):.2f}")         # 39.60
```

### Covarianza y correlación

**Covarianza** mide cómo dos variables varían juntas:

$$\text{Cov}(X, Y) = \frac{1}{n} \sum (x_i - \bar{x})(y_i - \bar{y})$$

**Correlación de Pearson** normaliza la covarianza a [-1, 1]:

$$\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

```python
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

cov = np.cov(X, Y)[0, 1]
corr = np.corrcoef(X, Y)[0, 1]
print(f"Covarianza: {cov:.2f}")
print(f"Correlación: {corr:.2f}")  # 0.83 → correlación positiva fuerte
```

> 💡 **Caso real:** En feature selection, eliminas features con alta correlación entre sí (multicollinearity) porque aportan información redundante.

---

## 4. Maximum Likelihood Estimation (MLE)

MLE es el principio de encontrar los parámetros que **maximizan** la probabilidad de observar los datos.

### Ejemplo: estimar la media de una normal

Supón que tienes datos `x₁, x₂, ..., xₙ` que crees que vienen de `N(μ, σ²)`. ¿Cuál es el mejor `μ`?

$$\hat{\mu}_{MLE} = \arg\max_\mu \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu)^2}{2\sigma^2}}$$

Tomando logaritmo (log-likelihood):

$$\log L(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2$$

Derivando e igualando a cero:

$$\hat{\mu}_{MLE} = \frac{1}{n}\sum x_i = \bar{x}$$

¡La media muestral es el estimador de máxima verosimilitud para la media de una normal!

```python
def log_likelihood_normal(mu, sigma, datos):
    n = len(datos)
    return -0.5 * n * np.log(2 * np.pi * sigma**2) - \
           (1 / (2 * sigma**2)) * np.sum((datos - mu)**2)

# Verificar que mu=media maximiza la log-likelihood
datos = np.random.normal(loc=5, scale=2, size=1000)
mu_mle = datos.mean()

print(f"mu_MLE = {mu_mle:.4f}")
print(f"Log-likelihood en mu_MLE: {log_likelihood_normal(mu_mle, 2, datos):.2f}")
print(f"Log-likelihood en mu=0: {log_likelihood_normal(0, 2, datos):.2f}")
```

> 💡 **Caso real:** Entrenar una red neuronal con cross-entropy loss es equivalente a hacer MLE para una distribución categórica.

---

## 5. Teorema del Límite Central (CLT)

El CLT dice que, sin importar la distribución original, la media muestral de `n` observaciones independientes tiende a una distribución normal cuando `n` es grande:

$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right)$$

**Implicaciones en ML:**
- Los promedios de batches en SGD se comportan aproximadamente normales.
- Permite construir intervalos de confianza para métricas.
- Justifica el uso de la distribución normal en muchos modelos.

```python
# Demostración visual del CLT
from scipy import stats

# Distribución original: uniforme (no normal)
muestras = [np.random.uniform(0, 1, 100).mean() for _ in range(10000)]

plt.hist(muestras, bins=50, density=True, alpha=0.7, label='Media de n=100')
# Superponer normal teórica
mu, sigma = 0.5, np.sqrt(1/12 / 100)
x = np.linspace(0.3, 0.7, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal teórica')
plt.legend()
plt.title('Teorema del Límite Central')
```

---

## 6. Distribuciones multivariadas

### Vector aleatorio y matriz de covarianza

Para un vector aleatorio `X = [X₁, X₂, ..., Xₙ]ᵀ`, la matriz de covarianza `Σ` contiene todas las covarianzas por pares:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

Propiedades:
- Es simétrica: `Σ = Σᵀ`.
- Es positivo semi-definida: `vᵀ Σ v ≥ 0` para todo `v`.
- La diagonal contiene las varianzas de cada variable.

### Distribución Normal Multivariada

$$f(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

```python
from scipy.stats import multivariate_normal

mu = np.array([0, 0])
Sigma = np.array([[1, 0.5], [0.5, 1]])

# Muestrear
muestras = multivariate_normal.rvs(mean=mu, cov=Sigma, size=1000)
print(f"Covarianza empírica:\n{np.cov(muestras.T)}")
```

> 💡 **Caso real:** Los Gaussian Mixture Models (GMM) asumen que los datos provienen de una mezcla de normales multivariadas. Se usan en clustering y generación de datos.

---

## 📦 Código de compresión: Naive Bayes desde cero

```python
"""
Naive Bayes Gaussiano implementado desde cero.
Demuestra probabilidad condicional, MLE y el teorema de Bayes.
"""
import numpy as np

class GaussianNaiveBayes:
    """Asume que cada feature sigue una normal dada la clase."""

    def fit(self, X, y):
        self.clases = np.unique(y)
        self.params = {}

        for c in self.clases:
            X_c = X[y == c]
            self.params[c] = {
                'prior': len(X_c) / len(X),
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0) + 1e-9  # Suavización
            }

    def _pdf_gaussiana(self, x, mean, var):
        """PDF de la normal."""
        return (1 / np.sqrt(2 * np.pi * var)) * \
               np.exp(-0.5 * ((x - mean) ** 2) / var)

    def predict(self, X):
        predicciones = []
        for x in X:
            scores = {}
            for c in self.clases:
                p = self.params[c]
                # log(P(y=c)) + sum(log(P(x_i|y=c)))
                log_prior = np.log(p['prior'])
                log_likelihood = np.sum(
                    np.log(self._pdf_gaussiana(x, p['mean'], p['var']))
                )
                scores[c] = log_prior + log_likelihood
            predicciones.append(max(scores, key=scores.get))
        return np.array(predicciones)

# --- Uso ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

modelo = GaussianNaiveBayes()
modelo.fit(X_train, y_train)
preds = modelo.predict(X_test)

accuracy = np.mean(preds == y_test)
print(f"Accuracy Naive Bayes: {accuracy:.2%}")
```


