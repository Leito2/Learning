# 🏷️ Etiquetado y Parsing

El etiquetado morfosintáctico y el análisis sintáctico constituyen la base estructural sobre la cual se construye la comprensión del lenguaje. Un modelo de lenguaje que no distingue entre un sustantivo y un verbo, o que ignora que un pronombre se refiere a una entidad mencionada tres oraciones atrás, está condenado a cometer errores semánticos graves. En ML/AI, estas tareas proporcionan features estructurales que mejoran significativamente el rendimiento de clasificadores, sistemas de preguntas-respuestas y extracción de información.

Caso real: el motor de búsqueda de Bloomberg LP utiliza dependency parsing para identificar relaciones entre entidades financieras. En la oración «Apple acquired Beats for $3 billion», el parser debe reconocer que «Apple» es el sujeto de «acquired», «Beats» es el objeto directo, y «$3 billion» modifica la transacción, no a Beats directamente.

---

## 1. POS Tagging (Part-of-Speech Tagging)

POS tagging asigna a cada token una categoría gramatical. Los tagsets más comunes son Penn Treebank (45 tags para inglés) y Universal Dependencies (17 tags multilingües).

### 1.1 Modelos Ocultos de Markov (HMM)

Un HMM para POS tagging modela la secuencia de tags como una cadena de Markov oculta y las palabras como observaciones emítidas por los estados.

Las probabilidades que gobiernan el modelo son:

**Probabilidad de transición entre tags:**

$$
P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}
$$

**Probabilidad de emisión de palabra dado un tag:**

$$
P(w_i | t_i) = \frac{C(t_i, w_i)}{C(t_i)}
$$

**Decodificación con el algoritmo de Viterbi:**

Dada una secuencia de palabras $w_1, w_2, \dots, w_n$, buscamos la secuencia de tags óptima $t^*$:

$$
t^* = \arg\max_{t_1, \dots, t_n} P(t_1) \prod_{i=2}^{n} P(t_i | t_{i-1}) \prod_{i=1}^{n} P(w_i | t_i)
$$

```python
import nltk
from nltk.tag import hmm

train_data = [
    [("The", "DT"), ("dog", "NN"), ("runs", "VBZ")],
    [("A", "DT"), ("cat", "NN"), ("sleeps", "VBZ")]
]

tagger = nltk.HiddenMarkovModelTagger.train(train_data)
print(tagger.tag(["The", "cat", "runs"]))
# [('The', 'DT'), ('cat', 'NN'), ('runs', 'VBZ')]
```

⚠️ **Advertencia**: Los HMM sufren de data sparsity. Si una palabra no aparece en el corpus de entrenamiento, su probabilidad de emisión es cero para todos los tags, lo que anula toda la secuencia. El smoothing de Laplace es obligatorio en producción.

### 1.2 Conditional Random Fields (CRF)

Los CRF son modelos discriminativos que modelan directamente $P(\mathbf{t} | \mathbf{w})$ en lugar de $P(\mathbf{w}, \mathbf{t})$.

La probabilidad condicional de una secuencia de tags dada una secuencia de palabras se define como:

$$
P(\mathbf{t} | \mathbf{w}) = \frac{1}{Z(\mathbf{w})} \exp\left( \sum_{i=1}^{n} \sum_{k} \lambda_k f_k(t_{i-1}, t_i, \mathbf{w}, i) \right)
$$

Donde:
- $f_k$ son funciones de características (features) que capturan propiedades locales (palabra actual, prefijos, sufijos, mayúsculas).
- $\lambda_k$ son pesos aprendidos.
- $Z(\mathbf{w})$ es la función de partición (normalización sobre todas las secuencias posibles).

```python
# Ejemplo conceptual de features para CRF
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        features['BOS'] = False
    else:
        features['BOS'] = True
    return features
```

| Aspecto | HMM | CRF |
|---------|-----|-----|
| Naturaleza | Generativo | Discriminativo |
| Features | Emisión y transición simples | Arbitrariamente ricas |
| Manejo de OOV | Débil (requiere smoothing) | Medio (feature engineering) |
| Velocidad de entrenamiento | Rápido | Lento |
| Precisión | ~95% (PTB) | ~97% (PTB) |
| Uso moderno | Educativo, baselines | Producción antes de BERT |

💡 **Tip**: En la práctica actual, la mayoría de los sistemas de POS tagging utilizan arquitecturas BiLSTM-CRF o directamente transformers fine-tuned. Sin embargo, entender HMM y CRF es esencial para debuggear errores de etiquetado y para implementar soluciones ligeras en dispositivos de bajo consumo.

---

## 2. Named Entity Recognition (NER)

NER identifica y clasifica entidades nombradas en texto (personas, organizaciones, localizaciones, fechas, etc.).

### 2.1 Esquema de Etiquetado BIO

| Prefijo | Significado | Ejemplo |
|---------|-------------|---------|
| B- | Beginning (inicio de entidad) | B-PER en «Juan» |
| I- | Inside (dentro de entidad) | I-PER en «García» |
| O- | Outside (fuera de entidad) | O en «corre» |

```python
# Ejemplo BIO
text = "Juan García trabaja en Google"
# Juan    -> B-PER
# García  -> I-PER
# trabaja -> O
# en      -> O
# Google  -> B-ORG
```

⚠️ **Advertencia**: El esquema BIO impone una restricción de secuencia: una etiqueta I-X no puede aparecer sin una B-X previa (o I-X previa). Los decoders CRF respetan esta restricción automáticamente; un clasificador independiente por token viola esta restricción y requiere post-procesado.

### 2.2 CRF para NER

Los CRF son el estándar de oro para NER antes de la era transformer. La diferencia clave respecto al POS tagging es la riqueza de features: n-gramas de caracteres, embeddings de palabras, diccionarios de gazetteers, y features ortográficas.

```python
import sklearn_crfsuite

# Entrenamiento simplificado
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```

### 2.3 LSTM-CRF

Las redes LSTM capturan dependencias a largo plazo en la secuencia de embeddings, y la capa CRF final impone restricciones de secuencia BIO.

La probabilidad de una secuencia de tags $\mathbf{y}$ dada una entrada $\mathbf{x}$ es:

$$
P(\mathbf{y} | \mathbf{x}) = \frac{\exp(\text{score}(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}'} \exp(\text{score}(\mathbf{x}, \mathbf{y}'))}
$$

Donde el score combina emisiones de la LSTM y transiciones del CRF:

$$
\text{score}(\mathbf{x}, \mathbf{y}) = \sum_{i=0}^{n} \mathbf{W}_{y_i}^\top \mathbf{h}_i + \sum_{i=1}^{n} \mathbf{T}_{y_{i-1}, y_i}
$$

Caso real: spaCy v2.x implementa NER mediante una red CNN sobre caracteres seguida de una capa de predicción con softmax (no CRF). La transición a transformer-based NER en spaCy v3.x (utilizando modelos como roBERTa) mejoró el F1-score en dominios biomédicos en más de 8 puntos, pero aumentó el tiempo de inferencia por documento de 5 ms a 150 ms.

![Ejemplo de NER](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Example_of_Named_Entity_Recognition.jpg/440px-Example_of_Named_Entity_Recognition.jpg)

---

## 3. Dependency Parsing

El análisis de dependencias modela las relaciones sintácticas entre palabras como un árbol dirigido, donde cada palabra (excepto la raíz) tiene exactamente un padre.

### 3.1 Parsing Transition-Based

El algoritmo más conocido es el de **Arc-Eager** (Nivre, 2003). Utiliza una configuración $(S, B, A)$ donde:
- $S$ es la pila de palabras procesadas.
- $B$ es el buffer de palabras pendientes.
- $A$ es el conjunto de arcos dependencia creados.

Las transiciones son:
- **SHIFT**: mueve la primera palabra del buffer a la pila.
- **LEFT-ARC**: crea un arco desde la palabra superior de la pila hacia la primera del buffer.
- **RIGHT-ARC**: crea un arco desde la primera palabra del buffer hacia la superior de la pila.
- **REDUCE**: elimina la palabra superior de la pila.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat on the mat")

for token in doc:
    print(f"{token.text} <--{token.dep_}-- {token.head.text}")
# The <--det-- cat
# cat <--nsubj-- sat
# sat <--ROOT-- sat
# on <--prep-- sat
# the <--det-- mat
# mat <--pobj-- on
```

### 3.2 Parsing Graph-Based

Los métodos graph-based (como MSTParser o el de Chu-Liu/Edmonds) asignan un score a cada posible árbol de dependencias y seleccionan el de máximo score global.

El score de un árbol $T$ para una oración $x$ es:

$$
\text{score}(x, T) = \sum_{(i, j) \in T} \mathbf{w}^\top \mathbf{f}(i, j, x)
$$

Donde $\mathbf{f}(i, j, x)$ es un vector de características del arco potencial entre los nodos $i$ y $j$.

⚠️ **Advertencia**: Los parsers transition-based son más rápidos ($O(n)$ en promedio) pero sufren de error de propagación: una decisión incorrecta temprana no puede corregirse. Los graph-based son más lentos ($O(n^2)$ o $O(n^3)$) pero garantizan optimalidad global si el scoring es exacto.

![Ejemplo de Dependency Parsing](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Dependency_parsing_example.svg/440px-Dependency_parsing_example.svg.png)

---

## 4. Constituency Parsing

A diferencia del dependency parsing, el constituency parsing (o phrase-structure parsing) organiza la oración en constituyentes anidados (frases nominales, verbales, preposicionales).

La gramática se representa típicamente en **Forma Normal de Chomsky (CNF)**, donde cada regla de producción tiene la forma $A \rightarrow B C$ o $A \rightarrow a$.

**Algoritmo CKY (Cocke-Kasami-Younger)**:

Dada una oración de longitud $n$ y una gramática en CNF, CKY utiliza programación dinámica para llenar una tabla triangular $P[i, j, A]$ que indica si el constituyente $A$ puede generar la subcadena $w_i \dots w_j$.

$$
P[i, j, A] = \bigvee_{A \rightarrow B C} \bigvee_{k=i}^{j-1} P[i, k, B] \wedge P[k+1, j, C]
$$

```python
# Ejemplo simplificado con NLTK
import nltk
from nltk import CFG

grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | 'John'
    VP -> V NP
    Det -> 'the' | 'a'
    N -> 'dog' | 'cat'
    V -> 'chased' | 'saw'
""")

parser = nltk.ChartParser(grammar)
tree = next(parser.parse(['John', 'saw', 'the', 'dog']))
tree.pretty_print()
#         S
#    _____|___
#   NP       VP
#   |      ___|___
#  John   V      NP
#         |    ___|___
#        saw  Det     N
#             |       |
#            the    dog
```

💡 **Tip**: Los constituency parsers modernos utilizan modelos neuronales con RNNs o transformers para predecir la estructura del árbol. Sin embargo, CKY sigue siendo el algoritmo de decodificación subyacente en muchas implementaciones por su garantía de optimalidad con gramáticas probabilísticas.

---

## 5. Coreference Resolution

La resolución de correferencia identifica qué expresiones en un texto se refieren a la misma entidad del mundo real.

### 5.1 Modelo Básico: Mention-Pair

Para cada par de menciones $(m_i, m_j)$, se predice si son correferenciales. Las features incluyen:

- Coincidencia de strings exacta.
- Distancia en número de oraciones.
- Concordancia de género y número (si están disponibles).
- Tipo de mención (pronombre, nombre propio, descripción definida).

```python
# Ejemplo conceptual
mentions = [
    ("Juan", 0), ("él", 1), ("el ingeniero", 2), ("su", 3)
]
# Juan - él: coreferencial
# Juan - el ingeniero: coreferencial (si el contexto lo confirma)
# Juan - su: coreferencial (posesivo)
```

Caso real: el sistema de resumen automático de artículos científicos de Semantic Scholar (Allen Institute for AI) utiliza coreference resolution para agrupar todas las menciones a un mismo método experimental. Sin esta agrupación, un modelo de resumen generaría redundancias como «BERT fue evaluado... BERT obtuvo... el modelo de Devlin et al. logró...» en lugar de consolidar la información.

---

## 6. Implementación Integral con spaCy y NLTK

```python
import spacy

# Cargar modelo con todo el pipeline
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# POS Tagging
print("POS Tags:")
for token in doc:
    print(f"{token.text:12} {token.pos_:8} {token.tag_:8}")

# NER
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_:10}")

# Dependency Parsing
print("\nDependencies:")
for token in doc:
    print(f"{token.text:10} <--{token.dep_:8}-- {token.head.text}")
```

| Tarea | Librería recomendada | Velocidad | Precisión | Customización |
|-------|---------------------|-----------|-----------|---------------|
| POS Tagging | spaCy | Muy rápido | Alta | Media |
| NER | spaCy / transformers | Rápido | Muy alta | Alta |
| Dependency Parsing | spaCy | Muy rápido | Alta | Baja |
| Constituency Parsing | NLTK / Benepar | Media | Alta | Baja |
| Coreference Resolution | neuralcoref / CoreNLP | Lento | Media | Baja |

---

📦 **Código de compresión**

```python
# Pipeline completo de etiquetado y parsing
import spacy
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# POS + NER + Dependencies en una sola pasada
pos = [(t.text, t.pos_) for t in doc]
ents = [(e.text, e.label_) for e in doc.ents]
deps = [(t.text, t.dep_, t.head.text) for t in doc]

# NER con NLTK (alternativa)
nltk_tags = pos_tag(word_tokenize(text))
nltk_tree = ne_chunk(nltk_tags)
nltk_ents = tree2conlltags(nltk_tree)
```

🎯 **Proyecto documentado: Extractor de Relaciones Semánticas**

Construye un sistema que, dado un corpus de noticias económicas:

1. Aplique NER para extraer entidades tipo ORG, PER, MONEY, DATE.
2. Utilice dependency parsing para identificar relaciones del tipo `ORG --acquired--> ORG`, `PER --appointed_to--> ORG`, `ORG --invested--> MONEY`.
3. Genere una base de datos de tripletas (sujeto, predicado, objeto) en formato CSV.
4. Evalúe la precisión del extractor comparando contra un gold standard de 50 oraciones anotadas manualmente.

El sistema debe funcionar completamente con spaCy/NLTK (sin transformers) para demostrar la viabilidad de pipelines tradicionales en entornos de baja latencia.
