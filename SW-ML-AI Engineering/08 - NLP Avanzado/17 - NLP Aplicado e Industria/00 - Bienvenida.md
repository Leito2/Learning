# 🌍 Bienvenida a NLP Aplicado e Industria

Bienvenido al curso **"NLP Aplicado e Industria"**, el módulo donde consolidamos todo el conocimiento de procesamiento de lenguaje natural en soluciones reales, escalables y de alto impacto empresarial.

Este curso es fundamental para cualquier ingeniero de ML/IA que aspire a desplegar sistemas NLP en producción. No basta con entender transformers; hay que dominar la traducción automática, la generación de resúmenes, la extracción estructurada de información y la búsqueda semántica a escala industrial.

![NLP Applications](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Natural_language_processing.jpg/640px-Natural_language_processing.jpg)

---

## 1. Objetivos del Curso

Al finalizar este módulo serás capaz de:

1. Diseñar pipelines de traducción automática (NMT) multilingües con modelos como mT5 y XLM-R.
2. Construir sistemas de resumen extractivo y abstractivo con control de longitud y evaluación de fidelidad.
3. Implementar pipelines de extracción de información (NER, RE) y construir grafos de conocimiento.
4. Desarrollar motores de búsqueda semántica empresariales con dense retrieval, re-ranking y generación de respuestas.

---

## 2. Mapa del Módulo

| Clase | Título | Descripción |
|---|---|---|
| 01 | [[01 - Traduccion y Multilinguismo]] | SMT vs NMT, atención, modelos multilingües, zero-shot, back-translation |
| 02 | [[02 - Resumen y Generacion de Texto]] | Extractivo, abstractivo, métricas, faithfulness, modelos de largo alcance |
| 03 | [[03 - Extraccion de Informacion y Grafos de Conocimiento]] | NER, RE, entity linking, KG, RDF/OWL, embeddings de grafo |
| 04 | [[04 - Caso Practico - Motor de Busqueda Semantica Empresarial]] | Pipeline completo de búsqueda semántica con dense retrieval y re-ranking |

---

## 3. Glosario

A continuación se definen los términos clave que utilizaremos a lo largo del curso:

- **MT (Machine Translation)**: Tarea de traducción automática de texto de un idioma a otro.
- **NMT (Neural Machine Translation)**: Traducción automática basada en redes neuronales, típicamente encoder-decoder con atención.
- **Back-translation**: Técnica de aumento de datos donde se traduce texto monolingüe al idioma objetivo y se usa como paralelo sintético.
- **Zero-shot translation**: Capacidad de un modelo multilingüe para traducir entre pares de idiomas no vistos durante el entrenamiento.
- **Summarization**: Tarea de condensar un documento en un resumen más corto, manteniendo la información relevante.
- **Abstractive summarization**: Generación de resúmenes con palabras nuevas, no necesariamente presentes en el texto fuente.
- **Extractive summarization**: Selección de oraciones o frases directamente del documento original.
- **IE (Information Extraction)**: Extracción automática de información estructurada (entidades, relaciones) de texto no estructurado.
- **NER (Named Entity Recognition)**: Identificación y clasificación de entidades nombradas (personas, organizaciones, lugares, etc.).
- **RE (Relation Extraction)**: Identificación de relaciones semánticas entre entidades.
- **KG (Knowledge Graph)**: Grafo que representa entidades como nodos y relaciones como aristas, con conocimiento estructurado.
- **Entity linking**: Proceso de vincular menciones de entidades en texto con nodos canónicos en una base de conocimiento.
- **Semantic search**: Búsqueda basada en el significado semántico de las consultas, no solo en coincidencias léxicas.
- **Dense retrieval**: Recuperación de documentos mediante representaciones vectoriales densas (embeddings) en espacios de alta dimensionalidad.

---

## 4. Diagrama de Flujo del Curso

El siguiente diagrama ilustra cómo se conectan los módulos del curso en un pipeline industrial típico:

```mermaid
graph LR
    A[Documentos Multilingues] --> B[Traduccion NMT]
    B --> C[Resumen Abstractive]
    C --> D[Extraccion de Informacion]
    D --> E[Grafo de Conocimiento]
    E --> F[Busqueda Semantica]
    F --> G[Respuesta Generada]
```

---

## 5. Relevancia Industrial

Según Gartner, más del 80% de los datos empresariales son no estructurados, y gran parte de ellos son texto. Las organizaciones que logran estructurar, resumir y buscar semánticamente este contenido obtienen ventajas competitivas significativas en atención al cliente, cumplimiento normativo e investigación de mercados.

💡 **Tip**: Siempre comienza un proyecto NLP industrial identificando la métrica de negocio que impactarás (tiempo de resolución, precisión de búsqueda, cobertura de cumplimiento), no solo las métricas técnicas.

⚠️ **Advertencia**: Los modelos más grandes no siempre son la mejor opción. En producción, la latencia, el costo y la privacidad suelen pesar más que el último punto de mejora en BLEU o ROUGE.

---

## 6. Proyecto Final del Módulo

🎯 **Proyecto documentado**: El proyecto integrador de este curso es el desarrollo de un **Motor de Búsqueda Semántica Empresarial** documentado en [[04 - Caso Practico - Motor de Busqueda Semantica Empresarial]]. Aquí aplicarás:

- Traducción multilingüe para normalizar documentos.
- Resumen automático para previsualizar resultados.
- Extracción de entidades para enriquecer metadatos.
- Dense retrieval con FAISS/Elasticsearch para recuperación eficiente.

```python
# Esqueleto del pipeline integrado
class EnterpriseNLPPipeline:
    def __init__(self):
        self.translator = None   # HuggingFace NMT
        self.summarizer = None   # Abstractive model
        self.ner = None          # spaCy / transformers NER
        self.retriever = None    # FAISS index

    def process(self, document: str):
        translated = self.translate(document)
        summary = self.summarize(translated)
        entities = self.extract_entities(translated)
        return {"summary": summary, "entities": entities}
```

---

## 7. Recursos y Referencias

- Hugging Face Transformers Documentation
- spaCy Industrial Applications
- Neo4j Graph Data Science Library
- FAISS Wiki & Tutorials

📦 **Código de compresión**:

```python
# Instalación de dependencias del módulo
!pip install transformers sentence-transformers faiss-cpu spacy neo4j
!python -m spacy download es_core_news_lg
!python -m spacy download en_core_web_lg
```
