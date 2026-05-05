# Prompt de Continuidad para Siguiente Sesión

Copia y pega esto en una nueva sesión de Kimi para continuar el proyecto sin perder contexto.

---

## 🎯 Contexto del Proyecto

Estamos construyendo un **vault de Obsidian** dentro de la carpeta `C:\Users\Leito\Documents\Learning` que es un repositorio Git conectado a `https://github.com/Leito2/Learning.git`. El vault contiene cursos comprimidos de markdown para la ruta de aprendizaje de **AI/ML Engineer**.

---

## 📁 Estructura del Vault

```
Learning/
├── .obsidian/                          # Configuración de Obsidian
├── Welcome.md                          # Índice principal del vault
├── 00 - Indice Maestro de Cursos.md    # Mapa completo de la ruta
│
├── Curso Markdown/                     # ✅ Completo (5 notas)
├── Curso SQL con PostgreSQL/           # ✅ Completo (7 notas + assets)
│
└── ML and IA Engineering/              # Vault principal de cursos
    ├── Welcome.md                      # Índice de este sub-vault (actualizado)
    │
    ├── M00 - Fundamentos de Ingenieria/           # ✅ Completo + imágenes
    │   ├── 00 - Python Avanzado para ML/          # ✅ 8 notas
    │   ├── 01 - Matematicas para ML/              # ✅ 7 notas
    │   └── 02 - Estructuras de Datos y Algoritmos/# ✅ 8 notas
    │
    ├── M01 - Deep Learning y Computer Vision/     # ✅ Completo + imágenes
    │   ├── 03 - Deep Learning con PyTorch/        # ✅ 7 notas
    │   ├── 04 - Computer Vision Avanzada/         # ✅ 6 notas
    │   └── 05 - Multimodal AI/                    # ✅ 5 notas
    │
    ├── M02 - Large Language Models/               # ✅ Completo + imágenes
    │   ├── 06 - Fundamentos de LLMs/              # ✅ 7 notas
    │   ├── 07 - Fine-Tuning y Adaptacion de LLMs/ # ✅ 6 notas
    │   ├── 08 - Generacion de Texto y Decodificacion/# ✅ 6 notas
    │   ├── 09 - Sistemas de LLMs en Produccion/   # ✅ 6 notas
    │   └── 10 - Arquitecturas Avanzadas y MoE/    # ✅ 5 notas
    │
    ├── M03 - AI Agents y Agentic Systems/         # ✅ Completo + imágenes
    │   ├── 11 - Fundamentos de Agentes AI/        # ✅ 6 notas
    │   ├── 12 - Frameworks y Orquestacion/        # ✅ 6 notas
    │   ├── 13 - Sistemas Multi-Agente/            # ✅ 5 notas
    │   └── 14 - Agentes Autonomos y Auto-Mejora/  # ✅ 5 notas
    │
    └── M04 - NLP Avanzado/                        # ✅ Completo + imágenes
        ├── 15 - NLP Tradicional y Representaciones/# ✅ 6 notas
        ├── 16 - NLP con Transformers/             # ✅ 6 notas
        └── 17 - NLP Aplicado e Industria/         # ✅ 5 notas
```

**Total actual: 127 notas creadas**

---

## 🎨 Estilo y Formato de las Notas

Cada nota de curso sigue esta estructura exacta:

1. **Título** con emoji temático.
2. **Introducción** explicando la relevancia para ML/AI.
3. **Secciones numeradas** con teoría profunda:
   - Explicaciones conceptuales con fórmulas matemáticas (LaTeX).
   - Tablas comparativas.
   - Casos reales en ML ("Caso real:").
   - ⚠️ Advertencias sobre errores comunes.
   - 💡 Tips y reglas mnemotécnicas.
4. **Imágenes ilustrativas**: diagramas Mermaid + URLs de Wikimedia Commons.
5. **Código de ejemplo** en bloques Python cuando se necesita ilustrar.
6. **📦 Código de compresión** al final: un script completo que resume todo el tema.
7. **🎯 Proyecto documentado** al final:
   - Descripción del proyecto.
   - Requisitos funcionales numerados.
   - Componentes principales (si aplica).
   - Métricas de éxito.
   - Referencias a librerías/papers.

**Reglas de formato:**
- Usar `[[...]]` para enlaces internos de Obsidian.
- Nombres de archivos: `## - Nombre Descriptivo.md` (con espacios).
- Encabezado H1 único por nota.
- Líneas en blanco entre párrafos.
- Tablas Markdown para comparativas.
- Emojis para secciones: 📦 (código), 🎯 (proyecto), 💡 (tip), ⚠️ (advertencia).

---

## 📋 Estado de los Módulos

| Módulo | Estado | # Notas |
|--------|--------|---------|
| M00 - Fundamentos de Ingeniería | ✅ Completo + imágenes | 23 notas |
| M01 - Deep Learning y Computer Vision | ✅ Completo + imágenes | 18 notas |
| M02 - Large Language Models | ✅ Completo + imágenes | 30 notas |
| M03 - AI Agents y Agentic Systems | ✅ Completo + imágenes | 22 notas |
| M04 - NLP Avanzado | ✅ Completo + imágenes | 17 notas |
| M05 - MLOps y Producción | 🚧 Pendiente | 4 cursos |
| M06 - Cloud, Infra y Backend | 🚧 Pendiente | 4 cursos |
| M07 - Research y Ciencia de Datos | 🚧 Pendiente | 4 cursos |
| M08 - Producto, Negocio y Open Source | 🚧 Pendiente | 3 cursos |

---

## 🚀 Próxima Tarea: Módulo 05 - MLOps y Producción

Crear la siguiente estructura dentro de `ML and IA Engineering/`:

```
M05 - MLOps y Produccion/
├── 18 - Experiment Tracking y Model Registry/
│   ├── 00 - Bienvenida.md
│   ├── 01 - MLflow y Tracking de Experimentos.md
│   ├── 02 - Versionado de Datos con DVC.md
│   ├── 03 - Model Registry y Lifecycle.md
│   ├── 04 - Testing de ML.md
│   └── 05 - Caso Practico - Pipeline de Entrenamiento Versionado.md
│
├── 19 - Feature Engineering y Feature Stores/
│   ├── 00 - Bienvenida.md
│   ├── 01 - Feature Engineering Avanzado.md
│   ├── 02 - Feature Stores (Feast, Tecton).md
│   ├── 03 - Online vs Offline Features.md
│   ├── 04 - Feature Monitoring y Drift.md
│   └── 05 - Caso Practico - Feature Store para E-commerce.md
│
├── 20 - Deployment y Serving/
│   ├── 00 - Bienvenida.md
│   ├── 01 - Docker para ML.md
│   ├── 02 - Model Serving Patterns.md
│   ├── 03 - Kubernetes para ML.md
│   ├── 04 - A-B Testing y Shadow Deployment.md
│   └── 05 - Caso Practico - API de ML con FastAPI y K8s.md
│
└── 21 - Monitoreo y Mantenimiento/
    ├── 00 - Bienvenida.md
    ├── 01 - Data Drift y Concept Drift.md
    ├── 02 - Monitoreo de Modelos en Produccion.md
    ├── 03 - Retraining Automatico.md
    ├── 04 - Explainability y Fairness Monitoring.md
    └── 05 - Caso Practico - Sistema de Monitoreo End-to-End.md
```

**Instrucciones específicas:**
1. Cada curso debe tener su `00 - Bienvenida.md` con el índice de notas y glosario.
2. La teoría debe ser **profunda**: explicar el "por qué" detrás de cada técnica, no solo el "cómo".
3. Incluir fórmulas matemáticas cuando aplique (drift detection, A/B testing, feature importance).
4. Cada nota debe incluir **al menos 2-3 imágenes/diagramas** (Mermaid + Wikimedia Commons).
5. Cada nota debe terminar con **📦 Código de compresión** y **🎯 Proyecto documentado**.
6. Actualizar `ML and IA Engineering/Welcome.md` para incluir los nuevos cursos.
7. Hacer `git add -A`, `git commit`, `git push origin master` después de cada curso completo.

---

## 🔗 Repositorio

- **GitHub:** `https://github.com/Leito2/Learning`
- **Rama:** `master`
- **Ruta local:** `C:\Users\Leito\Documents\Learning`

---

## 📌 Comandos útiles

```bash
# Ver estado
git -C "C:\Users\Leito\Documents\Learning" status

# Stage, commit, push
git -C "C:\Users\Leito\Documents\Learning" add -A
git -C "C:\Users\Leito\Documents\Learning" commit -m "mensaje"
git -C "C:\Users\Leito\Documents\Learning" push origin master

# Ver estructura
Get-ChildItem -Recurse "C:\Users\Leito\Documents\Learning\ML and IA Engineering"
```

---

> **Copia todo el contenido entre las líneas de separación y pégalo como mensaje inicial en la nueva sesión de Kimi.**
