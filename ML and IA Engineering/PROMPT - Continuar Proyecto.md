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
├── Advanced Python/                    # ✅ Completo (62 notas + imágenes)
│   ├── 01 - Python Basico/             # ✅ 13 notas
│   ├── 02 - Python Intermedio/         # ✅ 13 notas
│   ├── 03 - Python Avanzado/           # ✅ 13 notas
│   ├── 04 - Librerias Basicas de Python/# ✅ 8 notas
│   └── 05 - Librerias Especificas/     # ✅ 7 notas
│
├── Software Engineering/               # 🚧 En progreso
│   └── 01 - Docker Profesional/        # ✅ 7 notas
│
└── ML and IA Engineering/              # Vault principal de cursos
    ├── Welcome.md                      # Índice de este sub-vault
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

**Total actual: 204 notas creadas**

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

## ⚠️ REGLA CRÍTICA: Uso de Subagentes (Task Tool)

**PROBLEMA IDENTIFICADO:** Los subagentes (`task` tool) a veces completan el trabajo (archivos creados en disco) pero **no retornan el mensaje de resultado**, aparentando quedarse "colgados". Esto sucede cuando:
- Se lanzan más de 3 subagentes en paralelo.
- Un subagente debe crear más de 7 notas de golpe.
- Los prompts son muy extensos.

**WORKFLOW OBLIGATORIO para crear cursos:**
1. **SIEMPRE usar subagentes** para crear lotes de notas (crear una por una consume demasiado contexto del modelo principal).
2. **Máximo 2 subagentes en paralelo** por lote.
3. **Máximo 7 notas por subagente**. Si el curso tiene más notas, dividir en 2 subagentes secuenciales (verificar filesystem del primero antes de lanzar el segundo).
4. **INMEDIATAMENTE después** de lanzar subagentes, verificar el filesystem con bash:
   ```powershell
   Get-ChildItem -Recurse -File "RUTA\DEL\CURSO" | ForEach-Object { $_.FullName }
   ```
5. **Si los archivos existen**, considerar la tarea completada aunque el subagente no haya reportado nada.
6. **Si faltan archivos**, relanzar SOLO los faltantes.
7. **El filesystem es la única fuente de verdad.**

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
| Advanced Python | ✅ Completo + imágenes | 62 notas |
| Software Engineering / Docker | ✅ Completo + imágenes | 7 notas |

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

**Instrucciones específicas (aplicar reglas de subagentes):**
1. Crea los directorios con bash primero.
2. Lanza **máximo 2 subagentes** en paralelo para crear los cursos (ej: curso 18 + 19).
3. **Inmediatamente verifica** los archivos con bash.
4. Si faltan, lanza los cursos faltantes (20 + 21).
5. Cada curso tiene 6 notas → **1 subagente por curso es seguro**.
6. La teoría debe ser **profunda**: explicar el "por qué" detrás de cada técnica.
7. Incluir fórmulas matemáticas cuando aplique.
8. Cada nota debe incluir **al menos 2-3 imágenes/diagramas**.
9. Cada nota debe terminar con **📦 Código de compresión** y **🎯 Proyecto documentado**.
10. Actualizar `ML and IA Engineering/Welcome.md` para incluir los nuevos cursos.
11. Hacer `git add -A`, `git commit`, `git push origin master` después de cada curso completo.

---

## 🔗 Repositorio

- **GitHub:** `https://github.com/Leito2/Learning`
- **Rama:** `master`
- **Ruta local:** `C:\Users\Leito\Documents\Learning`

---

## 📌 Comandos útiles

```bash
# Ver estado git
git -C "C:\Users\Leito\Documents\Learning" status

# Stage, commit, push
git -C "C:\Users\Leito\Documents\Learning" add -A
git -C "C:\Users\Leito\Documents\Learning" commit -m "mensaje"
git -C "C:\Users\Leito\Documents\Learning" push origin master

# Verificar archivos de un curso (USAR SIEMPRE después de subagentes)
Get-ChildItem -Recurse -File "C:\Users\Leito\Documents\Learning\RUTA\DEL\CURSO" | ForEach-Object { $_.FullName }

# Ver estructura completa
Get-ChildItem -Recurse "C:\Users\Leito\Documents\Learning\ML and IA Engineering"
```

---

> **Copia todo el contenido entre las líneas de separación y pégalo como mensaje inicial en la nueva sesión de Kimi.**
