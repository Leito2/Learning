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
├── Software Engineering/               # ✅ En progreso
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
    ├── M04 - NLP Avanzado/                        # ✅ Completo + imágenes
    │   ├── 15 - NLP Tradicional y Representaciones/# ✅ 6 notas
    │   ├── 16 - NLP con Transformers/             # ✅ 6 notas
    │   └── 17 - NLP Aplicado e Industria/         # ✅ 5 notas
    │
    ├── M05 - MLOps y Produccion/                  # ✅ Completo + imágenes
    │   ├── 18 - Experiment Tracking y Model Registry/ # ✅ 6 notas
    │   ├── 19 - Feature Engineering y Feature Stores/ # ✅ 6 notas
    │   ├── 20 - Deployment y Serving/             # ✅ 6 notas
    │   └── 21 - Monitoreo y Mantenimiento/        # ✅ 6 notas
    │
    ├── M06 - Cloud, Infra y Backend/              # ✅ Completo + imágenes
    │   ├── 22 - Cloud Computing/                  # ✅ 6 notas
    │   ├── 23 - Infraestructura como Codigo/      # ✅ 6 notas
    │   ├── 24 - Backend para ML/                  # ✅ 6 notas
    │   └── 25 - Bases de Datos y Message Queues/  # ✅ 6 notas
    │
    ├── M07 - Research y Ciencia de Datos/         # ✅ Completo + imágenes
    │   ├── 26 - Metodologia de Investigacion en ML/ # ✅ 6 notas
    │   ├── 27 - Visualizacion de Datos y Storytelling/ # ✅ 6 notas
    │   ├── 28 - ETL y Data Engineering/           # ✅ 6 notas
    │   └── 29 - Estadistica Avanzada y Causalidad/# ✅ 6 notas
    │
    └── M08 - Producto, Negocio y Open Source/     # 🚧 Pendiente
```

**Total actual: 276 notas creadas**

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
- **IMPORTANTE: NO usar `/` en nombres de archivos. Usar `-` en su lugar** (Windows no lo permite).
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
| M05 - MLOps y Producción | ✅ Completo + imágenes | 24 notas |
| M06 - Cloud, Infra y Backend | ✅ Completo + imágenes | 24 notas |
| M07 - Research y Ciencia de Datos | ✅ Completo + imágenes | 24 notas |
| M08 - Producto, Negocio y Open Source | 🚧 Pendiente | 3 cursos |
| Advanced Python | ✅ Completo + imágenes | 62 notas |
| Software Engineering / Docker | ✅ Completo + imágenes | 7 notas |

---

## 🚀 Próxima Tarea: Módulo 08 - Producto, Negocio y Open Source

Crear la siguiente estructura dentro de `ML and IA Engineering/`:

```
M08 - Producto, Negocio y Open Source/
├── 30 - Producto y Estrategia de IA/
│   ├── 00 - Bienvenida.md
│   ├── 01 - Diseño de Productos con IA.md
│   ├── 02 - Estrategia y Roadmap de ML.md
│   ├── 03 - Ética y Responsabilidad en IA.md
│   ├── 04 - Legal y Compliance en IA.md
│   └── 05 - Caso Practico - Lanzamiento de Producto IA.md
│
├── 31 - Negocio y Metricas de ML/
│   ├── 00 - Bienvenida.md
│   ├── 01 - ROI de Proyectos de ML.md
│   ├── 02 - Métricas de Negocio vs Métricas Técnicas.md
│   ├── 03 - Costos y Presupuesto de ML.md
│   ├── 04 - Comunicación con Stakeholders.md
│   └── 05 - Caso Practico - Business Case para ML.md
│
└── 32 - Open Source y Comunidad/
    ├── 00 - Bienvenida.md
    ├── 01 - Contribución a Open Source.md
    ├── 02 - Construcción de Comunidad Técnica.md
    ├── 03 - Publicación de Librerías y Papers.md
    ├── 04 - Carrera y Crecimiento Profesional.md
    └── 05 - Caso Practico - Lanzar una Librería Open Source.md
```

**Instrucciones específicas (aplicar reglas de subagentes):**
1. Crea los directorios con bash primero.
2. Lanza **máximo 2 subagentes** en paralelo para crear los cursos (ej: curso 30 + 31).
3. **Inmediatamente verifica** los archivos con bash.
4. Si faltan, lanza el curso faltante (32).
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
