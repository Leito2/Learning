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
    ├── Welcome.md                      # Índice de este sub-vault
    │
    └── M00 - Fundamentos de Ingenieria/    # ✅ Módulo completo
        ├── 00 - Python Avanzado para ML/   # ✅ 7 notas + teoría expandida
        ├── 01 - Matematicas para ML/       # ✅ 6 notas
        └── 02 - Estructuras de Datos y Algoritmos/  # ✅ 7 notas
```

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
4. **Código de ejemplo** en bloques Python cuando se necesita ilustrar.
5. **📦 Código de compresión** al final: un script completo que resume todo el tema.
6. **🎯 Proyecto documentado** al final:
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
| M00 - Fundamentos de Ingeniería | ✅ Completo | 20 notas |
| M01 - Deep Learning y Computer Vision | 🚧 Pendiente | 3 cursos |
| M02 - Large Language Models | 🚧 Pendiente | 5 cursos |
| M03 - AI Agents y Agentic Systems | 🚧 Pendiente | 4 cursos |
| M04 - NLP Avanzado | 🚧 Pendiente | 3 cursos |
| M05 - MLOps y Producción | 🚧 Pendiente | 4 cursos |
| M06 - Cloud, Infra y Backend | 🚧 Pendiente | 4 cursos |
| M07 - Research y Ciencia de Datos | 🚧 Pendiente | 4 cursos |
| M08 - Producto, Negocio y Open Source | 🚧 Pendiente | 3 cursos |

---

## 🚀 Próxima Tarea: Módulo 01 - Deep Learning y Computer Vision

Crear la siguiente estructura dentro de `ML and IA Engineering/`:

```
M01 - Deep Learning y Computer Vision/
├── 03 - Deep Learning con PyTorch/
│   ├── 00 - Bienvenida.md
│   ├── 01 - Introduccion a Redes Neuronales.md
│   ├── 02 - CNNs y Arquitecturas Modernas.md
│   ├── 03 - RNNs y Secuencias.md
│   ├── 04 - Training Strategies.md
│   ├── 05 - Transfer Learning.md
│   └── 06 - Caso Practico - Clasificador de Imagenes Medico.md
│
├── 04 - Computer Vision Avanzada/
│   ├── 00 - Bienvenida.md
│   ├── 01 - Object Detection.md
│   ├── 02 - Segmentacion Semantica.md
│   ├── 03 - Vision Transformers.md
│   ├── 04 - OCR y Document Understanding.md
│   └── 05 - Caso Practico - Sistema de Document Intelligence.md
│
└── 05 - Multimodal AI/
    ├── 00 - Bienvenida.md
    ├── 01 - CLIP y Representaciones Conjuntas.md
    ├── 02 - Modelos de Image Captioning.md
    ├── 03 - Multimodal Transformers.md
    └── 04 - Caso Practico - Buscador Visual Multimodal.md
```

**Instrucciones específicas:**
1. Cada curso debe tener su `00 - Bienvenida.md` con el índice de notas y glosario.
2. La teoría debe ser **profunda**: explicar el "por qué" detrás de cada técnica, no solo el "cómo".
3. Incluir fórmulas matemáticas cuando aplique (backpropagation, convolución, attention).
4. Cada nota debe terminar con **📦 Código de compresión** (PyTorch) y **🎯 Proyecto documentado**.
5. Actualizar `ML and IA Engineering/Welcome.md` para incluir los nuevos cursos.
6. Hacer `git add -A`, `git commit`, `git push origin master` después de cada curso completo.

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
