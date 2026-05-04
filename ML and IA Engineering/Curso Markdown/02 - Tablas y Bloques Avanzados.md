# 02 - Tablas y Bloques Avanzados

Markdown estándar es simple, pero soporta estructuras más complejas cuando las necesitas.

---

## 1. Tablas

Usa pipes `|` para separar columnas y guiones `-` para la fila de encabezado.

```markdown
| Columna 1 | Columna 2 | Columna 3 |
|-----------|-----------|-----------|
| Dato 1    | Dato 2    | Dato 3    |
| Dato 4    | Dato 5    | Dato 6    |
```

**Resultado:**

| Columna 1 | Columna 2 | Columna 3 |
|-----------|-----------|-----------|
| Dato 1    | Dato 2    | Dato 3    |
| Dato 4    | Dato 5    | Dato 6    |

### Alineación de texto

```markdown
| Izquierda | Centrado  | Derecha   |
|:----------|:---------:|----------:|
| Texto     | Texto     | Texto     |
```

**Resultado:**

| Izquierda | Centrado  | Derecha   |
|:----------|:---------:|----------:|
| Texto     | Texto     | Texto     |

- `|:---|` → Izquierda
- `|:---:|` → Centro
- `|---:|` → Derecha

---

## 2. Bloques de código

Para mostrar varias líneas de código, usa tres comillas invertidas. Puedes especificar el lenguaje para resaltado de sintaxis.

```markdown
    ```python
    def saludar():
        print("Hola, Markdown")
    ```
```

**Resultado:**

```python
def saludar():
    print("Hola, Markdown")
```

**Otros lenguajes comunes:** `bash`, `javascript`, `json`, `yaml`, `html`, `css`, `sql`.

---

## 3. Citas (Blockquotes)

Usa `>` al inicio de la línea.

```markdown
> Esta es una cita.
> Puede ocupar varias líneas.
>
> > Y también puede anidarse.
```

**Resultado:**

> Esta es una cita.
> Puede ocupar varias líneas.
>
> > Y también puede anidarse.

---

## 4. HTML en Markdown

Cuando Markdown no alcanza, puedes usar HTML directamente.

```markdown
Este texto tiene <mark>resaltado amarillo</mark>.

<p align="center">Texto centrado</p>

<details>
<summary>Haz clic para expandir</summary>

Contenido oculto aquí.

</details>
```

> ⚠️ No todos los renderizadores soportan todo el HTML. Úsalo con moderación si necesitas compatibilidad máxima.

---

## 5. Listas de tareas

```markdown
- [x] Tarea completada
- [ ] Tarea pendiente
- [ ] Otra tarea pendiente
```

**Resultado:**

- [x] Tarea completada
- [ ] Tarea pendiente
- [ ] Otra tarea pendiente

---

## 6. Escapar caracteres

Si necesitas mostrar un símbolo de Markdown literalmente, antepón una barra invertida `\`.

```markdown
\* Esto no será cursiva \*
\# Esto no será un encabezado
```

**Resultado:**

\* Esto no será cursiva \*
\# Esto no será un encabezado

---

## 📋 Resumen rápido

| Elemento | Sintaxis |
|----------|----------|
| Tabla | `\| Col1 \| Col2 \|` |
| Bloque de código | ` ```lenguaje` |
| Cita | `> texto` |
| HTML | `<tag>...</tag>` |
| Tarea | `- [ ] tarea` |
| Escapar | `\*texto\*` |
