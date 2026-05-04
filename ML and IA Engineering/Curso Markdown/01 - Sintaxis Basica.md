# 01 - Sintaxis Básica

Markdown se basa en caracteres comunes del teclado. Con muy pocos símbolos, logras mucho formato.

---

## 1. Encabezados

Usa `#` seguido de un espacio. Más `#`, menor el nivel.

```markdown
# Encabezado 1
## Encabezado 2
### Encabezado 3
#### Encabezado 4
##### Encabezado 5
###### Encabezado 6
```

**Resultado:**

# Encabezado 1
## Encabezado 2
### Encabezado 3

---

## 2. Párrafos y saltos de línea

Escribe texto normal. Para un nuevo párrafo, deja una línea en blanco.

```markdown
Este es el primer párrafo.

Este es el segundo párrafo.
```

Para un salto de línea sin nuevo párrafo, deja dos espacios al final de la línea.

---

## 3. Énfasis: Negrita y Cursiva

| Estilo | Sintaxis | Resultado |
|--------|----------|-----------|
| Cursiva | `*texto*` o `_texto_` | *texto* |
| Negrita | `**texto**` o `__texto__` | **texto** |
| Negrita + Cursiva | `***texto***` | ***texto*** |
| Tachado | `~~texto~~` | ~~texto~~ |

---

## 4. Listas

### Lista desordenada

```markdown
- Elemento 1
- Elemento 2
  - Sub-elemento 2.1
  - Sub-elemento 2.2
- Elemento 3
```

**Resultado:**
- Elemento 1
- Elemento 2
  - Sub-elemento 2.1
  - Sub-elemento 2.2
- Elemento 3

### Lista ordenada

```markdown
1. Primer paso
2. Segundo paso
   1. Sub-paso 2.1
   2. Sub-paso 2.2
3. Tercer paso
```

**Resultado:**
1. Primer paso
2. Segundo paso
   1. Sub-paso 2.1
   2. Sub-paso 2.2
3. Tercer paso

---

## 5. Enlaces

```markdown
[Texto del enlace](https://ejemplo.com)
[Texto del enlace](https://ejemplo.com "Título opcional")
```

**Resultado:**
[Texto del enlace](https://ejemplo.com)

---

## 6. Imágenes

```markdown
![Texto alternativo](ruta/a/la/imagen.png)
![Texto alternativo](https://url.de.la/imagen.jpg)
```

> 💡 El texto alternativo se muestra si la imagen no carga y es leído por lectores de pantalla.

---

## 7. Código en línea

Usa comillas invertidas (backticks) para resaltar código dentro de una línea.

```markdown
Usa el comando `git status` para ver el estado.
```

**Resultado:**
Usa el comando `git status` para ver el estado.

---

## 8. Línea horizontal

Tres o más guiones, asteriscos o guiones bajos:

```markdown
---
***
___
```

**Resultado:**

---

## 📋 Resumen rápido

| Elemento | Sintaxis |
|----------|----------|
| Encabezado | `# Título` |
| Negrita | `**texto**` |
| Cursiva | `*texto*` |
| Lista | `- item` o `1. item` |
| Enlace | `[texto](url)` |
| Imagen | `![alt](url)` |
| Código | `` `código` `` |
| Línea | `---` |
