# 04 - Buenas Prácticas

Escribir Markdown es fácil. Escribir Markdown **bien** requiere un poco de disciplina. Estas prácticas te harán más productivo y tus documentos más profesionales.

---

## 1. Estructura de archivos

Mantén una jerarquía clara. Usa carpetas para agrupar temas.

```
Proyecto/
├── 00 - Índice.md
├── 01 - Introducción.md
├── 02 - Desarrollo/
│   ├── 01 - Conceptos.md
│   └── 02 - Ejemplos.md
├── 03 - Conclusiones.md
└── assets/
    └── imagen.png
```

**Consejos:**
- Usa números al inicio (`01-`, `02-`) para ordenar notas alfabéticamente.
- Separa palabras con guiones bajos o guiones medios.
- Evita espacios si trabajarás mucho con la terminal.

---

## 2. Un encabezado H1 por nota

Cada archivo debería tener un único `# Título Principal`. Esto ayuda a Obsidian y a otros renderizadores a identificar el nombre de la nota.

---

## 3. Usa líneas en blanco

Separa párrafos, listas y bloques de código con líneas en blanco. Mejora mucho la legibilidad en texto plano.

**Malo:**
```markdown
# Título
Texto.
- Item 1
- Item 2
```

**Bueno:**
```markdown
# Título

Texto.

- Item 1
- Item 2
```

---

## 4. Texto alternativo en imágenes

Nunca dejes el texto alternativo vacío a menos que sea puramente decorativo.

```markdown
<!-- Bien -->
![Diagrama de flujo del algoritmo](diagrama.png)

<!-- Mal -->
![](diagrama.png)
```

---

## 5. Consistencia en la sintaxis

Elige un estilo y respétalo en todo el documento.

| Decide | Opción A | Opción B |
|--------|----------|----------|
| Listas | `- item` | `* item` |
| Énfasis | `**negrita**` | `__negrita__` |
| Encabezados | `## Título` | `Título\n---` |

No mezcles ambas en el mismo archivo.

---

## 6. Compatibilidad

Si escribes para GitHub, blogs o documentación técnica, **evita** las funciones exclusivas de Obsidian (como `[[wikilinks]]` o callouts) a menos que estés seguro de que el renderizador las soporta.

Para GitHub, usa:
- Enlaces estándar: `[texto](ruta)`
- Tablas Markdown nativas
- HTML simple cuando sea necesario

---

## 7. Commits y versionado

Si usas Git (como en este proyecto), cada cambio en tus notas debería ir con un commit descriptivo.

```bash
git add -A
git commit -m "Add: Curso completo de Markdown"
git push origin master
```

---

## 8. Revisión final

Antes de compartir o publicar:

- [ ] ¿Todos los enlaces funcionan?
- [ ] ¿Las imágenes se ven correctamente?
- [ ] ¿No hay errores ortográficos?
- [ ] ¿La estructura de encabezados es lógica?
- [ ] ¿El código tiene resaltado de sintaxis correcto?

---

## 📋 Checklist de calidad

| Criterio | Estado |
|----------|--------|
| Estructura clara | ✅ |
| Encabezado H1 único | ✅ |
| Espaciado consistente | ✅ |
| Imágenes con alt text | ✅ |
| Sintaxis uniforme | ✅ |
| Enlaces verificados | ✅ |
| Commits descriptivos | ✅ |
