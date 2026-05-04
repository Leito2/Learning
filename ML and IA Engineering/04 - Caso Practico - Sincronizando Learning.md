# 04 - Caso Práctico: Sincronizando Learning

Este es el registro paso a paso de lo que hicimos con este repositorio. Úsalo como guía para futuras sincronizaciones.

---

## 🎯 Situación inicial

- Carpeta: `C:\Users\Leito\Documents\Learning`
- Ya era un repositorio Git vinculado a `https://github.com/Leito2/Learning.git`
- Había archivos viejos eliminados y una nueva carpeta sin rastrear.

---

## 📋 Paso 1: Explorar y diagnosticar

```bash
ls "C:\Users\Leito\Documents\Learning"
```
Listamos el contenido y encontramos:
- `ML ` ← carpeta con espacio al final (problema futuro)
- `ML and IA Engineering` ← carpeta nueva sin seguimiento

```bash
git -C "C:\Users\Leito\Documents\Learning" status
```
Detectamos:
- Archivos borrados en la raíz (`.obsidian/`, `Welcome.md`, etc.)
- Carpeta nueva sin rastrear (`ML and IA Engineering/`)

---

## 📋 Paso 2: Sincronizar con GitHub

```bash
# Agregar todo al staging
git -C "C:\Users\Leito\Documents\Learning" add -A

# Crear commit
git -C "C:\Users\Leito\Documents\Learning" commit -m "Sync local changes: remove old files and add ML and IA Engineering"

# Subir a GitHub
git -C "C:\Users\Leito\Documents\Learning" push origin master
```

**Resultado:** Rama `master` actualizada. Los cambios quedaron en GitHub.

---

## 📋 Paso 3: Eliminar carpeta problemática

La carpeta `ML ` tenía un espacio al final, lo que la hacía difícil de eliminar.

**Intentos fallidos:**
```powershell
Remove-Item "...\ML"          # No encontrada
rmdir "...\ML"                # No encontrada
cmd /c "rd /s /q \"...\ML\"" # Comportamiento impredecible
```

**Diagnóstico correcto:**
```powershell
Get-ChildItem "..." | ForEach-Object {
    [System.BitConverter]::ToString(
        [System.Text.Encoding]::Unicode.GetBytes($_.Name)
    )
}
# Resultado: 4D-00-4C-00-20-00 → "ML " con espacio
```

**Solución segura:**
```python
python -c "import shutil; shutil.rmtree(r'\\?\C:\Users\Leito\Documents\Learning\ML ')"
```

**Verificación:**
```powershell
Get-ChildItem "C:\Users\Leito\Documents\Learning"
# Resultado: Solo queda "ML and IA Engineering"
```

---

## 📋 Paso 4: Crear vault de Obsidian

Creamos la estructura de un vault funcional dentro de la misma carpeta `Learning`:

```
Learning/
├── .obsidian/
│   ├── app.json
│   ├── appearance.json
│   ├── core-plugins.json
│   └── workspace.json
├── Welcome.md
└── Curso Comprimido de Comandos/
    ├── 00 - Bienvenida.md
    ├── 01 - Navegacion y Exploracion.md
    ├── 02 - Git - Conexion y Sincronizacion.md
    ├── 03 - Eliminacion Segura de Archivos y Carpetas.md
    └── 04 - Caso Practico - Sincronizando Learning.md
```

---

## ✅ Estado final

- Repositorio sincronizado con GitHub.
- Carpeta `ML` eliminada correctamente.
- Vault de Obsidian creado y funcional.
- Curso documentado en Markdown.

---

## 🔄 Próximos pasos recomendados

1. Abre Obsidian.
2. Selecciona "Open folder as vault".
3. Elige `C:\Users\Leito\Documents\Learning`.
4. Explora las notas del curso.
5. Cuando hagas cambios, repite el flujo: `add -A` → `commit` → `push`.
