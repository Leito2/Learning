# 01 - Navegación y Exploración

Antes de hacer cualquier cambio, necesitas **ver** qué hay en el sistema. Estos comandos son tus ojos.

---

## 📂 Listar archivos y carpetas

### PowerShell / Windows

```powershell
ls
# o
Get-ChildItem
```
**Qué hace:** Muestra el contenido del directorio actual (archivos y carpetas).

```powershell
ls -Recurse
# o
Get-ChildItem -Recurse
```
**Qué hace:** Muestra el contenido de forma recursiva (incluye subcarpetas).

```powershell
Get-ChildItem | Select-Object Name
```
**Qué hace:** Lista solo los nombres de archivos y carpetas.

```powershell
Get-ChildItem | Format-List *
```
**Qué hace:** Muestra toda la información detallada (modo, fechas, atributos, etc.).

---

## 🔍 Ver bytes exactos de un nombre

```powershell
Get-ChildItem "C:\ruta" | ForEach-Object {
    [System.BitConverter]::ToString(
        [System.Text.Encoding]::Unicode.GetBytes($_.Name)
    )
}
```
**Qué hace:** Revela los caracteres ocultos de un nombre (como espacios al final). Útil para diagnosticar por qué un archivo no se encuentra.

**Ejemplo real:**
- Nombre visible: `ML`
- Bytes reales: `4D-00-4C-00-20-00` → ¡Hay un espacio al final (`20-00`)!

---

## 🖥️ Ejecutar comandos de CMD desde PowerShell

```powershell
cmd /c "comando_aqui"
```
**Qué hace:** Ejecuta un comando del antiguo símbolo del sistema (CMD) dentro de PowerShell.

> ⚠️ **Cuidado:** Mezclar comillas y rutas con espacios puede causar comportamientos inesperados. Preferir soluciones nativas de PowerShell o Python.

---

## 🐍 Ejecutar Python inline

```python
python -c "codigo_aqui"
```
**Qué hace:** Ejecuta una línea de Python sin crear un archivo `.py`.

```python
python -c "import shutil; shutil.rmtree(r'\\?\C:\ruta\carpeta ')"
```
**Qué hace:** Usa la **sintaxis de ruta extendida** (`\\?\`) para manejar nombres con espacios al final en Windows.

---

## 📝 Resumen visual

| Comando | Sistema | Propósito |
|---------|---------|-----------|
| `ls` / `Get-ChildItem` | PowerShell | Listar contenido |
| `ls -Recurse` | PowerShell | Listar recursivamente |
| `cmd /c` | PowerShell → CMD | Ejecutar comando antiguo |
| `python -c` | Cualquiera | Script rápido inline |
| `BitConverter` | PowerShell | Revelar caracteres ocultos |
