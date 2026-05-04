# 03 - Eliminación Segura de Archivos y Carpetas

Eliminar parece simple, pero en Windows puede ser peligroso si no se hace con precisión. Aquí aprendemos las formas correctas y los errores a evitar.

---

## 🗑️ Formas de eliminar en Windows

### 1. PowerShell nativo

```powershell
Remove-Item -Path "C:\ruta\carpeta" -Recurse -Force
```
**Qué hace:** Elimina carpetas y su contenido de forma recursiva.

```powershell
Remove-Item -LiteralPath "C:\ruta\carpeta" -Recurse -Force
```
**Qué hace:** Igual que arriba, pero interpreta el nombre literalmente (útil para caracteres especiales).

**Problema conocido:** Si el nombre de la carpeta tiene un espacio al final (`ML `), PowerShell puede no encontrarla aunque exista.

---

### 2. CMD antiguo

```cmd
rd /s /q "C:\ruta\carpeta"
```
**Qué hace:** Elimina directorios (`/s` = recursivo, `/q` = silencioso).

> ⚠️ **PELIGRO:** Si las comillas o el espacio final se interpretan mal, el comando puede **salirse de la ruta** y tocar archivos del sistema. **Evitar si hay caracteres especiales.**

---

### 3. Python (La forma más segura)

```python
python -c "import shutil; shutil.rmtree(r'C:\ruta\carpeta')"
```
**Qué hace:** Usa Python para eliminar la carpeta de forma recursiva.

#### Para nombres con espacios al final (ruta extendida)

```python
python -c "import shutil; shutil.rmtree(r'\\?\C:\ruta\carpeta ')"
```
**Qué hace:** Usa la sintaxis `\\?\` (ruta extendida de Windows) que permite manejar nombres con espacios finales, puntos o más de 260 caracteres.

---

## 🔍 Cómo detectar espacios ocultos

Si un archivo o carpeta no se puede eliminar ni encontrar, revisa sus bytes reales:

```powershell
Get-ChildItem "C:\ruta\padre" | ForEach-Object {
    [System.BitConverter]::ToString(
        [System.Text.Encoding]::Unicode.GetBytes($_.Name)
    )
}
```

**Interpretación del resultado:**
```
4D-00-4C-00-20-00       = "ML "  (hay espacio al final: 20-00)
4D-00-4C-00             = "ML"   (nombre limpio)
```

---

## 🛡️ Reglas de oro

| ✅ Hacer | ❌ Evitar |
|----------|-----------|
| Verificar el nombre exacto con `Get-ChildItem` | Usar `rd /s /q` con rutas no verificadas |
| Usar `\?\` para nombres problemáticos | Ignorar espacios al final de nombres |
| Usar Python para casos extremos | Forzar eliminación sin confirmar la ruta |
| Revisar los bytes del nombre si falla | Ejecutar `cmd /c` con comillas complejas |

---

## ⚠️ Lección aprendida

En esta sesión, la carpeta se llamaba `ML ` (con espacio al final). Intentar usar `cmd /c rd /s /q` causó que el comando intentara acceder a archivos del sistema (Anaconda) por mala interpretación de la ruta.

**La solución segura fue:**
```python
python -c "import shutil; shutil.rmtree(r'\\?\C:\Users\Leito\Documents\Learning\ML ')"
```
