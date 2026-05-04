# 02 - Git: Conexión y Sincronización

Git es el sistema de control de versiones que conecta tu carpeta local con GitHub.

---

## 🔗 Ver conexión remota

```bash
git remote -v
```
**Qué hace:** Muestra a qué URL de GitHub está vinculado tu repositorio local.

**Ejemplo de salida:**
```
origin  https://github.com/Leito2/Learning.git (fetch)
origin  https://github.com/Leito2/Learning.git (push)
```

---

## 📊 Ver estado del repositorio

```bash
git status
```
**Qué hace:** Muestra qué archivos cambiaron, cuáles están listos para commit y cuáles no están rastreados.

**Posibles estados:**
- `Changes not staged for commit`: Archivos modificados pero no agregados.
- `Untracked files`: Archivos nuevos que Git no conoce.
- `nothing to commit, working tree clean`: Todo está sincronizado.

---

## ➕ Preparar cambios (Staging)

```bash
git add -A
```
**Qué hace:** Agrega **todos** los cambios (modificaciones, eliminaciones, archivos nuevos) al área de preparación.

**Variantes:**
```bash
git add nombre_archivo.md     # Agregar un archivo específico
git add .                       # Agregar todos los cambios del directorio actual
```

---

## 💾 Guardar cambios (Commit)

```bash
git commit -m "Mensaje descriptivo del cambio"
```
**Qué hace:** Crea un punto de guardado en el historial local con un mensaje descriptivo.

**Buenas prácticas para el mensaje:**
- Sé claro y conciso.
- Usa verbos en infinitivo: "Add", "Fix", "Update", "Remove".
- Ejemplo: `"Add ML course notes and update README"`.

---

## 🚀 Subir a GitHub (Push)

```bash
git push origin master
```
**Qué hace:** Envía tus commits locales a la rama `master` del repositorio remoto (`origin`).

**Si usas la rama `main`:**
```bash
git push origin main
```

---

## ⬇️ Descargar cambios (Pull)

```bash
git pull origin master
```
**Qué hace:** Descarga los cambios nuevos de GitHub y los fusiona con tu copia local.

---

## 🔄 Flujo de trabajo completo

```bash
# 1. Verificar estado
git status

# 2. Agregar cambios
git add -A

# 3. Guardar en local
git commit -m "Describe el cambio"

# 4. Subir a GitHub
git push origin master
```

---

## 📋 Tabla resumen

| Comando | Función |
|---------|---------|
| `git remote -v` | Ver URL de GitHub |
| `git status` | Ver estado de cambios |
| `git add -A` | Preparar todos los cambios |
| `git commit -m "msg"` | Guardar en historial local |
| `git push origin master` | Subir a GitHub |
| `git pull origin master` | Descargar de GitHub |
