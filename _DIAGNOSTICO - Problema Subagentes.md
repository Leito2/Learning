# 🔧 Diagnóstico: Problema con Subagentes

> Fecha: 2026-05-04  
> Contexto: Construcción del vault de Obsidian `Learning`

---

## 🐛 Síntoma

Al lanzar múltiples `task` (subagentes) en paralelo para crear notas de cursos, ocurre lo siguiente:

1. El subagente **sí ejecuta el trabajo** (archivos creados en disco).
2. El subagente **no retorna el mensaje de resultado** al agente padre.
3. Desde la perspectiva del usuario, parece que "se quedó ejecutando la nada".
4. Al verificar el filesystem manualmente, los archivos están completos.

---

## 🔍 Causa Raíz

Este es un **problema de infraestructura de la plataforma**, no de código:

- Los subagentes son instancias aisladas que ejecutan prompts largos con múltiples operaciones `Write`.
- Cuando un subagente escribe **muchos archivos grandes** (>10 notas con contenido extenso), el canal de comunicación de vuelta al padre puede **cerrarse o perderse**.
- El subagente termina su ejecución internamente pero el resultado nunca llega al contexto del padre.
- Esto se agrava cuando se lanzan **5+ subagentes simultáneamente** competiendo por I/O de disco.

**No es un bug que podamos arreglar con código.** Es un límite operacional de la herramienta `task`.

---

## ✅ Soluciones Implementadas

### 1. Verificación Forzada Post-Subagente

NUNCA confiar en el reporte del subagente. SIEMPRE ejecutar:

```powershell
Get-ChildItem -Recurse -File "RUTA\DEL\CURSO" | ForEach-Object { $_.FullName }
```

Si los archivos existen y tienen tamaño >100 bytes, la tarea está completa.

### 2. Límite de Subagentes en Paralelo

**Antes:** Lanzar 5 subagentes simultáneos (uno por curso).  
**Ahora:** Máximo **2-3 subagentes en paralelo**.

### 3. Límite de Notas por Subagente

**Antes:** Un subagente creaba 12 notas de golpe.  
**Ahora:** Máximo **7 notas por subagente**.

### 4. Fragmentación de Prompts

Si un curso tiene >7 notas, dividir en 2 subagentes:
- Subagente A: Notas 00-06
- Subagente B: Notas 07-12

### 5. Subagentes SÍ se Usan (Con Límites)

**NO evitar subagentes completamente** — crear notas una por una con `Write` directo consume **demasiado contexto** del modelo principal y es ineficiente.

**Regla:**
- ✅ **Usar subagentes** para crear lotes de 3–7 notas.
- ✅ **Máximo 2 subagentes en paralelo**.
- ✅ **Verificar filesystem** inmediatamente después.
- ✅ **Relanzar** si faltan archivos.
- ❌ **NO** lanzar 5+ subagentes simultáneos.
- ❌ **NO** pedir a un subagente que cree más de 7 notas.

---

## 📋 Casos Confirmados

| Curso | Subagente Reportó | Archivos Existían | Acción Tomada |
|-------|-------------------|-------------------|---------------|
| M03-C12 Frameworks | ❌ Vacío | ✅ 6 notas | Verificación manual + commit |
| M04-C04 Librerías Básicas | ❌ Vacío | ✅ 7 notas | Verificación manual + commit |
| Docker Profesional | ❌ Vacío | ✅ 7 notas | Verificación manual + commit |

---

## 🛡️ Workflow Recomendado para Futuras Sesiones

```
1. Crear directorios con bash
2. Lanzar máximo 2 subagentes en paralelo
3. Inmediatamente después, verificar filesystem con bash
4. Si faltan archivos, relanzar SOLO esos faltantes
5. Si existen todos, continuar directamente (ignorar reporte vacío)
6. Nunca esperar activamente la respuesta del subagente
```

---

## 📝 Nota para Agentes Futuros

> **Regla de oro:** `task` tool es útil para paralelización, pero es **inestable con tareas masivas**.  
> Trátalo como un "fire and forget": lanzas, verificas disco, y sigues.  
> El filesystem es la única fuente de verdad.
