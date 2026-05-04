# 03 - DML: Manipulando Datos

**DML** (Data Manipulation Language) son los comandos para trabajar con los datos: insertar, actualizar, eliminar y consultar.

---

## 1. INSERT INTO

Agrega nuevas filas a una tabla.

```sql
-- Insertar una fila con todas las columnas
INSERT INTO clientes (nombre, email, edad)
VALUES ('Ana Pérez', 'ana@ejemplo.com', 28);

-- Insertar múltiples filas
INSERT INTO clientes (nombre, email, edad)
VALUES
    ('Luis García', 'luis@ejemplo.com', 35),
    ('María López', 'maria@ejemplo.com', 22);

-- Insertar usando SELECT (copiar datos de otra tabla)
INSERT INTO clientes_backup
SELECT * FROM clientes WHERE activo = TRUE;
```

> 💡 Si omites la lista de columnas, debes proporcionar valores para **todas** las columnas en el orden correcto.

---

## 2. UPDATE

Modifica datos existentes.

```sql
-- Actualizar una columna
UPDATE clientes
SET email = 'ana.nueva@ejemplo.com'
WHERE id = 1;

-- Actualizar múltiples columnas
UPDATE clientes
SET email = 'nuevo@ejemplo.com', activo = FALSE
WHERE id = 2;

-- Actualizar con condición
UPDATE productos
SET precio = precio * 1.10
WHERE categoria = 'Electrónica';
```

> ⚠️ **Si omites `WHERE`, afectas TODAS las filas.**

---

## 3. DELETE

Elimina filas de una tabla.

```sql
-- Eliminar filas específicas
DELETE FROM clientes
WHERE id = 3;

-- Eliminar con condición
DELETE FROM pedidos
WHERE fecha < '2024-01-01';
```

> ⚠️ **Sin `WHERE`, vacías toda la tabla.**

---

## 4. TRUNCATE

Vacia una tabla completamente, más rápido que `DELETE` sin `WHERE`.

```sql
TRUNCATE TABLE log_eventos;

-- Varias tablas a la vez
TRUNCATE TABLE tabla1, tabla2;

-- Reinicia secuencias asociadas
TRUNCATE TABLE clientes RESTART IDENTITY;
```

> 💡 `TRUNCATE` no ejecuta triggers `ON DELETE` y no se puede usar con `WHERE`.

---

## 5. RETURNING

PostgreSQL permite devolver los datos afectados en la misma operación.

```sql
-- Devolver la fila insertada (útil con SERIAL)
INSERT INTO clientes (nombre, email)
VALUES ('Carlos Ruiz', 'carlos@ejemplo.com')
RETURNING id, nombre;

-- Ver qué se actualizó
UPDATE productos
SET stock = stock - 1
WHERE id = 10
RETURNING id, nombre, stock;

-- Ver qué se eliminó
DELETE FROM clientes
WHERE id = 5
RETURNING *;
```

---

## 📋 Comparativa rápida

| Operación | Comando | Reversible | Usa WHERE |
|-----------|---------|------------|-----------|
| Insertar filas | `INSERT` | No aplica | No |
| Modificar filas | `UPDATE` | Sí (con backup) | Sí |
| Eliminar filas | `DELETE` | Sí (con backup) | Sí |
| Vaciar tabla | `TRUNCATE` | No | No |

---

## 🛡️ Buenas prácticas

1. **Siempre usa `WHERE`** en `UPDATE` y `DELETE`.
2. **Haz un `SELECT` primero** para verificar qué filas afectarás.
3. **Usa transacciones** (`BEGIN; ... COMMIT;`) para operaciones críticas.
4. **Usa `RETURNING`** para confirmar inserciones y actualizaciones.
