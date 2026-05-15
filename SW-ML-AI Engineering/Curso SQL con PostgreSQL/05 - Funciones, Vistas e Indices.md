# 05 - Funciones, Vistas e Índices

PostgreSQL ofrece herramientas poderosas para reutilizar lógica, simplificar consultas y acelerar el acceso a datos.

---

## 1. Funciones definidas por el usuario

### Funciones SQL simples

```sql
CREATE FUNCTION calcular_iva(precio NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    RETURN precio * 0.21;
END;
$$ LANGUAGE plpgsql;

-- Usarla
SELECT nombre, precio, calcular_iva(precio) AS iva FROM productos;
```

### Funciones con múltiples parámetros

```sql
CREATE FUNCTION descuento(precio NUMERIC, porcentaje NUMERIC)
RETURNS NUMERIC AS $$
BEGIN
    RETURN precio * (1 - porcentaje / 100);
END;
$$ LANGUAGE plpgsql;
```

### Eliminar función

```sql
DROP FUNCTION calcular_iva(NUMERIC);
```

---

## 2. Vistas (Views)

Una vista es una consulta almacenada que se comporta como una tabla virtual.

```sql
-- Crear vista
CREATE VIEW clientes_activos AS
SELECT id, nombre, email
FROM clientes
WHERE activo = TRUE;

-- Usarla como si fuera una tabla
SELECT * FROM clientes_activos;

-- Actualizar vista (con cuidado)
CREATE OR REPLACE VIEW clientes_activos AS ...;

-- Eliminar vista
DROP VIEW IF EXISTS clientes_activos;
```

> 💡 Las vistas simplifican consultas complejas y pueden restringir qué datos ven ciertos usuarios.

---

## 3. Índices (Indexes)

Los índices aceleran las búsquedas, pero ralentizan las inserciones y actualizaciones.

```sql
-- Índice simple
CREATE INDEX idx_clientes_email ON clientes(email);

-- Índice único
CREATE UNIQUE INDEX idx_clientes_email_unico ON clientes(email);

-- Índice compuesto
CREATE INDEX idx_pedidos_cliente_fecha ON pedidos(cliente_id, fecha);

-- Índice para búsquedas de texto (ILIKE)
CREATE INDEX idx_productos_nombre ON productos USING gin(nombre gin_trgm_ops);
```

### Ver índices existentes

```sql
\d clientes
```

### Eliminar índice

```sql
DROP INDEX idx_clientes_email;
```

### ¿Cuándo crear índices?

| ✅ Crear índice | ❌ Evitar índice |
|-----------------|------------------|
| Columnas en `WHERE` | Tablas pequeñas |
| Columnas en `JOIN` | Columnas que cambian constantemente |
| Columnas en `ORDER BY` | Tablas de solo escritura |
| Columnas únicas (`UNIQUE`) | Sin análisis previo |

---

## 4. CTE (Common Table Expressions)

Subconsultas nombradas que mejoran la legibilidad.

```sql
WITH ventas_por_cliente AS (
    SELECT cliente_id, SUM(total) AS total_gastado
    FROM pedidos
    GROUP BY cliente_id
)
SELECT c.nombre, v.total_gastado
FROM clientes c
JOIN ventas_por_cliente v ON c.id = v.cliente_id
WHERE v.total_gastado > 1000;
```

### CTE recursivas

```sql
WITH RECURSIVE subordinados AS (
    -- Caso base
    SELECT id, nombre, jefe_id
    FROM empleados
    WHERE id = 1

    UNION ALL

    -- Caso recursivo
    SELECT e.id, e.nombre, e.jefe_id
    FROM empleados e
    INNER JOIN subordinados s ON e.jefe_id = s.id
)
SELECT * FROM subordinados;
```

---

## 📋 Resumen

| Objeto | Propósito | Comando |
|--------|-----------|---------|
| **Función** | Reutilizar lógica | `CREATE FUNCTION` |
| **Vista** | Simplificar consultas | `CREATE VIEW` |
| **Índice** | Acelerar búsquedas | `CREATE INDEX` |
| **CTE** | Consultas legibles y recursivas | `WITH ...` |
