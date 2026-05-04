# 04 - Consultas Avanzadas

`SELECT` es el comando más poderoso de SQL. Aquí exploramos técnicas para extraer exactamente lo que necesitas.

---

## 1. SELECT básico

```sql
-- Seleccionar columnas específicas
SELECT nombre, email FROM clientes;

-- Seleccionar todo
SELECT * FROM clientes;

-- Alias de columnas
SELECT nombre AS nombre_cliente, precio * 1.21 AS precio_con_iva FROM productos;

-- Eliminar duplicados
SELECT DISTINCT ciudad FROM clientes;
```

---

## 2. Filtrar con WHERE

```sql
SELECT * FROM productos WHERE precio > 50;
SELECT * FROM clientes WHERE edad BETWEEN 25 AND 35;
SELECT * FROM pedidos WHERE estado IN ('enviado', 'entregado');
SELECT * FROM clientes WHERE nombre LIKE 'A%';   -- Empieza con A
SELECT * FROM clientes WHERE email LIKE '%@gmail.com';
SELECT * FROM productos WHERE descripcion IS NULL;
```

**Operadores:**

| Operador | Significado |
|----------|-------------|
| `=` | Igual |
| `!=` / `<>` | Diferente |
| `>` / `<` / `>=` / `<=` | Comparación |
| `BETWEEN` | Dentro de un rango |
| `IN` | Dentro de una lista |
| `LIKE` | Patrón de texto (`%` = cualquier cosa, `_` = un carácter) |
| `IS NULL` / `IS NOT NULL` | Valor nulo |
| `AND` / `OR` / `NOT` | Lógicos |

---

## 3. Ordenar y limitar

```sql
-- Orden ascendente (por defecto)
SELECT * FROM productos ORDER BY precio;

-- Orden descendente
SELECT * FROM productos ORDER BY precio DESC;

-- Orden múltiple
SELECT * FROM clientes ORDER BY ciudad ASC, nombre ASC;

-- Limitar resultados
SELECT * FROM productos ORDER BY precio DESC LIMIT 10;

-- Paginación (LIMIT + OFFSET)
SELECT * FROM productos LIMIT 10 OFFSET 20;  -- Página 3
```

---

## 4. Funciones de agregación

```sql
SELECT
    COUNT(*) AS total_clientes,
    AVG(edad) AS edad_promedio,
    MAX(precio) AS precio_maximo,
    MIN(precio) AS precio_minimo,
    SUM(total) AS ventas_totales
FROM pedidos;
```

**Funciones:** `COUNT`, `SUM`, `AVG`, `MAX`, `MIN`.

---

## 5. GROUP BY y HAVING

Agrupa filas y filtra grupos.

```sql
-- Contar clientes por ciudad
SELECT ciudad, COUNT(*) AS total
FROM clientes
GROUP BY ciudad;

-- Promedio de precio por categoría
SELECT categoria, AVG(precio) AS promedio
FROM productos
GROUP BY categoria;

-- Filtrar grupos (HAVING va después de GROUP BY)
SELECT categoria, AVG(precio) AS promedio
FROM productos
GROUP BY categoria
HAVING AVG(precio) > 100;
```

> 💡 `WHERE` filtra **filas** antes de agrupar. `HAVING` filtra **grupos** después.

---

## 6. Subconsultas (Subqueries)

Consultas dentro de consultas.

```sql
-- Subconsulta en WHERE
SELECT * FROM productos
WHERE precio > (SELECT AVG(precio) FROM productos);

-- Subconsulta en FROM
SELECT categoria, promedio
FROM (SELECT categoria, AVG(precio) AS promedio FROM productos GROUP BY categoria) AS sub;

-- Subconsulta correlacionada
SELECT nombre FROM clientes c
WHERE EXISTS (SELECT 1 FROM pedidos p WHERE p.cliente_id = c.id);
```

---

## 7. CASE (Sentencias condicionales)

```sql
SELECT
    nombre,
    precio,
    CASE
        WHEN precio < 50 THEN 'Económico'
        WHEN precio BETWEEN 50 AND 150 THEN 'Medio'
        ELSE 'Premium'
    END AS categoria_precio
FROM productos;
```

---

## 📋 Orden de ejecución lógico

```sql
SELECT columnas       -- 5
FROM tabla            -- 1
WHERE condicion       -- 2
GROUP BY columnas     -- 3
HAVING condicion      -- 4
ORDER BY columnas     -- 6
LIMIT n;              -- 7
```

1. `FROM` → De dónde traemos datos.
2. `WHERE` → Filtramos filas.
3. `GROUP BY` → Agrupamos.
4. `HAVING` → Filtramos grupos.
5. `SELECT` → Elegimos columnas.
6. `ORDER BY` → Ordenamos.
7. `LIMIT` / `OFFSET` → Acotamos resultados.
