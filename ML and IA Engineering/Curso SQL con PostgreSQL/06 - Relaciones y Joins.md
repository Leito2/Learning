# 06 - Relaciones y Joins

Las bases de datos relacionales se basan en conectar tablas. Aquí aprendes a diseñar esas conexiones y a consultarlas.

---

## 1. Claves foráneas (Foreign Keys)

Una clave foránea vincula una columna de una tabla con la clave primaria de otra.

```sql
CREATE TABLE pedidos (
    id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total NUMERIC(12, 2),
    CONSTRAINT fk_cliente
        FOREIGN KEY (cliente_id)
        REFERENCES clientes(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);
```

**Acciones `ON DELETE` / `ON UPDATE`:**

| Acción | Comportamiento |
|--------|----------------|
| `RESTRICT` | Rechaza la operación si hay filas relacionadas |
| `CASCADE` | Propaga la operación (elimina/actualiza filas hijas) |
| `SET NULL` | Pone NULL en la clave foránea |
| `SET DEFAULT` | Pone el valor por defecto |
| `NO ACTION` | Similar a RESTRICT, pero verifica al final |

---

## 2. Tipos de relaciones

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **1:1** (Uno a uno) | Una fila en A → una fila en B | Usuario y perfil |
| **1:N** (Uno a muchos) | Una fila en A → muchas filas en B | Cliente y pedidos |
| **N:M** (Muchos a muchos) | Muchas filas en A ↔ muchas filas en B | Estudiantes y cursos |

### Tabla intermedia para N:M

```sql
CREATE TABLE estudiantes_cursos (
    estudiante_id INTEGER REFERENCES estudiantes(id),
    curso_id INTEGER REFERENCES cursos(id),
    fecha_inscripcion DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (estudiante_id, curso_id)
);
```

---

## 3. JOINs

Combina filas de dos o más tablas.

### INNER JOIN

Solo filas que coinciden en **ambas** tablas.

```sql
SELECT c.nombre, p.id, p.total
FROM clientes c
INNER JOIN pedidos p ON c.id = p.cliente_id;
```

### LEFT JOIN

Todas las filas de la tabla **izquierda**, y las coincidencias de la derecha (NULL si no hay).

```sql
SELECT c.nombre, p.id AS pedido_id
FROM clientes c
LEFT JOIN pedidos p ON c.id = p.cliente_id;
```

### RIGHT JOIN

Todas las filas de la tabla **derecha**, y las coincidencias de la izquierda.

```sql
SELECT c.nombre, p.id AS pedido_id
FROM clientes c
RIGHT JOIN pedidos p ON c.id = p.cliente_id;
```

### FULL OUTER JOIN

Todas las filas de **ambas** tablas, con NULL donde no hay coincidencia.

```sql
SELECT c.nombre, p.id AS pedido_id
FROM clientes c
FULL OUTER JOIN pedidos p ON c.id = p.cliente_id;
```

### CROSS JOIN

Producto cartesiano: cada fila de A con cada fila de B.

```sql
SELECT p.nombre, c.nombre
FROM productos p
CROSS JOIN categorias c;
```

---

## 4. Diagrama visual de JOINs

```
A INNER JOIN B     → Solo la intersección
A LEFT JOIN B      → Todo A + intersección
A RIGHT JOIN B     → Intersección + todo B
A FULL JOIN B      → Todo A + todo B
A CROSS JOIN B     → Toda la cuadrícula
```

---

## 5. Auto-join

Una tabla unida consigo misma.

```sql
-- Empleados y sus jefes
SELECT
    e.nombre AS empleado,
    j.nombre AS jefe
FROM empleados e
LEFT JOIN empleados j ON e.jefe_id = j.id;
```

---

## 6. UNION, INTERSECT, EXCEPT

Combinan resultados de múltiples `SELECT`.

```sql
-- UNION: combina y elimina duplicados
SELECT nombre FROM clientes_mexico
UNION
SELECT nombre FROM clientes_argentina;

-- UNION ALL: combina conservando duplicados
SELECT nombre FROM clientes_mexico
UNION ALL
SELECT nombre FROM clientes_argentina;

-- INTERSECT: solo lo común
SELECT producto_id FROM ventas_2024
INTERSECT
SELECT producto_id FROM ventas_2025;

-- EXCEPT: en A pero no en B
SELECT producto_id FROM catalogo_completo
EXCEPT
SELECT producto_id FROM productos_descatalogados;
```

---

## 📋 Checklist de diseño relacional

- [ ] ¿Cada tabla tiene una clave primaria?
- [ ] ¿Las claves foráneas tienen el tipo de dato correcto?
- [ ] ¿Se eligió la acción `ON DELETE` adecuada?
- [ ] ¿Se usan índices en las columnas de `JOIN`?
- [ ] ¿Las tablas N:M tienen clave primaria compuesta?
