# 02 - DDL: Definiendo la Estructura

**DDL** (Data Definition Language) son los comandos que definen y modifican la estructura de la base de datos.

---

## 1. CREATE TABLE

Crea una nueva tabla con columnas y restricciones.

```sql
CREATE TABLE clientes (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE,
    edad INTEGER CHECK (edad >= 18),
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activo BOOLEAN DEFAULT TRUE
);
```

**Restricciones comunes:**

| Restricción | Descripción |
|-------------|-------------|
| `PRIMARY KEY` | Identificador único y no nulo |
| `UNIQUE` | Valor único en toda la tabla |
| `NOT NULL` | No permite valores vacíos |
| `DEFAULT` | Valor por defecto si no se especifica |
| `CHECK` | Condición que debe cumplirse |
| `REFERENCES` | Clave foránea (ver módulo 06) |

---

## 2. Tipos de datos principales

| Tipo | Uso | Ejemplo |
|------|-----|---------|
| `INTEGER` / `BIGINT` / `SMALLINT` | Números enteros | `42` |
| `SERIAL` / `BIGSERIAL` | Autoincremental | `1, 2, 3...` |
| `NUMERIC(p, s)` | Decimales exactos | `NUMERIC(10, 2)` → `12345678.90` |
| `REAL` / `DOUBLE PRECISION` | Decimales aproximados | `3.14159` |
| `VARCHAR(n)` | Texto variable | `'Hola'` |
| `TEXT` | Texto ilimitado | `'Largo párrafo...'` |
| `BOOLEAN` | Verdadero/Falso | `TRUE`, `FALSE` |
| `DATE` | Fecha | `'2025-05-03'` |
| `TIME` | Hora | `'14:30:00'` |
| `TIMESTAMP` | Fecha y hora | `'2025-05-03 14:30:00'` |
| `JSON` / `JSONB` | Datos JSON | `'{"key": "value"}'` |
| `ARRAY` | Arreglos | `ARRAY[1, 2, 3]` |

---

## 3. ALTER TABLE

Modifica una tabla existente sin perder datos.

```sql
-- Agregar columna
ALTER TABLE clientes ADD COLUMN telefono VARCHAR(20);

-- Eliminar columna
ALTER TABLE clientes DROP COLUMN telefono;

-- Renombrar columna
ALTER TABLE clientes RENAME COLUMN nombre TO nombre_completo;

-- Cambiar tipo de dato
ALTER TABLE clientes ALTER COLUMN edad TYPE SMALLINT;

-- Agregar restricción
ALTER TABLE clientes ADD CONSTRAINT chk_edad CHECK (edad >= 18);

-- Eliminar restricción
ALTER TABLE clientes DROP CONSTRAINT chk_edad;
```

---

## 4. DROP TABLE

Elimina una tabla y todos sus datos permanentemente.

```sql
DROP TABLE clientes;

-- Solo si existe (evita errores)
DROP TABLE IF EXISTS clientes;

-- Elimina también objetos dependientes (cuidado)
DROP TABLE clientes CASCADE;
```

---

## 5. CREATE SCHEMA

Organiza objetos en grupos lógicos.

```sql
CREATE SCHEMA ventas;

-- Crear tabla dentro del schema
CREATE TABLE ventas.facturas (
    id SERIAL PRIMARY KEY,
    total NUMERIC(12, 2)
);

-- Listar schemas
\dn
```

---

## 6. CREATE SEQUENCE

Control manual de secuencias numéricas.

```sql
CREATE SEQUENCE pedido_id_seq START 1000;

-- Usarla
INSERT INTO pedidos (id, descripcion) VALUES (nextval('pedido_id_seq'), 'Pedido A');
```

---

## 📋 Checklist de diseño de tablas

- [ ] ¿Cada tabla tiene una clave primaria?
- [ ] ¿Los tipos de datos son apropiados para el contenido?
- [ ] ¿Se usaron restricciones (`NOT NULL`, `UNIQUE`, `CHECK`) donde corresponde?
- [ ] ¿Hay valores por defecto (`DEFAULT`) para campos comunes?
- [ ] ¿El nombre de la tabla y columnas es claro y consistente?
