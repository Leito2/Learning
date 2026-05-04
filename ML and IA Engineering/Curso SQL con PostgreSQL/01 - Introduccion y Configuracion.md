# 01 - Introducción y Configuración

Antes de escribir código SQL, necesitas entender el entorno y cómo interactuar con PostgreSQL.

---

## 1. Instalación

### Windows
1. Descarga el instalador desde [postgresql.org/download](https://www.postgresql.org/download/).
2. Ejecuta el instalador y sigue el asistente.
3. Guarda la contraseña del usuario `postgres`.
4. El instalador incluye **pgAdmin** (interfaz gráfica) y **psql** (terminal).

### Verificar instalación

```bash
psql --version
```

---

## 2. Conectarse a PostgreSQL

### Desde la terminal (psql)

```bash
psql -U postgres -h localhost -p 5432
```

**Parámetros:**
- `-U`: usuario
- `-h`: host (localhost por defecto)
- `-p`: puerto (5432 por defecto)

### Desde pgAdmin
1. Abre pgAdmin.
2. Crea un nuevo servidor con los datos de conexión.
3. Expande **Databases** para ver las bases disponibles.

---

## 3. Comandos básicos de psql

| Comando | Descripción |
|---------|-------------|
| `\l` | Listar todas las bases de datos |
| `\c nombre_db` | Conectarse a una base de datos |
| `\dt` | Listar tablas del schema actual |
| `\d nombre_tabla` | Describir una tabla (columnas, tipos, constraints) |
| `\dn` | Listar schemas |
| `\q` | Salir de psql |
| `\?` | Ayuda de comandos psql |
| `\h CREATE TABLE` | Ayuda del comando SQL |

---

## 4. Crear y eliminar una base de datos

```sql
-- Crear base de datos
CREATE DATABASE tienda;

-- Eliminar base de datos
DROP DATABASE tienda;
```

> ⚠️ No puedes estar conectado a una base de datos si vas a eliminarla.

---

## 5. Estructura lógica de PostgreSQL

```
Cluster PostgreSQL
└── Base de datos (Database)
    └── Schema (público por defecto)
        └── Tablas, vistas, funciones, índices...
```

- Una instancia de PostgreSQL puede tener **múltiples bases de datos**.
- Cada base de datos tiene un **schema** `public` por defecto.
- Los objetos (tablas, etc.) viven dentro de un schema.

---

## 6. Tu primera tabla

```sql
CREATE TABLE productos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    precio NUMERIC(10, 2)
);
```

**Conceptos clave:**
- `SERIAL`: entero autoincremental.
- `PRIMARY KEY`: identificador único de cada fila.
- `NOT NULL`: el campo no puede quedar vacío.
- `NUMERIC(10, 2)`: número con hasta 10 dígitos, 2 decimales.

---

## 📋 Resumen

| Paso | Comando / Acción |
|------|------------------|
| Conectar | `psql -U postgres` |
| Listar DBs | `\l` |
| Cambiar DB | `\c nombre_db` |
| Listar tablas | `\dt` |
| Describir tabla | `\d tabla` |
| Salir | `\q` |
