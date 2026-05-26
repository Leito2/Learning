# 01 - Introducción y Configuración

Antes de escribir código SQL, necesitas entender el entorno y cómo interactuar con PostgreSQL. En este curso usaremos **Supabase**, una plataforma que te da acceso a una base de datos PostgreSQL completamente funcional en la nube, sin necesidad de instalar nada localmente.

---

## 1. ¿Qué es Supabase?

**Supabase** es una alternativa de código abierto a Firebase. Proporciona:

- Una base de datos **PostgreSQL** completa (no una versión reducida).
- Autenticación de usuarios.
- APIs REST y GraphQL automáticas.
- Almacenamiento de archivos.
- Funciones Edge (serverless).
- Panel de control web para gestionar tu base de datos.

**Ventajas para aprender SQL:**
- No necesitas instalar PostgreSQL localmente.
- Acceso desde cualquier dispositivo con internet.
- Interfaz visual para tablas + editor SQL integrado.
- Plan gratuito generoso para proyectos personales y aprendizaje.

---

## 2. Crear tu cuenta y proyecto

### Paso 1: Registro

1. Ve a [supabase.com](https://supabase.com).
2. Haz clic en **"Start your project"** o **"Sign In"**.
3. Puedes registrarte con **GitHub**, Google o email.

![Registro en Supabase](assets/supabase-og.jpg)

### Paso 2: Crear una organización

Después de registrarte, Supabase te pedirá crear una **organización**. Esto es solo un agrupador de proyectos. Puedes ponerle cualquier nombre, por ejemplo: `Mi Aprendizaje`.

### Paso 3: Crear un proyecto

1. Dentro de tu organización, haz clic en **"New project"**.
2. Configura tu proyecto:
   - **Name**: Un nombre para tu base de datos (ej. `sql-curso`).
   - **Database Password**: Crea una contraseña segura y guárdala bien (no se puede recuperar fácilmente).
   - **Region**: Elige la región más cercana a ti (ej. `East US` si estás en América).
3. Haz clic en **"Create new project"**.
4. Espera unos segundos a que se aprovisione tu base de datos.

---

## 3. El Dashboard de Supabase

Una vez creado el proyecto, verás el **Dashboard**. Esta es tu centro de control:

### Barra lateral (izquierda)

| Sección | Para qué sirve |
|---------|----------------|
| **Table Editor** | Ver y editar tablas como si fueran hojas de cálculo |
| **SQL Editor** | Escribir y ejecutar consultas SQL |
| **Database** | Gestión avanzada: triggers, funciones, extensiones |
| **Authentication** | Gestión de usuarios y permisos |
| **Storage** | Almacenamiento de archivos |
| **Edge Functions** | Funciones serverless |
| **Settings** | Configuración del proyecto |

### Panel principal

En el centro verás información de tu proyecto, estadísticas de uso y conexiones recientes.

---

## 4. Conectar con el SQL Editor

Esta es la herramienta que más usarás para practicar.

### Abrir el SQL Editor

1. En la barra lateral, haz clic en **"SQL Editor"**.
2. Se abre un panel dividido:
   - **Izquierda**: Historial de consultas y favoritos.
   - **Centro**: Editor de código SQL.
   - **Abajo/Derecha**: Resultados de la ejecución.

### Tu primera consulta

En el editor, escribe:

```sql
SELECT version();
```

Luego haz clic en el botón **"Run"** (o presiona `Ctrl + Enter`).

Verás algo como:
```
PostgreSQL 15.1 on x86_64-pc-linux-gnu ...
```

¡Listo! Estás conectado a tu base de datos PostgreSQL real.

---

## 5. Conectar desde la terminal local (opcional)

Si quieres usar `psql` desde tu computadora, Supabase te da los datos de conexión.

### Obtener credenciales

1. Ve a **Settings** → **Database**.
2. Busca la sección **"Connection string"**.
3. Elige el formato **"psql"** o **"URI"**.
4. Copia la cadena de conexión (reemplaza `[YOUR-PASSWORD]` por tu contraseña real).

```bash
psql "postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxx.supabase.co:5432/postgres"
```

> ⚠️ Si no tienes `psql` instalado localmente, instala PostgreSQL desde [postgresql.org/download](https://www.postgresql.org/download/) solo para obtener las herramientas de cliente.

---

## 6. Comandos básicos de psql

Si te conectas desde terminal, estos comandos te serán útiles:

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

## 7. Estructura lógica de PostgreSQL

```
Proyecto Supabase
└── Base de datos (Database)
    └── Schema (público por defecto)
        └── Tablas, vistas, funciones, índices...
```

- Tu proyecto de Supabase contiene **una base de datos PostgreSQL**.
- Cada base de datos tiene un **schema** `public` por defecto.
- Los objetos (tablas, etc.) viven dentro de un schema.

---

## 8. Tu primera tabla (en el SQL Editor)

En el SQL Editor de Supabase, ejecuta:

```sql
CREATE TABLE productos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    precio NUMERIC(10, 2)
);
```

Después, ve a **Table Editor** y verás tu tabla `productos` lista para agregar datos manualmente o por SQL.

**Conceptos clave:**
- `SERIAL`: entero autoincremental.
- `PRIMARY KEY`: identificador único de cada fila.
- `NOT NULL`: el campo no puede quedar vacío.
- `NUMERIC(10, 2)`: número con hasta 10 dígitos, 2 decimales.

---

## 📋 Resumen

| Paso | Acción |
|------|--------|
| Registro | [supabase.com](https://supabase.com) → Sign Up |
| Crear proyecto | New project → Configurar nombre, password, región |
| SQL Editor | Barra lateral → SQL Editor → Escribir y ejecutar |
| Table Editor | Barra lateral → Table Editor → Ver tablas como spreadsheet |
| Conexión externa | Settings → Database → Connection string |
| Listar DBs (psql) | `\l` |
| Listar tablas (psql) | `\dt` |
| Describir tabla (psql) | `\d tabla` |

---

> 💡 **Consejo:** Guarda tus consultas frecuentes como **favoritos** en el SQL Editor. Haz clic en la ⭐ junto al nombre de la consulta.
