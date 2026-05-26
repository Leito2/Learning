# 07 - Supabase en Profundidad

Esta nota es una guía visual y práctica para dominar **Supabase** como plataforma de desarrollo con PostgreSQL. Aprenderás a navegar por su interfaz, gestionar datos sin escribir SQL y aprovechar sus herramientas avanzadas.

---

## 1. El Dashboard de Supabase

Cuando entras a tu proyecto en [app.supabase.com](https://app.supabase.com), lo primero que ves es el **Dashboard Principal**.

### Secciones principales

![Vista de Tablas en Supabase](assets/supabase-table-view-hd.png)

**Home / Overview**
- Muestra métricas de uso: consultas por segundo, usuarios activos, almacenamiento.
- Estado de los servicios: Database, Auth, Storage, Realtime.

**Table Editor**
- Es el corazón visual de Supabase.
- Muestra tus tablas como hojas de cálculo (spreadsheet).
- Puedes filtrar, ordenar, editar celdas y crear nuevas filas sin escribir código.

**SQL Editor**
- Editor completo con sintaxis resaltada.
- Guarda consultas favoritas.
- Historial de ejecuciones.
- Resultados exportables a CSV.

---

## 2. Table Editor en detalle

El **Table Editor** convierte PostgreSQL en algo tan fácil como Excel.

### Crear una tabla visualmente

1. Ve a **Table Editor** → **New table**.
2. Se abre un panel lateral donde defines:
   - **Name**: nombre de la tabla.
   - **Columns**: agrega columnas definiendo nombre, tipo y restricciones.
   - **Enable Row Level Security (RLS)**: activa seguridad por filas (recomendado para producción).
3. Haz clic en **Save** y tu tabla se crea automáticamente con el SQL correspondiente.

### Funciones del Table Editor

| Función | Cómo usarla |
|---------|-------------|
| **Filter** | Icono de embudo → Filtra por columna y valor |
| **Sort** | Icono de flechas → Ordena ascendente/descendente |
| **Insert row** | Botón verde **+** → Agrega una fila manualmente |
| **Edit cell** | Doble clic en cualquier celda |
| **Duplicate table** | Menú de tres puntos → Duplicate |
| **Relationships** | Mira los iconos de enlaces para ver claves foráneas |

> 💡 Aunque el Table Editor no requiere SQL, Supabase muestra el comando SQL generado en la parte inferior. Es una excelente forma de aprender.

---

## 3. SQL Editor avanzado

El **SQL Editor** es donde pondrás en práctica todo el curso.

![SQL Editor de Supabase](assets/supabase-sql-editor.png)

### Características

- **Sintaxis resaltada**: colores para palabras clave, funciones, strings.
- **Autocompletado**: sugerencias mientras escribes.
- **Tabs múltiples**: trabaja en varias consultas a la vez.
- **Favoritos**: guarda consultas que usas frecuentemente.
- **Historial**: recupera consultas ejecutadas anteriormente.

### Ejecutar y exportar

1. Escribe tu consulta.
2. Presiona `Ctrl + Enter` o haz clic en **Run**.
3. Los resultados aparecen en una tabla debajo.
4. Usa el botón **Export** para descargar como CSV.

---

## 4. Authentication (Autenticación)

Supabase incluye un sistema de autenticación completo.

### Características principales

- **Registro e inicio de sesión**: Email, Magic Link, OAuth (Google, GitHub, etc.).
- **Políticas de seguridad (RLS)**: Controla quién puede leer/escribir qué datos.
- **Users**: Lista de usuarios registrados con metadata.

### Row Level Security (RLS)

Las políticas RLS permiten que una misma tabla sirva datos diferentes según el usuario.

```sql
-- Solo los usuarios autenticados pueden ver sus propios pedidos
CREATE POLICY "Users can view own orders"
ON pedidos FOR SELECT
USING (auth.uid() = usuario_id);
```

---

## 5. Database (Gestión avanzada)

En la sección **Database** del menú lateral encontrarás:

### Triggers

Acciones automáticas que ocurren ante eventos (INSERT, UPDATE, DELETE).

```sql
-- Ejemplo: actualizar fecha de modificación automáticamente
CREATE TRIGGER update_modtime
BEFORE UPDATE ON productos
FOR EACH ROW
EXECUTE FUNCTION moddatetime (updated_at);
```

### Functions

Gestión de funciones PL/pgSQL almacenadas.

### Extensions

PostgreSQL tiene cientos de extensiones. En Supabase puedes activarlas con un clic:

| Extensión | Para qué sirve |
|-----------|----------------|
| `pgcrypto` | Encriptación y funciones hash |
| `uuid-ossp` | Generar UUIDs |
| `postgis` | Datos geoespaciales (mapas) |
| `pg_trgm` | Búsqueda de texto aproximada |
| `vector` | Embeddings para IA (pgvector) |

---

## 6. Storage (Almacenamiento)

Supabase te permite almacenar archivos (imágenes, PDFs, videos) con reglas de acceso.

```sql
-- Ejemplo de bucket público para avatares
INSERT INTO storage.buckets (id, name, public)
VALUES ('avatars', 'avatars', true);
```

---

## 7. API REST automática

Una de las características más potentes: cada tabla que creas genera automáticamente una **API REST**.

### Ejemplo de uso con JavaScript

```javascript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient('https://xxx.supabase.co', 'tu-api-key')

// SELECT * FROM productos
const { data } = await supabase
  .from('productos')
  .select('*')

// INSERT INTO productos (nombre, precio) VALUES ('Mouse', 25.99)
await supabase
  .from('productos')
  .insert({ nombre: 'Mouse', precio: 25.99 })
```

> 💡 Para ver tu URL y claves API: **Settings** → **API**.

---

## 8. Realtime (Datos en tiempo real)

Escucha cambios en la base de datos sin necesidad de refrescar.

```javascript
supabase
  .channel('productos-cambios')
  .on('postgres_changes', { event: '*', schema: 'public', table: 'productos' }, payload => {
    console.log('Cambio detectado:', payload)
  })
  .subscribe()
```

---

## 9. Conexión externa

Supabase te permite conectarte desde herramientas externas.

### Desde DBeaver / TablePlus

1. Ve a **Settings** → **Database**.
2. Copia el **Connection string** (URI format).
3. Pégalo en tu cliente SQL favorito.

```
postgresql://postgres:[PASSWORD]@db.xxxxxx.supabase.co:5432/postgres
```

### Pool de conexiones (PGBouncer)

Para aplicaciones con muchos usuarios simultáneos, usa la **Transaction pooler** URL en lugar de la conexión directa.

---

## 📋 Tabla comparativa: Supabase vs PostgreSQL local

| Característica | PostgreSQL local | Supabase |
|----------------|------------------|----------|
| Instalación | Descargar e instalar | Cuenta web, 2 minutos |
| Interfaz visual | pgAdmin (complejo) | Dashboard intuitivo |
| APIs automáticas | No | Sí (REST + GraphQL) |
| Autenticación | Manual | Integrada completa |
| Almacenamiento de archivos | No | Sí (buckets) |
| Realtime | Configuración extra | Nativo |
| Serverless functions | No | Edge Functions |
| Backup automático | Configurar manual | Incluido |

---

## 🎯 Flujo de trabajo recomendado en Supabase

1. **Diseña** tus tablas en el **Table Editor** (modo visual).
2. **Practica** consultas SQL en el **SQL Editor**.
3. **Guarda** tus queries favoritas para reutilizarlas.
4. **Activa** RLS y define políticas antes de poner tu app en producción.
5. **Usa** la API REST para conectar tu frontend sin escribir backend.

---

> 🔗 **Recursos oficiales:**
> - Sitio web: [supabase.com](https://supabase.com)
> - Documentación: [supabase.com/docs](https://supabase.com/docs)
> - Comunidad (GitHub): [github.com/supabase/supabase](https://github.com/supabase/supabase)
