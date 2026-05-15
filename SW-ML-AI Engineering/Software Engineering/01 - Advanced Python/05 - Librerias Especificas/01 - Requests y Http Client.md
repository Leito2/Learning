# 🌐 Requests y HTTP Client

Las APIs REST son el tejido conectivo de la arquitectura moderna de microservicios y la principal vía de ingesta de datos en pipelines de ML. Desde la descarga de datasets hasta la invocación de modelos servidos vía HTTP, dominar el cliente HTTP es una competencia transversal entre AI Engineering y Backend.

![Globo HTTP](https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/1200px-Wikipedia-logo-v2.svg.png)

## 1. Fundamentos del protocolo HTTP y REST

HTTP (HyperText Transfer Protocol) es un protocolo sin estado que opera sobre TCP. En el contexto REST (Representational State Transfer), cada recurso se identifica mediante una URI y se manipula con verbos semánticos:

- **GET:** Lectura idempotente.
- **POST:** Creación de recursos.
- **PUT:** Reemplazo completo.
- **PATCH:** Modificación parcial.
- **DELETE:** Eliminación.

Los códigos de estado más relevantes son: `200 OK`, `201 Created`, `204 No Content`, `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `500 Internal Server Error`.

## 2. El módulo requests

`requests` es la librería de facto para HTTP en Python. Abstrae la complejidad de `urllib` y ofrece una API intuitiva.

```python
import requests

# GET con parámetros de consulta
respuesta = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'machine+learning', 'sort': 'stars'}
)
print(respuesta.status_code)
print(respuesta.json()['total_count'])
```

```python
# POST con payload JSON
payload = {'model': 'resnet50', 'version': '1.0'}
respuesta = requests.post(
    'https://httpbin.org/post',
    json=payload
)
print(respuesta.json())
```

## 3. Headers personalizados y payloads

Los headers controlan el comportamiento del cliente y del servidor. Es común enviar `Content-Type`, `Accept` y headers de autenticación.

```python
headers = {
    'User-Agent': 'ml-pipeline/1.0',
    'Accept': 'application/vnd.github.v3+json'
}
respuesta = requests.get('https://api.github.com/user', headers=headers)
```

Para payloads complejos, `requests` soporta `data` (form-encoded), `json` (serialización automática) y `files` (multipart).

## 4. Autenticación

### Basic Auth

```python
from requests.auth import HTTPBasicAuth
requests.get('https://httpbin.org/basic-auth/user/pass', auth=HTTPBasicAuth('user', 'pass'))
```

### Bearer Token

```python
headers = {'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'}
requests.get('https://api.example.com/protected', headers=headers)
```

### OAuth2 (mención)

En flujos OAuth2, se intercambia un `code` por un `access_token`. Librerías como `requests-oauthlib` simplifican este flujo, pero el concepto clave es la obtención y renovación del token antes de cada request.

## 5. Sessions: cookies y keep-alive

`requests.Session()` permite persistir cookies, reutilizar conexiones TCP (keep-alive) y configurar headers por defecto.

```python
with requests.Session() as s:
    s.headers.update({'Authorization': 'Bearer TOKEN'})
    r1 = s.get('https://api.example.com/data')
    r2 = s.get('https://api.example.com/more-data')  # Reusa la conexión
```

Caso real: Un backend Django consulta repetidamente un microservicio de predicción de ML. Sin Session, cada request requiere un handshake TCP completo; con Session, el rendimiento mejora notablemente.

## 6. Manejo de errores

```python
from requests.exceptions import HTTPError, ConnectionError, Timeout

try:
    r = requests.get('https://api.example.com/data', timeout=(3, 27))
    r.raise_for_status()
except HTTPError as e:
    print(f"Error HTTP: {e}")
except ConnectionError:
    print("Error de conexión")
except Timeout:
    print("La petición excedió el tiempo máximo")
```

## 7. Rate limiting

Las APIs suelen limitar el número de peticiones por ventana de tiempo. Estrategias comunes:

- **Backoff exponencial:** Incrementar el tiempo de espera tras cada error 429.
- **Token Bucket:** Mantener un contenedor de tokens que se consume por request.
- **Headers de respuesta:** Respetar `X-RateLimit-Remaining` y `Retry-After`.

```python
import time

def get_with_retry(url, max_retries=3):
    for i in range(max_retries):
        r = requests.get(url)
        if r.status_code == 429:
            time.sleep(2 ** i)
        else:
            return r
    raise Exception("Rate limit persistido")
```

## 8. urllib como alternativa stdlib

El módulo `urllib` de la stdlib no requiere instalación externa, pero su API es verbosa.

```python
from urllib import request, parse

data = parse.urlencode({'key': 'value'}).encode()
req = request.Request('https://httpbin.org/post', data=data, method='POST')
with request.urlopen(req) as response:
    print(response.read().decode())
```

| Característica | requests | urllib3 | httpx |
|---|---|---|---|
| Instalación externa | Sí | Sí | Sí |
| API amigable | ✅ Alta | Media | ✅ Alta |
| Soporta HTTP/2 | No | No | Sí |
| Async nativo | No | No | ✅ Sí |
| Keep-alive | Sí | Sí | Sí |
| Rendimiento raw | Bueno | Muy bueno | Muy bueno |

⚠️ **Advertencia:** Nunca almacenes tokens o credenciales directamente en el código fuente. Usa variables de entorno o gestores de secretos como `python-dotenv` o Vault.

💡 **Tip:** Utiliza `requests.Session()` siempre que vayas a realizar más de una petición al mismo host. Reduce la latencia de red y el consumo de recursos.

Caso real: Un pipeline de ingeniería de características descarga diariamente millones de registros desde una API REST paginada. Usar Sessions + manejo de rate limits + headers de autenticación permite completar la ingesta en horas en lugar de días.

```mermaid
sequenceDiagram
    participant Cliente
    participant API
    Cliente->>API: GET /data (Auth Bearer)
    API-->>Cliente: 200 OK + JSON
    Cliente->>API: GET /data?page=2 (misma sesión)
    API-->>Cliente: 200 OK + JSON
    Cliente->>API: POST /predict
    API-->>Cliente: 201 Created
```

📦 **Código de compresión**

```python
import requests
import zipfile
import pathlib

def descargar_y_comprimir(url: str, salida: pathlib.Path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    temp = salida.with_suffix('.tmp')
    with open(temp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(salida, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp, temp.name)
    temp.unlink()
    print(f"📦 Descarga comprimida en {salida}")

if __name__ == '__main__':
    descargar_y_comprimir(
        'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
        pathlib.Path('descarga.zip')
    )
```
