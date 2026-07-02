# 📂 Local File Handling

## 🎯 Learning Objectives

- Accept file uploads in FastAPI with `UploadFile` and `File`
- Validate uploads: size limits, MIME types, content sniffing
- Stream large files without loading them into memory
- Save files to local storage with proper directory structure
- Build a clean abstraction for "the storage backend" so S3 / GCS can replace local later
- Avoid the four most common local file-handling mistakes

## Introduction

The local filesystem is the simplest "object store" for development and small deployments. For production, the right answer is S3 / GCS (covered in [[02 - S3 GCS with Presigned URLs|the next note]]), but the patterns for accepting, validating, and streaming uploads apply regardless of where the file ends up. This note covers the local case and the abstractions that make the S3 migration a one-line change.

The naïve approach — `with open(path, "wb") as f: f.write(await file.read())` — works for small files but breaks for large ones (the entire file is in memory). The right approach: stream the file in chunks; never load the whole file; validate as you go.

---

## 1. The Basic Upload

### 1.1 The `UploadFile` type

```python
from fastapi import FastAPI, UploadFile, File
from typing import Annotated


@app.post("/upload")
async def upload_file(file: Annotated[UploadFile, File(...)]):
    """Accept a single file upload."""
    contents = await file.read()  # bytes
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
    }
```

`UploadFile` is FastAPI's wrapper around Python's `SpooledTemporaryFile`. It supports:
- `file.read()`: read the entire file (memory-bound).
- `file.read(chunk_size)`: read in chunks.
- `file.seek(pos)`: seek to a position.
- `file.close()`: close the file (usually automatic).

### 1.2 Multiple files

```python
@app.post("/upload-multiple")
async def upload_multiple(
    files: Annotated[list[UploadFile], File(...)],
):
    return [{"filename": f.filename, "size": len(await f.read())} for f in files]
```

A list of `UploadFile`. FastAPI handles the multipart parsing.

### 1.3 File + form data

```python
from fastapi import Form


@app.post("/upload-with-metadata")
async def upload_with_metadata(
    file: Annotated[UploadFile, File(...)],
    description: Annotated[str, Form(...)],
    is_public: Annotated[bool, Form()] = False,
):
    return {
        "filename": file.filename,
        "description": description,
        "is_public": is_public,
        "size": len(await file.read()),
    }
```

The form fields are parsed alongside the file. Useful for "upload a file and tell me what it is".

### 1.4 The multipart format

A multipart request looks like:

```
POST /upload HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Length: 12345

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="hello.txt"
Content-Type: text/plain

Hello, world!
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

FastAPI parses the multipart and exposes the file as `UploadFile`. The form fields are also parsed.

---

## 2. Validation

### 2.1 The three validations

Every upload must be validated for:
1. **Size**: the file is not too large.
2. **Type**: the file is of an expected MIME type.
3. **Content**: the file's bytes match its claimed type (content sniffing).

### 2.2 Size limits

```python
from fastapi import HTTPException
import os


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post("/upload")
async def upload_file(file: Annotated[UploadFile, File(...)]):
    # Read in chunks to enforce the limit
    total = 0
    chunks = []
    while chunk := await file.read(1024 * 64):  # 64 KB chunks
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(413, f"File too large; max {MAX_UPLOAD_SIZE} bytes")
        chunks.append(chunk)
    contents = b"".join(chunks)
    return {"size": total}
```

The chunk-by-chunk read lets you abort early if the file exceeds the limit. Reading the whole file first is too late — you've already used 50 MB of memory.

### 2.3 The "in-memory" warning

By default, FastAPI stores small files in memory (up to a threshold; the default is 1 MB) and large files in a spooled temp file. For files between 1 MB and your chunk size, the behavior is unpredictable. Always read in chunks; never `await file.read()` for files you don't trust.

### 2.4 Content-Type validation

```python
import magic  # python-magic; pip install python-magic (and libmagic)


ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "application/pdf",
    "text/plain", "text/csv",
}


def validate_content_type(file: UploadFile, content: bytes) -> None:
    """Validate the file's MIME type using the actual content."""
    # The browser's content_type is untrusted; the actual type comes from the bytes
    sniffed = magic.from_buffer(content, mime=True)
    if sniffed not in ALLOWED_TYPES:
        raise HTTPException(
            415,
            f"Disallowed file type: {sniffed}. Allowed: {ALLOWED_TYPES}",
        )
    # Optional: warn if the claimed type differs from the actual type
    if file.content_type and file.content_type != sniffed:
        # Could be a renamed file; allow it but log
        ...
```

The `Content-Type` header is set by the client; it can be anything. The actual type is determined by the bytes (via `libmagic`). Always validate the actual type.

### 3.5 File extension validation

```python
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf", ".txt", ".csv"}


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Disallowed extension: {ext}")
    return ext
```

The extension is from the filename, which is untrusted. Combine with content-type validation: a `.jpg` with a `text/plain` body is suspicious.

---

## 3. The Storage Abstraction

### 3.1 The interface

The right pattern is an abstraction: the handler calls `storage.save(file)`, and the storage implementation decides where the bytes go. In development, `storage` is local disk; in production, it's S3.

```python
# app/storage/base.py
from abc import ABC, abstractmethod
from typing import BinaryIO
from dataclasses import dataclass


@dataclass
class SavedFile:
    """Metadata about a saved file."""
    key: str  # The unique key (e.g., "uploads/2024/01/15/abc123.jpg")
    size: int
    content_type: str
    url: str | None = None  # The public URL (if available)


class Storage(ABC):
    """The storage abstraction."""
    
    @abstractmethod
    async def save(
        self,
        *,
        key: str,
        content: BinaryIO,
        content_type: str,
        metadata: dict | None = None,
    ) -> SavedFile:
        """Save a file. The `key` is the desired path; the implementation chooses the storage."""
        ...
    
    @abstractmethod
    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        """Get a URL for the file. For local: a static URL. For S3: a presigned URL."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a file."""
        ...
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a file exists."""
        ...
```

The handler is unaware of where the file lives.

### 3.2 The local implementation

```python
# app/storage/local.py
import os
import shutil
from pathlib import Path
from fastapi import UploadFile


class LocalStorage(Storage):
    def __init__(self, base_dir: str = "/var/lib/myapp/uploads", base_url: str = "/uploads"):
        self.base_dir = Path(base_dir)
        self.base_url = base_url
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def save(self, *, key: str, content, content_type, metadata=None) -> SavedFile:
        full_path = self.base_dir / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Stream the content to disk
        with open(full_path, "wb") as f:
            shutil.copyfileobj(content, f)
        size = full_path.stat().st_size
        return SavedFile(
            key=key,
            size=size,
            content_type=content_type,
            url=f"{self.base_url}/{key}",
        )
    
    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        # Local files don't expire; just return the static URL
        return f"{self.base_url}/{key}"
    
    async def delete(self, key: str) -> None:
        full_path = self.base_dir / key
        if full_path.exists():
            full_path.unlink()
    
    async def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()
```

### 3.3 The S3 implementation (preview)

The S3 implementation is covered in [[02 - S3 GCS with Presigned URLs|the next note]]; the interface is the same. The handler code doesn't change when you switch from local to S3.

### 3.4 The handler

```python
from app.storage.base import Storage
from app.storage import get_storage


@router.post("/upload")
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    storage: Storage = Depends(get_storage),
    uow: UnitOfWork = Depends(get_uow),
):
    # Read in chunks
    total = 0
    chunks = []
    while chunk := await file.read(1024 * 64):
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(413, "File too large")
        chunks.append(chunk)
    contents = b"".join(chunks)
    # Validate
    ext = validate_extension(file.filename)
    validate_content_type(file, contents)
    # Save
    key = f"uploads/{datetime.now():%Y/%m/%d}/{uuid4()}{ext}"
    saved = await storage.save(
        key=key, content=io.BytesIO(contents), content_type=file.content_type,
    )
    # Record the metadata
    upload = Upload(
        user_id=current_user.id, key=saved.key, size=saved.size,
        content_type=saved.content_type, original_filename=file.filename,
    )
    await uow.uploads.create(upload)
    await uow.commit()
    return {"key": saved.key, "url": saved.url, "size": saved.size}
```

The handler reads in chunks, validates, and saves via the storage abstraction. The same code works with local storage in dev and S3 in production.

---

## 4. Streaming Large Files

### 4.1 The problem

A 1 GB file upload with `await file.read()` loads the entire file into memory. If 10 users upload 1 GB files simultaneously, the server uses 10 GB of RAM. With a 2 GB file, the server crashes.

### 4.2 The solution: streaming to disk

```python
@app.post("/upload-streaming")
async def upload_streaming(
    file: Annotated[UploadFile, File(...)],
    storage: Storage = Depends(get_storage),
):
    key = f"uploads/{datetime.now():%Y/%m/%d}/{uuid4()}"
    # Stream the file to the storage backend
    saved = await storage.save_streaming(
        key=key, content=file.file,  # SpooledTemporaryFile; iterable in chunks
        content_type=file.content_type,
    )
    return {"key": saved.key, "size": saved.size}
```

The storage backend's `save_streaming` reads in chunks and writes directly to the destination. The bytes are never in memory.

### 4.3 The streaming save

```python
class LocalStorage(Storage):
    async def save_streaming(
        self, *, key: str, content, content_type: str, chunk_size: int = 1024 * 1024,
    ) -> SavedFile:
        full_path = self.base_dir / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # `content` is a file-like object; iterate in chunks
        with open(full_path, "wb") as f:
            while chunk := content.read(chunk_size):
                f.write(chunk)
        size = full_path.stat().st_size
        return SavedFile(key=key, size=size, content_type=content_type, url=f"{self.base_url}/{key}")
```

The streaming save reads `chunk_size` bytes at a time and writes to the destination. Memory usage is `chunk_size`, not the file size.

### 4.4 The S3 streaming save (preview)

```python
class S3Storage(Storage):
    async def save_streaming(
        self, *, key: str, content, content_type: str, chunk_size: int = 1024 * 1024,
    ) -> SavedFile:
        # boto3 supports multipart upload; for single-part, this works
        self.s3.upload_fileobj(content, self.bucket, key, ExtraArgs={"ContentType": content_type})
        return SavedFile(key=key, ...)
```

`boto3.upload_fileobj` streams the file to S3. The client never loads the whole file.

---

## 5. Downloading Files

### 5.1 The streaming download

```python
from fastapi.responses import StreamingResponse
import os


@app.get("/files/{key:path}")
async def download_file(key: str, storage: Storage = Depends(get_storage)):
    """Stream a file download."""
    if not await storage.exists(key):
        raise HTTPException(404, "File not found")
    
    async def file_stream():
        # Open the file and stream chunks
        with open(f"/var/lib/myapp/uploads/{key}", "rb") as f:
            while chunk := f.read(1024 * 64):  # 64 KB
                yield chunk
    
    return StreamingResponse(
        file_stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{key}"'},
    )
```

The `StreamingResponse` yields chunks of the file. The client receives the file as it's being read. Memory usage is constant (one chunk at a time).

### 5.2 The S3 download (preview)

```python
class S3Storage(Storage):
    def get_object_stream(self, key: str):
        # boto3 returns a streaming body
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"]


@app.get("/files/{key:path}")
async def download_file(key: str, storage: Storage = Depends(get_storage)):
    if not await storage.exists(key):
        raise HTTPException(404, "File not found")
    
    body = storage.get_object_stream(key)
    return StreamingResponse(
        body.iter_chunks(chunk_size=1024 * 64),
        media_type="application/octet-stream",
    )
```

S3 streams the file to FastAPI, which streams it to the client. Memory usage is constant.

### 5.3 The Content-Disposition header

The `Content-Disposition` header tells the browser how to handle the file:
- `attachment; filename="..."`: download as a file.
- `inline; filename="..."`: display in the browser (e.g., a PDF).
- `attachment; filename*=UTF-8''...`: the modern, RFC 5987-encoded version for non-ASCII filenames.

```python
import urllib.parse


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """Generate a Content-Disposition header value."""
    # ASCII fallback
    ascii_filename = filename.encode("ascii", "ignore").decode("ascii")
    # RFC 5987 encoding for the full filename
    encoded = urllib.parse.quote(filename)
    return f"{disposition}; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded}"
```

The `filename*` parameter handles non-ASCII filenames correctly.

---

## 6. The Four Common Local File Mistakes

### 6.1 Loading the entire file into memory

```python
# ❌ Loads 1 GB into RAM
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)
```

The fix: stream in chunks. Always.

### 6.2 Storing files in the FastAPI process's working directory

```python
# ❌ The files are lost when the container restarts
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with open(f"./uploads/{file.filename}", "wb") as f:
        # ...
```

A stateless container's disk is ephemeral. Use a persistent volume (in Kubernetes) or object storage (S3 / GCS).

### 6.3 Trusting the filename

```python
# ❌ The user provides a path: ../../etc/passwd
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with open(f"./uploads/{file.filename}", "wb") as f:
        f.write(await file.read())
```

The fix: generate a unique filename; never use the user-provided filename as the storage path.

```python
import uuid
from pathlib import Path


async def save_file(file: UploadFile) -> SavedFile:
    ext = Path(file.filename).suffix
    safe_name = f"{uuid.uuid4()}{ext}"  # e.g., "abc123.jpg"
    # ...
```

### 6.4 No size limit

```python
# ❌ A 100 GB upload fills the disk
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 64):
            f.write(chunk)
```

The fix: enforce a size limit during the upload, not after.

---

## 7. The Production Configuration

### 7.1 The server limits

```python
# uvicorn or gunicorn
# Limit the maximum request body size
uvicorn app.main:app --limit-max-requests 100000

# Nginx as reverse proxy
client_max_body_size 50M;
```

Even with a FastAPI-level size check, the server has to read the request to know its size. The reverse proxy enforces a hard limit.

### 7.2 The Kubernetes limits

```yaml
# k8s deployment
spec:
  containers:
  - name: api
    resources:
      limits:
        memory: "512Mi"  # The API server's memory limit
```

A file upload reads into memory (even in chunks, the request is buffered). A memory limit prevents a single upload from OOMing the pod.

### 7.3 The async client

```python
import httpx


async def upload_to_remote(url: str, file_path: str) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:  # 5 min timeout
        with open(file_path, "rb") as f:
            response = await client.post(
                url, files={"file": f}, timeout=300,
            )
    return response.json()
```

The async client streams the upload; the timeout is generous (large files take time).

---

## 8. The Test Patterns

### 8.1 Test the upload

```python
@pytest.mark.asyncio
async def test_upload_file(client, storage):
    content = b"Hello, world!"
    files = {"file": ("hello.txt", io.BytesIO(content), "text/plain")}
    response = await client.post("/upload", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["size"] == len(content)
    # The file is in the storage
    saved = await storage.get(body["key"])
    assert saved == content
```

### 8.2 Test the size limit

```python
@pytest.mark.asyncio
async def test_size_limit(client):
    content = b"x" * (MAX_UPLOAD_SIZE + 1)
    files = {"file": ("big.bin", io.BytesIO(content), "application/octet-stream")}
    response = await client.post("/upload", files=files)
    assert response.status_code == 413
```

### 8.3 Test the content-type validation

```python
@pytest.mark.asyncio
async def test_disallowed_content_type(client):
    # A file claimed as image/jpeg but containing plain text
    content = b"Not an image"
    files = {"file": ("fake.jpg", io.BytesIO(content), "image/jpeg")}
    response = await client.post("/upload", files=files)
    assert response.status_code == 415
```

### 8.4 Test the download

```python
@pytest.mark.asyncio
async def test_download_file(client, storage):
    content = b"Hello, world!"
    key = await storage.save("test.txt", io.BytesIO(content), "text/plain")
    response = await client.get(f"/files/{key}")
    assert response.status_code == 200
    assert response.content == content
```

---

## 9. The Storage Interface in Practice

A complete storage abstraction that supports both local and S3:

```python
# app/storage/__init__.py
from app.core.config import settings


def get_storage():
    """Factory: returns the configured storage backend."""
    if settings.STORAGE_BACKEND == "local":
        from app.storage.local import LocalStorage
        return LocalStorage(
            base_dir=settings.LOCAL_STORAGE_DIR,
            base_url=settings.LOCAL_STORAGE_URL,
        )
    elif settings.STORAGE_BACKEND == "s3":
        from app.storage.s3 import S3Storage
        return S3Storage(
            bucket=settings.S3_BUCKET,
            region=settings.AWS_REGION,
        )
    else:
        raise ValueError(f"Unknown storage backend: {settings.STORAGE_BACKEND}")


# app/api/uploads.py
from app.storage import get_storage


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    storage = Depends(get_storage),
):
    # ... use the abstract interface
    saved = await storage.save_streaming(
        key=f"uploads/{uuid4()}", content=file.file, content_type=file.content_type,
    )
    return saved
```

The handler is the same regardless of backend. The configuration switches between local and S3.

---

## 10. Código de Compresión

```python
"""
Compresión: Local File Handling
Covers: UploadFile, validation, streaming, storage abstraction, downloads.
"""
import io
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.responses import StreamingResponse


# 1) The storage interface
@dataclass
class SavedFile:
    key: str
    size: int
    content_type: str
    url: Optional[str] = None


class Storage(ABC):
    @abstractmethod
    async def save(self, *, key, content, content_type, metadata=None) -> SavedFile: ...
    @abstractmethod
    async def save_streaming(self, *, key, content, content_type, chunk_size=1024*1024) -> SavedFile: ...
    @abstractmethod
    async def get_url(self, key, *, expires_in=3600) -> str: ...
    @abstractmethod
    async def delete(self, key) -> None: ...
    @abstractmethod
    async def exists(self, key) -> bool: ...


# 2) The local implementation
class LocalStorage(Storage):
    def __init__(self, base_dir="/var/lib/myapp/uploads", base_url="/uploads"):
        self.base_dir = Path(base_dir)
        self.base_url = base_url
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_streaming(self, *, key, content, content_type, chunk_size=1024*1024) -> SavedFile:
        full_path = self.base_dir / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(full_path, "wb") as f:
            while chunk := content.read(chunk_size):
                f.write(chunk)
                total += len(chunk)
        return SavedFile(key=key, size=total, content_type=content_type, url=f"{self.base_url}/{key}")

    async def save(self, *, key, content, content_type, metadata=None) -> SavedFile:
        return await self.save_streaming(key=key, content=content, content_type=content_type)

    async def get_url(self, key, *, expires_in=3600) -> str:
        return f"{self.base_url}/{key}"

    async def delete(self, key) -> None:
        (self.base_dir / key).unlink(missing_ok=True)

    async def exists(self, key) -> bool:
        return (self.base_dir / key).exists()


# 3) The handler
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf", "text/plain"}


def validate_size_and_type(file: UploadFile, contents: bytes) -> None:
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large; max {MAX_UPLOAD_SIZE} bytes")
    # Optional: content sniffing with python-magic
    # sniffed = magic.from_buffer(contents, mime=True)
    # if sniffed not in ALLOWED_TYPES:
    #     raise HTTPException(415, f"Disallowed type: {sniffed}")


app = FastAPI()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    storage: Storage = Depends(lambda: LocalStorage()),
):
    # Read in chunks to enforce size limit
    total = 0
    chunks = []
    while chunk := await file.read(1024 * 64):
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(413, "File too large")
        chunks.append(chunk)
    contents = b"".join(chunks)
    validate_size_and_type(file, contents)
    ext = Path(file.filename).suffix
    key = f"uploads/{datetime.now():%Y/%m/%d}/{uuid.uuid4()}{ext}"
    saved = await storage.save(key=key, content=io.BytesIO(contents), content_type=file.content_type)
    return saved


@app.get("/files/{key:path}")
async def download_file(key: str, storage: Storage = Depends(lambda: LocalStorage())):
    if not await storage.exists(key):
        raise HTTPException(404, "File not found")
    full_path = Path("/var/lib/myapp/uploads") / key
    def file_stream():
        with open(full_path, "rb") as f:
            while chunk := f.read(1024 * 64):
                yield chunk
    return StreamingResponse(file_stream(), media_type="application/octet-stream")
```

---

## Key Takeaways

- **The local filesystem is for development and small deployments.** For production at scale, use object storage (S3 / GCS) with the same `Storage` interface.
- **Stream uploads in chunks.** Never `await file.read()` for files you don't trust; a 1 GB upload loads 1 GB into memory.
- **Validate size, type, and content.** The size limit prevents disk fills; the type validation prevents malicious files; the content sniffing prevents renamed files.
- **Generate unique filenames.** Never use the user-provided filename as the storage path. Use `{uuid}{ext}` for the storage key; preserve the original filename as metadata.
- **Stream downloads too.** Use `StreamingResponse` with `yield chunk` to avoid loading the whole file into memory.
- **The `Storage` abstraction** is the right pattern. The handler is unaware of where the file lives; switching from local to S3 is a configuration change.
- **The reverse proxy enforces size limits**. A 50 MB nginx `client_max_body_size` prevents a 1 GB upload from ever reaching FastAPI.
- **Trust the bytes, not the headers.** The `Content-Type` header is set by the client; use `python-magic` to sniff the actual type.

## References

- [FastAPI — Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [FastAPI — StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Starlette — UploadFile](https://www.starlette.io/requests/#request-files)
- [python-magic — Content type detection](https://pypi.org/project/python-magic/)
- [Pillow — Image format support](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
- [RFC 7578 — Multipart form data](https://www.rfc-editor.org/rfc/rfc7578)
- [RFC 5987 — filename* encoding](https://www.rfc-editor.org/rfc/rfc5987)
- [Nginx — client_max_body_size](https://nginx.org/en/docs/http/ngx_http_core_module.html#client_max_body_size)
