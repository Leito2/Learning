# 🖼️ Image Processing Pipeline

## 🎯 Learning Objectives

- Use Pillow for image processing: resize, convert, strip EXIF
- Build a thumbnail generation pipeline
- Implement virus scanning with ClamAV
- Process images in the background with ARQ
- Avoid the four most common image handling mistakes

## Introduction

Once a file is in S3, the work is often not done. An uploaded avatar needs a 256x256 thumbnail. A user photo needs EXIF stripping (privacy: GPS coordinates, camera model). A document upload needs virus scanning. A product image needs format conversion (WebP for web, AVIF for modern browsers). All of this is **background work** that should not block the upload response.

The pattern is well-established: the upload confirms the file is in S3, the response is sent, and a job (via [[../40 - Background Jobs and Workers for FastAPI/00 - Welcome|ARQ]] or Celery) processes the file. The job generates thumbnails, runs the virus scan, strips EXIF, and writes the results back to S3. The user sees the upload as instant; the processing happens in the background.

This note covers the image processing patterns: Pillow basics, thumbnail generation, EXIF stripping, virus scanning, and the ARQ integration. The patterns apply to any file type that needs post-processing (video transcoding, document OCR, etc.).

---

## 1. Pillow: The Standard Library

### 1.1 Installation

```bash
pip install Pillow
```

Pillow is the standard Python imaging library. It supports reading, writing, and transforming images in 30+ formats.

### 1.2 The basic usage

```python
from PIL import Image
import io


async def process_image(content: bytes) -> dict:
    """Open an image, get metadata, generate a thumbnail."""
    img = Image.open(io.BytesIO(content))
    metadata = {
        "format": img.format,
        "size": img.size,
        "mode": img.mode,
    }
    # Generate a thumbnail
    img.thumbnail((256, 256))
    # Save the thumbnail
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=85)
    thumbnail = buf.getvalue()
    return {
        "metadata": metadata,
        "thumbnail": thumbnail,
    }
```

The `Image.open` reads the file; `img.thumbnail` resizes in place (preserving aspect ratio); `img.save` writes to a buffer.

### 1.3 The supported formats

Pillow supports a wide range of formats:

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| JPEG | ✅ | ✅ | Most common, lossy |
| PNG | ✅ | ✅ | Lossless, supports alpha |
| WebP | ✅ | ✅ | Modern, lossy or lossless |
| AVIF | ✅ | ✅ | Modern, better compression than WebP |
| GIF | ✅ | ✅ | Animation support |
| TIFF | ✅ | ✅ | Print, archival |
| HEIC | ✅ | ❌ | Apple format; reading requires libheif |

For web applications, the recommended formats are JPEG (photos), WebP (modern), and AVIF (cutting edge).

### 1.4 The format conversion

```python
async def convert_to_webp(content: bytes, quality: int = 85) -> bytes:
    """Convert any input format to WebP."""
    img = Image.open(io.BytesIO(content))
    # Convert to RGB (WebP doesn't support CMYK)
    if img.mode == "CMYK":
        img = img.convert("RGB")
    if img.mode == "RGBA":
        # Preserve alpha by saving as PNG; WebP also supports alpha but with different options
        pass
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()
```

`method=6` is the slowest but best compression in libwebp. For high-quality, use `quality=90+`; for small files, `quality=70-80`.

---

## 2. Thumbnail Generation

### 2.1 The pattern

A typical web service needs multiple thumbnail sizes:
- `xs` (50x50): profile pic in chat
- `sm` (150x150): list view avatar
- `md` (300x300): profile page
- `lg` (600x600): detail view

Each is a separate file in S3 with a different key.

```python
THUMBNAIL_SIZES = {
    "xs": (50, 50),
    "sm": (150, 150),
    "md": (300, 300),
    "lg": (600, 600),
}


async def generate_thumbnails(content: bytes) -> dict[str, bytes]:
    """Generate thumbnails at multiple sizes."""
    img = Image.open(io.BytesIO(content))
    # Convert to RGB (avoids issues with RGBA, CMYK)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    
    thumbnails = {}
    for size_name, (width, height) in THUMBNAIL_SIZES.items():
        # Make a copy; thumbnail modifies in place
        thumb = img.copy()
        thumb.thumbnail((width, height))
        buf = io.BytesIO()
        # Save as WebP for modern browsers; fall back to JPEG for compatibility
        ext = "webp"
        mime = "image/webp"
        thumb.save(buf, format="WEBP", quality=85)
        thumbnails[size_name] = {
            "content": buf.getvalue(),
            "content_type": mime,
            "size": (thumb.width, thumb.height),
        }
    return thumbnails
```

The pattern: open the image once, copy for each size, save. Avoids re-decoding the image for each thumbnail.

### 2.2 The aspect ratio

`img.thumbnail((width, height))` resizes the image to fit within the box, preserving the aspect ratio. The image is not cropped; it fits.

If you need a square crop (common for avatars), use `img.crop`:

```python
def make_square(img: Image.Image, size: int) -> Image.Image:
    """Crop the image to a square, then resize."""
    width, height = img.size
    # Center crop
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)
```

The center crop is the most common; for product images, you might use a "smart crop" (face detection) — that's a more advanced topic.

### 2.3 The high-quality resampling

```python
img.thumbnail((256, 256), Image.LANCZOS)
```

`Image.LANCZOS` is the highest-quality resampling filter in Pillow. Other options:
- `Image.BILINEAR`: fast, lower quality
- `Image.BICUBIC`: slower, better quality
- `Image.LANCZOS`: slowest, best quality

For thumbnails, `LANCZOS` is the right choice. The processing is off the request path (it's a background job), so the slight speed cost is acceptable.

---

## 3. EXIF Stripping

### 3.1 Why strip EXIF

EXIF metadata can include:
- GPS coordinates (privacy: "this user is at this address")
- Camera model and serial number
- Timestamps (when the photo was taken)
- Thumbnail of the original (which might be a cropped version)

Most production services strip EXIF on upload. The user can re-add it later if they want; the default is no EXIF.

### 3.2 The implementation

```python
from PIL import Image
import piexif  # pip install piexif


def strip_exif(content: bytes) -> bytes:
    """Remove all EXIF metadata from an image."""
    img = Image.open(io.BytesIO(content))
    # Method 1: Pillow's built-in
    # When you don't pass exif= to save(), Pillow doesn't include it
    buf = io.BytesIO()
    img.save(buf, format=img.format)
    return buf.getvalue()


def strip_exif_strict(content: bytes) -> bytes:
    """Aggressive EXIF removal; uses piexif for completeness."""
    img = Image.open(io.BytesIO(content))
    # Remove all EXIF tags
    if "exif" in img.info:
        img.info["exif"] = None
    if hasattr(img, "_getexif") and img._getexif() is not None:
        # Clear the EXIF
        piexif.remove(img.info["exif"])
    buf = io.BytesIO()
    img.save(buf, format=img.format)
    return buf.getvalue()
```

The strict version uses `piexif` to ensure all EXIF data is removed. The Pillow-only version is simpler but may not catch all metadata.

### 3.3 The selective stripping

Sometimes you want to keep some metadata (e.g., the orientation) but strip the rest. Pillow supports this:

```python
def strip_exif_keep_orientation(content: bytes) -> bytes:
    """Strip EXIF but keep the orientation tag."""
    img = Image.open(io.BytesIO(content))
    if "exif" in img.info:
        try:
            exif_dict = piexif.load(img.info["exif"])
            # Keep only the orientation
            orientation = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation)
            # Apply the rotation
            if orientation:
                img = apply_orientation(img, orientation)
            # Save without EXIF
            buf = io.BytesIO()
            img.save(buf, format=img.format)
            return buf.getvalue()
        except Exception:
            pass
    # Fallback: just save without EXIF
    buf = io.BytesIO()
    img.save(buf, format=img.format)
    return buf.getvalue()
```

The orientation tag is important: without it, the image might be rotated wrong. Strip everything except orientation, then apply the rotation.

---

## 4. Virus Scanning

### 4.1 Why scan uploads

A user can upload a malicious file disguised as an image. The most common attack: a polyglot file that is a valid image AND a valid executable. The browser displays it as an image; the attacker exploits a vulnerability to execute it as code.

The defense: scan every upload for known malware signatures. ClamAV is the standard open-source antivirus.

### 4.2 The ClamAV integration

```bash
# Install ClamAV
apt-get install clamav clamav-daemon

# Update the virus database
freshclam

# Start the daemon
systemctl start clamav-daemon
```

ClamAV runs as a daemon on port 3310. The `clamd` daemon handles requests from clients.

### 4.3 The Python client

```bash
pip install clamd
```

```python
import clamd
import io


async def scan_file(content: bytes) -> dict:
    """Scan a file with ClamAV. Returns {'clean': True/False, 'reason': str}."""
    # Connect to the ClamAV daemon
    cd = clamd.ClamdAsyncNetworkSocket(host="clamav", port=3310)
    # Scan the file
    result = await cd.instream(io.BytesIO(content))
    # Parse the result
    if result["stream"] is None:
        return {"clean": True, "reason": None}
    # The result is a tuple: (status, reason)
    return {"clean": False, "reason": result["stream"][1]}


async def scan_via_file_path(path: str) -> dict:
    """Scan a file by path (faster for large files)."""
    cd = clamd.ClamdAsyncNetworkSocket(host="clamav", port=3310)
    result = await cd.scan(path)
    if result is None:
        return {"clean": True, "reason": None}
    # The result is a dict: {path: (status, reason)}
    for file_path, (status, reason) in result.items():
        if status == "FOUND":
            return {"clean": False, "reason": reason}
    return {"clean": True, "reason": None}
```

The `instream` method scans a file from a stream. The `scan` method scans a file by path (faster for large files).

### 4.4 The pattern: scan before processing

```python
async def process_upload(upload_id: int, storage: S3Storage, uow: UnitOfWork):
    """Background job: scan + process an uploaded file."""
    upload = await uow.uploads.get(upload_id)
    
    # 1) Download the file
    content = await storage.download(upload.key)
    
    # 2) Virus scan
    scan_result = await scan_file(content)
    if not scan_result["clean"]:
        # Mark the upload as malicious; notify the user
        upload.status = "rejected_malware"
        upload.rejection_reason = scan_result["reason"]
        await uow.commit()
        # Send a notification
        await notify_user_of_rejected_upload(upload)
        # Optionally: delete the file
        await storage.delete(upload.key)
        return
    
    # 3) Process (image, document, video, etc.)
    if upload.content_type.startswith("image/"):
        await process_image_upload(upload, content, storage, uow)
    elif upload.content_type.startswith("video/"):
        await process_video_upload(upload, content, storage, uow)
    # ...
```

The pattern: scan first, then process. If the file is malicious, reject it; the user is notified; the file is optionally deleted.

### 4.5 The performance consideration

ClamAV scanning is fast (~10 ms per MB). For a 100 MB file, the scan takes 1 second. The scan is in the background job; the upload response is sent immediately.

The cost of skipping the scan: a malicious file gets to the user. The cost of scanning: 1 second of CPU per file. The trade-off is obvious.

---

## 5. The Background Job Pipeline

### 5.1 The ARQ job

```python
# app/jobs/image.py
from PIL import Image
import io
from app.core.storage import get_storage
from app.db.session import async_session_maker
from app.models.upload import Upload
from sqlalchemy import select


async def process_upload(ctx, upload_id: int):
    """Background job: scan, process, and finalize an upload."""
    storage = get_storage()
    async with async_session_maker()() as session:
        upload = (await session.execute(
            select(Upload).where(Upload.id == upload_id)
        )).scalar_one_or_none()
        if not upload or upload.status != "in_progress":
            ctx["logger"].info(f"Upload {upload_id} not in_progress; skipping")
            return
        
        # 1) Download
        content = await storage.download(upload.key)
        if not content:
            ctx["logger"].error(f"Could not download {upload.key}")
            upload.status = "error"
            await session.commit()
            return
        
        # 2) Virus scan
        scan_result = await scan_file(content)
        if not scan_result["clean"]:
            upload.status = "rejected_malware"
            upload.rejection_reason = scan_result["reason"]
            await storage.delete(upload.key)
            await session.commit()
            return
        
        # 3) Process the file
        if upload.content_type.startswith("image/"):
            # Generate thumbnails
            thumbnails = await generate_thumbnails(content)
            for size_name, thumb in thumbnails.items():
                thumb_key = f"thumbnails/{upload.key}/{size_name}.webp"
                await storage.save(
                    key=thumb_key, content=io.BytesIO(thumb["content"]),
                    content_type=thumb["content_type"],
                )
            upload.thumbnails = {size: f"thumbnails/{upload.key}/{size}.webp" for size in thumbnails}
        
        elif upload.content_type.startswith("video/"):
            # Extract a frame at 1 second
            ...
        
        upload.status = "processed"
        await session.commit()
```

The job: download, scan, process, finalize. The processing is image-specific; for other types, the logic differs.

### 5.2 The enqueue

```python
@app.post("/upload/confirm")
async def confirm_upload(
    key: str,
    file_size: int,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
    arq: ArqRedis = Depends(get_arq),
):
    # Verify the upload
    if not await storage.exists(key):
        raise HTTPException(404, "File not found in storage")
    
    # Record the upload
    upload = Upload(
        user_id=current_user.id, key=key, size=file_size, status="in_progress",
        content_type=...,  # from the presigned URL
    )
    await uow.uploads.create(upload)
    await uow.commit()
    
    # Enqueue the processing job
    await arq.enqueue_job("process_upload", upload_id=upload.id)
    
    return {"key": key, "status": "processing"}
```

The confirm step records the metadata and enqueues the processing. The user gets a "processing" status immediately; the actual processing happens in the background.

### 5.3 The status endpoint

```python
@app.get("/uploads/{upload_id}")
async def get_upload_status(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    upload = await uow.uploads.get(upload_id)
    if not upload or upload.user_id != current_user.id:
        raise HTTPException(404, "Upload not found")
    return {
        "status": upload.status,
        "thumbnails": upload.thumbnails if hasattr(upload, "thumbnails") else None,
        "created_at": upload.created_at.isoformat(),
    }
```

The user polls the status. The job updates the status when it's done.

---

## 6. The Production Patterns

### 6.1 The Pillow security

Pillow has had security vulnerabilities in the past (image parsing exploits). Mitigations:

1. **Keep Pillow updated**. Subscribe to security advisories.
2. **Use the latest stable version**. Newer Pillow versions have more bounds checks.
3. **Limit image size**. A 50,000 x 50,000 pixel image is a DoS vector.

```python
from PIL import Image
MAX_PIXELS = 10_000_000  # 10 megapixels


def safe_open(content: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(content))
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f"Image too large: {img.width}x{img.height}")
    return img
```

### 6.2 The memory limit

Image processing is memory-intensive. A 4K image is ~32 MB in memory. A 50-megapixel image is ~200 MB. Multiple concurrent jobs can OOM the worker.

The right fix: limit the input size (above) and use a worker pool with memory limits.

```python
# Kubernetes: limit the worker's memory
resources:
  limits:
    memory: "2Gi"
```

The worker is killed if it exceeds 2 GB; the job is retried.

### 6.3 The async Pillow

Pillow is sync. For async usage, use `asyncio.to_thread` or `loop.run_in_executor`:

```python
import asyncio
from PIL import Image


async def process_image_async(content: bytes) -> dict:
    """Run Pillow in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(_process_image_sync, content)


def _process_image_sync(content: bytes) -> dict:
    img = Image.open(io.BytesIO(content))
    # ... processing
    return result
```

The ARQ worker uses async; Pillow is sync. The thread is the bridge.

### 6.4 The retry policy

Pillow can fail in subtle ways (corrupt files, missing dependencies). The job should retry with exponential backoff:

```python
async def process_upload(ctx, upload_id: int):
    for attempt in range(3):
        try:
            await _process_upload_inner(ctx, upload_id)
            return
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                # Mark as failed
                ...
                raise
```

The retry is built into ARQ's `max_tries` setting; the explicit loop is for custom retry logic.

---

## 7. The Four Common Image Processing Mistakes

### 7.1 Synchronous processing in the request

```python
# ❌ The user waits 5 seconds while the thumbnail generates
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    # Generate thumbnails (5 seconds)
    thumbnails = await generate_thumbnails(content)
    # Save
    ...
    return {"thumbnails": thumbnails}
```

The fix: enqueue the processing; return immediately.

### 7.2 Storing the original without EXIF stripping

```python
# ❌ The original (with GPS coordinates) is stored
async def save_image(content: bytes):
    # Just save the original
    await storage.save(key=key, content=content, content_type="image/jpeg")
```

The fix: strip EXIF (or at least the privacy-sensitive fields) before storing.

### 7.3 No virus scan

```python
# ❌ A malicious file is saved and processed
async def process_upload(upload_id: int):
    # ... process the file
    # No scan
```

The fix: always scan. The cost is milliseconds; the benefit is a major attack vector closed.

### 7.4 Unbounded image size

```python
# ❌ A 1-gigapixel image consumes all memory
img = Image.open(content)
img.thumbnail((256, 256))  # Pillow tries to load the whole image
```

The fix: limit the input size; use a streaming image decoder for very large images.

---

## 8. The Test Patterns

### 8.1 Test the thumbnail generation

```python
def test_generate_thumbnails():
    # Create a test image
    img = Image.new("RGB", (1000, 1000), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()
    # Generate thumbnails
    thumbnails = await generate_thumbnails(content)
    # Check the thumbnails
    assert "sm" in thumbnails
    assert thumbnails["sm"]["size"] == (150, 100)  # aspect ratio preserved
```

### 8.2 Test the EXIF stripping

```python
def test_strip_exif():
    # Create an image with EXIF data (a real JPEG from a camera)
    with open("test_with_exif.jpg", "rb") as f:
        content = f.read()
    # Strip
    stripped = strip_exif(content)
    # The EXIF is gone
    img = Image.open(io.BytesIO(stripped))
    assert img._getexif() is None
```

### 8.3 Test the virus scan

```python
async def test_scan_clean_file():
    content = b"Hello, world!"  # Not a real file; will be scanned as clean
    result = await scan_file(content)
    # The result depends on ClamAV; for a real test, use the EICAR test file
    assert "clean" in result


async def test_scan_malicious_file():
    # The EICAR test file is a standard antivirus test string
    eicar = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    result = await scan_file(eicar)
    assert result["clean"] is False
    assert "EICAR" in result["reason"]
```

The EICAR test file is a standard way to verify antivirus. Most ClamAV installations detect it.

### 8.4 Test the full pipeline

```python
async def test_process_upload(arq_worker, db_with_upload):
    # Start the worker (runs in the same process)
    await arq_worker.async_run()
    # Enqueue the job
    await arq_worker.enqueue_job("process_upload", upload_id=1)
    # Wait for the job to complete
    await asyncio.sleep(1)
    # Check the upload
    async with SessionLocal()() as session:
        upload = (await session.execute(
            select(Upload).where(Upload.id == 1)
        )).scalar_one()
        assert upload.status == "processed"
        # The thumbnails exist
        assert "sm" in upload.thumbnails
```

The test runs the worker in-process, enqueues a job, and verifies the result.

---

## 9. The Image Processing Checklist

| Concern | Where | Verified by |
|---------|-------|-------------|
| Thumbnail generation | `generate_thumbnails` | Test the thumbnails |
| EXIF stripping | `strip_exif` | Test the EXIF is gone |
| Virus scan | `scan_file` | Test with EICAR |
| Background processing | `process_upload` job | Test the full pipeline |
| Pillow security | Limit input size, update Pillow | Subscribe to advisories |
| Memory limit | Kubernetes memory limit | K8s resource limit |
| Async | `asyncio.to_thread` | Test the job runs |

---

## 10. The Cost Optimization

### 10.1 The format choice

| Format | Compression | Use case |
|--------|-------------|----------|
| JPEG | Lossy, good for photos | Default for photos |
| WebP | Better than JPEG, smaller files | Modern web apps |
| AVIF | Even better than WebP | Cutting-edge browsers |
| PNG | Lossless, large files | Graphics, screenshots |

For most web services, WebP is the right choice. AVIF is better but only supported by modern browsers (2024+).

### 10.2 The quality level

| Quality | Use case |
|---------|----------|
| 95+ | Master copy, archival |
| 85-90 | High-quality display (default) |
| 70-80 | Small thumbnails, fast load |
| <70 | Avoid; visible artifacts |

The default of 85 is the sweet spot: smaller than JPEG quality 90 but visually indistinguishable for most content.

### 10.3 The thumbnail strategy

A single 256x256 thumbnail is 5-15 KB. A full 1024x1024 is 100-300 KB. Generating 4 thumbnail sizes (xs, sm, md, lg) adds ~500 KB of storage per image.

For most services, 2 sizes (sm and md) is enough. The xs (50x50) is rarely needed (the sm is fine for most UIs). The lg is for detail views; many services skip it and use the original.

---

## 11. Código de Compresión

```python
"""
Compresión: Image Processing Pipeline
Covers: Pillow basics, thumbnail generation, EXIF stripping, virus scanning.
"""
import io
from typing import Optional

from PIL import Image
import piexif
import clamd


# 1) Thumbnail generation
THUMBNAIL_SIZES = {
    "sm": (150, 150),
    "md": (300, 300),
    "lg": (600, 600),
}


async def generate_thumbnails(content: bytes) -> dict[str, bytes]:
    img = Image.open(io.BytesIO(content))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    thumbnails = {}
    for size_name, (w, h) in THUMBNAIL_SIZES.items():
        thumb = img.copy()
        thumb.thumbnail((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format="WEBP", quality=85)
        thumbnails[size_name] = buf.getvalue()
    return thumbnails


# 2) EXIF stripping
def strip_exif(content: bytes) -> bytes:
    img = Image.open(io.BytesIO(content))
    buf = io.BytesIO()
    # Pillow doesn't include EXIF when saving unless explicitly told
    img.save(buf, format=img.format)
    return buf.getvalue()


def strip_exif_aggressive(content: bytes) -> bytes:
    img = Image.open(io.BytesIO(content))
    exif_bytes = img.info.get("exif", None)
    if exif_bytes:
        try:
            piexif.remove(exif_bytes)
        except Exception:
            pass
    buf = io.BytesIO()
    img.save(buf, format=img.format)
    return buf.getvalue()


# 3) Virus scan
async def scan_file(content: bytes) -> dict:
    cd = clamd.ClamdAsyncNetworkSocket(host="clamav", port=3310)
    result = await cd.instream(io.BytesIO(content))
    if result["stream"] is None:
        return {"clean": True, "reason": None}
    return {"clean": False, "reason": result["stream"][1]}


# 4) The background job
async def process_upload(ctx, upload_id: int):
    from app.db.session import async_session_maker
    from app.models.upload import Upload
    from sqlalchemy import select
    from app.core.storage import get_storage
    
    storage = get_storage()
    async with async_session_maker()() as session:
        upload = (await session.execute(
            select(Upload).where(Upload.id == upload_id)
        )).scalar_one_or_none()
        if not upload or upload.status != "in_progress":
            return
        
        # Download
        content = await storage.download(upload.key)
        if not content:
            upload.status = "error"
            await session.commit()
            return
        
        # Scan
        scan = await scan_file(content)
        if not scan["clean"]:
            upload.status = "rejected_malware"
            await storage.delete(upload.key)
            await session.commit()
            return
        
        # Process
        if upload.content_type.startswith("image/"):
            thumbnails = await generate_thumbnails(content)
            thumb_urls = {}
            for name, data in thumbnails.items():
                thumb_key = f"thumbnails/{upload.key}/{name}.webp"
                await storage.save(key=thumb_key, content=io.BytesIO(data), content_type="image/webp")
                thumb_urls[name] = thumb_key
            upload.thumbnails = thumb_urls
        
        upload.status = "processed"
        await session.commit()
```

---

## Key Takeaways

- **Image processing is background work.** The upload confirms the file in S3; the processing happens in an ARQ job. The user sees the upload as instant.
- **Pillow is the standard library.** Use `Image.LANCZOS` for high-quality resampling. Convert to RGB before saving as JPEG/WebP.
- **Strip EXIF** to protect user privacy. GPS coordinates and camera models are sensitive.
- **Always virus scan** uploads. ClamAV is the standard open-source antivirus. The cost is milliseconds; the benefit is closing a major attack vector.
- **Limit image size** to prevent DoS. A 1-gigapixel image is a memory bomb.
- **Generate multiple thumbnail sizes** (sm, md, lg) for different UI contexts. WebP is the right format for modern web apps.
- **Pillow is sync**; use `asyncio.to_thread` or a thread executor to avoid blocking the event loop.
- **The processing is a job**; the result is a status field that the user polls. The job is idempotent; retries are safe.

## References

- [Pillow Documentation](https://pillow.readthedocs.io/)
- [piexif Documentation](https://pypi.org/project/piexif/)
- [clamd — Python ClamAV client](https://pypi.org/project/clamd/)
- [ClamAV Documentation](https://docs.clamav.net/)
- [ARQ Documentation](https://arq-docs.helpmanual.io/)
- [WebP Container Specification (Google)](https://developers.google.com/speed/webp/docs/api)
- [AVIF Image Format (Alliance for Open Media)](https://aomediacodec.github.io/av1-avif/)
- [ImageMagick vs Pillow for Production (2023)](https://www.smashingmagazine.com/2023/02/large-image-pipelines/)
- [Pillow Security Disclosures](https://github.com/python-pillow/Pillow/security)
