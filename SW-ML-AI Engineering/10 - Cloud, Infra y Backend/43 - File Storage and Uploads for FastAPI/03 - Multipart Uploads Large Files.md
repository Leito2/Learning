# 📤 Multipart Uploads (Large Files)

## 🎯 Learning Objectives

- Implement S3 multipart upload: start, upload parts, complete, abort
- Build a client-side chunked upload for gigabyte files
- Handle part failures with retries and resume
- Avoid the four most common multipart upload mistakes

## Introduction

Files larger than 5 GB cannot be uploaded to S3 in a single PUT request — the API has a hard limit. The solution is multipart upload: split the file into parts (each up to 5 GB), upload each part independently, then combine them. Multipart upload has additional benefits for smaller files too: resumability (retry only the failed part), parallelism (upload parts in parallel), and faster uploads (multiple parts hit S3 at once).

This note covers the multipart flow on both the server side (presigned URLs for each part) and the client side (chunked reading, retries, progress). The patterns apply to S3, GCS, and any S3-compatible storage.

---

## 1. When to Use Multipart

### 1.1 The decision

| File size | Use multipart? | Why |
|-----------|----------------|-----|
| <100 MB | No | Single PUT is fine; multipart adds complexity |
| 100 MB - 5 GB | Yes | Resumability, parallelism, faster |
| >5 GB | Required | Single PUT would fail (5 GB hard limit) |

The 5 GB limit is hard: S3 rejects single PUTs over 5 GB. The 100 MB threshold is a soft recommendation: multipart becomes worth the complexity around 100 MB for the resumability and parallelism benefits.

### 1.2 The multipart API

The S3 multipart API has four operations:

| Operation | When |
|-----------|------|
| `CreateMultipartUpload` | Start the upload; returns an upload ID |
| `UploadPart` | Upload one part; returns an ETag |
| `CompleteMultipartUpload` | Combine the parts; returns the final object |
| `AbortMultipartUpload` | Cancel; cleans up the parts |

The flow:

```
1. Client: POST /multipart/start
   Server: returns { key, upload_id }
2. Client: For each part:
   a. Reads chunk N from the file
   b. Uploads to S3 with a presigned URL for that part
3. Client: POST /multipart/complete with the parts list
   Server: calls S3.CompleteMultipartUpload
4. Server: returns the final key
```

The client never uploads the whole file to the FastAPI server. The presigned URLs are for each part.

---

## 2. The Server-Side Endpoints

### 2.1 The start endpoint

```python
@app.post("/multipart/start")
async def start_multipart_upload(
    filename: str,
    content_type: str,
    size: int,  # The client knows the size; helps with validation
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
):
    """Start a multipart upload and return the upload ID."""
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Disallowed extension: {ext}")
    if size > MAX_UPLOAD_SIZE * 10:  # 500 MB max for now
        raise HTTPException(413, f"File too large; max {MAX_UPLOAD_SIZE * 10} bytes")
    
    key = f"uploads/{current_user.id}/{datetime.now():%Y/%m/%d}/{uuid.uuid4()}{ext}"
    response = storage.s3.create_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        ContentType=content_type,
    )
    upload_id = response["UploadId"]
    
    # Record the upload in the database
    upload = Upload(
        user_id=current_user.id, key=key, size=size, status="in_progress",
        upload_id=upload_id, content_type=content_type,
    )
    await uow.uploads.create(upload)
    await uow.commit()
    
    # Return the key, upload_id, and recommended part size
    return {
        "key": key,
        "upload_id": upload_id,
        "part_size": 5 * 1024 * 1024,  # 5 MB
        "max_parts": 1000,
    }
```

The server records the upload in the database. If the client abandons, the upload can be aborted later (covered in §6).

### 2.2 The part URL endpoint

For direct browser uploads, each part needs a presigned URL:

```python
@app.post("/multipart/part-url")
async def get_part_upload_url(
    key: str,
    upload_id: str,
    part_number: int,
    storage: S3Storage = Depends(get_s3_storage),
):
    """Generate a presigned URL for uploading a specific part."""
    if part_number < 1 or part_number > 10000:
        raise HTTPException(400, f"Invalid part number: {part_number}")
    
    url = storage.s3.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": storage.bucket,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=3600,  # 1 hour
    )
    return {"url": url, "part_number": part_number, "expires_in": 3600}
```

The browser uses this URL to PUT the part directly to S3. S3 returns an ETag; the client collects the ETags for the complete step.

### 2.3 The complete endpoint

```python
class CompletedPart(BaseModel):
    part_number: int
    etag: str


@app.post("/multipart/complete")
async def complete_multipart_upload(
    key: str,
    upload_id: str,
    parts: list[CompletedPart],
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
    arq: ArqRedis = Depends(get_arq),
):
    """Complete the multipart upload."""
    # Verify the upload exists and belongs to the user
    upload = await uow.uploads.get_by(key=key, user_id=current_user.id)
    if not upload or upload.upload_id != upload_id:
        raise HTTPException(404, "Upload not found")
    
    # Sort parts by part number (S3 requires this)
    sorted_parts = sorted(parts, key=lambda p: p.part_number)
    
    # Complete the multipart upload
    response = storage.s3.complete_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": p.part_number, "ETag": p.etag} for p in sorted_parts]},
    )
    
    # Update the upload record
    upload.status = "uploaded"
    upload.size = response.get("size", upload.size)
    await uow.commit()
    
    # Trigger background processing (e.g., thumbnail generation)
    await arq.enqueue_job("process_upload", upload_id=upload.id)
    
    return {"key": key, "status": "uploaded", "size": upload.size}
```

The server uses the ETags (returned by S3 for each part) to combine the parts. S3 ensures the parts are in order; the response includes the final object size.

### 2.4 The abort endpoint

```python
@app.post("/multipart/abort")
async def abort_multipart_upload(
    key: str,
    upload_id: str,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
):
    """Abort an in-progress multipart upload."""
    upload = await uow.uploads.get_by(key=key, user_id=current_user.id)
    if not upload or upload.upload_id != upload_id:
        raise HTTPException(404, "Upload not found")
    
    storage.s3.abort_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        UploadId=upload_id,
    )
    upload.status = "aborted"
    await uow.commit()
    return {"status": "aborted"}
```

The abort cleans up the parts in S3. Without this, the parts linger (and you pay for them) until the lifecycle rule kicks in.

---

## 3. The Client-Side Implementation

### 3.1 The browser-based upload (JavaScript)

```javascript
async function uploadLargeFile(file) {
    // 1. Start the multipart upload
    const startRes = await fetch("/multipart/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            filename: file.name,
            content_type: file.type,
            size: file.size,
        }),
    });
    const {key, upload_id, part_size} = await startRes.json();
    
    // 2. Upload each part
    const parts = [];
    const partCount = Math.ceil(file.size / part_size);
    for (let i = 0; i < partCount; i++) {
        const start = i * part_size;
        const end = Math.min(start + part_size, file.size);
        const chunk = file.slice(start, end);
        const partNumber = i + 1;
        
        // Get a presigned URL for this part
        const urlRes = await fetch("/multipart/part-url", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({key, upload_id, part_number: partNumber}),
        });
        const {url} = await urlRes.json();
        
        // Upload the chunk directly to S3
        const uploadRes = await fetch(url, {
            method: "PUT",
            body: chunk,
        });
        const etag = uploadRes.headers.get("ETag");
        parts.push({part_number: partNumber, etag: etag});
        
        // Update progress
        const progress = (i + 1) / partCount;
        updateProgressBar(progress);
    }
    
    // 3. Complete the upload
    const completeRes = await fetch("/multipart/complete", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({key, upload_id, parts}),
    });
    return await completeRes.json();
}
```

The client reads the file in chunks (using `file.slice`), uploads each chunk to S3, and reports progress.

### 3.2 The Python-based upload (server-to-server)

```python
async def upload_large_file_to_s3(
    file_path: str,
    key: str,
    content_type: str,
    storage: S3Storage,
):
    """Server-side multipart upload with parallelism."""
    import math
    import aiohttp
    
    # Get the file size
    file_size = os.path.getsize(file_path)
    part_size = 5 * 1024 * 1024  # 5 MB
    part_count = math.ceil(file_size / part_size)
    
    # Start
    response = storage.s3.create_multipart_upload(
        Bucket=storage.bucket, Key=key, ContentType=content_type,
    )
    upload_id = response["UploadId"]
    
    # Upload parts in parallel (max 4 at a time)
    semaphore = asyncio.Semaphore(4)
    parts = []
    
    async def upload_part(part_number, start, end):
        async with semaphore:
            for attempt in range(3):
                try:
                    presigned = storage.s3.generate_presigned_url(
                        "upload_part",
                        Params={
                            "Bucket": storage.bucket,
                            "Key": key,
                            "UploadId": upload_id,
                            "PartNumber": part_number,
                        },
                        ExpiresIn=3600,
                    )
                    # Read the chunk
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        chunk = f.read(end - start)
                    # Upload
                    async with aiohttp.ClientSession() as session:
                        async with session.put(presigned, data=chunk) as resp:
                            etag = resp.headers["ETag"]
                    return {"PartNumber": part_number, "ETag": etag}
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
    
    # Launch all parts
    tasks = [
        upload_part(i + 1, i * part_size, min((i + 1) * part_size, file_size))
        for i in range(part_count)
    ]
    parts = await asyncio.gather(*tasks)
    
    # Complete
    parts_sorted = sorted(parts, key=lambda p: p["PartNumber"])
    response = storage.s3.complete_multipart_upload(
        Bucket=storage.bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": parts_sorted},
    )
    return {"key": key, "size": file_size, "status": "uploaded"}
```

Parallel upload (4 parts at a time) is 4x faster than serial for large files on good networks.

### 3.3 The chunk size

The right part size depends on the file size and the network:

| File size | Recommended part size | Reason |
|-----------|---------------------|--------|
| <50 MB | 5 MB | Single chunk is enough |
| 50-500 MB | 5-10 MB | 5-100 parts; manageable |
| 500 MB - 5 GB | 10-50 MB | 10-500 parts; faster uploads |
| >5 GB | 50-100 MB | Required; fewer parts |

Smaller parts mean more requests (more overhead); larger parts mean fewer requests but slower to recover from a failed part. 5-10 MB is the sweet spot for most cases.

---

## 4. Resumability

### 4.1 The pattern

A multipart upload can fail mid-stream (network drops, browser closes). The client should be able to resume: figure out which parts have been uploaded and upload only the rest.

```python
async def resume_multipart_upload(
    key: str,
    upload_id: str,
    file_path: str,
    storage: S3Storage,
):
    """Resume an in-progress multipart upload."""
    # 1. Ask S3 which parts have been uploaded
    parts_info = storage.s3.list_parts(
        Bucket=storage.bucket, Key=key, UploadId=upload_id,
    )
    uploaded_parts = {p["PartNumber"]: p["ETag"] for p in parts_info.get("Parts", [])}
    
    # 2. Calculate the parts we still need
    file_size = os.path.getsize(file_path)
    part_size = 5 * 1024 * 1024
    part_count = math.ceil(file_size / part_size)
    
    # 3. Upload only the missing parts
    semaphore = asyncio.Semaphore(4)
    parts = []
    for i in range(part_count):
        part_number = i + 1
        if part_number in uploaded_parts:
            parts.append({"PartNumber": part_number, "ETag": uploaded_parts[part_number]})
            continue
        # ... upload the part as before
```

The `list_parts` call returns the parts that have been uploaded. The client uploads only the missing parts. The complete step combines them.

### 4.2 The browser-based resume

The browser-based resume uses localStorage to track progress:

```javascript
async function uploadWithResume(file) {
    // Check if there's a previous attempt
    const saved = localStorage.getItem(`upload-${file.name}`);
    let key, uploadId, partSize;
    if (saved) {
        const state = JSON.parse(saved);
        key = state.key;
        uploadId = state.upload_id;
        partSize = state.part_size;
    } else {
        // Start a new upload
        const startRes = await fetch("/multipart/start", { ... });
        const data = await startRes.json();
        key = data.key;
        uploadId = data.upload_id;
        partSize = data.part_size;
        localStorage.setItem(`upload-${file.name}`, JSON.stringify(data));
    }
    
    // Continue uploading
    // ... (same as before, but skip already-uploaded parts)
    
    // On success, clear the saved state
    localStorage.removeItem(`upload-${file.name}`);
}
```

The user can close the browser, reopen it, and the upload resumes from where it stopped.

---

## 5. The Server-Side Cleanup

### 5.1 The orphaned uploads problem

A client starts a multipart upload, then disconnects. The parts in S3 are orphaned: the upload will never complete, but the parts are still there, costing money.

The fix: a background job that aborts uploads older than 24 hours.

```python
async def cleanup_orphaned_uploads():
    """Find uploads in 'in_progress' for more than 24 hours and abort them."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    async with async_session_maker()() as session:
        result = await session.execute(
            select(Upload).where(
                Upload.status == "in_progress",
                Upload.created_at < cutoff,
            )
        )
        orphaned = result.scalars().all()
    
    storage = get_s3_storage()
    for upload in orphaned:
        try:
            storage.s3.abort_multipart_upload(
                Bucket=storage.bucket, Key=upload.key, UploadId=upload.upload_id,
            )
        except Exception as e:
            logger.warning(f"Failed to abort {upload.key}: {e}")
        upload.status = "aborted"
        await session.commit()
    logger.info(f"Cleaned up {len(orphaned)} orphaned uploads")
```

### 5.2 The lifecycle rule

A backup: a lifecycle rule on the bucket that aborts in-progress multipart uploads older than 7 days.

```json
{
  "Rules": [
    {
      "ID": "abort-stale-uploads",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "uploads/"
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
```

S3 automatically aborts uploads older than 7 days. This is a safety net; the application-level cleanup is the primary defense.

---

## 6. The Four Common Multipart Mistakes

### 6.1 The too-small part size

```python
# ❌ 1 MB parts for a 1 GB file = 1000 parts
part_size = 1024 * 1024
# 1000 presigned URLs, 1000 PUTs, slow
```

The fix: 5-50 MB parts. Fewer requests; faster uploads.

### 6.2 The too-large part size

```python
# ❌ 5 GB parts (the max) for a 6 GB file = only 2 parts
# A failed part = retry 5 GB
part_size = 5 * 1024 * 1024 * 1024
```

The fix: 10-100 MB parts. Smaller parts are faster to retry.

### 6.3 The lost upload ID

```python
# ❌ The upload ID is lost; the client can't complete or abort
const uploadId = response.upload_id;
# The page reloads; uploadId is gone
# The parts sit in S3 forever
```

The fix: persist the upload ID in localStorage (or the server's database) so it survives a page reload.

### 6.4 The non-resumable client

```python
# ❌ The client starts from scratch on every failure
for part in parts:
    if upload_failed:
        # restart from part 0
```

The fix: use `list_parts` to figure out which parts are uploaded; resume from there.

---

## 7. The Test Patterns

### 7.1 Test the multipart flow end-to-end

```python
@pytest.mark.asyncio
async def test_multipart_upload(s3_storage, httpx_client):
    # Start
    start = await httpx_client.post("/multipart/start", json={
        "filename": "test.bin", "content_type": "application/octet-stream", "size": 12 * 1024 * 1024,
    })
    data = start.json()
    key, upload_id, part_size = data["key"], data["upload_id"], data["part_size"]
    # Upload parts
    parts = []
    for i in range(3):
        part_number = i + 1
        # Get the URL
        url_res = await httpx_client.post("/multipart/part-url", json={
            "key": key, "upload_id": upload_id, "part_number": part_number,
        })
        url = url_res.json()["url"]
        # Upload the part
        chunk = b"x" * part_size
        upload_res = await httpx_client.put(url, content=chunk)
        parts.append({"part_number": part_number, "etag": upload_res.headers["ETag"]})
    # Complete
    complete = await httpx_client.post("/multipart/complete", json={
        "key": key, "upload_id": upload_id, "parts": parts,
    })
    assert complete.status_code == 200
    # The file is in S3
    assert await s3_storage.exists(key)
    await s3_storage.delete(key)
```

The test exercises the full multipart flow against a real S3 (or MinIO).

### 7.2 Test the abort

```python
@pytest.mark.asyncio
async def test_multipart_abort(s3_storage, httpx_client):
    # Start
    start = await httpx_client.post("/multipart/start", json={
        "filename": "test.bin", "content_type": "application/octet-stream", "size": 1024 * 1024,
    })
    data = start.json()
    # Abort
    response = await httpx_client.post("/multipart/abort", json={
        "key": data["key"], "upload_id": data["upload_id"],
    })
    assert response.status_code == 200
    # The upload is aborted; parts are cleaned up
    assert not await s3_storage.exists(data["key"])
```

The test verifies that abort cleans up the parts.

### 7.3 Test the resume

```python
@pytest.mark.asyncio
async def test_multipart_resume(s3_storage, httpx_client):
    # Start
    start = await httpx_client.post("/multipart/start", json={
        "filename": "test.bin", "content_type": "application/octet-stream", "size": 12 * 1024 * 1024,
    })
    data = start.json()
    key, upload_id, part_size = data["key"], data["upload_id"], data["part_size"]
    # Upload only the first 2 of 3 parts
    parts = []
    for i in range(2):
        part_number = i + 1
        url = (await httpx_client.post("/multipart/part-url", json={
            "key": key, "upload_id": upload_id, "part_number": part_number,
        })).json()["url"]
        upload = await httpx_client.put(url, content=b"x" * part_size)
        parts.append({"part_number": part_number, "etag": upload.headers["ETag"]})
    # ... simulate a network failure ...
    # Resume: ask S3 which parts are uploaded
    list_parts = s3_storage.s3.list_parts(Bucket=s3_storage.bucket, Key=key, UploadId=upload_id)
    uploaded = {p["PartNumber"] for p in list_parts.get("Parts", [])}
    # Upload the missing part
    for part_number in range(1, 4):
        if part_number in uploaded:
            continue
        # ... upload and append
    # Complete
    # ...
```

The test verifies that `list_parts` returns the correct parts and the resume works.

---

## 8. The Production Considerations

### 8.1 The CORS for direct browser uploads

```xml
<CORSConfiguration>
  <CORSRule>
    <AllowedOrigin>https://app.example.com</AllowedOrigin>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>POST</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
    <ExposeHeader>ETag</ExposeHeader>
    <MaxAgeSeconds>3000</MaxAgeSeconds>
  </CORSRule>
</CORSConfiguration>
```

The browser sends an OPTIONS preflight; the bucket returns the CORS headers. The `ExposeHeader: ETag` is required so the JavaScript client can read the ETag from the response.

### 8.2 The KMS encryption

S3 encrypts files at rest by default (SSE-S3). For sensitive data, use SSE-KMS:

```python
response = storage.s3.create_multipart_upload(
    Bucket=storage.bucket,
    Key=key,
    ContentType=content_type,
    ServerSideEncryption="aws:kms",
    SSEKMSKeyId=settings.KMS_KEY_ID,
)
```

S3 uses the KMS key to encrypt each part. The key is managed in AWS KMS; access is logged.

### 8.3 The transfer acceleration

S3 Transfer Acceleration uses CloudFront edge locations to speed up uploads. The client uploads to the nearest edge; the edge forwards to S3.

```python
# Enable Transfer Acceleration on the bucket
storage.s3.put_bucket_accelerate_configuration(
    Bucket=storage.bucket,
    AccelerateConfiguration={"Status": "Enabled"},
)

# The presigned URL uses the acceleration endpoint
url = storage.s3.generate_presigned_url(
    "put_object",
    Params={"Bucket": storage.bucket, "Key": key},
    ExpiresIn=600,
)
# The client uses the bucket's accelerate endpoint: <bucket>.s3-accelerate.amazonaws.com
```

The benefit: faster uploads for users far from the bucket's region. The cost: per-GB transfer fees to the edge.

---

## 9. Código de Compresión

```python
"""
Compresión: Multipart Uploads (Large Files)
Covers: multipart start/upload/complete/abort, presigned part URLs,
       server-side cleanup, resumability.
"""
import asyncio
import math
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


# 1) Models
class CompletedPart(BaseModel):
    part_number: int
    etag: str


# 2) Start endpoint
@app.post("/multipart/start")
async def start_multipart(
    filename: str,
    content_type: str,
    size: int,
    storage = None,  # S3Storage
    uow = None,  # UnitOfWork
):
    key = f"uploads/{uuid.uuid4()}{Path(filename).suffix}"
    response = storage.s3.create_multipart_upload(
        Bucket=storage.bucket, Key=key, ContentType=content_type,
    )
    return {
        "key": key,
        "upload_id": response["UploadId"],
        "part_size": 5 * 1024 * 1024,
    }


# 3) Part URL endpoint
@app.post("/multipart/part-url")
async def get_part_url(
    key: str, upload_id: str, part_number: int, storage = None,
):
    url = storage.s3.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": storage.bucket, "Key": key,
            "UploadId": upload_id, "PartNumber": part_number,
        },
        ExpiresIn=3600,
    )
    return {"url": url, "part_number": part_number}


# 4) Complete endpoint
@app.post("/multipart/complete")
async def complete_multipart(
    key: str, upload_id: str, parts: list[CompletedPart], storage = None, uow = None,
):
    sorted_parts = sorted(parts, key=lambda p: p.part_number)
    response = storage.s3.complete_multipart_upload(
        Bucket=storage.bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": p.part_number, "ETag": p.etag} for p in sorted_parts]},
    )
    upload = await uow.uploads.get_by(key=key)
    upload.status = "uploaded"
    upload.size = response.get("size", upload.size)
    await uow.commit()
    return {"key": key, "status": "uploaded"}


# 5) Abort endpoint
@app.post("/multipart/abort")
async def abort_multipart(
    key: str, upload_id: str, storage = None, uow = None,
):
    storage.s3.abort_multipart_upload(Bucket=storage.bucket, Key=key, UploadId=upload_id)
    upload = await uow.uploads.get_by(key=key)
    upload.status = "aborted"
    await uow.commit()
    return {"status": "aborted"}


# 6) Cleanup orphaned uploads
async def cleanup_orphaned_uploads(storag, uow):
    cutoff = datetime.utcnow() - timedelta(hours=24)
    async with async_session_maker()() as session:
        result = await session.execute(
            select(Upload).where(Upload.status == "in_progress", Upload.created_at < cutoff)
        )
        orphaned = result.scalars().all()
    for upload in orphaned:
        try:
            storage.s3.abort_multipart_upload(
                Bucket=storage.bucket, Key=upload.key, UploadId=upload.upload_id,
            )
            upload.status = "aborted"
        except Exception:
            pass
    await session.commit()
```

---

## Key Takeaways

- **Multipart upload** is required for files >5 GB and beneficial for files >100 MB (resumability, parallelism).
- **The four-endpoint flow** is the standard: start, get part URL, upload part, complete (or abort). The server only generates URLs; the client uploads directly to S3.
- **Part size matters**: 5-50 MB is the sweet spot. Too small = many requests; too large = slow to retry.
- **The presigned URL is the right pattern** for both single-part and multipart uploads. The server never sees the bytes.
- **Resumability** is critical for large files. Use `list_parts` to figure out which parts are uploaded; upload only the rest.
- **The cleanup** is the often-forgotten part. Abort in-progress uploads older than 24 hours; use a lifecycle rule as a safety net.
- **CORS** is required for direct browser uploads. The bucket's CORS config allows the browser to PUT files.
- **The lifecycle rule** is the safety net. S3 auto-aborts uploads older than 7 days (or whatever you set).

## References

- [AWS S3 Multipart Upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)
- [boto3 S3 Multipart Upload Examples](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html)
- [S3 Abort Incomplete Multipart Upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/abort-mpu.html)
- [S3 Transfer Acceleration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/transfer-acceleration.html)
- [S3 CORS Configuration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html)
- [resumable.js — Client-side resumable upload](https://github.com/23/resumable.js)
- [tus — Open protocol for resumable uploads](https://tus.io/)
