# ☁️ S3 / GCS with Presigned URLs

## 🎯 Learning Objectives

- Implement the S3 / GCS storage backend for the `Storage` interface
- Use presigned URLs for direct client upload (bypassing your server)
- Configure bucket policies, IAM, and lifecycle rules
- Handle multipart uploads via the S3 API
- Avoid the three most common S3 / GCS security mistakes

## Introduction

The local filesystem is fine for development, but production at scale needs object storage: S3 (AWS), GCS (Google Cloud), or Azure Blob. The benefits: 11 9s of durability, unlimited scale, pay-per-use, and a rich security model. The cost: an extra hop and a new set of patterns to learn.

The right pattern is **presigned URLs**: the client uploads directly to S3, bypassing your FastAPI server. Your server only generates the URL; the bytes never touch your infrastructure. This scales to any size, doesn't burden your server, and is the standard for production.

This note covers the S3 and GCS implementations, presigned URL generation, IAM policies, and the security patterns that keep your bucket safe. The local `Storage` interface from [[01 - Local File Handling|the previous note]] is the basis; S3 / GCS implement the same interface.

---

## 1. The S3 Storage Implementation

### 1.1 The setup

```python
# app/storage/s3.py
import boto3
from botocore.config import Config
from app.storage.base import Storage, SavedFile


class S3Storage(Storage):
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,  # For S3-compatible services (MinIO, etc.)
        presign_expires: int = 3600,
    ):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        self.presign_expires = presign_expires
    
    async def save_streaming(
        self, *, key: str, content, content_type: str, chunk_size: int = 1024 * 1024,
    ) -> SavedFile:
        """Stream the file to S3."""
        # For single-part upload, upload_fileobj is fine
        # (boto3 streams the file in chunks internally)
        extra_args = {"ContentType": content_type}
        # Run in a thread to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.s3.upload_fileobj(content, self.bucket, key, ExtraArgs=extra_args),
        )
        # Get the size
        head = await loop.run_in_executor(
            None, lambda: self.s3.head_object(Bucket=self.bucket, Key=key),
        )
        return SavedFile(
            key=key,
            size=head["ContentLength"],
            content_type=content_type,
            url=f"https://{self.bucket}.s3.{self.s3.meta.region_name}.amazonaws.com/{key}",
        )
    
    async def save(self, *, key, content, content_type, metadata=None) -> SavedFile:
        return await self.save_streaming(
            key=key, content=content, content_type=content_type,
        )
    
    async def get_url(self, key, *, expires_in=None) -> str:
        expires = expires_in or self.presign_expires
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )
    
    async def delete(self, key) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self.s3.delete_object(Bucket=self.bucket, Key=key),
        )
    
    async def exists(self, key) -> bool:
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self.s3.head_object(Bucket=self.bucket, Key=key),
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
```

The S3 client is **synchronous** (boto3). All calls run in a thread executor to avoid blocking the FastAPI event loop.

### 1.2 The boto3 sync-vs-async issue

`boto3` is sync; `aioboto3` is the async wrapper. For a FastAPI service, either works, but `aioboto3` requires more setup. The thread executor pattern is simpler and has the same performance characteristics for I/O-bound operations.

```python
# If you want pure async, use aioboto3
import aioboto3


session = aioboto3.Session()
async with session.client("s3", region_name="us-east-1") as s3:
    await s3.upload_fileobj(content, bucket, key, ExtraArgs={"ContentType": content_type})
```

For most services, the thread executor pattern is enough.

---

## 2. Presigned URLs

### 2.1 The pattern

A presigned URL is a temporary URL that grants a client permission to upload or download a specific file. The client uses the URL directly with S3; your server never sees the bytes.

```python
@app.post("/upload-url")
async def get_upload_url(
    filename: str,
    content_type: str,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
):
    """Generate a presigned URL for direct upload to S3."""
    # Validate
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Disallowed extension: {ext}")
    # Generate a unique key
    key = f"uploads/{current_user.id}/{datetime.now():%Y/%m/%d}/{uuid.uuid4()}{ext}"
    # Generate the presigned URL
    upload_url = storage.s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": storage.bucket,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=600,  # 10 minutes
    )
    return {
        "upload_url": upload_url,
        "key": key,
        "expires_in": 600,
    }
```

The client receives the URL and uploads directly:

```javascript
// Client-side
const response = await fetch("/upload-url", { method: "POST" });
const { upload_url, key } = await response.json();
const file = document.getElementById("fileInput").files[0];
await fetch(upload_url, {
    method: "PUT",
    body: file,
    headers: {"Content-Type": file.type},
});
// The file is now in S3 at the given key
```

The server's bandwidth is zero; S3 handles the upload directly.

### 2.2 The constraints

A presigned URL has limits:
- **Expiration**: the URL expires (default 1 hour, max 7 days for S3).
- **Scope**: the URL is for a specific bucket + key.
- **Method**: PUT (upload) or GET (download).
- **Size**: the URL can include `Content-Length` to enforce a max size.

```python
# Limit the upload size to 100 MB
upload_url = storage.s3.generate_presigned_url(
    "put_object",
    Params={
        "Bucket": storage.bucket,
        "Key": key,
        "ContentType": content_type,
        "ContentLength": 100 * 1024 * 1024,  # max 100 MB
    },
    ExpiresIn=600,
)
```

If the client tries to upload more, S3 rejects the upload with a 403.

### 2.3 The post-upload callback

The presigned URL returns 200 on success, but the client still needs to know the URL is valid. The standard pattern:

```python
@app.post("/upload-url")
async def get_upload_url(...) -> dict:
    key = ...
    upload_url = ...
    # The client uploads; once done, the client calls /upload/confirm
    return {"upload_url": upload_url, "key": key}


@app.post("/upload/confirm")
async def confirm_upload(
    key: str,
    file_size: int,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
):
    """Verify the upload succeeded and record the metadata."""
    # Verify the file exists in S3
    if not await storage.exists(key):
        raise HTTPException(404, "File not found in storage")
    actual_size = await storage.get_size(key)
    if actual_size != file_size:
        # The client lied; reject (or just record the actual size)
        ...
    # Record the metadata in the database
    upload = Upload(
        user_id=current_user.id, key=key, size=actual_size, status="uploaded",
    )
    await uow.uploads.create(upload)
    await uow.commit()
    # Trigger background processing (e.g., thumbnail generation)
    await arq.enqueue_job("process_upload", upload_id=upload.id)
    return {"key": key, "status": "uploaded"}
```

The confirm step records the metadata. The presigned URL is just the upload mechanism.

### 2.4 The download URL

```python
@app.get("/files/{key:path}/download")
async def get_download_url(
    key: str,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
):
    """Generate a presigned URL for downloading a file."""
    if not await storage.exists(key):
        raise HTTPException(404, "File not found")
    url = storage.s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": storage.bucket,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{key.split("/")[-1]}"',
        },
        ExpiresIn=600,
    )
    return {"download_url": url, "expires_in": 600}
```

The client receives a URL; the file is downloaded directly from S3.

---

## 3. The GCS Storage Implementation

### 3.1 The setup

```python
# app/storage/gcs.py
from google.cloud import storage
from google.cloud.storage.blob import Blob
import asyncio


class GCSStorage(Storage):
    def __init__(
        self,
        bucket_name: str,
        project_id: str,
        presign_expires: int = 3600,
    ):
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        self.presign_expires = presign_expires
    
    async def save_streaming(
        self, *, key: str, content, content_type: str, chunk_size: int = 1024 * 1024,
    ) -> SavedFile:
        blob = self.bucket.blob(key)
        blob.content_type = content_type
        loop = asyncio.get_event_loop()
        # Upload in chunks
        await loop.run_in_executor(
            None, lambda: blob.upload_from_file(content, content_type=content_type, rewind=True),
        )
        return SavedFile(
            key=key,
            size=blob.size,
            content_type=content_type,
            url=f"https://storage.googleapis.com/{self.bucket.name}/{key}",
        )
    
    async def get_url(self, key, *, expires_in=None) -> str:
        expires = expires_in or self.presign_expires
        blob = self.bucket.blob(key)
        return blob.generate_signed_url(expiration=timedelta(seconds=expires), method="GET")
    
    # ... (other methods similar to S3)
```

GCS's API is similar to S3's. The presigned URL is a "signed URL" with a 7-day max expiration.

### 3.2 The Storage interface

```python
# app/storage/__init__.py
from app.core.config import settings


def get_storage():
    if settings.STORAGE_BACKEND == "local":
        from app.storage.local import LocalStorage
        return LocalStorage(...)
    elif settings.STORAGE_BACKEND == "s3":
        from app.storage.s3 import S3Storage
        return S3Storage(bucket=settings.S3_BUCKET, region=settings.AWS_REGION)
    elif settings.STORAGE_BACKEND == "gcs":
        from app.storage.gcs import GCSStorage
        return GCSStorage(
            bucket_name=settings.GCS_BUCKET,
            project_id=settings.GCP_PROJECT_ID,
        )
    else:
        raise ValueError(f"Unknown storage backend: {settings.STORAGE_BACKEND}")
```

The handler is the same regardless of backend.

---

## 4. Bucket Configuration and Security

### 4.1 The bucket policy

The bucket should not be public. The presigned URL is the only way to access the files.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAll",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    }
  ]
}
```

The bucket is locked down. Only IAM users with the right policy can access it. The presigned URL is the only way for clients to access files.

### 4.2 The IAM policy

The FastAPI server's IAM role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:HeadObject"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```

The role can read, write, delete, and head objects in the bucket. It cannot list the bucket, change the policy, or do anything else. The principle of least privilege.

### 4.3 The CORS configuration

For browser-based uploads (direct from the browser to S3), the bucket needs CORS:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <CORSRule>
    <AllowedOrigin>https://app.example.com</AllowedOrigin>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>POST</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
    <MaxAgeSeconds>3000</MaxAgeSeconds>
  </CORSRule>
</CORSConfiguration>
```

The browser sends an OPTIONS preflight; the bucket returns the CORS headers. Without this, the browser refuses to upload.

### 4.4 The lifecycle rules

```xml
<LifecycleConfiguration>
  <Rule>
    <ID>delete-temp-uploads</ID>
    <Status>Enabled</Status>
    <Filter>
      <Prefix>uploads/temp/</Prefix>
    </Filter>
    <Expiration>
      <Days>1</Days>
    </Expiration>
  </Rule>
  <Rule>
    <ID>archive-old-uploads</ID>
    <Status>Enabled</Status>
    <Filter>
      <Prefix>uploads/</Prefix>
    </Filter>
    <Transition>
      <Days>90</Days>
      <StorageClass>GLACIER</StorageClass>
    </Transition>
  </Rule>
</LifecycleConfiguration>
```

Temp uploads (from a presigned URL that was never confirmed) are deleted after 1 day. Old uploads are moved to Glacier (cheaper storage) after 90 days.

---

## 5. The Multipart Upload Pattern

### 5.1 The use case

Files larger than 5 GB must use multipart upload in S3. The multipart API also has benefits for smaller files: resumability, parallelism, and faster uploads.

### 5.2 The flow

```python
@app.post("/upload-multipart/start")
async def start_multipart_upload(
    filename: str,
    content_type: str,
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
):
    """Start a multipart upload and return the upload ID."""
    ext = Path(filename).suffix
    key = f"uploads/{current_user.id}/{uuid.uuid4()}{ext}"
    response = storage.s3.create_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        ContentType=content_type,
    )
    return {
        "key": key,
        "upload_id": response["UploadId"],
        "part_size": 5 * 1024 * 1024,  # 5 MB parts
    }


@app.post("/upload-multipart/part")
async def upload_part(
    key: str,
    upload_id: str,
    part_number: int,
    file: UploadFile = File(...),
    storage: S3Storage = Depends(get_s3_storage),
):
    """Upload a single part."""
    response = storage.s3.upload_part(
        Bucket=storage.bucket,
        Key=key,
        PartNumber=part_number,
        UploadId=upload_id,
        Body=await file.read(),
    )
    return {"etag": response["ETag"], "part_number": part_number}


@app.post("/upload-multipart/complete")
async def complete_multipart_upload(
    key: str,
    upload_id: str,
    parts: list[dict],  # [{"part_number": 1, "etag": "..."}, ...]
    current_user: User = Depends(get_current_user),
    storage: S3Storage = Depends(get_s3_storage),
    uow: UnitOfWork = Depends(get_uow),
):
    """Complete the multipart upload."""
    response = storage.s3.complete_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    # Record the metadata
    upload = Upload(user_id=current_user.id, key=key, size=response["size"], status="uploaded")
    await uow.uploads.create(upload)
    await uow.commit()
    return {"key": key, "status": "uploaded"}


@app.post("/upload-multipart/abort")
async def abort_multipart_upload(
    key: str,
    upload_id: str,
    storage: S3Storage = Depends(get_s3_storage),
):
    """Abort an in-progress multipart upload."""
    storage.s3.abort_multipart_upload(
        Bucket=storage.bucket,
        Key=key,
        UploadId=upload_id,
    )
    return {"status": "aborted"}
```

The four-endpoint flow: start, upload parts, complete (or abort). The client uses the part upload endpoint for each chunk.

### 5.3 The presigned part URL

For direct browser uploads, each part is uploaded with a presigned URL:

```python
@app.post("/upload-multipart/part-url")
async def get_part_upload_url(
    key: str,
    upload_id: str,
    part_number: int,
    storage: S3Storage = Depends(get_s3_storage),
):
    """Generate a presigned URL for uploading a specific part."""
    url = storage.s3.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": storage.bucket,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=3600,
    )
    return {"url": url, "part_number": part_number}
```

The client uploads each part with the presigned URL. The FastAPI server never sees the bytes.

---

## 6. The Three Common S3 / GCS Security Mistakes

### 6.1 The public bucket

```json
// ❌ Anyone can list and download all files
{
  "Effect": "Allow",
  "Principal": "*",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::my-bucket/*"
}
```

A public bucket is a data breach waiting to happen. The bucket should be private; presigned URLs are the only public access.

### 6.2 Long-lived presigned URLs

```python
# ❌ A presigned URL that never expires
url = storage.s3.generate_presigned_url(
    "put_object",
    Params={"Bucket": storage.bucket, "Key": key},
    ExpiresIn=86400 * 30,  # 30 days
)
```

A long-lived URL is a stolen-credential goldmine. Keep URLs short: 10-60 minutes for uploads, 1-24 hours for downloads.

### 6.3 The wildcard IAM policy

```json
// ❌ The server can do anything in S3
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*"
}
```

The principle of least privilege: the FastAPI server should only be able to read/write to its specific bucket. The wildcard policy is convenient but dangerous.

---

## 7. The Production Configuration

### 7.1 The boto3 config

```python
import boto3
from botocore.config import Config


s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",  # The current signing version
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=50,  # Connection pool size
        connect_timeout=5,
        read_timeout=10,
    ),
)
```

The boto3 config controls connection pooling, retries, and timeouts. The defaults are reasonable; tune for your workload.

### 7.2 The CORS for direct uploads

```python
# app/storage/s3.py
def set_bucket_cors(s3_client, bucket: str, allowed_origins: list[str]):
    """Configure CORS for direct browser uploads."""
    s3_client.put_bucket_cors(
        Bucket=bucket,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedHeaders": ["*"],
                    "AllowedMethods": ["PUT", "POST", "GET"],
                    "AllowedOrigins": allowed_origins,
                    "ExposeHeaders": ["ETag"],
                    "MaxAgeSeconds": 3000,
                }
            ]
        },
    )
```

Set CORS once at deployment. The browser respects the CORS rules; the FastAPI server is bypassed for the actual upload.

### 7.3 The error handling

```python
from botocore.exceptions import ClientError


async def save_with_retries(storage: S3Storage, key: str, content) -> SavedFile:
    """Save with exponential backoff on transient errors."""
    for attempt in range(5):
        try:
            return await storage.save_streaming(key=key, content=content, content_type="...")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("Throttling", "ServiceUnavailable"):
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("S3 save failed after 5 attempts")
```

S3 is reliable but not infallible. Transient errors (throttling, 503) are retried; permanent errors (NoSuchBucket, AccessDenied) raise immediately.

---

## 8. The Test Patterns

### 8.1 Test with a real S3 (or MinIO)

```python
# tests/test_s3_storage.py
import pytest
from app.storage.s3 import S3Storage


@pytest.fixture
def s3_storage():
    # Use a local MinIO instance for tests
    return S3Storage(
        bucket="test-bucket",
        endpoint_url="http://localhost:9000",  # MinIO
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )


@pytest.mark.asyncio
async def test_save_and_get(s3_storage):
    content = io.BytesIO(b"Hello, world!")
    saved = await s3_storage.save(key="test.txt", content=content, content_type="text/plain")
    assert saved.size == 13
    assert await s3_storage.exists(saved.key)
    await s3_storage.delete(saved.key)
```

### 8.2 Test the presigned URL

```python
@pytest.mark.asyncio
async def test_presigned_upload(s3_storage, httpx_client):
    # Generate a presigned URL
    key = "test-presigned.txt"
    url = s3_storage.s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": s3_storage.bucket, "Key": key},
        ExpiresIn=600,
    )
    # Upload via the presigned URL
    response = await httpx_client.put(url, content=b"Test content", headers={"Content-Type": "text/plain"})
    assert response.status_code == 200
    # The file is in S3
    assert await s3_storage.exists(key)
    await s3_storage.delete(key)
```

The test uses a real S3 (or MinIO) to verify the presigned URL works.

### 8.3 Test the multipart flow

```python
@pytest.mark.asyncio
async def test_multipart_upload(s3_storage):
    # Start
    response = s3_storage.s3.create_multipart_upload(Bucket=s3_storage.bucket, Key="test-mp.bin")
    upload_id = response["UploadId"]
    # Upload parts
    parts = []
    for i in range(1, 4):
        part = s3_storage.s3.upload_part(
            Bucket=s3_storage.bucket, Key="test-mp.bin",
            PartNumber=i, UploadId=upload_id,
            Body=b"x" * (5 * 1024 * 1024),
        )
        parts.append({"PartNumber": i, "ETag": part["ETag"]})
    # Complete
    response = s3_storage.s3.complete_multipart_upload(
        Bucket=s3_storage.bucket, Key="test-mp.bin",
        UploadId=upload_id, MultipartUpload={"Parts": parts},
    )
    assert response["size"] == 15 * 1024 * 1024
    await s3_storage.delete("test-mp.bin")
```

The test exercises the full multipart flow.

---

## 9. The Cost Optimization

### 9.1 Storage classes

| Class | Cost | Availability | Use case |
|-------|------|--------------|----------|
| Standard | $$$$ | 99.99% | Hot data, recent uploads |
| Intelligent-Tiering | $$ | Auto | Mixed access patterns |
| Standard-IA | $$ | 99.9% | Infrequently accessed |
| One Zone-IA | $ | 99.5% | Re-creatable data |
| Glacier Instant Retrieval | $ | 99.9% | Archive, accessed <1x/month |
| Glacier Flexible Retrieval | ¢ | 99.99% | Archive, 1x/year |
| Glacier Deep Archive | ¢ | 99.99% | Long-term archive, 1x/10-years |

For most production services: Standard for the first 30 days, Intelligent-Tiering after. The lifecycle rules automate the transition.

### 9.2 The request cost

S3 charges per request. A presigned URL is one PUT request. A 100 MB file in 10 MB parts is 10 PUT requests. For a high-throughput service, the request cost can exceed the storage cost.

The optimization: use larger parts (5-50 MB) to reduce the number of requests. The trade-off: larger parts are slower to recover from a failed part.

### 9.3 The transfer cost

S3 charges for data transfer out (to the internet). Data transfer in is free. A presigned URL that the client uses to download the file is transfer-out; a presigned URL for upload is free.

For a service with high download traffic, consider CloudFront (the AWS CDN). CloudFront caches the file at edge locations; the transfer cost is lower.

---

## 10. Código de Compresión

```python
"""
Compresión: S3 / GCS with Presigned URLs
Covers: S3Storage, presigned URLs, multipart upload, security.
"""
import asyncio
import uuid
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import Depends, File, HTTPException, UploadFile


# 1) The S3Storage class
class S3Storage:
    def __init__(self, bucket, region="us-east-1", endpoint_url=None, presign_expires=3600):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3", region_name=region, endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        self.presign_expires = presign_expires
    
    async def save_streaming(self, *, key, content, content_type, chunk_size=1024*1024):
        extra_args = {"ContentType": content_type}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self.s3.upload_fileobj(content, self.bucket, key, ExtraArgs=extra_args),
        )
        head = await loop.run_in_executor(
            None, lambda: self.s3.head_object(Bucket=self.bucket, Key=key),
        )
        return {
            "key": key, "size": head["ContentLength"],
            "content_type": content_type,
            "url": f"https://{self.bucket}.s3.amazonaws.com/{key}",
        }
    
    async def exists(self, key) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self.s3.head_object(Bucket=self.bucket, Key=key),
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    
    async def delete(self, key):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self.s3.delete_object(Bucket=self.bucket, Key=key),
        )


# 2) Presigned upload URL endpoint
@app.post("/upload-url")
async def get_upload_url(
    filename: str,
    content_type: str,
    storage: S3Storage = Depends(lambda: S3Storage(bucket="my-bucket")),
):
    ext = Path(filename).suffix
    if ext not in {".jpg", ".png", ".pdf"}:
        raise HTTPException(400, f"Disallowed extension: {ext}")
    key = f"uploads/{datetime.now():%Y/%m/%d}/{uuid.uuid4()}{ext}"
    url = storage.s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": storage.bucket,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=600,
    )
    return {"upload_url": url, "key": key, "expires_in": 600}


# 3) Multipart upload
@app.post("/upload-multipart/start")
async def start_multipart(
    filename: str, content_type: str, storage: S3Storage = Depends(lambda: S3Storage(bucket="my-bucket")),
):
    key = f"uploads/{uuid.uuid4()}{Path(filename).suffix}"
    response = storage.s3.create_multipart_upload(Bucket=storage.bucket, Key=key, ContentType=content_type)
    return {"key": key, "upload_id": response["UploadId"]}


@app.post("/upload-multipart/part")
async def upload_part(
    key: str, upload_id: str, part_number: int, file: UploadFile = File(...),
    storage: S3Storage = Depends(lambda: S3Storage(bucket="my-bucket")),
):
    response = storage.s3.upload_part(
        Bucket=storage.bucket, Key=key, PartNumber=part_number,
        UploadId=upload_id, Body=await file.read(),
    )
    return {"etag": response["ETag"], "part_number": part_number}
```

---

## Key Takeaways

- **The `Storage` interface** is the right abstraction. The handler is the same regardless of local, S3, or GCS.
- **Presigned URLs** are the production pattern. The client uploads directly to S3; your server never sees the bytes. Scales to any size, doesn't burden your server.
- **Presigned URLs are short-lived** (10-60 minutes for uploads). A long-lived URL is a stolen-credential goldmine.
- **The bucket is private**. Presigned URLs are the only public access. CORS is configured for direct browser uploads.
- **Multipart upload** is required for files >5 GB and beneficial for resumability, parallelism, and faster uploads.
- **The IAM policy is least-privilege**. The FastAPI server can only PutObject, GetObject, DeleteObject, HeadObject on its specific bucket. Nothing more.
- **boto3 is sync**. Use the thread executor pattern to avoid blocking the event loop.
- **Lifecycle rules** automate cost optimization. Standard for hot data, Intelligent-Tiering or IA for cold data, Glacier for archive.

## References

- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS S3 Multipart Upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)
- [Google Cloud Storage Python Client](https://cloud.google.com/python/docs/reference/storage/latest)
- [S3 Presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)
- [S3 CORS Configuration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html)
- [S3 Lifecycle Rules](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)
- [MinIO — S3-compatible local storage](https://min.io/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
