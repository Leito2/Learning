# 🔒 MFA, Passkeys, and WebAuthn

## 🎯 Learning Objectives

- Implement TOTP-based MFA using `pyotp` and QR codes
- Understand WebAuthn and the FIDO2 ceremony: registration, authentication, attestation
- Support passkeys (the FIDO2-discoverable credential)
- Build recovery flows: backup codes, account recovery, admin reset
- Avoid the seven most common MFA mistakes in production

## Introduction

Passwords alone are not enough. The 2024 Verizon Data Breach Report shows that 68% of breaches involve a human element, and the most common attack vector is credential theft. Multi-factor authentication (MFA) is the most effective single control: it reduces account takeover by 99.9% according to Microsoft's research. The challenge is implementing MFA in a way that is secure, usable, and recoverable.

The two dominant MFA technologies are **TOTP** (time-based one-time passwords, the "Google Authenticator" model) and **WebAuthn** (the W3C standard for FIDO2, used by passkeys, hardware keys, and platform authenticators like Touch ID and Windows Hello). TOTP is the "something you have" factor (the phone or device running the authenticator app); WebAuthn is the modern evolution: the device itself authenticates, no codes to type, phishing-resistant.

This note covers both, with a focus on the production patterns that make MFA usable at scale. The capstone at the end of the course shows them in a real system.

---

## 1. TOTP: Time-Based One-Time Passwords

### 1.1 The algorithm

TOTP (RFC 6238) is a shared-secret algorithm:

```
TOTP = HOTP(K, T)
where:
  K = shared secret (stored on the server, given to the user once)
  T = floor((now - T0) / X)
    T0 = Unix time of the start (0)
    X = time step (typically 30 seconds)
  HOTP = HMAC-SHA1(K, T) truncated to 6 digits
```

The server and the authenticator app share the secret `K`. They both compute the same 6-digit code at the same time. The user types the code; the server verifies it.

The 30-second window means codes expire quickly; an attacker who observes one code cannot reuse it. A tolerance of ±1 step (so ±30 seconds on each side) accommodates clock drift.

### 1.2 Setup

```python
import pyotp
import qrcode
from io import BytesIO


def generate_totp_secret() -> str:
    """Generate a new TOTP secret for a user."""
    return pyotp.random_base32()


def get_totp_uri(secret: str, user_email: str, issuer: str) -> str:
    """Build the otpauth:// URI for the QR code."""
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_email,
        issuer_name=issuer,
    )


def generate_qr_code(uri: str) -> bytes:
    """Render the URI as a PNG for the user to scan."""
    img = qrcode.make(uri)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

The server generates a secret, builds an `otpauth://` URI, renders it as a QR code, and shows it to the user. The user scans with Google Authenticator, Authy, 1Password, or any RFC 6238-compliant app.

### 1.3 Verification

```python
def verify_totp(secret: str, code: str) -> bool:
    """Verify a 6-digit TOTP code. Allows ±1 step of clock drift."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)
```

The `valid_window=1` accepts the current step and one step on either side (so a code from the previous or next 30-second window is also accepted). This trades a small amount of security (an attacker has three chances in 90 seconds) for usability (a user who types their code 10 seconds late isn't locked out).

### 1.4 The FastAPI endpoints

```python
class MFASetupResponse(BaseModel):
    secret: str  # shown to the user as a fallback; they can type it manually
    qr_code: bytes  # PNG image to scan


@router.post("/auth/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    ctx: TenantContext = Depends(get_tenant_context),
):
    # Generate a new secret
    secret = generate_totp_secret()
    # Store it (unverified) on the user
    ctx.user.mfa_secret_encrypted = encrypt(secret)
    await uow.commit()
    # Generate the QR code
    uri = get_totp_uri(secret, ctx.user.email, "MyApp")
    qr = generate_qr_code(uri)
    return MFASetupResponse(secret=secret, qr_code=qr)


@router.post("/auth/mfa/verify")
async def verify_mfa(
    code: str,
    ctx: TenantContext = Depends(get_tenant_context),
):
    secret = decrypt(ctx.user.mfa_secret_encrypted)
    if not verify_totp(secret, code):
        raise HTTPException(401, "Invalid code")
    ctx.user.mfa_enabled = True
    await uow.commit()
    return {"status": "verified"}


@router.post("/auth/mfa/disable")
async def disable_mfa(
    password: str,
    ctx: TenantContext = Depends(get_tenant_context),
):
    # Re-verify the password before disabling MFA
    if not verify_password(password, ctx.user.hashed_password):
        raise HTTPException(401, "Invalid password")
    ctx.user.mfa_enabled = False
    ctx.user.mfa_secret_encrypted = None
    await uow.commit()
    return {"status": "disabled"}
```

The `mfa_secret_encrypted` is encrypted at rest (the secret is sensitive; if the database is compromised, the attacker should not get the secrets). Use `cryptography.fernet` or AWS KMS for the encryption.

### 1.5 The login flow with MFA

```python
@router.post("/auth/login")
async def login(payload: LoginRequest):
    user = await authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    if not user.mfa_enabled:
        # No MFA: issue token directly
        token, _ = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
        return {"access_token": token}
    # MFA is enabled: return a "challenge" token that allows /auth/mfa/check
    challenge_token = create_mfa_challenge_token(user.id)
    return {"mfa_required": True, "mfa_token": challenge_token}


@router.post("/auth/mfa/check")
async def mfa_check(payload: MFARequest):
    # Verify the challenge token
    user_id = verify_mfa_challenge_token(payload.mfa_token)
    user = await uow.users.get(user_id)
    secret = decrypt(user.mfa_secret_encrypted)
    if not verify_totp(secret, payload.code):
        raise HTTPException(401, "Invalid code")
    # Issue the real token
    token, _ = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
    return {"access_token": token}
```

The login is a two-step flow: first the password (proves identity), then the MFA code (proves possession of the device). The intermediate "mfa_token" is a short-lived JWT with a single purpose; it cannot be used to access the API.

---

## 2. Recovery Codes

### 2.1 The pattern

Users lose phones. Hardware keys break. The TOTP secret is gone with the device. Recovery codes are the fallback: a set of one-time codes generated at MFA setup that the user can use to log in when they lose their second factor.

```python
import secrets


def generate_recovery_codes(n: int = 10) -> list[str]:
    """Generate n one-time recovery codes."""
    return [secrets.token_hex(5).upper() for _ in range(n)]  # e.g., "A3F2C9B1E0"


def hash_recovery_code(code: str) -> str:
    """Hash a recovery code for storage (don't store plaintext)."""
    return hashlib.sha256(code.upper().encode()).hexdigest()
```

Codes are 10 characters of hex, uppercase, formatted in groups of 5 for readability: `A3F2C-9B1E0`. The user downloads or prints them at MFA setup. The server stores only the hashes.

### 2.2 Setup and use

```python
@router.post("/auth/mfa/setup")
async def setup_mfa(ctx: TenantContext = Depends(get_tenant_context)):
    secret = generate_totp_secret()
    recovery_codes = generate_recovery_codes()
    ctx.user.mfa_secret_encrypted = encrypt(secret)
    ctx.user.mfa_recovery_codes_hashed = [hash_recovery_code(c) for c in recovery_codes]
    await uow.commit()
    return {
        "secret": secret,
        "qr_code": generate_qr_code(get_totp_uri(secret, ctx.user.email, "MyApp")),
        "recovery_codes": recovery_codes,  # shown ONCE at setup
    }


@router.post("/auth/mfa/recover")
async def mfa_recover(payload: MFARecoverRequest):
    # Verify the recovery code
    code_hash = hash_recovery_code(payload.code)
    user = await uow.users.get_by(email=payload.email)
    if not user or code_hash not in user.mfa_recovery_codes_hashed:
        raise HTTPException(401, "Invalid recovery code")
    # Consume the code (remove it)
    user.mfa_recovery_codes_hashed.remove(code_hash)
    # Issue a token that allows the user to reset MFA
    token = create_mfa_reset_token(user.id)
    await uow.commit()
    return {"reset_token": token, "remaining_codes": len(user.mfa_recovery_codes_hashed)}
```

The recovery code is consumed on use. The user gets a token that lets them re-enroll MFA (or skip it for the rest of this session).

### 2.3 Code storage security

Recovery codes are sensitive: anyone with a code can bypass MFA. Store them hashed (not in plaintext). The number of remaining codes is logged but not the codes themselves.

### 2.4 Recovery code regeneration

When the user has used most of their codes, prompt them to regenerate:

```python
@router.post("/auth/mfa/recovery-codes/regenerate")
async def regenerate_recovery_codes(
    password: str,
    ctx: TenantContext = Depends(get_tenant_context),
):
    if not verify_password(password, ctx.user.hashed_password):
        raise HTTPException(401, "Invalid password")
    new_codes = generate_recovery_codes()
    ctx.user.mfa_recovery_codes_hashed = [hash_recovery_code(c) for c in new_codes]
    await uow.commit()
    return {"recovery_codes": new_codes}
```

Generating new codes invalidates the old ones. The user must re-download or re-print.

---

## 3. WebAuthn: The Modern Standard

### 3.1 What WebAuthn is

WebAuthn is the W3C standard for FIDO2 authentication. It uses public-key cryptography:

- The server stores a public key per user (per credential).
- The device holds the private key, protected by biometrics (Touch ID, Face ID) or a PIN.
- To log in, the server sends a challenge; the device signs it with the private key; the server verifies the signature.

Three properties make WebAuthn superior to TOTP:

1. **Phishing-resistant**: the browser binds the credential to the origin. A phishing site cannot reuse the credential.
2. **No codes to type**: the user authenticates with biometrics. UX is "tap your fingerprint" or "look at the camera".
3. **No shared secret to leak**: the server holds only a public key. A database breach yields no usable credentials.

### 3.2 The registration ceremony

```
User clicks "Add passkey"
  ↓
Browser / app generates a key pair on the device
  ↓
Browser sends the public key + challenge to the server
  ↓
Server stores: {user_id, public_key, credential_id, sign_count, ...}
  ↓
Done. The user can now log in with this passkey.
```

The server side:

```python
from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
    options_to_json,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
import secrets


@router.post("/auth/webauthn/register/options")
async def webauthn_register_options(
    ctx: TenantContext = Depends(get_tenant_context),
):
    challenge = secrets.token_bytes(32)
    options = generate_registration_options(
        rp_id="example.com",
        rp_name="MyApp",
        user_id=str(ctx.user.id).encode(),
        user_name=ctx.user.email,
        user_display_name=ctx.user.name,
        challenge=challenge,
        supported_pub_key_algs=[COSEAlgorithmIdentifier.ECDSA_SHA_256],
    )
    # Store the challenge for verification
    await uow.webauthn_challenges.create(
        user_id=ctx.user.id,
        challenge=base64url_encode(challenge),
        purpose="register",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    return options_to_json(options)


@router.post("/auth/webauthn/register/verify")
async def webauthn_register_verify(
    credential: dict,
    ctx: TenantContext = Depends(get_tenant_context),
):
    # Look up the challenge
    stored = await uow.webauthn_challenges.get_for_user(ctx.user.id, purpose="register")
    if not stored:
        raise HTTPException(400, "No registration in progress")
    challenge_bytes = base64url_decode(stored.challenge)
    # Verify the registration response
    verification = verify_registration_response(
        credential=credential,
        expected_challenge=challenge_bytes,
        expected_origin="https://example.com",
        expected_rp_id="example.com",
    )
    # Store the credential
    await uow.webauthn_credentials.create(
        user_id=ctx.user.id,
        credential_id=verification.credential_id,
        public_key=verification.credential_public_key,
        sign_count=verification.sign_count,
    )
    await uow.webauthn_challenges.delete(stored.id)
    return {"status": "registered"}
```

The `py_webauthn` library handles the cryptographic details. The application code provides the configuration and the database storage.

### 3.3 The authentication ceremony

```python
@router.post("/auth/webauthn/login/options")
async def webauthn_login_options(payload: WebAuthnLoginOptions):
    user = await uow.users.get_by(email=payload.email)
    if not user:
        # Don't leak which emails are registered
        # Return a fake challenge
        return {"challenge": base64url_encode(secrets.token_bytes(32))}
    credentials = await uow.webauthn_credentials.list_for_user(user.id)
    challenge = secrets.token_bytes(32)
    options = generate_authentication_options(
        rp_id="example.com",
        challenge=challenge,
        allow_credentials=[{"id": c.credential_id, "type": "public-key"} for c in credentials],
    )
    await uow.webauthn_challenges.create(
        user_id=user.id,
        challenge=base64url_encode(challenge),
        purpose="authenticate",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    return options_to_json(options)


@router.post("/auth/webauthn/login/verify")
async def webauthn_login_verify(payload: WebAuthnLoginVerify):
    # 1) Decode the credential ID to find the user
    credential_id = base64url_decode(payload.credential_id)
    credential = await uow.webauthn_credentials.get_by(credential_id=credential_id)
    if not credential:
        raise HTTPException(401, "Unknown credential")
    user = await uow.users.get(credential.user_id)
    # 2) Look up the challenge
    stored = await uow.webauthn_challenges.get_for_user(user.id, purpose="authenticate")
    if not stored:
        raise HTTPException(400, "No authentication in progress")
    challenge_bytes = base64url_decode(stored.challenge)
    # 3) Verify the signature
    verification = verify_authentication_response(
        credential=payload.credential,
        expected_challenge=challenge_bytes,
        expected_origin="https://example.com",
        expected_rp_id="example.com",
        credential_public_key=credential.public_key,
        credential_current_sign_count=credential.sign_count,
    )
    # 4) Update the sign count (defense against cloned authenticators)
    credential.sign_count = verification.new_sign_count
    credential.last_used_at = datetime.now(timezone.utc)
    await uow.commit()
    # 5) Issue the access token
    membership = await uow.memberships.get_primary_for_user(user.id)
    token, _ = create_access_token(sub=user.id, tenant_id=membership.tenant_id, role=membership.role)
    return {"access_token": token}
```

The sign count must be monotonically increasing. If a cloned authenticator is used twice (once by the legitimate user, once by an attacker), the counts collide and the server detects the clone.

### 3.4 Passkeys: the discoverable credential

A "passkey" (Apple's term) or "discoverable credential" (the FIDO2 term) is a WebAuthn credential that:
- Is stored on a user's device and synced across the user's devices (iCloud Keychain, Google Password Manager, 1Password).
- Can be presented without first specifying a username (the browser shows a list of available passkeys).

The user experience is the famous "Touch ID to log in" with no username or password.

```python
# Discoverable credentials: do not pre-specify allow_credentials
options = generate_authentication_options(
    rp_id="example.com",
    challenge=challenge,
    # No allow_credentials parameter → discoverable
)
```

The browser enumerates the passkeys available for the relying party (your site) and lets the user pick.

---

## 4. Backup and Recovery

### 4.1 Backup strategies

Users lose devices. The recovery plan depends on the type of MFA:

| MFA type | Recovery | Trade-off |
|----------|----------|-----------|
| TOTP | Recovery codes | Code list can be lost; codes can be stolen |
| WebAuthn (synced) | Login on another synced device | Apple/Google hold the keys |
| WebAuthn (hardware key) | Second hardware key, recovery codes | Multiple keys = more friction |
| WebAuthn (platform) | Recovery codes + another WebAuthn method | Most users have one device |

The best practice: **always offer at least two methods**, with one being recoverable. A user with one TOTP and a set of recovery codes can recover. A user with only one hardware key is one lost key away from being locked out.

### 4.2 Admin-initiated reset

For users who lose all options, an admin can reset their MFA. The reset:
- Requires the admin to re-verify their own credentials (password + MFA).
- Is logged in the audit log.
- Sends a notification to the user's email and phone.
- Does NOT unlock the account; the user still needs to set up MFA again at next login.

```python
@router.post("/admin/users/{user_id}/mfa/reset")
async def admin_reset_mfa(
    user_id: int,
    admin_ctx: AdminContext = Depends(get_admin_context),
):
    user = await uow.users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    user.mfa_enabled = False
    user.mfa_secret_encrypted = None
    user.mfa_recovery_codes_hashed = []
    # Log it
    await log_admin_action(
        actor_user_id=admin_ctx.user.id,
        action="mfa_reset",
        affected_tenant_id=user.primary_tenant_id,
        details={"target_user_id": user_id},
    )
    # Notify the user
    await send_mfa_reset_notification(user.email)
    await uow.commit()
    return {"status": "reset"}
```

### 4.3 Account lockout prevention

Multiple consecutive failed MFA attempts should lock the account temporarily:

```python
@router.post("/auth/mfa/check")
async def mfa_check(payload: MFARequest):
    user_id = verify_mfa_challenge_token(payload.mfa_token)
    user = await uow.users.get(user_id)
    # Check the failed-attempt counter
    if user.mfa_failed_attempts >= 5:
        if user.mfa_lockout_until and user.mfa_lockout_until > datetime.now(timezone.utc):
            raise HTTPException(429, "Account temporarily locked. Try again later.")
        # Lockout expired; reset
        user.mfa_failed_attempts = 0
        user.mfa_lockout_until = None
    # Verify the code
    secret = decrypt(user.mfa_secret_encrypted)
    if not verify_totp(secret, payload.code):
        user.mfa_failed_attempts += 1
        if user.mfa_failed_attempts >= 5:
            user.mfa_lockout_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        await uow.commit()
        raise HTTPException(401, "Invalid code")
    # Success
    user.mfa_failed_attempts = 0
    user.mfa_lockout_until = None
    await uow.commit()
    # Issue token
    token, _ = create_access_token(sub=user.id, tenant_id=user.tenant_id, role=user.role)
    return {"access_token": token}
```

The lockout is exponential: 5 fails → 15 min, 10 fails → 1 hour, 20 fails → 24 hours. Without a lockout, an attacker can brute-force a 6-digit code (1M possibilities) in a few hours.

---

## 5. The Seven MFA Mistakes

### 5.1 Sending TOTP codes by SMS as a fallback

SMS-based 2FA is better than nothing but vulnerable to SIM swapping, SS7 attacks, and phishing. **Never use SMS as the primary second factor.** If you need a fallback, use TOTP (in an authenticator app) or WebAuthn.

### 5.2 Storing TOTP secrets in plaintext

The TOTP secret is as sensitive as a password. If the database is compromised, the attacker has the keys to bypass MFA for every user. **Encrypt the secret at rest** (Fernet, KMS, or `cryptography.fernet`).

### 5.3 Skipping rate limiting on MFA

A 6-digit TOTP code has 1M possibilities. Without rate limiting, an attacker can brute-force in 1-2 hours. **Always rate limit** the MFA check (5 attempts, then lockout).

### 5.4 Allowing MFA bypass for "convenience"

Some apps let users mark devices as "trusted" and skip MFA on subsequent logins. This is convenient but creates a hole: a stolen device is permanent access. The trade-off is real; if you implement it, bound the trust to a specific device fingerprint and re-verify on suspicious activity.

### 5.5 Not invalidating recovery codes on use

A recovery code is single-use. If you don't remove it from the user's account after use, the attacker who stole it can use it repeatedly. The code is consumed; the user is told how many remain.

### 5.6 Storing WebAuthn challenges too long

The challenge must expire (5 minutes is standard). A long-lived challenge window is an attack surface: a stolen challenge + private key is enough to authenticate.

### 5.7 Not updating the sign count

The WebAuthn sign count detects cloned authenticators. If you don't update it, you lose the protection. A cloned key can be used indefinitely.

---

## 6. Testing MFA

### 6.1 TOTP tests

```python
import pyotp
import time


def test_totp_secret_is_valid():
    secret = generate_totp_secret()
    totp = pyotp.TOTP(secret)
    code = totp.now()
    assert totp.verify(code, valid_window=1)


def test_recovery_code_is_hashed():
    code = "A3F2C9B1E0"
    hashed = hash_recovery_code(code)
    assert hashed != code  # not plaintext
    assert hash_recovery_code(code) == hashed  # deterministic


def test_recovery_code_is_consumed():
    user.mfa_recovery_codes_hashed = [hash_recovery_code("A3F2C9B1E0")]
    # Verify and consume
    assert hash_recovery_code("A3F2C9B1E0") in user.mfa_recovery_codes_hashed
    user.mfa_recovery_codes_hashed.remove(hash_recovery_code("A3F2C9B1E0"))
    assert hash_recovery_code("A3F2C9B1E0") not in user.mfa_recovery_codes_hashed
```

### 6.2 WebAuthn tests

WebAuthn testing is tricky because the ceremonies require a real authenticator. Use the `py_webauthn` test fixtures or a virtual authenticator (Chrome DevTools has one):

```python
def test_webauthn_registration_creates_credential():
    # Use a virtual authenticator to simulate the device
    options = generate_registration_options(...)
    # Simulate the authenticator's response
    credential = simulate_authenticator_response(options, ...)
    # Verify
    verification = verify_registration_response(credential, ...)
    assert verification.verified
```

For end-to-end tests, use a real device or a CI tool that emulates WebAuthn.

---

## 7. Código de Compresión

```python
"""
Compresión: MFA, Passkeys, and WebAuthn
Covers: TOTP setup, recovery codes, WebAuthn ceremonies, lockout.
"""
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
import pyotp
import qrcode
from io import BytesIO
from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
    options_to_json,
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier


# 1) TOTP setup
def generate_totp_secret() -> str:
    return pyotp.random_base32()


def get_totp_uri(secret: str, email: str, issuer: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name=issuer)


def generate_qr_code(uri: str) -> bytes:
    img = qrcode.make(uri)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def verify_totp(secret: str, code: str) -> bool:
    return pyotp.TOTP(secret).verify(code, valid_window=1)


# 2) Recovery codes
def generate_recovery_codes(n: int = 10) -> list[str]:
    return [secrets.token_hex(5).upper() for _ in range(n)]


def hash_recovery_code(code: str) -> str:
    return hashlib.sha256(code.upper().encode()).hexdigest()


# 3) WebAuthn registration
def make_registration_options(user_id: int, email: str) -> dict:
    return generate_registration_options(
        rp_id="example.com",
        rp_name="MyApp",
        user_id=str(user_id).encode(),
        user_name=email,
        user_display_name=email,
        challenge=secrets.token_bytes(32),
        supported_pub_key_algs=[COSEAlgorithmIdentifier.ECDSA_SHA_256],
    )


def verify_registration(credential: dict, challenge: bytes) -> dict:
    return verify_registration_response(
        credential=credential,
        expected_challenge=challenge,
        expected_origin="https://example.com",
        expected_rp_id="example.com",
    )


# 4) WebAuthn authentication
def make_authentication_options(credentials: list) -> dict:
    return generate_authentication_options(
        rp_id="example.com",
        challenge=secrets.token_bytes(32),
        allow_credentials=[{"id": c.credential_id, "type": "public-key"} for c in credentials],
    )


def verify_authentication(
    credential: dict, challenge: bytes, public_key: bytes, current_sign_count: int
) -> dict:
    return verify_authentication_response(
        credential=credential,
        expected_challenge=challenge,
        expected_origin="https://example.com",
        expected_rp_id="example.com",
        credential_public_key=public_key,
        credential_current_sign_count=current_sign_count,
    )


# 5) Lockout
class LockoutPolicy:
    MAX_ATTEMPTS = 5
    LOCKOUT_DURATIONS = [  # exponential
        timedelta(minutes=15),
        timedelta(hours=1),
        timedelta(hours=24),
    ]

    def record_failure(self, user) -> None:
        user.mfa_failed_attempts += 1
        level = min(user.mfa_failed_attempts // self.MAX_ATTEMPTS, len(self.LOCKOUT_DURATIONS) - 1)
        if user.mfa_failed_attempts % self.MAX_ATTEMPTS == 0:
            user.mfa_lockout_until = datetime.now(timezone.utc) + self.LOCKOUT_DURATIONS[level]

    def is_locked(self, user) -> bool:
        if not user.mfa_lockout_until:
            return False
        if user.mfa_lockout_until > datetime.now(timezone.utc):
            return True
        # Expired; clear
        user.mfa_failed_attempts = 0
        user.mfa_lockout_until = None
        return False
```

---

## Key Takeaways

- MFA reduces account takeover by 99.9%. The cost of not having it is data breaches; the cost of having it is a slightly slower login.
- TOTP is the standard "authenticator app" factor. The secret must be stored encrypted; the code verification must be rate-limited; the recovery codes must be hashed and consumed on use.
- WebAuthn is the modern standard: phishing-resistant, no codes to type, no shared secret to leak. Passkeys (synced WebAuthn credentials) are the consumer-friendly evolution.
- Always offer at least two recovery methods: recovery codes, a second WebAuthn credential, an admin reset, or another device. Users lose devices.
- The WebAuthn sign count detects cloned authenticators. Always update it on every authentication.
- Never use SMS as the primary second factor. SIM swapping and SS7 attacks make it insecure.
- Rate limiting is mandatory for the MFA check. A 6-digit code has 1M possibilities; without rate limiting, brute force is feasible.
- MFA bypass for "trusted devices" is a real risk. If you implement it, bind to a device fingerprint and re-verify on suspicious activity.
- Admin-initiated MFA reset must require the admin's own MFA, log the action, and notify the user. A "silent" reset is a security hole.

## References

- [RFC 6238 — TOTP](https://www.rfc-editor.org/rfc/rfc6238)
- [W3C WebAuthn Level 2](https://www.w3.org/TR/webauthn-2/)
- [FIDO Alliance — Passkeys](https://fidoalliance.org/passkeys/)
- [pyotp Documentation](https://pyauth.github.io/pyotp/)
- [py_webauthn](https://github.com/duo-labs/py_webauthn)
- [OWASP MFA Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Multifactor_Authentication_Cheat_Sheet.html)
- [NIST SP 800-63B — Authentication and Lifecycle Management](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [Microsoft — One Simple Action You Can Take to Prevent 99.9% of Account Attacks](https://www.microsoft.com/security/blog/2019/08/20/one-simple-action-you-can-take-to-prevent-99-9-percent-of-account-attacks/)
