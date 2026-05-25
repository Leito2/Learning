# 🔐 Rust Cryptography and Security

## Introduction

Rust's memory safety and lack of undefined behavior make it an excellent choice for cryptographic code, where bugs can have catastrophic security consequences. The ecosystem offers high‑quality, audited crates for symmetric/asymmetric encryption, hashing, digital signatures, and transport‑layer security. By leveraging Rust's type system and ownership model, these libraries avoid common pitfalls like buffer overflows, side‑channel leaks, and use‑after‑free errors.

The Rust cryptography ecosystem is dominated by a few key crates: `ring` (a safe wrapper around BoringSSL), `rustls` (a modern TLS library), and pure‑Rust implementations like `ed25519‑dalek` and `x25519‑dalek`. Unlike C‑based libraries (e.g., OpenSSL), these crates are memory‑safe by construction and often provide constant‑time operations to mitigate timing attacks.

This deep dive explores the architecture of major crypto crates, compares them with traditional alternatives, and shows how to build a secure TLS server/client stack. We’ll also examine real‑world deployments, such as Cloudflare’s migration from OpenSSL to `rustls`, and highlight security pitfalls to avoid.

## 1. Crypto Crates Ecosystem

The Rust crypto ecosystem is organized around the RustCrypto project, but several standalone crates stand out for performance and safety.

| Crate | Type | Memory Safety | Performance | Notes |
|-------|------|---------------|-------------|-------|
| `ring` | Symmetric, AEAD, signing | Safe (C bindings) | High (assembly‑optimized) | Based on BoringSSL, limited algorithm set |
| `rustls` | TLS 1.2/1.3 | Pure Rust | High | No C dependencies, safe by default |
| `ed25519‑dalek` | EdDSA signatures | Pure Rust | Medium‑High | RFC 8032 compliant |
| `x25519‑dalek` | Elliptic‑curve Diffie‑Hellman | Pure Rust | Medium‑High | Curve25519 implementation |
| `sodiumoxide` | Various (libsodium bindings) | Safe (C bindings) | High | Comprehensive but C dependency |
| `openssl` | TLS, crypto (C bindings) | Unsafe (C bindings) | High | Full OpenSSL API, but unsafe |

**Real case:** Signal uses `ed25519‑dalek` for its key‑exchange protocol, benefiting from pure Rust and constant‑time operations.

⚠️ **Warning:** Using `unsafe` to call C crypto libraries (e.g., OpenSSL directly) can introduce memory safety bugs. Always use safe wrappers like `rustls` or `ring`.

💡 **Tip:** For maximum security, prefer pure Rust implementations (dalek crates) over C bindings. They are easier to audit and have no hidden side effects.

## 2. TLS: rustls vs. OpenSSL Bindings

TLS (Transport Layer Security) is the backbone of secure internet communication. Rust offers two main approaches:

### rustls

- **Pure Rust**: No C code, no unsafe.
- **Memory‑safe**: Buffer overflows are impossible.
- **Modern**: Supports TLS 1.3, ALPN, SNI, and certificate pinning.
- **Performance**: Comparable to OpenSSL, often faster for small records.

### OpenSSL Bindings (e.g., `openssl` crate)

- **Unsafe**: Relies on C code, which may have bugs.
- **Feature‑rich**: Supports legacy protocols and algorithms.
- **Binding overhead**: Requires careful memory management.

**Real case:** Cloudflare switched its edge proxy from OpenSSL to `rustls` in 2019, reducing memory usage and eliminating a class of vulnerabilities. The switch improved TLS handshake performance by 10‑20%.

⚠️ **Warning:** `rustls` does not support older TLS 1.0/1.1 or certain legacy cipher suites. Ensure client compatibility before switching.

💡 **Tip:** Use `rustls` for new projects; it's safer and simpler. Only fall back to OpenSSL bindings if you need specific legacy features.

## 3. Signatures, Hashing, and AEAD

Modern cryptography relies on three primitives:

### Digital Signatures

- **Ed25519** (`ed25519‑dalek`): Schnorr‑based, fast, small keys.
- **ECDSA** (`ecdsa` crate): NIST standards, widely supported.
- **RSA** (`rsa` crate): Legacy, slow, large keys.

### Hashing

- **SHA‑2** (`sha2` crate): SHA‑256, SHA‑512.
- **SHA‑3** (`sha3` crate): Keccak‑based, resistant to length‑extension attacks.
- **BLAKE3** (`blake3` crate): Parallel, faster than SHA‑256.

### Authenticated Encryption with Associated Data (AEAD)

- **AES‑GCM** (`aes‑gcm` crate): NIST standard, hardware acceleration.
- **ChaCha20‑Poly1305** (`chacha20‑poly1305` crate): Pure Rust, constant‑time, mobile‑friendly.

**Real case:** The Matrix protocol uses `ed25519‑dalek` for device signing and `olm` (C library) for end‑to‑end encryption, but newer implementations are moving to pure Rust.

Formula: `Security_Strength = min(Hash_Output_Size, Key_Size) / 2` (for collision resistance).

## 4. Building a TLS Server and Client

Let's create a simple TLS server and client using `rustls` and `rcgen` for certificate generation.

```rust
// tls_example.rs
use rustls::ServerConfig;
use rustls::client::ServerCertVerifier;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use std::net::TcpListener;
use std::thread;

fn generate_self_signed_cert() -> (rustls::Certificate, rustls::PrivateKey) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let key = rustls::PrivateKey(cert.serialize_private_key_der());
    let cert = rustls::Certificate(cert.serialize_der().unwrap());
    (cert, key)
}

fn start_server() {
    let (cert, key) = generate_self_signed_cert();
    let config = ServerConfig::builder()
        .with_safe_default_cipher_suites()
        .with_safe_default_kx_groups()
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_single_cert(vec![cert], key)
        .expect("bad certificate/key");
    let config = Arc::new(config);
    let listener = TcpListener::bind("127.0.0.1:8443").unwrap();
    for stream in listener.incoming() {
        let stream = stream.unwrap();
        let config = config.clone();
        thread::spawn(move || {
            let mut conn = rustls::ServerConnection::new(config).unwrap();
            let mut rustls_stream = rustls::Stream::new(&mut conn, &mut &stream, &mut &stream);
            // Echo server
            let mut buf = [0; 1024];
            let n = rustls_stream.read(&mut buf).unwrap();
            rustls_stream.write_all(&buf[..n]).unwrap();
        });
    }
}

fn client_connect() {
    let mut config = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(NoCertificateVerification))
        .with_no_client_auth();
    let mut roots = rustls::RootCertStore::empty();
    // In real code, add server's certificate to roots
    let (cert, _) = generate_self_signed_cert();
    roots.add(&cert).unwrap();
    config.root_store = roots;
    let config = Arc::new(config);
    let server_name = rustls::ServerName::try_from("localhost").unwrap();
    let mut conn = rustls::ClientConnection::new(config, server_name).unwrap();
    let mut sock = std::net::TcpStream::connect("127.0.0.1:8443").unwrap();
    let mut tls = rustls::Stream::new(&mut conn, &mut sock, &mut sock);
    tls.write_all(b"Hello TLS!").unwrap();
    let mut buf = Vec::new();
    tls.read_to_end(&mut buf).unwrap();
    println!("Server echoed: {:?}", String::from_utf8_lossy(&buf));
}

struct NoCertificateVerification;

impl ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::Certificate,
        intermediates: &[rustls::Certificate],
        server_name: &rustls::ServerName,
        scts: &mut dyn Iterator<Item = &[u8]>,
        ocsp_response: &[u8],
        now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

fn main() {
    thread::spawn(|| start_server());
    std::thread::sleep(std::time::Duration::from_millis(100));
    client_connect();
}
```

---

## 📦 Compression Code

Complete Rust script implementing a secure key exchange and encryption using `x25519‑dalek` and `chacha20‑poly1305`.

```rust
// key_exchange.rs
use x25519_dalek::{EphemeralSecret, PublicKey};
use chacha20_poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20_poly1305::aead::{Aead, NewAead};
use rand::rngs::OsRng;

fn main() {
    // Generate ephemeral keys
    let alice_secret = EphemeralSecret::new(OsRng);
    let alice_public = PublicKey::from(&alice_secret);
    let bob_secret = EphemeralSecret::new(OsRng);
    let bob_public = PublicKey::from(&bob_secret);

    // Perform Diffie‑Hellman
    let shared_alice = alice_secret.diffie_hellman(&bob_public);
    let shared_bob = bob_secret.diffie_hellman(&alice_public);
    assert_eq!(shared_alice.as_bytes(), shared_bob.as_bytes());

    // Derive symmetric key
    let key = Key::from_slice(shared_alice.as_bytes());
    let cipher = ChaCha20Poly1305::new(key);
    let nonce = Nonce::from_slice(b"unique nonce"); // In practice, use random nonce

    // Encrypt
    let plaintext = b"Secret message";
    let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).expect("encryption failed");
    println!("Ciphertext: {:?}", ciphertext);

    // Decrypt
    let decrypted = cipher.decrypt(nonce, ciphertext.as_ref()).expect("decryption failed");
    println!("Decrypted: {:?}", String::from_utf8_lossy(&decrypted));
}
```

## 🎯 Documented Project

### Description

A secure, zero‑copy chat server using TLS 1.3 for transport encryption and Ed25519 for message authentication.

### Functional Requirements

1. TLS 1.3 with perfect forward secrecy.
2. Mutual authentication using client certificates.
3. Message signing with Ed25519 to prevent tampering.
4. Zero‑copy parsing of binary protocol messages.
5. Support for 10,000 concurrent connections.

### Main Components

- **TLS Layer**: `rustls` with custom certificate verifier.
- **Auth Module**: `ed25519‑dalek` for key generation and signing.
- **Protocol**: Binary format with length‑prefixed fields.
- **Connection Pool**: `tokio::sync::Semaphore` for backpressure.
- **Metrics**: Prometheus exporter for handshake latency.

### Success Metrics

- TLS handshake < 5 ms (99th percentile).
- Message throughput > 100,000 msgs/sec.
- Zero memory safety bugs (verified via Miri and fuzzing).
- No data corruption after 24‑hour soak test.

### References

- RustCrypto project: [[03 - Rust Cryptography and Security|crypto crates]]
- TLS 1.3 RFC 8446
- Ed25519 RFC 8032
- Cloudflare's rustls adoption blog post