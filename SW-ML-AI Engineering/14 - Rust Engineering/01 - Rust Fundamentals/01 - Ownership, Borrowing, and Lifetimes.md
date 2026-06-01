# 🦀 1 - Ownership, Borrowing, and Lifetimes

## 🎯 Learning Objectives

- Explain how Rust's ownership system guarantees memory safety without a garbage collector
- Distinguish between moves, copies, clones, and borrows in Rust code
- Use immutable and mutable references correctly according to the borrowing rules
- Annotate lifetimes explicitly when elision rules are insufficient

## Introduction

Rust's ownership system is the language's most distinctive feature. Unlike C++ where memory safety is the programmer's responsibility, or Java and Go which rely on garbage collectors, Rust enforces memory safety at compile time through ownership rules. This eliminates dangling pointers, double frees, and data races without runtime overhead. For systems engineers, this means you can build data loaders that process terabyte-scale datasets without segfaults, and inference engines with deterministic latency because there is no GC to pause execution.

Every value in Rust has a single owner. When that owner goes out of scope, the value is automatically dropped. This seemingly simple rule has profound implications for how you structure programs, pass data between functions, and design APIs. This connects directly with [[02 - Types, Traits, and Generics|type-safe abstractions]] and [[03 - Cargo, Crates, and the Module System|project architecture]].

---

## 1. Ownership Rules

**Three rules:**
1. Each value has exactly one owner at a time.
2. When the owner goes out of scope, the value is dropped.
3. Ownership can be moved to a new owner.

```rust
fn main() {
    // String is heap-allocated — does NOT implement Copy.
    let s1 = String::from("hello");

    // Ownership moves from s1 to s2. Heap data is NOT duplicated.
    let s2 = s1;

    // ❌ ERROR: borrow of moved value
    // println!("{}", s1);

    // ✅ OK: s2 owns the string
    println!("{}", s2);

    // Ownership moves into the function; s2 is invalid after this call
    takes_ownership(s2);

    // i32 implements Copy — bitwise duplication is safe
    let x = 5;
    let y = x;
    println!("x = {}, y = {}", x, y); // Both valid
}

fn takes_ownership(s: String) {
    println!("{}", s);
} // s dropped here; heap memory freed
```

⚠️ **Copy types** (integers, bools, floats, chars, tuples of Copy types) are duplicated on assignment. **Move types** (String, Vec, Box) transfer ownership — the source becomes invalid.

💡 Use `.clone()` explicitly when you need two independent copies. Writing `.clone()` makes the cost visible in your code:

```rust
let s1 = String::from("hello");
let s2 = s1.clone(); // ✅ Explicit deep copy; both are valid
println!("{} {}", s1, s2);
```

**Caso real:** An ML data loader reading a 4GB feature matrix must free it exactly once. Rust's ownership guarantees this at compile time. Double-free is impossible in safe Rust. If you pass the matrix to a worker thread, ownership moves — the original thread can no longer access it, preventing use-after-free races.

---

## 2. Borrowing and References

Borrowing provides temporary access without transferring ownership. The core principle: **aliasing XOR mutation** — any number of immutable references (`&T`) XOR exactly one mutable reference (`&mut T`).

```rust
fn main() {
    let mut s = String::from("hello");

    // Multiple immutable borrows are fine
    let r1 = &s;
    let r2 = &s;
    println!("{} {}", r1, r2);

    // r1 and r2 are no longer used, so mutable borrow is allowed
    let r3 = &mut s;
    r3.push_str(" world");
    println!("{}", r3);

    // ✅ After r3's last use, new immutable borrows are fine
    let r4 = &s;
    println!("{}", r4);

    // ❌ ERROR: cannot borrow `s` as mutable — it's already borrowed as immutable
    // let r5 = &mut s;
    // r5.push_str("!!");
    // println!("{}", r1);
}
```

**Reference rules:**
- At any moment, you have **either** one mutable reference **or** any number of immutable references.
- References must always be valid (never dangle).

```rust
// ❌ Dangling reference — rejected at compile time
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s   // s dropped here, reference would dangle
// }

// ✅ Fix: return the owned value
fn no_dangle() -> String {
    String::from("hello")
}
```

⚠️ Non-Lexical Lifetimes (NLL) allow the compiler to be smart about when borrows end. A borrow's scope ends at its *last use*, not at the closing brace. This enables patterns like:

```rust
let mut data = vec![1, 2, 3];
let r = &data[0];
println!("{r}");    // last use of r
data.push(4);       // ✅ OK: r is no longer alive
```

💡 When designing APIs, prefer borrowing over cloning. `fn process(data: &[f32])` tells callers the function only reads the data without taking ownership.

**Caso real:** A tokenizer processing a large corpus uses `&str` slices into the input text. Multiple tokenizer workers can read the same buffer concurrently via immutable references. A preprocessing step that needs to normalize text uses `&mut` exclusively, and the borrow checker prevents overlapping mutation.

---

## 3. Lifetimes

Lifetimes are Rust's way of tracking how long references are valid. Every reference has a lifetime — the span during which it is guaranteed to point to valid data. The compiler infers lifetimes via **elision rules** most of the time; explicit annotation is needed when the output lifetime cannot be deduced from inputs.

```rust
// 'a ties the output lifetime to the shorter of the two inputs
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Struct holding references must declare lifetimes
struct Book<'a> {
    title: &'a str,
    author: &'a str,
}

impl<'a> Book<'a> {
    fn new(title: &'a str, author: &'a str) -> Self {
        Book { title, author }
    }

    fn description(&self) -> String {
        format!("{} by {}", self.title, self.author)
    }
}

// 'static lives for the entire program — string literals are 'static
const GREETING: &'static str = "Hello, world!";

fn main() {
    let title = "The Rust Programming Language";
    let author = "Steve Klabnik and Carol Nichols";
    let book = Book::new(title, author); // book cannot outlive title/author
    println!("{}", book.description());

    let a = "short";
    let b = "looooooooong";
    println!("Longest: {}", longest(a, b));
}
```

⚠️ **Don't overuse `'static`.** Marking a reference as `'static` claims it lives forever, which is only true for compile-time constants. For dynamic data, tie the lifetime to a specific scope.

❌ **Antipattern:** Fighting the borrow checker by converting everything to owned types (`String` instead of `&str`). This defeats zero-copy borrowing.

✅ **Preferred:** Start with owned data and introduce references after the architecture is stable. It's easier to optimize toward borrows than to debug lifetime errors in a tangled design.

💡 **Lifetime elision rules:**
1. Each input reference gets its own lifetime parameter.
2. If there's exactly one input lifetime, it's assigned to all output references.
3. If `&self` or `&mut self` is an input, its lifetime is assigned to all output references.

**Caso real:** A CSV parser uses lifetimes to ensure parsed records cannot outlive the input buffer they reference. The `Record<'a>` struct carries the buffer's lifetime, so the compiler rejects any code that attempts to use a record after freeing the buffer — exactly the class of bug that causes use-after-free in C parsers.

---

## 🎯 Key Takeaways

- Every value has exactly one owner; ownership moves on assignment for non-Copy types
- Borrowing follows the XOR rule: many readers OR one writer, never both
- Lifetimes prevent dangling references; the compiler enforces them at compile time
- Prefer `&T` for read-only access, `&mut T` for exclusive write, and owned values for long-term storage
- Start with owned types, then introduce borrows for optimization

## References

- [The Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Rust By Example - Lifetimes](https://doc.rust-lang.org/rust-by-example/scope/lifetime.html)
- [[02 - Types, Traits, and Generics]]
- [[03 - Cargo, Crates, and the Module System]]

## 📦 Código de compresión

```rust
// Run-length encoding with ownership, borrowing, and lifetimes
use std::fs;

fn main() {
    let content = fs::read_to_string("input.txt").expect("File not found");

    // borrows content immutably
    let compressed = compress(&content);
    println!("Original: {} bytes", content.len());
    println!("Compressed: {} bytes", compressed.len());

    // parser holds a reference with lifetime tied to content
    let mut parser = WordParser::new(&content);
    while let Some(word) = parser.next_word() {
        println!("Word: {}", word);
    }
}

fn compress(input: &str) -> Vec<u8> {
    let mut result = Vec::new();
    let bytes = input.as_bytes();
    if bytes.is_empty() { return result; }
    let mut current = bytes[0];
    let mut count = 1u8;
    for &byte in &bytes[1..] {
        if byte == current && count < 255 { count += 1; }
        else {
            result.push(current); result.push(count);
            current = byte; count = 1;
        }
    }
    result.push(current); result.push(count);
    result
}

struct WordParser<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> WordParser<'a> {
    fn new(text: &'a str) -> Self { WordParser { text, pos: 0 } }
    fn next_word(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while self.pos < self.text.len() && !self.text[self.pos..].starts_with(' ') {
            self.pos += 1;
        }
        if start == self.pos { return None; }
        let word = &self.text[start..self.pos];
        self.pos += 1; // skip space
        Some(word)
    }
}
```
