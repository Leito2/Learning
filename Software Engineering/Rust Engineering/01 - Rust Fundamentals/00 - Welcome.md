# 👋 Welcome to Rust Fundamentals

## What will you learn?

This course is your comprehensive journey into the heart of Rust — a systems programming language that guarantees memory safety and thread safety without a garbage collector. You will learn how Rust's unique ownership model eliminates entire classes of bugs at compile time, how to build robust applications using Cargo and the crate ecosystem, and how to leverage zero-cost abstractions for high-performance code.

By the end of this course, you will be able to:

- Reason about memory management using [[01 - Ownership, Borrowing, and Lifetimes|ownership and lifetimes]] rather than relying on a garbage collector
- Design expressive APIs with [[02 - Types, Traits, and Generics|traits and generics]] that are both flexible and performant
- Structure large projects using [[03 - Cargo, Crates, and the Module System|Cargo workspaces and modules]]
- Handle failures gracefully with [[04 - Error Handling and Pattern Matching|Result, Option, and pattern matching]]
- Process data efficiently with [[05 - Collections and Iterators|collections and zero-cost iterators]]
- Model complex domains with [[06 - Structs, Enums, and Advanced Patterns|enums, structs, and advanced patterns]]

## Course Structure

- [[01 - Ownership, Borrowing, and Lifetimes|🔒 01 - Ownership]]
- [[02 - Types, Traits, and Generics|📐 02 - Types and Traits]]
- [[03 - Cargo, Crates, and the Module System|📦 03 - Cargo]]
- [[04 - Error Handling and Pattern Matching|⚠️ 04 - Error Handling]]
- [[05 - Collections and Iterators|🗃️ 05 - Collections]]
- [[06 - Structs, Enums, and Advanced Patterns|🏗️ 06 - Structs and Enums]]

## Capstone Project

Throughout this course, you will build a **Safe File Indexer** — a command-line tool that recursively scans directories, indexes file metadata, and provides fast queries. Each module of the course contributes a layer to this project: ownership ensures safe buffer handling, traits enable pluggable storage backends, modules organize the codebase, error handling manages I/O failures, collections power the index, and enums model the different file types.

By the final module, you will have a production-ready Rust application that demonstrates every fundamental concept in a real-world context.

---

💡 **Tip:** The borrow checker is your friend, not your enemy. Every error it catches is a potential segfault or data race prevented at compile time.
