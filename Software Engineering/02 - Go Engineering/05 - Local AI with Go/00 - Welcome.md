# 🤖 Welcome to Local AI with Go

## What will you learn?

This course teaches you how to build production-ready AI applications entirely in Go, running Large Language Models (LLMs) locally without depending on Python or cloud APIs. You will learn to architect, integrate, and deploy intelligent systems from chatbots to edge devices.

- **Local LLM Inference:** Understand how to run [[Llama 3]], [[Mistral]], and [[CodeLlama]] on your own hardware using [[Ollama]].
- **Go SDK Integration:** Build robust HTTP clients, streaming parsers, and wrappers for Ollama's REST API.
- **Conversational AI:** Implement stateful chatbots with message history, context window management, and Server-Sent Events (SSE).
- **Retrieval-Augmented Generation (RAG):** Construct end-to-end RAG pipelines in Go using vector databases like [[Qdrant]] and [[Weaviate]].
- **Desktop Applications:** Create cross-platform desktop AI assistants using [[Wails]] with Go backends.
- **Edge Deployment:** Deploy quantized models to ARM devices like Raspberry Pi using Go and ONNX Runtime.

## Course Structure

- [[01 - Running LLMs Locally with Ollama|🦙 01 - Ollama]]
- [[02 - Ollama Go SDK and API Integration|🔌 02 - Ollama SDK]]
- [[03 - Building Chatbots with Go + LLMs|💬 03 - Chatbots]]
- [[04 - RAG Pipelines with Go and Vector DBs|🔍 04 - RAG]]
- [[05 - Desktop Apps with Wails|🖥️ 05 - Wails]]
- [[06 - Edge AI Deployment with Go|📱 06 - Edge AI]]

## Capstone Project

By the end of this course, you will build a fully functional **Local AI Knowledge Assistant**: a Wails desktop application that connects to a local Ollama instance, maintains conversational history, retrieves context from a local vector database (Qdrant), and streams responses in real time. This project demonstrates the complete stack: local inference, Go backend architecture, RAG pipelines, and desktop deployment.

---

💡 **Tip:** Ollama + Go is the fastest way to build production-ready local AI tools. No Python environment needed for inference.
