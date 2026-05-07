# 🛡️ Welcome to DevSecOps and CLI Tools

## What will you learn?

This course teaches you how to build secure, automated, and production-ready Go applications using modern DevSecOps practices and CLI tooling. You will learn to:

- Design and implement powerful command-line interfaces using the Cobra framework
- Integrate security scanning, dependency checks, and secret detection into your workflow
- Build CI/CD pipelines tailored for Go projects with caching, matrix builds, and artifact management
- Automate releases, testing, and deployments using GitHub Actions and GoReleaser
- Harden container images and apply runtime security policies for Go microservices
- Assemble a complete DevOps observability toolkit with Prometheus, Grafana, and alerting pipelines

By the end of this course, you will be able to ship Go binaries and containers that are fast, secure, and fully observable.

## Course Structure

- [[01 - Building CLIs with Cobra|⌨️ 01 - CLIs with Cobra]]
- [[02 - Security Scanning and Hardening|🔒 02 - Security Scanning]]
- [[03 - CI-CD Pipelines for Go Projects|🔄 03 - CI-CD Pipelines]]
- [[04 - GitHub Actions and Automation|🤖 04 - GitHub Actions]]
- [[05 - Container Security with Go|🐳 05 - Container Security]]
- [[06 - Building a DevOps Toolkit|🧰 06 - DevOps Toolkit]]

## Capstone Project

You will build a production-grade DevSecOps CLI tool called `secops-cli` — a single Go binary that can scan Dockerfiles for security issues, run SAST checks on Go source code, generate CI/CD YAML templates, and export Prometheus metrics about scan results. The tool will be distributed via GitHub Releases using GoReleaser, packaged in a distroless container, and deployed through an automated GitHub Actions pipeline.

---

💡 **Tip:** Go compiles to a single binary. This makes CLI tools and DevOps utilities incredibly portable — just copy the binary and run.
