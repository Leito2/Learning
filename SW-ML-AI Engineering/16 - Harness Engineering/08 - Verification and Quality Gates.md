# 08 — Verification and Quality Gates

## 🎯 Learning Objectives

- Distinguish between human-in-the-loop gates and automated verification gates.
- Explain the TDD Harness pattern: red, green, triangulate.
- Apply the Review Workload Harness to decompose oversized changes.
- Verify implemented code against specification documents mechanically.
- Place quality gates at the correct phase based on failure mode analysis.
- Interpret skill resolution feedback as a harness quality telemetry signal.

---

## Introduction

A harness guarantees that agents produce output, but output alone is not quality. Without verification, an agent can generate working code that solves the wrong problem, passes tests but ignores the spec, or ships a 2000-line diff that no human can review. Quality gates solve this.

The key insight: verification must be automated and integrated into the harness, not performed as a separate step by a human after the fact. A post-hoc review is too late — the code exists, the context is stale, and the cost of rework is high. A gate that blocks the phase before it completes forces the agent to correct course immediately, when the context is still fresh in the conversation window.

The verification layer is the quality assurance department of the harness. It sits between production and persistence. Every artifact — specification, design, code, test — must pass through a gate before the harness advances to the next phase. Some gates are human: slow, expensive, high-leverage. Most are automated: fast, mechanical, repeatable. The art is knowing which kind to put where.

If [[06 - Multi-Agent Orchestration and Capstone]] is the control structure and [[04 - Specification-Driven Development]] is the protocol, verification is the police officer at the intersection. It does not drive the car, but it decides whether traffic flows.

The [[07 - Complete Harness Taxonomy]] classifies twelve sub-harnesses. Verification is not one of them — it is a cross-cutting concern that applies to many. The TDD Harness, the Review Workload Harness, and the Verification Against Spec Harness all live in this layer. They are independent mechanisms with a shared purpose: ensure the output is worth keeping before the harness persists it.

Trust in an agent system is built gate by gate. Every passing check increases confidence. Every missing check introduces uncertainty. The goal of this layer is to make the certainty measurable and the gates explicit.

---

## 1. Two Kinds of Gates: Human and Automated

Quality gates divide into two categories, and confusing them is the most common design mistake in agent harnesses.

### Human Gates

Human gates require a person to read, assess, and approve. They are:

- **Slow** — minutes to hours, not milliseconds.
- **High-leverage** — catching a mistake here prevents whole cascades of downstream waste.
- **Expensive** — they consume human attention, the scarcest resource in the system.

In the SDD approach, there are exactly **two human gates**: spec approval and design approval. Everything else is automated. This ratio — 2 human, N automated — is not arbitrary. It reflects the leverage curve: catching a wrong requirement at spec time costs one conversation. Catching it at deployment costs a rollback, a hotfix, and a post-mortem.

### Automated Gates

Automated gates run without human intervention. They are:

- **Fast** — milliseconds to seconds.
- **Mechanical** — they check what they are told to check, nothing more.
- **Cheap** — marginal cost of one more check approaches zero.

Examples: linter passes, test suite green, spec coverage verified, diff size within threshold, all tasks marked complete. Each automated gate encodes a policy decision: *this condition must hold before the next phase starts.*

Consider what happens when a gate is missing. No lint check: the codebase accumulates inconsistent style, dead imports, and unused variables across agent runs. No diff size check: a single agent session produces a 1500-line change that takes a human reviewer an entire afternoon to read. No spec coverage check: the agent builds something clever that the user never asked for. Each missing gate creates a failure mode that the harness cannot detect until it reaches a human — at which point the cost of fixing it is already high.

### Why This Ratio Holds

More than two human gates creates a bottleneck. If every phase requires a human sign-off, the harness cannot run unattended — it becomes a ticket system with AI drafting. Fewer automated gates creates blind deployment: the harness ships code whose quality is unknown.

The ratio is stable because of the asymmetry of error cost:

| Error type | Caught by | Cost if missed |
|---|---|---|
| Wrong requirement | Human gate | Full rebuild |
| Bad architecture | Human gate | Major refactor |
| Bug in implementation | Automated gate | Hotfix |
| Spec non-compliance | Automated gate | Rejection at verify |
| Unreviewable diff | Automated gate | Context-switching tax |

---

## 2. The TDD Harness: Red, Green, Triangulate

The TDD Harness enforces test-driven development as a **harness constraint**, not a developer preference. The agent does not choose to write tests first — the harness rejects code that does not follow the pattern.

### The Three-Step Pattern

**Step 1 — Red.** The harness instructs the agent to write tests for the desired behavior *before* any implementation exists. The agent creates the test file, runs it, and confirms it fails. The harness checks: did the test run? Did it fail for the expected reason? If the test passes, the behavior already exists — the agent is testing legacy code, not driving new design.

**Step 2 — Green.** The agent writes the minimum implementation required to make the red test pass. The harness checks: does the test pass now? Did the agent add production code proportional to the test? If the agent adds unrelated refactoring, the harness flags it.

**Step 3 — Triangulate.** This is the distinguishing feature. The harness instructs the agent: *find two additional test cases that could break this implementation.* Not the happy path — the edge cases. The agent proposes new tests, adds them, and verifies all three (original + two) pass.

```
❌ Agent submits: implementation passes the one test written.
✅ Agent submits: implementation passes original test, plus two triangulation
   tests that exercise boundary conditions and error states.
```

Triangulation is the mechanism that produces what the transcript calls "ultra cubierto" — ultra-covered code. One test proves the feature works. Two more prove it does not break under stress.

A concrete example: the feature is a function that extracts the domain from an email address. The red test asserts `extract_domain("user@example.com")` returns `"example.com"`. The green implementation splits on `@` and returns the second part. Triangulation then asks: what breaks this? The agent adds `extract_domain("user@sub.example.com")` returning `"sub.example.com"` (multi-level domain), and `extract_domain("invalid")` raising an error (no `@` sign). The original test proved existence. The two triangulation tests proved the function handles subdomains and bad input — cases the agent would not have considered without the harness demanding them.

### Why the Harness Enforces This

Without harness enforcement, agents optimize for speed. They write one test, make it pass, and move on. The harness must be the entity that demands rigor because the agent, left to itself, will always take the shortest path to "done."

The TDD Harness also generates a durable artifact: the test file. That file becomes part of the project's permanent test suite, integrated into the project's CI pipeline by the File Architecture layer. The harness does not just verify in the moment — it leaves behind a verification asset.

---

## 3. The Review Workload Harness: Human-Scale Changes

The Review Workload Harness addresses a problem that pure automation cannot solve: *someone has to read this.* The transcript describes it as "one of my favorites because it thinks about the human." Most agent harnesses optimize for throughput — how fast can the machine produce. This harness optimizes for *reviewability*.

### The Workload Check

Before the harness applies code to the project (the "apply" step in the phase DAG), it evaluates the change:

1. **Line count.** Does the diff exceed 400 lines? If yes, the change is likely too large for a focused review.
2. **Scope.** Does the change touch multiple areas within the project — model, view, controller, database, config? If yes, the mental load of reviewing across contexts is high.
3. **Dependency impact.** Does the change modify shared interfaces or public APIs? If yes, the reviewer must trace ripple effects.

If any of these conditions trigger, the harness does not reject the change. It **decomposes** it. The agent is instructed to split the change into smaller, independently reviewable units and re-enter the phase.

The decomposition is not arbitrary. The harness provides decomposition prompts: "Separate the database migration from the business logic change. Submit the migration first, then the logic change in a second phase." Each decomposed unit must pass the workload check independently before it can proceed. A large change that touches auth, billing, and UI becomes three sequential changes: one for auth, one for billing, one for UI. The human reviews each in isolation, which takes three focused 15-minute sessions instead of one overwhelming hour.

```
❌ Diff: 680 lines across 14 files (auth, billing, UI, migration).
   Harness: "This change exceeds review workload threshold.
   Decompose into 3-4 sequential changes before proceeding."

✅ Diff: 120 lines across 2 files (billing module only).
   Harness: "Review workload within threshold. Proceed to apply."
```

### Why This Is a Harness, Not a Suggestion

A code review workflow *recommends* small diffs. A harness *enforces* them. The distinction matters:

- If the agent can bypass the check, the check is decoration.
- If the check runs before every apply, it becomes a structural constraint.
- If the check forces decomposition, it shapes the entire output of the system.

The harness asks one question: *Can a human review this without context-switching for an hour?* If the answer is no, the change is not ready. This respects human cognitive limits while leveraging AI speed. The agent does the heavy lifting of decomposition; the human reviews smaller, focused changes at their own pace.

This harness is most valuable in production systems where code must be reviewed by a human before deployment. In fully automated pipelines (where review is skipped), this harness is optional. But the moment a human is in the loop, this harness becomes critical.

---

## 4. Verification Against Spec: The Contract Check

The most common failure in agent-generated code is not bugs — it is *spec drift*. The code works, the tests pass, but the implementation does not match the specification. Verification against the spec catches this.

### The 3-File Check

The harness maintains references to three specification files, as defined in [[04 - Specification-Driven Development]]:

| File | Question checked | Failure mode |
|---|---|---|
| `requirements.md` | Are all EARS statements addressed by the implementation? | Missing feature |
| `design.md` | Does the code match the architecture decisions? | Structural mismatch |
| `tasks.md` | Are all tasks marked complete, with verification evidence? | Incomplete delivery |

The harness reads each file mechanically, line by line, and cross-references the implementation. This is not a human review — it is a structured comparison:

```
❌ requirements.md has: "REQ-003: System SHALL reject invalid tokens."
   Implementation: accepts all tokens, no validation.
   Harness: "REQ-003 not addressed. Verify against spec: FAIL."

✅ requirements.md has: "REQ-003: System SHALL reject invalid tokens."
   Implementation: validates JWT, returns 401 on invalid signature.
   Harness: "REQ-003 addressed. Verify against spec: PASS."
```

### What Verification Is Not

Verification is **not** running tests. Tests prove the code behaves as the developer expected. Verification proves the code behaves as the *spec* requires. These are different standards, and they diverge when:

- The developer misread the spec but wrote correct tests against their misinterpretation.
- The spec changed during implementation but the code did not update.
- Tests cover unit behavior but miss integration requirements stated in the spec.

The harness runs both: test suite **and** spec verification. One without the other is incomplete.

A test suite that passes against misread requirements produces green builds and wrong software. Spec verification that passes without a test suite produces compliant code that may crash at runtime. The harness requires both because they cover different error modes.

### Verification and the Result Contract

Each verification check feeds into the phase's result contract — the consistent envelope described in the transcript: status, executive summary, artifact, next recommended action, risk, and skill resolution. The verification step populates the `risk` field: if spec coverage is below 100%, the contract carries a `risk: medium` or `risk: high` tag that the orchestrator reads before deciding whether to advance to the next phase.

```
Phase result contract (verification section):
  spec_coverage: 87% (REQ-012 not addressed in implementation)
  test_suite: PASS (47/47)
  risk: medium
  next_action: re-enter implementation with REQ-012 context
```

This makes verification results machine-readable. The orchestrator does not need to parse human language to understand why a phase failed — the contract tells it in a structured format.

### The Verification Artifact

The verification step produces an artifact: a structured report mapping each EARS requirement to its implementation evidence. This artifact is stored alongside the code in the File Architecture, forming the audit trail. If a requirement is not met, the harness rejects the phase output and the agent re-enters the implementation phase — not from scratch, but with the verification failure as context.

---

## 5. Quality Gate Placement Theory

Gates are not placed arbitrarily. Each gate location is determined by the failure mode it prevents and the cost of detecting that failure at that point.

### Gate at Spec

**What it catches:** Wrong problem definition. The implementation may be correct, but the requirements are wrong.

**What it costs:** One conversation (human gate). Minutes to hours.

**Failure mode prevented:** Building the wrong thing entirely. This is the most expensive failure to fix later — it invalidates everything downstream.

### Gate at Design

**What it catches:** Bad architecture decisions. The requirements may be correct, but the design does not satisfy them efficiently.

**What it costs:** One conversation (human gate). Minutes to hours.

**Failure mode prevented:** Implementing a design that must be refactored immediately. This is the second most expensive failure — the code exists but the structure is wrong.

### Gate at Verify

**What it catches:** Spec non-compliance, missing requirements, broken tests.

**What it costs:** Automated, near-zero marginal cost.

**Failure mode prevented:** Deploying code that does not meet requirements. Without this gate, "tests pass" becomes the only quality signal, which is insufficient.

### Gate at Archive

**What it catches:** Missing artifacts, incomplete documentation, uncommitted work.

**What it costs:** Automated, near-zero marginal cost.

**Failure mode prevented:** Losing work. The archive gate ensures the phase output is persisted correctly before the next phase starts.

### The Placement Principle

The principle is: **detect failure at the lowest-cost point where it can be identified.** Human gates exist where humans are the best detectors (is this the right thing to build?). Automated gates exist everywhere else. This minimizes the cycle time of feedback — the interval between making a mistake and discovering it.

### Gate Cost Economics

Every gate has a cost: human gates cost attention, automated gates cost compute (tokens and time). The optimal gate placement minimizes the *total cost of quality* — the sum of prevention cost (running gates) plus failure cost (fixing bugs found later). A missing gate shifts costs from prevention to failure. An unnecessary gate shifts costs from nothing to prevention for no benefit.

Practical rule of thumb: if an automated gate has never failed in 50 consecutive runs, it is probably not checking anything useful and should be retired. If a human gate has not rejected a spec in 10 projects, the human is rubber-stamping and the gate is ceremonial. Gates earn their keep by catching things. A gate that always passes is not a gate — it is decoration.

```
Phase:  Spec → Design → Implement → Verify → Archive
Gate:   HUMAN     HUMAN    NONE      AUTO      AUTO
Cost:    $$$       $$$      $          $         $
```

No gate at implementation is intentional: the agent works freely within the boundaries set by spec and design. The gates are at the decision points, not the execution points.

### How Gates Compose

Gates compose in sequence. A failure at the spec gate prevents design work from starting. A failure at the verify gate prevents archiving. But gates also compose in time — a gate that passes now may be rechecked later if the context changes. For example, the verify gate runs at the end of implementation, but if the implementation phase re-enters (because verification failed), the design gate does not re-run — design was already approved. The composition is monotonic: once a gate passes, it stays passed unless its inputs change.

This monotonicity is what makes the harness efficient. The orchestrator caches gate results and only re-evaluates when the artifacts feeding the gate have changed. Without this, every phase re-entry would start from scratch, making multi-pass refinement prohibitively expensive.

---

## 6. Skill Resolution Feedback as Quality Signal

Every phase in the harness returns a result contract. One field in that contract — `skill_resolution` — is a critical quality signal that the transcript describes as "the feedback that keeps the orchestrator from going blind."

### What Skill Resolution Tells You

| Resolution state | Meaning | Quality implication |
|---|---|---|
| `resolved` | The skill executed and produced its expected output. | System healthy. |
| `fallback` | The skill was unavailable; a default handler ran instead. | Degraded quality. A skill is missing or failed to load. |
| `compact_rules` | The skill resolved but used compact rules instead of full project standards. | Quality is reduced, but the system adapted. |
| `no_standards` | The subagent operated without project standards entirely. | Unmonitored quality. The orchestrator cannot assess the output. |

### Why This Matters for Verification

If skills consistently resolve, the harness is functioning as designed. If skills consistently fall back, the harness quality is degrading — and the orchestrator should alert before the degradation compounds across phases.

The resolution field turns skill execution into telemetry. Without it, the orchestrator sees that a phase completed but cannot tell whether the phase used the right tools, followed the project standards, or improvised.

### Practical Integration

The skill resolution check becomes an automated gate at the end of each phase:

```
Phase output:
  status: completed
  skill_resolution: fallback (skill "ears-validator" not available)
  artifact: /build/output/v1/

Harness action:
  Logs warning. Compares resolution against threshold.
  If >30% of skills resolve as fallback across recent phases → alert.
  If any skill resolves as "no_standards" → block archive, require human review.
```

This prevents silent quality degradation. The system does not wait for a human to notice that output quality slipped — the harness detects the pattern in the resolution metadata and surfaces it.

### Resolution Trend as a Health Metric

A single fallback is noise. A trend of fallbacks is a signal. The harness tracks resolution outcomes over time and computes a running health score:

| Window | Healthy threshold | Warning threshold | Critical threshold |
|---|---|---|---|
| Last 10 phases | <10% fallback | 10-30% fallback | >30% fallback |
| Last 50 phases | <5% fallback | 5-20% fallback | >20% fallback |

When a window crosses the warning threshold, the harness logs an advisory. When it crosses critical, the harness escalates to the human operator with a recommendation: "Skill resolution degradation detected. Recommended action: audit skill registry for missing or outdated entries."

This turns quality from a subjective assessment into a quantitative metric. The human operator does not need to guess whether the system is degrading — the harness tells them, with data.

---

## 7. init.sh: The First Quality Gate

Before any agent runs, before any file is read, before any spec is parsed — `init.sh` executes. This is the first quality gate and the most fundamental. If the environment is wrong, everything downstream fails.

### What init.sh Verifies

The script encodes the project's minimum requirements as executable assertions:

- **Directory structure.** Do the required directories (`spec/`, `design/`, `tasks/`, `build/`) exist?
- **Tool availability.** Are the required tools installed? (Node, Python, Docker, etc.)
- **Configuration presence.** Does `opencode.json` or equivalent configuration exist?
- **Provider connectivity.** Can the system reach the LLM provider?
- **Permission model.** Does the runner have the correct filesystem permissions?

```
# Minimal init.sh quality gate
if [ ! -d "spec" ]; then
  echo "FAIL: spec/ directory missing. Project not initialized."
  exit 1
fi

if ! command -v node &> /dev/null; then
  echo "FAIL: Node.js not found. Required for project toolchain."
  exit 1
fi

echo "PASS: Environment verified."
```

### Why It Is a Quality Gate

Most quality gates validate the *output* of a phase. init.sh validates the *input* — the conditions under which every subsequent phase will operate. If init.sh fails, no agent should proceed because the foundation is unsound.

In the gate placement framework, init.sh is an **unconditional automated gate**. It runs once at harness startup and blocks everything until it passes. It has no human override — if the environment is broken, the harness must not work in it.

### From init.sh to Phase Gates

init.sh is the only gate that runs before the phase DAG begins. Once it passes, the orchestrator enters the spec phase, where the first human gate waits. The progression is:

```
init.sh (auto) → Spec (human gate) → Design (human gate) → Implement (no gate)
→ Verify & TDD & Review Workload (auto gates) → Archive (auto gate)
```

The init.sh gate establishes the preconditions. The human gates establish the direction. The automated gates establish the quality floor. Each gate type protects against a different class of failure, and together they form a pipeline that catches errors at the earliest possible moment.

---

## 🎯 Key Takeaways

- Quality gates divide into human (spec, design) and automated (verify, archive) — the 2 + N ratio balances leverage and throughput. More human gates create bottlenecks; fewer automated gates create blind deployment.
- The TDD Harness enforces red → green → triangulate, producing "ultra cubierto" code through harness-mandated edge case testing. Triangulation is what distinguishes coverage from confidence.
- The Review Workload Harness enforces human-scale diffs by decomposing changes that exceed 400 lines or touch multiple areas. It is a harness enforcement, not a suggestion — if the change cannot be reviewed in one focused session, the agent must split it.
- Verification against spec is a separate concern from running tests. Tests prove internal correctness; the 3-file check (requirements, design, tasks) proves alignment with the specification. Both are required.
- Gate placement follows one principle: detect failure at the lowest-cost point where it can be identified. Human gates at spec and design catch expensive errors early; automated gates everywhere else catch mechanical errors cheaply.
- Skill resolution feedback is quality telemetry. Consistent fallback or no-standards states indicate harness degradation. init.sh is the first gate — if the environment is wrong, no agent should proceed.

## 🔗 Production Integration

Verification connects to every layer of the harness. The Phase DAG in [[06 - Multi-Agent Orchestration and Capstone]] defines where gates are placed — spec gate after spec phase, design gate after design phase, verify gate before archive. The orchestrator enforces these gates by reading each phase's result contract and comparing it against the gate's pass/fail criteria. The File Architecture in [[05 - File Architecture]] stores the verification artifacts: spec coverage reports, test results, review workload assessments, and skill resolution logs.

Quality gates are not an afterthought bolted onto a working system. They are designed into the harness from Layer 1. The harness taxonomy in [[07 - Complete Harness Taxonomy]] classifies each sub-harness by its gate type — TDD Harness (automated pre-apply), Review Workload Harness (automated pre-apply), Verification Harness (automated post-implement). Together they form a verification pipeline that is as structured as the code production pipeline.

The verification layer also closes the feedback loop. When a gate fails, the result contract carries the failure reason, which feeds back into the orchestrator's next action decision. The orchestrator may re-enter the same phase with the failure context appended to the agent prompt, or escalate if the gate has failed multiple times. This creates a self-correcting system: failures are not dead ends, they are data for the next iteration.

A harness without verification layers is a cargo conveyor belt with no inspection station. Product moves fast, but defective units reach the end of the line before anyone notices. The verification layer installs those inspection stations at every critical junction, ensuring that by the time code reaches archive, it has passed every check the project considers essential. Speed is useless without direction; throughput is meaningless without quality.

The next layer, [[09 - Tools, Provider Abstraction, and Memory]], introduces the tooling substrate that makes these gates possible — the actual linters, test runners, and spec parsers that execute the checks.

## References

- Transcript: "20 Agent Harness" (5Q7jV8TpMXA) — TDD Harness, Review Workload Harness, Verification Harness, Skill Resolution Feedback.
- [[04 - Specification-Driven Development]] — EARS requirements, spec files, the SDD protocol.
- [[05 - File Architecture]] — Where verification artifacts are stored.
- [[06 - Multi-Agent Orchestration and Capstone]] — Phase DAG and gate placement.
- [[07 - Complete Harness Taxonomy]] — Classification of all sub-harnesses by gate type.
- [[09 - Tools, Provider Abstraction, and Memory]] — The tooling substrate for gate execution.
