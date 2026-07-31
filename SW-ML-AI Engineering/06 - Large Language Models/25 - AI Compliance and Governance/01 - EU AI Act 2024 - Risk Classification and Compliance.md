# 🎯 01 - EU AI Act 2024 — Risk Classification and Compliance

> **The most consequential AI regulation ever enacted. Every AI engineer deploying to enterprise customers must understand the risk tiers, the obligations per tier, and the penalties for non-compliance.**

## 🎯 Learning Objectives
- Classify any AI system into one of four EU AI Act risk tiers
- Identify the obligations that apply at each tier
- Calculate the penalties for non-compliance (up to 7% global revenue)
- Recognize which existing AI patterns are explicitly prohibited
- Build a risk-classification tool for your portfolio projects
- Distinguish EU AI Act from NIST AI RMF, ISO 42001, and other frameworks

## Introduction

The **EU AI Act** (Regulation (EU) 2024/1689) entered into force on August 1, 2024. It is the first horizontal regulation of artificial intelligence by any major jurisdiction. Its extraterritorial reach means any company whose AI system affects EU residents — regardless of where the company is based — must comply.

The Act classifies AI systems into **four risk tiers**, each with different obligations. The penalties reach **7% of global annual turnover or €35 million** (whichever is higher). The Act also explicitly prohibits certain AI practices (social scoring, real-time biometric identification in public spaces for law enforcement, with limited exceptions).

For AI engineers, understanding this regulation is no longer optional. It is table-stakes for any enterprise role.

![EU AI Act risk pyramid](https://example.com/eu-ai-act.png)

---

## 1. The Four Risk Tiers

### 1.1 Unacceptable Risk — Prohibited (Article 5)

These AI practices are **banned outright** in the EU:

| Practice | Example | Status |
|----------|---------|--------|
| **Social scoring by public authorities** | Government citizen score | Banned |
| **Real-time remote biometric ID in public spaces (law enforcement)** | Mass facial recognition | Banned (with narrow exceptions) |
| **Emotion recognition in workplace and education** | Sentiment analysis of students | Banned |
| **Predictive policing based solely on profiling** | Pre-crime risk scores | Banned |
| **Untargeted scraping of facial images from internet/CCTV** | Facial recognition training sets | Banned |
| **Inferring emotions in sensitive contexts** | Detecting lying in interrogations | Banned |

Penalty: **€35M or 7% of global turnover**.

### 1.2 High Risk — Strict Compliance (Annex III)

These AI systems must comply with strict requirements before deployment:

| Domain | Example | Article |
|-------|---------|---------|
| **Biometric ID** | One-to-one face verification | Annex III(1) |
| **Critical infrastructure** | Traffic management | Annex III(2) |
| **Education and vocational training** | Exam scoring | Annex III(3) |
| **Employment, HR** | Resume screening | Annex III(4) |
| **Access to essential services** | Credit scoring, insurance pricing | Annex III(5) |
| **Law enforcement** | Evidence reliability assessment | Annex III(6) |
| **Migration and border control** | Visa application assessment | Annex III(7) |
| **Justice and democratic processes** | Court sentencing assistance | Annex III(8) |

High-risk systems must:

- Have a **risk management system** (Article 9)
- Use **high-quality datasets** (Article 10)
- Maintain **technical documentation** (Article 11)
- Keep **automatic logs** (Article 12)
- Provide **transparency** to users (Article 13)
- Ensure **human oversight** (Article 14)
- Meet **accuracy, robustness, cybersecurity** standards (Article 15)

Penalty: **€15M or 3% of global turnover**.

### 1.3 Limited Risk — Transparency Requirements (Article 50)

These systems must disclose AI involvement to users:

| Domain | Example | Requirement |
|-------|---------|-------------|
| **Chatbots** | Customer service bots | Disclose AI nature |
| **Emotion recognition** | Marketing analytics | Inform subjects |
| **Biometric categorization** | Demographic inference | Inform subjects |
| **Deepfakes** | Synthetic media | Disclose synthetic nature |

Penalty: **€15M or 3% of global turnover**.

### 1.4 Minimal Risk — Voluntary Compliance

Most AI systems fall here: spam filters, recommendation engines, content moderation. No mandatory obligations, but **voluntary codes of conduct** are encouraged.

Penalty: None for non-compliance with the Act itself.

---

## 2. Compliance Obligations by Tier

### 2.1 High-Risk System Compliance Checklist

A high-risk AI system must comply with:

| Obligation | Implementation | Reference |
|------------|---------------|-----------|
| **Risk management** | Documented risk assessment, updated throughout lifecycle | Article 9 |
| **Data quality** | Training data documented, representative, free of errors | Article 10 |
| **Technical documentation** | Detailed system specs, intended purpose | Article 11 |
| **Logging** | Automatic event logging for traceability | Article 12 |
| **Transparency** | User-facing instructions for use | Article 13 |
| **Human oversight** | Effective human-in-the-loop design | Article 14 |
| **Accuracy, robustness, security** | Performance metrics, adversarial robustness, cybersecurity | Article 15 |
| **Quality management system** | Documented procedures for design, development, deployment | Article 17 |
| **Conformity assessment** | Self-assessment or third-party audit before deploy | Article 43 |
| **CE marking** | Conformity with the Act | Article 48 |
| **EU database registration** | Register in the EU AI database | Article 49 |
| **Post-market monitoring** | Continuous monitoring of deployed systems | Article 72 |

For an LLM system used for credit scoring (high-risk), all 12 obligations apply.

### 2.2 GPAI (General Purpose AI) Models — Separate Regime

The Act also regulates **GPAI models** (foundation models like GPT-4, Llama 3) separately:

| Tier | Threshold | Obligation |
|------|-----------|-------------|
| **Standard GPAI** | All GPAI | Technical documentation, copyright compliance, training data summary |
| **Systemic-risk GPAI** | >10^25 FLOPs training compute | Risk assessment, incident reporting, adversarial testing, cybersecurity |

Most frontier LLMs (GPT-4, Claude 3.5, Llama 3.1 405B) are systemic-risk GPAI. Smaller models are standard GPAI.

---

## 3. Penalties — The Real Numbers

### 3.1 Penalty structure

| Violation | Max Penalty |
|-----------|-------------|
| Prohibited practice (Article 5) | €35M or **7% of global turnover** |
| High-risk non-compliance | €15M or **3% of global turnover** |
| Misleading information to regulators | €7.5M or **1% of global turnover** |

For a company with $10B revenue:
- Prohibited practice: up to **$700M fine**
- High-risk non-compliance: up to **$300M fine**
- Misleading info: up to **$100M fine**

### 3.2 Real case: Clearview AI (illustrative)

Clearview AI was fined €30M by Dutch DPA for facial recognition database. Under the EU AI Act, similar systems would face up to 7% of global turnover. Clearview's revenue is ~$100M, so the fine would be **€7M** under the Act. For larger companies, the difference is dramatic.

### 3.3 Enforcement timeline

| Date | Provision |
|------|-----------|
| August 2024 | Act entered into force |
| February 2025 | Prohibited practices (Article 5) enforceable |
| August 2025 | GPAI obligations enforceable |
| August 2026 | High-risk obligations enforceable (most provisions) |
| August 2027 | High-risk obligations for embedded systems (Annex III) |

The **August 2026 deadline** is the critical one. From that date, deploying a high-risk AI system without full compliance is a regulatory violation.

---

## 4. Risk Classification Tool

Build a tool to classify any AI system:

```python
from dataclasses import dataclass
from enum import Enum


class RiskTier(Enum):
    UNACCEPTABLE = "unacceptable"  # Banned
    HIGH = "high"                  # Strict compliance
    LIMITED = "limited"            # Transparency
    MINIMAL = "minimal"            # Voluntary


@dataclass
class AISystem:
    name: str
    purpose: str
    domain: str  # e.g., "education", "credit", "healthcare"
    uses_biometrics: bool
    affects_rights: bool  # Decisions about individuals
    uses_personal_data: bool
    automates_decisions: bool
    is_safety_critical: bool
    is_categorized_gpai: bool
    

# Annex III categories — high risk
HIGH_RISK_DOMAINS = {
    "biometric_id",
    "critical_infrastructure",
    "education",
    "employment",
    "credit_scoring",
    "law_enforcement",
    "migration",
    "justice",
}


# Article 5 prohibited practices
PROHIBITED_PRACTICES = {
    "social_scoring_public",
    "real_time_biometric_law_enforcement",
    "emotion_recognition_workplace",
    "predictive_policing",
    "untargeted_face_scraping",
}


def classify_ai_system(system: AISystem) -> RiskTier:
    """Classify an AI system under EU AI Act risk tiers."""
    
    # Check prohibited practices first
    if system.uses_biometrics and system.domain == "law_enforcement":
        return RiskTier.UNACCEPTABLE
    
    if "emotion_recognition" in system.purpose.lower() and system.domain in ("workplace", "education"):
        return RiskTier.UNACCEPTABLE
    
    # High risk: domain in Annex III
    if system.domain in HIGH_RISK_DOMAINS:
        return RiskTier.HIGH
    
    if system.affects_rights and system.automates_decisions:
        return RiskTier.HIGH
    
    # Limited risk: transparency required
    if "chatbot" in system.purpose.lower() or "deepfake" in system.purpose.lower():
        return RiskTier.LIMITED
    
    # Default: minimal
    return RiskTier.MINIMAL


def required_obligations(tier: RiskTier) -> list[str]:
    """Return the obligations for a given tier."""
    if tier == RiskTier.UNACCEPTABLE:
        return ["PROHIBITED — Do not deploy"]
    
    if tier == RiskTier.HIGH:
        return [
            "Risk management system (Art. 9)",
            "Data quality and governance (Art. 10)",
            "Technical documentation (Art. 11)",
            "Automatic logging (Art. 12)",
            "Transparency to users (Art. 13)",
            "Human oversight (Art. 14)",
            "Accuracy, robustness, security (Art. 15)",
            "Quality management system (Art. 17)",
            "Conformity assessment (Art. 43)",
            "CE marking (Art. 48)",
            "EU database registration (Art. 49)",
            "Post-market monitoring (Art. 72)",
        ]
    
    if tier == RiskTier.LIMITED:
        return ["Transparency: disclose AI nature (Art. 50)"]
    
    return ["Voluntary codes of conduct (Art. 95)"]


# Usage
system = AISystem(
    name="CreditScoringLLM",
    purpose="Score loan applications",
    domain="credit_scoring",
    uses_biometrics=False,
    affects_rights=True,
    uses_personal_data=True,
    automates_decisions=True,
    is_safety_critical=False,
    is_categorized_gpai=True,
)

tier = classify_ai_system(system)
print(f"Tier: {tier.value}")
print("Obligations:")
for ob in required_obligations(tier):
    print(f"  - {ob}")
```

Output:

```
Tier: high
Obligations:
  - Risk management system (Art. 9)
  - Data quality and governance (Art. 10)
  ...
  - Post-market monitoring (Art. 72)
```

---

## 5. EU AI Act vs Other Frameworks

| Aspect | EU AI Act | NIST AI RMF | ISO 42001 |
|--------|-----------|-------------|-----------|
| **Type** | Binding regulation | Voluntary framework | Certifiable standard |
| **Penalty** | €35M / 7% revenue | None | Certification withdrawal |
| **Geographic** | EU + extraterritorial | US/global | Global |
| **Risk tiers** | 4 (unacceptable/high/limited/minimal) | Continuous (LOW/MEDIUM/HIGH) | Process-based |
| **Focus** | Outcome (banned practices) | Process (risk management) | Management system |
| **Required for** | EU AI deployment | US federal procurement | Enterprise certification |

For a global company, you typically need **all three**:
- EU AI Act for any EU exposure
- NIST AI RMF for US federal contracts
- ISO 42001 for enterprise customers who require certification

The compliance artifacts (model cards, risk assessments, monitoring) overlap significantly across frameworks — build once, use for all.

---

## 6. Real Cases

### 6.1 Clearview AI (NL) — facial recognition database

Built a 30B image facial recognition database from public sources. **EU AI Act violation**: untargeted scraping + biometric identification. Result: €30M fine (under DPA); under the Act, would face €35M / 7% turnover.

### 6.2 HireVue (US) — AI video interviews

AI-powered video interview analysis used by employers. **EU AI Act violation**: emotion recognition in workplace + employment decisions. Result: under the Act, would be classified high-risk; would face 3% of global turnover fine + conformity assessment + EU database registration.

### 6.3 ChatGPT (Italy) — banned temporarily

Italian Garante banned ChatGPT in March 2023 over privacy concerns. Under the Act, GPAI models must comply with **transparency (Art. 50)** and provide **opt-outs**. OpenAI added these features.

### 6.4 Real case: a bank deploying a credit-scoring LLM

A US bank deploys Llama 3 70B as a credit-scoring assistant. The system:
- Affects rights (loan decisions)
- Automates decisions
- Uses personal data
- Domain is credit_scoring (Annex III)

**Risk tier**: HIGH. Required obligations: 12. Failure to comply = up to 3% of global turnover.

The bank must:
- Conduct conformity assessment before deploy
- Generate technical documentation
- Establish human oversight (loan officer reviews AI recommendations)
- Register in the EU AI database
- Implement post-market monitoring
- Generate model card and datasheets

---

## 7. Antipatterns

### 7.1 Antipattern 1: Ignoring risk classification

```python
# ❌ Treating all AI systems the same
def deploy_ai_system(system):
    deploy(system)  # No risk classification

# ✅ Classify first, then deploy with appropriate compliance
def deploy_ai_system(system):
    tier = classify_ai_system(system)
    if tier == RiskTier.UNACCEPTABLE:
        raise ValueError("Cannot deploy prohibited system")
    
    if tier == RiskTier.HIGH:
        comply_with_high_risk_obligations(system)
    
    deploy(system)
```

### 7.2 Antipattern 2: Incomplete risk assessment

```python
# ❌ Single-question assessment
tier = "high" if affects_rights else "low"

# ✅ Multi-dimensional assessment
tier = classify_ai_system(system)
```

### 7.3 Antipattern 3: No human oversight

```python
# ❌ Pure automated decisions on rights
def loan_decision(application):
    return llm.score(application)  # No human review

# ✅ High-risk: human-in-the-loop
def loan_decision(application):
    score = llm.score(application)
    return loan_officer.review(score, application)  # Required for high-risk
```

### 7.4 Antipattern 4: No post-market monitoring

```python
# ❌ Deploy and forget
def deploy():
    serve()

# ✅ Continuous monitoring (Article 72)
def deploy():
    serve()
    monitor_drift()      # track data and concept drift
    monitor_fairness()   # track bias metrics
    monitor_quality()    # track accuracy
    report_to_board()    # quarterly report
```

### 7.5 Antipattern 5: Treating GPAI and downstream systems the same

```python
# ❌ Same compliance for base model and deployment
def deploy_gpai(model, deployment):
    if is_high_risk(deployment):
        comply_with_high_risk(model)
        comply_with_high_risk(deployment)

# ✅ GPAI provider (model) and deployer (system) have separate obligations
def deploy_gpai(model, deployment):
    # GPAI provider: technical documentation, copyright, training summary
    gpai_compliance(model)
    
    # Deployer: high-risk obligations for the deployment
    if is_high_risk(deployment):
        comply_with_high_risk(deployment)
```

---

## 🎯 Key Takeaways

- The EU AI Act has **four risk tiers**: unacceptable (banned), high (strict), limited (transparency), minimal (voluntary).
- High-risk systems face **12 mandatory obligations** (Articles 9-17, 43, 48, 49, 72).
- Penalties reach **7% of global turnover** (€35M minimum) for prohibited practices.
- Enforcement begins February 2025 (prohibited) → August 2026 (high-risk) → August 2027 (embedded).
- Build a classification tool that maps any AI system to a tier.
- Pair EU AI Act with NIST AI RMF and ISO 42001 for global coverage.
- Apply to portfolio: credit scoring = HIGH, hiring = HIGH, biometric ID = HIGH or PROHIBITED, customer service bots = LIMITED, content recommendation = MINIMAL.
- Avoid ignoring risk classification, incomplete assessment, no human oversight, no monitoring, conflating GPAI vs deployment obligations.

## References

- EU AI Act full text — [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)
- EU AI Act official portal — [artificialintelligenceact.eu](https://artificialintelligenceact.eu/)
- NIST AI RMF — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
- ISO/IEC 42001 — [iso.org/standard/81230.html](https://www.iso.org/standard/81230.html)
- Mitchell et al. — Model Cards for Model Reporting — [arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993)
- Gebru et al. — Datasheets for Datasets — [arxiv.org/abs/1803.09010](https://arxiv.org/abs/1803.09010)
- [[12 - Producto, Negocio y Open Source/30 - Producto y Estrategia de IA/04 - Legal y Compliance en IA|Legal y Compliance en IA (Spanish)]]
- [[06 - Large Language Models/15 - LLM Security and Guardrails|LLM Security and Guardrails]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/02 - Model Cards and Datasheets for Datasets|Note 02 — Model Cards]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/03 - Bias and Fairness - AI Fairness 360 and Demographic Parity|Note 03 — Bias and Fairness]]
- [[06 - Large Language Models/25 - AI Compliance and Governance/04 - Red-Teaming and Adversarial Testing|Note 04 — Red-Teaming]]