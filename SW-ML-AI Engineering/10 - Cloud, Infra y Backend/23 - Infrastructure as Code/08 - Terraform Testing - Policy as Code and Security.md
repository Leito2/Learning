# 🏷️ 08 - Terraform Testing, Policy as Code, and Security

## 🎯 Learning Objectives

- Explain why `terraform validate` is not a test — and design a defense-in-depth testing strategy: HCL syntax tests, plan assertion tests, and live infrastructure tests with Terratest
- Write `terraform test` files with plan-based assertions and integrate them into the CI pipeline as a mandatory gate
- Implement Policy as Code with OPA/Rego and Conftest — enforce encryption, least-privilege IAM, and networking compliance BEFORE apply
- Run security scanners (tfsec, checkov, trivy) in CI to catch misconfigurations: open security groups, unencrypted S3 buckets, public RDS instances
- Manage secrets correctly: AWS Secrets Manager, HashiCorp Vault, `sensitive = true`, and NEVER in plaintext variables
- Configure scheduled drift detection to catch manual console changes within hours

## Introduction

**"It applied without errors" is NOT the same as "it's correct."** This sentence is the thesis of infrastructure testing. A Terraform `validate` pass confirms that your HCL syntax is legal. A `plan` pass confirms that the provider API accepted your resource definitions. Neither confirms that your security group rules are correct, your IAM roles follow least privilege, your S3 buckets are encrypted, or your database has deletion protection enabled. Infrastructure bugs are production bugs — they affect real users immediately. A misconfigured security group that exposes port 5432 (PostgreSQL) to `0.0.0.0/0` will be exploited by automated scanners within hours.

The problem before systematic infrastructure testing was the "apply and pray" pattern. An engineer writes Terraform, runs `terraform validate` (it passes — the syntax is fine), runs `terraform plan` (it shows resources — the API accepts them), and runs `terraform apply`. Production now contains infrastructure that has never been tested against any correctness, security, or compliance standard. Three months later, a security audit reveals 47 S3 buckets without encryption, 12 security groups open to the world, and 3 IAM roles with `*:*` permissions. The remediation is a panic-driven sprint that blocks all feature work for two weeks.

Infrastructure testing, Policy as Code, and security scanning form a defense-in-depth strategy:
1. **Testing** verifies that infrastructure PRODUCES THE RIGHT RESULT (resource counts, attribute values, service health)
2. **Policy as Code** verifies that infrastructure is COMPLIANT (encryption, allowed instance types, required tags)
3. **Security scanning** verifies that infrastructure is SAFE (no open ports, no public databases, no leaked secrets)

This note connects the CI/CD pipelines from [[06 - CI-CD and GitOps for ML Infrastructure|Note 06]] with the observability patterns from [[07 - Infrastructure Observability - Prometheus, Grafana and Distributed Tracing|Note 07]] and assumes cloud fundamentals from [[10 - Cloud, Infra y Backend/22 - Cloud Computing/04 - Redes y Seguridad en Cloud|Cloud Networking]].

---

## 1. Testing Infrastructure — The Three Layers

Infrastructure testing operates at three layers of increasing fidelity (and increasing cost):

### Layer 1: terraform test (Fast, No Resources Created)

`terraform test`, GA in Terraform 1.6+, is HCL-native testing. Test files (`*.tftest.hcl`) live alongside your Terraform configuration and run in CI as a mandatory gate. They verify resource counts, attribute values, and output values:

```hcl
# tests/vpc_test.tftest.hcl — Test that VPC module creates correct resources
run "vpc_has_correct_subnets" {
  command = plan

  variables {
    vpc_cidr = "10.0.0.0/16"
    az_count = 3
  }

  assert {
    condition = length(aws_subnet.public) == var.az_count
    error_message = "Expected ${var.az_count} public subnets, got ${length(aws_subnet.public)}"
  }

  assert {
    condition = length(aws_subnet.private) == var.az_count
    error_message = "Private subnet count mismatch"
  }

  assert {
    condition = aws_vpc.main.cidr_block == "10.0.0.0/16"
    error_message = "VPC CIDR block did not match input"
  }
}

run "nat_gateways_per_az" {
  command = plan

  assert {
    condition = length(aws_nat_gateway.main) == var.az_count
    error_message = "Expected 1 NAT Gateway per AZ for production HA"
  }
  # 💡 This test encodes an architectural requirement: production VPCs
  # MUST have one NAT gateway per AZ. A single NAT gateway is a single
  # point of failure that takes down all private subnets.
}
```

```bash
# Run in CI: tests must pass before plan/apply
$ terraform test
# terraform test: 7 passed, 0 failed, 0 skipped
```

### Layer 2: Terratest (Slow, Tests Real Infrastructure)

**[Terratest](https://terratest.gruntwork.io/)** (by Gruntwork) is a Go library for testing real infrastructure. Unlike `terraform test`, Terratest actually applies Terraform, interacts with the live infrastructure, and then destroys it:

```go
// terratest/ml_infra_test.go — Verify entire ML infrastructure works
package test

import (
    "testing"
    "time"

    "github.com/gruntwork-io/terratest/modules/http_helper"
    "github.com/gruntwork-io/terratest/modules/k8s"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/stretchr/testify/assert"
)

func TestMLInferenceInfrastructure(t *testing.T) {
    t.Parallel()

    terraformOptions := &terraform.Options{
        TerraformDir: "../envs/staging",
        Vars: map[string]interface{}{
            "environment": "staging",
            "gpu_count":   1,
        },
    }

    // Always destroy, even if the test fails
    defer terraform.Destroy(t, terraformOptions)

    // Apply Terraform
    terraform.InitAndApply(t, terraformOptions)

    // Verify EKS cluster is healthy
    kubeConfig := terraform.Output(t, terraformOptions, "kubeconfig_path")
    kubectlOptions := k8s.NewKubectlOptions("", kubeConfig, "default")

    nodes := k8s.GetNodes(t, kubectlOptions)
    assert.GreaterOrEqual(t, len(nodes), 1, "Expected at least 1 worker node")

    // Verify inference endpoint responds
    inferenceURL := terraform.Output(t, terraformOptions, "inference_endpoint")
    http_helper.HttpGetWithRetryWithCustomValidation(
        t, inferenceURL, nil, 30, 5*time.Second,
        func(statusCode int, body string) bool {
            return statusCode == 200 || statusCode == 401
            // 401 = OK (endpoint exists, auth required)
        },
    )

    // Verify Prometheus is scraping metrics
    prometheusURL := terraform.Output(t, terraformOptions, "prometheus_endpoint")
    statusCode, body := http_helper.HttpGet(t, prometheusURL+"/api/v1/targets", nil)
    assert.Equal(t, 200, statusCode)
    assert.Contains(t, body, "dcgm_exporter", "DCGM exporter not found in Prometheus targets")
    // ¡Sorpresa! This test verifies that Prometheus successfully discovered
    // and scraped the DCGM exporter. If the GPU node didn't register properly,
    // this test catches it.
}
```

⚠️ Terratest creates REAL infrastructure. A full ML infrastructure test (VPC + EKS + RDS + GPU nodes) can cost $50-100 per run and take 20-30 minutes. Run Terratest on a schedule (nightly) or on merge to main, NOT on every PR.

### Layer 3: Scheduled Verification (Production Health Checks)

Beyond Terratest, continuously verify that production infrastructure is healthy:

```yaml
# .github/workflows/production-health-check.yml — Runs every hour
on:
  schedule:
    - cron: '0 * * * *'

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - name: Verify EKS cluster API is reachable
        run: |
          kubectl cluster-info || exit 1

      - name: Verify GPU nodes are Ready
        run: |
          gpu_nodes=$(kubectl get nodes -l node.kubernetes.io/instance-type=g5.xlarge -o name | wc -l)
          if [ "$gpu_nodes" -lt 2 ]; then
            echo "CRITICAL: Only $gpu_nodes GPU nodes available (expected >=2)"
            exit 1
          fi

      - name: Verify Prometheus is healthy
        run: |
          curl -sf http://prometheus.internal:9090/-/healthy || exit 1
```

---

## 2. Policy as Code — OPA and Sentinel

**Policy as Code** separates business/compliance rules from the engineers who write HCL. Engineers define WHAT infrastructure they want (a security group, an S3 bucket). Policy defines WHAT IS ALLOWED (encryption required, no open ports, approved instance types). Policy violations BLOCK the build — they are not warnings.

### OPA (Open Policy Agent)

[OPA](https://www.openpolicyagent.org/) is a CNCF-graduated general-purpose policy engine. Policies are written in **Rego**, a declarative language inspired by Datalog. OPA integrates with Terraform via **Conftest**, which evaluates Rego policies against `terraform plan` JSON output:

```rego
# policy/s3_encryption.rego — All S3 buckets MUST be encrypted
package terraform

deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_s3_bucket"
    not resource.change.after.server_side_encryption_configuration
    msg = sprintf("S3 bucket %s does not have encryption enabled", [resource.address])
}
```

```rego
# policy/security_groups.rego — No security group allows 0.0.0.0/0 on SSH
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_security_group_rule"
    resource.change.after.cidr_blocks[_] == "0.0.0.0/0"
    resource.change.after.from_port == 22
    msg = sprintf("Security group %s allows SSH from 0.0.0.0/0. Use bastion or SSM.", [resource.address])
}
```

```rego
# policy/iam_least_privilege.rego — No IAM policy with wildcard actions
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_iam_policy"
    statement := resource.change.after.policy[_]
    statement.Effect == "Allow"
    action := statement.Action[_]
    action == "*"
    msg = sprintf("IAM policy %s uses Action: *. Apply least privilege.", [resource.address])
}
```

```bash
# Run Conftest against terraform plan JSON
$ terraform plan -out=tfplan
$ terraform show -json tfplan | conftest test -
# FAIL - S3 bucket aws_s3_bucket.models does not have encryption enabled
# FAIL - Security group aws_security_group.bastion allows SSH from 0.0.0.0/0

# 2 policies failed. Block the build.
```

### Sentinel (HashiCorp, Terraform Cloud Only)

Sentinel is HashiCorp's policy-as-code language, available only in Terraform Cloud and Terraform Enterprise. It serves the same purpose as OPA but with tighter integration:

```sentinel
# sentinel/s3-encryption.sentinel
import "tfplan/v2" as tfplan

s3_buckets = filter tfplan.resource_changes as _, rc {
    rc.type is "aws_s3_bucket" and
    rc.change.actions contains "create"
}

main = rule {
    all s3_buckets as _, bucket {
        bucket.change.after.server_side_encryption_configuration is not null
    }
}
```

💡 The choice between OPA and Sentinel depends on your stack. OPA works with Terraform, Kubernetes, Envoy, Kafka, and any JSON/YAML-based tool. Sentinel works ONLY with Terraform. If your organization uses multiple infrastructure tools (Terraform + Kubernetes + CloudFormation), OPA is the better investment.

---

## 3. Security Scanning — tfsec, checkov, trivy

Security scanners are static analysis tools for Infrastructure as Code. They catch known misconfigurations WITHOUT requiring policy authoring:

### tfsec

Lightweight, fast, HCL-native. Scans Terraform code directly (no plan needed):

```bash
$ tfsec .
# CRITICAL: aws-s3-enable-bucket-encryption (aws_s3_bucket.datasets)
#   Bucket does not have encryption enabled
#   @ modules/data_storage/main.tf:14
#
# HIGH: aws-ec2-no-public-ingress-sgr (aws_security_group_rule.public_ssh)
#   Security group rule allows ingress from public internet
#   @ modules/networking/security.tf:23
```

### checkov

Comprehensive, multi-tool (Terraform, CloudFormation, Kubernetes, Helm, Dockerfile, GitHub Actions). 750+ built-in policies, including compliance frameworks (PCI-DSS, SOC2, HIPAA, CIS):

```bash
$ checkov -d infrastructure/
# Check: CKV_AWS_19: "Ensure all S3 buckets are encrypted at rest"
#   FAILED for resource: aws_s3_bucket.models
#   File: /modules/storage/main.tf:5-25
#
# Check: CKV_AWS_23: "Ensure every security group has a description"
#   FAILED for resource: aws_security_group.ml_inference
```

### trivy

Scans Terraform, Docker images, AND Kubernetes manifests in one tool — misconfigurations + vulnerabilities:

```bash
$ trivy config infrastructure/
# HIGH: Security group allows SSH from the internet
#   ────────────────────────────────────────────
#   terraform: [aws_security_group_rule.ssh_open]

$ trivy image nvcr.io/nvidia/tritonserver:24.05-py3
# MEDIUM: CVE-2024-1234 in libcurl4 (7.8 HIGH)
# HIGH:   CVE-2024-5678 in libssl1.1 (9.1 CRITICAL)
```

💡 Run `trivy` in CI for BOTH config scanning (misconfigurations) AND image scanning (vulnerabilities). A single tool that covers your Terraform modules, Docker images, and Kubernetes manifests reduces cognitive load and pipeline complexity.

---

## 4. Secrets Management — Never in Plaintext

The #1 infrastructure security mistake: API keys in Terraform variables. `terraform plan` prints ALL non-sensitive variables to stdout and CI logs. If `AWS_SECRET_ACCESS_KEY` is in a variable, it is in the CI log archive — which is typically accessible to every engineer in the organization.

```bash
# ⚠️ WHAT NOT TO DO
$ terraform plan
# An execution plan has been generated and is shown below.
#   + variable "db_password"
#       + default = "SuperSecretPassword123!"  ← PRINTED TO CI LOG
```

### The Hierarchy of Secrets Management

```
Level 0: Plaintext in terraform.tfvars (NEVER)
Level 1: variable "secret" { sensitive = true } (better, but still in statefile)
Level 2: CI secret variable (TF_VAR_db_password) (better, but CI secrets sprawl)
Level 3: External secret manager (AWS Secrets Manager, HashiCorp Vault) (BEST)
```

### Level 3 — External Secret Manager

```hcl
# main.tf — Pull DB password from Secrets Manager at apply time
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "prod/rds/ml-registry"
}

resource "aws_db_instance" "ml_registry" {
  engine         = "postgres"
  engine_version = "16.3"
  instance_class = "db.r5.large"
  username       = "ml_admin"
  password       = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)["password"]
  # ⚠️ This password is still in the Terraform statefile!
  # Always use encrypted remote state (S3 + SSE-KMS) and restrict statefile access.
}
```

¡Sorpresa! Even with `sensitive = true`, variable values ARE stored in the Terraform statefile. Any entity with read access to the statefile (S3 bucket, Terraform Cloud workspace) can extract secrets. Encrypted remote state is mandatory, and statefile access must be restricted to CI runners + senior engineers.

**Caso real: Capital One** mandates that every Terraform change pass OPA policy checks via Conftest before it can reach the `terraform apply` stage. Their OPA policies enforce:
- S3 buckets must have server-side encryption (AES256 or KMS)
- VPC flow logs must be enabled on every VPC
- RDS instances must have deletion protection enabled
- IAM roles must have permission boundaries
- Security groups must not allow 0.0.0.0/0 on any port (SSH via SSM only)

The result: 40% of initial PRs fail policy checks — catching security and compliance issues at code review time, not at audit time. Before Policy as Code, these issues were discovered months later by the security team during manual audits.

**Caso real: Coinbase** stores all infrastructure configurations in a monorepo with mandatory security scanning in CI. Every PR runs `checkov` with their custom policies. No PR can merge if it creates an unencrypted resource, an overly permissive IAM role, or a publicly accessible database. This automated enforcement reduced security review time from 3 days (manual review by the security team) to 12 minutes (automated check in CI).

---

## 🎯 Key Takeaways

- "terraform apply worked!" proves syntax correctness, NOT infrastructure correctness — testing verifies that your infrastructure produces the right resources with the right configuration
- Defense in depth: `terraform test` (syntax/plan assertions) → Terratest (live infrastructure) → scheduled health checks (continuous production verification)
- Policy as Code (OPA/Rego + Conftest) blocks non-compliant infrastructure BEFORE apply — encryption, least-privilege IAM, network security are enforced automatically, not by manual review
- Run `tfsec`, `checkov`, and `trivy` in CI on every PR — security scanning catches misconfigurations in seconds that would take a security engineer hours to find manually
- NEVER put secrets in Terraform variables — they appear in CI logs and statefiles. Use AWS Secrets Manager, HashiCorp Vault, with encrypted remote state
- Scheduled drift detection (`terraform plan -detailed-exitcode` every hour) catches manual console changes — alert on exit code 2
- A `terraform plan` showing "Plan: 0 to add, 0 to change, 0 to destroy" means the infrastructure matches the statefile — but the statefile could be wrong if someone used `terraform state rm`

## 📦 Código de Compresión

```hcl
# tests/eks_cluster_test.tftest.hcl — terraform test for EKS module
run "eks_cluster_basics" {
  command = plan

  variables {
    cluster_name    = "ml-prod"
    kubernetes_version = "1.30"
    vpc_id          = "vpc-0123456789abcdef0"
    subnet_ids      = ["subnet-abc", "subnet-def", "subnet-ghi"]
  }

  assert {
    condition = aws_eks_cluster.main.version == "1.30"
    error_message = "EKS cluster must run Kubernetes 1.30"
  }

  assert {
    condition = length(aws_eks_node_group.gpu.scaling_config[0].desired_size) != null
    error_message = "GPU node group must have desired_size configured"
  }

  assert {
    condition = aws_eks_cluster.main.encryption_config != null
    error_message = "EKS cluster must have envelope encryption enabled with KMS"
  }

  assert {
    condition = length(aws_eks_addon.vpc_cni) > 0
    error_message = "VPC CNI addon is required for pod networking"
  }
}

# tests/iam_roles_test.tftest.hcl — Enforce least privilege
run "iam_roles_no_wildcard" {
  command = plan

  assert {
    condition = alltrue([
      for policy in aws_iam_policy.this :
      !contains(flatten([for s in policy.policy.Statement :
        s.Action if contains(["Allow"], s.Effect)]), "*")
    ])
    error_message = "No IAM policy may use wildcard (*) Action with Allow effect"
  }
}
```

```yaml
# .github/workflows/security-scan.yml — CI security scanning pipeline
name: "Infrastructure Security Scan"

on: [pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: tfsec — Static analysis for security issues
        uses: aquasecurity/tfsec-action@v1
        with:
          working_directory: infrastructure/
          # 💡 tfsec scans HCL directly — no terraform init needed

      - name: checkov — Compliance and security scanning
        uses: bridgecrewio/checkov-action@v12
        with:
          directory: infrastructure/
          framework: terraform

      - name: Conftest — OPA policy evaluation
        run: |
          terraform init && terraform plan -out=tfplan
          terraform show -json tfplan | conftest test --policy policy/ -
        working-directory: infrastructure/envs/prod
        # ⚠️ Conftest requires terraform init + plan because it evaluates
        # the plan JSON, not raw HCL.

      - name: trivy — Misconfigurations + vulnerability scan
        uses: aquasecurity/trivy-action@0.24.0
        with:
          scan-type: config
          scan-ref: infrastructure/
          severity: HIGH,CRITICAL
```

---

## References

- Gruntwork. (2024). *Terratest Documentation*. https://terratest.gruntwork.io/docs/ — Go library for automated infrastructure testing.
- Open Policy Agent. (2024). *OPA Documentation*. https://www.openpolicyagent.org/docs/latest/ — Rego language reference and policy authoring guide.
- Conftest. (2024). *Conftest Documentation*. https://www.conftest.dev/ — Policy evaluation against structured configuration data.
- Aqua Security. (2024). *tfsec Documentation*. https://aquasecurity.github.io/tfsec/ — Static analysis security scanner for Terraform.
- Bridgecrew. (2024). *Checkov Documentation*. https://www.checkov.io/ — Compliance and security scanning for IaC.
- HashiCorp. (2024). *Terraform Test Documentation*. https://developer.hashicorp.com/terraform/language/tests — HCL-native testing framework.
- [[10 - Cloud, Infra y Backend/23 - Infrastructure as Code/06 - CI-CD and GitOps for ML Infrastructure|CI-CD and GitOps]]
- [[09/29 - CI-CD for ML]]
- [[09/21 - Monitoreo y Mantenimiento]]
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing/04 - Redes y Seguridad en Cloud|Cloud Networking]]
- [[13/04 - DevSecOps Go]]
