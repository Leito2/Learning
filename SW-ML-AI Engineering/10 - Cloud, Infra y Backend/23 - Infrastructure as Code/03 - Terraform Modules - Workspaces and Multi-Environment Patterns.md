# 🧩 03 - Terraform Modules, Workspaces, and Multi-Environment Patterns

## 🎯 Learning Objectives

- Design reusable, versioned Terraform modules with clean public interfaces (variables in, outputs out)
- Source modules from local paths, the Terraform Registry, and Git repositories — with precise version pinning
- Compose infrastructure using flat, layered, and Terragrunt-style module orchestration patterns
- Choose between workspaces, directory-per-environment, and branch-per-environment for production isolation
- Perform state surgery: `state mv`, `state rm`, `state pull/push`, and `import`
- Understand module composition anti-patterns: the 500-resource monolith and the over-fragmented micro-module
- Apply the directory-per-environment pattern — the industry standard for production Terraform

## Introduction

A single `main.tf` with 500 resources, 80 variables, and 45 outputs is not infrastructure as code — it is infrastructure as a hostage situation. Nobody can review it, nobody can test it in isolation, and a single typo can destroy the entire AWS footprint. **Modules** are Terraform's composition primitive: reusable, versioned building blocks that encapsulate a concern (VPC, EC2 cluster, RDS instance, IAM policy suite) behind a clean contract of input variables and output values. A module is to Terraform what a class is to object-oriented programming: a self-contained unit of abstraction with a public API and private internals.

The module ecosystem is one of Terraform's greatest strengths. The Terraform Registry hosts thousands of verified modules — the `terraform-aws-modules/vpc/aws` module alone has been downloaded over 15 million times and is maintained by AWS and HashiCorp jointly. But module composition requires discipline: flat composition (root calls independent modules), layered composition (networking → compute → application), and wrapper tools like Terragrunt each solve different scaling problems.

Multi-environment management is where the discipline matters most. Dev, staging, and production must be isolated — a test VPC change must not affect production. But they must also share as much code as possible to avoid drift. The directory-per-environment pattern (with shared modules) is the industry consensus for production workloads. Workspaces are simpler but weaker; branch-per-environment is GitOps-native but operationally complex. This note builds on the DAG and lifecycle foundations from [[02 - Advanced Terraform - Loops, Functions, Dynamic Blocks and Lifecycle|Note 02]] and connects to CI/CD pipelines covered in [[09/29 - CI-CD for ML|CI/CD for ML]].

---

## 1. Module Design Principles

### 1.1 Anatomy of a Module

A Terraform module is a directory containing `.tf` files. The module contract has three parts:

| Layer | Files | Purpose | Example |
|-------|-------|---------|---------|
| Public API | `variables.tf`, `outputs.tf` | Inputs the caller provides; outputs the caller consumes | `vpc_cidr` in, `vpc_id` out |
| Private internals | `main.tf`, `data.tf` | Resources, data sources, locals — invisible to the caller | `aws_vpc.main`, `aws_subnet.*` |
| Versioning | `versions.tf` | `required_providers`, `required_version` — declares dependencies | `terraform { required_version = ">= 1.5" }` |

A well-designed module does exactly one thing and does not leak implementation details through its outputs:

```hcl
# modules/vpc/variables.tf — The public interface
variable "cidr_block" {
  type        = string
  description = "CIDR block for the VPC"
}

variable "subnet_cidrs" {
  type        = map(string)
  description = "Map of subnet name to CIDR block"
  default     = {}
}

variable "enable_nat_gateway" {
  type        = bool
  description = "Create a NAT gateway for private subnets?"
  default     = false
}

# modules/vpc/outputs.tf — What callers can reference
output "vpc_id" {
  value       = aws_vpc.main.id
  description = "VPC ID for downstream resources"
}

output "subnet_ids" {
  value       = {for k, v in aws_subnet.main : k => v.id}
  description = "Map of subnet name to ID"
}
```

### 1.2 Module Sources and Version Pinning

The `source` argument in a `module` block tells Terraform where to find the module code:

```hcl
module "vpc_local" {
  source = "./modules/vpc"                  # Local path (relative to root module)
}

module "vpc_registry" {
  source  = "terraform-aws-modules/vpc/aws" # Terraform Registry
  version = "5.8.1"                         # ⚠️ ALWAYS pin a version — never "latest"
}

module "vpc_git" {
  source = "git::https://github.com/org/infra-modules.git//vpc?ref=v2.1.0"  # Git
}

module "vpc_s3" {
  source = "s3::https://s3.amazonaws.com/my-bucket/vpc-module.zip"  # S3 archive
}
```

💡 Always pin module versions — either with `version` (Registry), `?ref=<tag>` (Git), or a content hash (S3). The `~>` operator works for modules too: `version = "~> 5.8"` means `5.8 ≤ version < 6.0`.

¡Sorpresa! The Terraform Registry's `//` prefix in Git URLs is not a comment — it's a path separator that tells Terraform to look inside a subdirectory of the repository. `git::https://...//vpc` means "look in the `vpc/` subdirectory of this repo." Without `//`, Terraform looks at the repo root.

---

## 2. The Terraform Registry

The [Terraform Registry](https://registry.terraform.io/) is a public catalog of verified and community modules. Key modules for ML infrastructure:

| Module | Publisher | Downloads | Use Case |
|--------|-----------|-----------|----------|
| `terraform-aws-modules/vpc/aws` | AWS + HashiCorp | 15M+ | VPC with public/private subnets, NAT, VPC endpoints |
| `terraform-aws-modules/eks/aws` | AWS + HashiCorp | 8M+ | EKS cluster for containerized ML training |
| `terraform-aws-modules/rds/aws` | AWS + HashiCorp | 3M+ | RDS with multi-AZ, encryption, parameter groups |
| `terraform-aws-modules/s3-bucket/aws` | AWS + HashiCorp | 5M+ | S3 with versioning, lifecycle, encryption, CORS |
| `terraform-aws-modules/iam/aws` | AWS + HashiCorp | 4M+ | IAM roles, policies, instance profiles |

Before using a module, read its documentation: inputs (variables), outputs, and examples. Never trust a module blindly. Check:
1. Is it verified? (Blue "Verified" badge)
2. Does it pin provider versions?
3. Does it have a recent release? (More than 6 months stale is a red flag)
4. Does it expose exactly the inputs you need? Too many inputs = over-engineered; too few = inflexible.

---

## 3. Module Composition Patterns

### 3.1 Flat Composition

The root module calls several independent child modules — no module depends on another at the HCL level:

```hcl
# root/main.tf — Flat composition: VPC, compute, and storage are independent
module "networking" { source = "./modules/vpc"   cidr_block = "10.0.0.0/16" }
module "compute"    { source = "./modules/ec2"   vpc_id     = module.networking.vpc_id }
module "storage"    { source = "./modules/s3"    bucket     = "ml-models" }
```

Dependencies exist (compute needs networking), but each module is directly called by root — no module-to-module calls. This pattern works for small-to-medium infrastructure (< 50 resources).

### 3.2 Layered Composition

Modules call other modules, forming a dependency pyramid. The networking module is "layer 1," compute is "layer 2," application is "layer 3":

```hcl
# modules/compute/main.tf — Layer 2 calls layer 1
module "network" {
  source = "./modules/vpc"  # Child module dependency
  cidr_block = "10.0.0.0/16"
}

resource "aws_instance" "worker" {
  subnet_id = module.network.subnet_ids["private"]
}
```

Layered composition is powerful for ML infrastructure: the GPU training module internally provisions its own VPC, security groups, and EFS mount targets. The root module just calls `module "training_cluster"` and gets a complete, isolated training environment.

### 3.3 Terragrunt — DRY Across Environments

Terragrunt is a thin wrapper that eliminates boilerplate. Without it, every environment directory must repeat the same `terraform {}` backend block and the same `provider {}` block. With Terragrunt, you define these once and inherit them:

```hcl
# terragrunt.hcl — Root (shared across environments)
remote_state {
  backend = "s3"
  config = {
    bucket         = "ml-tfstate-${get_aws_account_id()}"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite"
  contents  = <<EOF
provider "aws" { region = "us-east-1" }
EOF
}
```

```hcl
# envs/prod/vpc/terragrunt.hcl — Environment-specific config
include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "git::https://github.com/org/modules.git//vpc?ref=v2.1.0"
}

inputs = {
  cidr_block = "10.100.0.0/16"
  enable_nat = true
}
```

**Caso real: Grubhub** rebuilt their entire AWS infrastructure using 40+ Terraform modules and Terragrunt. The networking team owns the VPC module; the compute team owns the ECS module; the data team owns the RDS module. Each module has its own CI pipeline that runs `terraform plan` and Terratest on every PR. Terragrunt generates environment configurations from a single root definition, eliminating copy-paste drift across dev, staging, and production.

---

## 4. Multi-Environment Patterns

### 4.1 Workspaces — Simple but Limited

```bash
terraform workspace new staging
terraform apply -var-file="staging.tfvars"
terraform workspace select production
terraform apply -var-file="production.tfvars"
```

Workspaces are adequate for personal projects or small teams with identical environment topologies. Limitations:

- All workspaces share the same `terraform {}` backend config (same bucket, same region)
- All workspaces share the same `provider {}` config (same region, same credentials)
- No workspace-specific module versions — you cannot pin dev to `v2.1.0` and prod to `v2.0.0`
- Deleting a workspace deletes its state — a catastrophic operational risk

**❌ Workspace antipattern**: Using `terraform.workspace` inside resource definitions to switch behavior:

```hcl
resource "aws_instance" "ml" {
  instance_type = terraform.workspace == "prod" ? "p4d.24xlarge" : "t3.medium"
  # ❌ Code branching by workspace name is fragile and hard to read
}
```

**✅ Directory-per-environment pattern**: Each environment has its own directory with explicit configs.

### 4.2 Directory-per-Environment — Industry Standard

```
infra/
├── modules/
│   ├── vpc/
│   ├── gpu-cluster/
│   └── model-registry/
├── envs/
│   ├── dev/
│   │   ├── main.tf          # module "vpc" { ... }
│   │   ├── terraform.tfvars # dev-specific values
│   │   └── backend.tf       # dev state bucket
│   ├── staging/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── prod/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
└── terragrunt.hcl            # Optional: Terragrunt root config
```

Each environment has its own state bucket, its own backend config, and its own `terraform.tfvars`. They share module code from `modules/`. A dev change cannot corrupt production state because the state files are in different S3 buckets.

```hcl
# envs/prod/main.tf — Calls the same modules as dev but with prod-sized inputs
module "vpc" {
  source = "../../modules/vpc"
  cidr_block   = "10.200.0.0/16"
  enable_nat   = true
  nat_count    = 3           # 3 NAT gateways for AZ redundancy
}

module "gpu_cluster" {
  source = "../../modules/gpu-cluster"
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnet_ids
  instance_type = "p4d.24xlarge"
  node_count    = 32
}
```

```hcl
# envs/prod/terraform.tfvars
environment = "prod"
region      = "us-east-1"
gpu_count   = 32
backup_retention_days = 90
enable_deletion_protection = true
```

**Caso real: HashiCorp** recommends directory-per-environment combined with Terraform Cloud workspaces. Each directory is linked to a Terraform Cloud workspace. CI/CD runs `terraform plan` on PR; merge triggers `terraform apply` in Terraform Cloud with Sentinel policy enforcement. HashiCorp's own internal infrastructure uses this exact pattern — dogfooding across the entire organization.

### 4.3 Branch-per-Environment (GitOps)

Each environment is a Git branch — `main` for dev, `staging` for staging, `production` for prod. Merge workflows promote changes: dev → staging → production. Full audit trail via git log, but operational complexity: you need separate CI pipelines per branch, and handling hotfixes that skip staging is messy. This pattern is native to GitOps tools like Flux and ArgoCD ([[06 - CI-CD and GitOps for Infrastructure]]) rather than standalone Terraform.

---

## 5. State Surgery

Production Terraform requires state manipulation to fix mistakes, migrate resources between modules, and import existing infrastructure:

```bash
# Rename a resource in state (without destroying it)
terraform state mv aws_instance.old_name aws_instance.new_name

# Stop managing a resource without destroying it
terraform state rm aws_s3_bucket.legacy_bucket

# Import an unmanaged resource into Terraform state
terraform import aws_instance.recovered i-0abcd1234efgh5678

# Export state to inspect or migrate
terraform state pull > state-backup.json

# Push a modified state (dangerous — use only for migration)
terraform state push state-migrated.json
```

¡Sorpresa! `terraform state mv` only renames in the state file — it does NOT touch the real resource. You can rename `aws_instance.web` to `aws_instance.api` and then `terraform apply` will show "no changes." But `aws_instance.web` must no longer exist in your `.tf` files — this is a rename, not a copy.

### 5.1 Refactoring a Resource into a Module

If you started with a flat root module and later want to refactor `aws_instance.ml_node` into `module.training.aws_instance.node`, the sequence is:

```bash
# 1. Create the module code in modules/training/main.tf
# 2. Add module block to root
# 3. Run state surgery:
terraform state mv aws_instance.ml_node module.training.aws_instance.node
# 4. Remove old resource from root main.tf
terraform plan  # Should show "No changes. Your infrastructure matches the configuration."
```

---

## 6. Module Anti-patterns

**❌ The Monolith**: One root module with 500+ resources, 80 variables, and no child modules. Impossible to review. A typo on line 3200 destroys production.

**❌ The Micro-Module**: A module that wraps a single resource with no added abstraction. `module "bucket" { source = "./bucket" }` where `bucket/` contains exactly one `aws_s3_bucket` and passes through every argument. This is indirection masquerading as abstraction.

**❌ The God Output**: A module that outputs 30 values, most of which are internal implementation details. Outputs should be the minimal set needed by callers.

**✅ Good module**: A VPC module that takes `cidr_block`, `subnet_count`, `enable_nat` as inputs and exposes `vpc_id`, `public_subnet_ids`, `private_subnet_ids` as outputs. The caller doesn't know or care about route tables, internet gateways, or NAT gateway allocation IDs — those are implementation details.

---

## 🎯 Key Takeaways

- A Terraform module is a directory with a public API (`variables.tf` + `outputs.tf`) and private internals (`main.tf`) — one concern per module
- Module sources include local paths, the Terraform Registry (always pin versions), Git repositories (use `?ref=` tags), and S3 archives
- Flat composition works for <50 resources; layered composition isolates concerns for large infrastructure; Terragrunt DRY-s up shared backend/provider configs across environments
- Directory-per-environment is the industry standard for production: each environment has its own state, backend config, and tfvars, but shares module code
- Workspaces are simpler but limited — all environments share backend config with no module version pinning per environment
- State surgery (`mv`, `rm`, `import`, `pull`/`push`) is an essential skill for refactoring, recovery, and importing brownfield infrastructure
- A good module is like a good function: one purpose, minimal interface, no side effects, and callers don't need to read the implementation

## 📦 Código de Compresión

```hcl
# envs/prod/main.tf — Root module calling three child modules
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket         = "ml-tfstate-prod"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

module "networking" {
  source = "../../modules/vpc"
  cidr_block   = "10.200.0.0/16"
  subnet_cidrs = { public = "10.200.1.0/24", private = "10.200.2.0/24" }
  enable_nat   = true
}

module "compute" {
  source = "../../modules/gpu-cluster"
  vpc_id           = module.networking.vpc_id
  subnet_id        = module.networking.public_subnet_id
  instance_type    = "g4dn.xlarge"
  node_count       = 4
  ssh_key_name     = var.ssh_key_name
}

module "storage" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "4.2.1"
  bucket  = "ml-models-prod"

  versioning = { enabled = true }
  lifecycle_rule = [{
    id      = "expire-old"
    enabled = true
    expiration = { days = 365 }
  }]
}

output "cluster_public_ips" {
  value       = module.compute.public_ips
  description = "Public IPs of GPU cluster nodes"
}

output "model_bucket_arn" {
  value       = module.storage.s3_bucket_arn
  description = "ARN of the model storage bucket"
}
```

💡 The `modules/` directory lives outside `envs/` and is shared across all environments. Each environment directory is small (~30–50 lines) because all the complexity lives in the modules. A new engineer can understand `envs/prod/main.tf` in under two minutes.

---

## References

- HashiCorp. (2024). *Terraform Modules Documentation*. https://developer.hashicorp.com/terraform/language/modules — Module creation, sources, versioning, and publishing.
- HashiCorp. (2024). *Terraform Registry*. https://registry.terraform.io/ — Verified AWS, GCP, Azure modules.
- Gruntwork. (2024). *Terragrunt Documentation*. https://terragrunt.gruntwork.io/docs/ — DRY Terraform configurations with Terragrunt.
- Brikman, Y. (2022). *Terraform: Up & Running*, 3rd ed. O'Reilly, Ch. 4: "Terraform Modules" and Ch. 8: "How to Use Terraform as a Team."
- [[01 - Terraform Fundamentals - HCL, State and Resource Graph|Note 01 — HCL and State]]
- [[02 - Advanced Terraform - Loops, Functions, Dynamic Blocks and Lifecycle|Note 02 — Advanced Terraform]]
- [[09/29 - CI-CD for ML]]
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing/01 - Fundamentos de Cloud y Modelos de Servicio|Cloud Fundamentals]]
