# 🔄 02 - Advanced Terraform — Loops, Functions, Dynamic Blocks, and Lifecycle

## 🎯 Learning Objectives

- Choose correctly between `count`, `for_each`, and `for` expressions — and understand why the wrong choice destroys production resources
- Use `for` expressions for collection transformation with filtering, grouping, and type conversion
- Generate nested blocks programmatically with `dynamic` blocks instead of copy-paste
- Apply lifecycle rules (`create_before_destroy`, `prevent_destroy`, `ignore_changes`) for zero-downtime deployments
- Leverage HCL's function library (`templatefile`, `try`, `can`, `flatten`, `jsonencode`, `merge`) to write DRY, resilient configurations
- Combine conditionals with `for_each` for conditional resource creation
- Understand when provisioners are a last resort and when to use cloud-init, Packer, or Ansible instead

## Introduction

Basic Terraform treats each resource as a unique, hand-written block. You write `resource "aws_instance" "node_0"`, then `resource "aws_instance" "node_1"`, then `resource "aws_instance" "node_2"`. This works for three instances. It does not work for a 64-GPU training cluster. Production Terraform requires iteration: create N resources from a data structure, conditionally skip resources based on environment flags, and generate nested blocks (20 ingress rules for a security group) from a single template. These patterns separate "demo Terraform" — the kind you write on day one of learning — from "production Terraform" that manages infrastructure at scale.

The three iteration mechanisms — `count`, `for_each`, and `for` — are superficially similar but have radically different semantic guarantees. Choosing `count` over `for_each` is the single most common cause of production destruction incidents in Terraform history. This note dissects each mechanism, explains when the wrong choice causes disasters, and builds toward a complete mental model of lifecycle rules, dynamic blocks, and the function library that glues it all together. Scaled ML infrastructure — where a single `terraform apply` might manage 200+ resources across VPCs, GPU clusters, model registries, and monitoring dashboards — demands these patterns. Foundational HCL and DAG concepts are covered in [[01 - Terraform Fundamentals - HCL, State and Resource Graph|Note 01]]; deployment orchestration connects to [[09/29 - CI-CD for ML|CI/CD for ML]].

---

## 1. The Three Iteration Mechanisms

### 1.1 `count` — Indexed Repetition (Use Sparingly)

`count = N` creates N identical resources distinguished by their index (0, 1, 2, ..., N-1). The index becomes part of the resource address: `aws_instance.nodes[0]`, `aws_instance.nodes[1]`.

```hcl
variable "subnet_cidrs" {
  type    = list(string)
  default = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

resource "aws_subnet" "by_count" {
  count      = length(var.subnet_cidrs)
  vpc_id     = aws_vpc.main.id
  cidr_block = var.subnet_cidrs[count.index]
}
```

The fatal flaw: **index instability**. If you remove `"10.0.1.0/24"` from the list (position 0), every element shifts. `subnet[0]` was the old `10.0.1.0/24` — it now points to `10.0.2.0/24`. Terraform sees: destroy `subnet[2]` (the last one, which "disappeared"), and recreate `subnet[0]` and `subnet[1]` with shifted values. You just destroyed and recreated subnets 1 and 2 — potentially taking down every EC2 instance in them.

**Caso real:** A fintech startup lost their production PostgreSQL RDS instance when they used `count` for their database replicas. Adding a read replica shifted all indices; Terraform destroyed the master (index 0 became the new replica) and recreated it from scratch. The `for_each` pattern with stable keys (db name as key) would have added the new replica as a separate resource without touching existing databases.

**❌ `count` antipattern**:
```hcl
# Adding or removing a subnet DESTROYS and RECREATES other subnets
resource "aws_subnet" "bad" {
  count      = length(var.subnets)    # Index-based — order matters
  cidr_block = var.subnets[count.index]
}
```

**✅ `for_each` pattern**:
```hcl
# Each subnet has a stable key — reordering does nothing
resource "aws_subnet" "good" {
  for_each   = var.subnets            # Map: { "web" = "10.0.1.0/24", "db" = "10.0.2.0/24" }
  cidr_block = each.value
}
```

### 1.2 `for_each` — Keyed Repetition (Always Prefer)

`for_each` iterates over a map or `set(string)` and uses each key as a stable resource identifier. The resource address becomes `aws_subnet.good["web"]`, `aws_subnet.good["db"]`. Removing the `"web"` key only destroys one resource — nothing shifts.

```hcl
variable "gpu_nodes" {
  type = map(object({
    instance_type = string
    gpu_count     = number
    zone          = string
  }))
  default = {
    "trainer-0" = { instance_type = "p3.2xlarge", gpu_count = 1, zone = "us-east-1a" }
    "trainer-1" = { instance_type = "p3.2xlarge", gpu_count = 1, zone = "us-east-1b" }
  }
}

resource "aws_instance" "gpu_cluster" {
  for_each      = var.gpu_nodes
  ami           = data.aws_ami.deep_learning.id
  instance_type = each.value.instance_type
  availability_zone = each.value.zone

  tags = {
    Name = each.key                          # "trainer-0", "trainer-1"
    GPUs = each.value.gpu_count
  }
}
```

¡Sorpresa! Removing a key from the `for_each` map **destroys** that specific resource — not just ignores it. Terraform tracks each keyed resource separately. If you want to stop managing a resource without destroying it, use `terraform state rm` first.

`for_each` also works with `toset()` for lists:

```hcl
resource "aws_iam_user" "team" {
  for_each = toset(["alice", "bob", "carol"])
  name     = each.key
}
```

`toset(["alice", "bob", "carol"])` creates `set(string)` with elements `"alice"`, `"bob"`, `"carol"`. Each becomes a stable resource key. But if a team member leaves, removing `"bob"` destroys Bob's IAM user on the next apply.

### 1.3 `for` Expressions — Collection Transformation

`for` expressions transform collections without creating resources. They are functional map/filter operations evaluated at plan time:

```hcl
# List transformation: extract one field
instance_types = [for node in var.gpu_nodes : node.instance_type]
# Result: ["p3.2xlarge", "p3.2xlarge"]

# Map transformation: build a lookup
node_by_zone = {for node in var.gpu_nodes : node.zone => node}
# Result: { "us-east-1a" = {...}, "us-east-1b" = {...} }

# Filter with if
public_nodes = [for node in var.gpu_nodes : node if node.zone == "us-east-1a"]

# Nested flatten
all_ports = flatten([for sg in var.security_groups : sg.ports])
```

The `...` grouping mode (with `...` after the value expression) groups multiple values under the same key:

```hcl
# Group by zone: { "us-east-1a" = [node1, node2], "us-east-1b" = [node3] }
nodes_by_zone = {for node in var.gpu_nodes : node.zone => node...}
```

---

## 2. Dynamic Blocks — Generating Nested Blocks

Security groups, IAM policies, load balancer listeners — all use nested blocks for repeated structure. Without dynamic blocks, you write each one manually:

```hcl
# ❌ Antipattern: copy-paste nested blocks
resource "aws_security_group" "ml_cluster" {
  ingress { from_port = 22 to_port = 22 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 443 to_port = 443 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 8080 to_port = 8080 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  # ... 20 more lines for a real ML cluster
}
```

Dynamic blocks generate nested blocks from a collection:

```hcl
# ✅ Dynamic block pattern
variable "ingress_rules" {
  type = list(object({
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
    description = string
  }))
  default = [
    { from_port = 22, to_port = 22, protocol = "tcp",
      cidr_blocks = ["10.0.0.0/8"], description = "SSH from VPN" },
    { from_port = 443, to_port = 443, protocol = "tcp",
      cidr_blocks = ["0.0.0.0/0"], description = "HTTPS" },
    { from_port = 8080, to_port = 8080, protocol = "tcp",
      cidr_blocks = ["0.0.0.0/0"], description = "Model serving" },
  ]
}

resource "aws_security_group" "ml_cluster" {
  name = "ml-cluster-sg"

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = ingress.value.description
    }
  }
}
```

💡 The `iterator` argument renames the block variable for readability when nesting dynamic blocks:

```hcl
dynamic "ingress" {
  for_each = var.ingress_rules
  iterator = rule   # Access as rule.value instead of ingress.value
  content {
    from_port = rule.value.from_port
  }
}
```

---

## 3. HCL Functions Deep Dive

HCL ships with ~60 built-in functions. The ones that matter most for production configurations:

### 3.1 `templatefile()` — Templated Configuration

`templatefile(path, vars)` reads a file from disk and substitutes variables using `${var}` interpolation. This is how you DRY up `user_data` scripts, IAM policy documents, and cloud-init configs:

```hcl
resource "aws_instance" "trainer" {
  user_data = templatefile("${path.module}/scripts/init.sh.tpl", {
    model_bucket    = aws_s3_bucket.models.bucket
    training_port   = var.training_port
    gpu_memory_fraction = var.gpu_memory_fraction
  })
}
```

```bash
# scripts/init.sh.tpl
#!/bin/bash
echo "Model bucket: ${model_bucket}" >> /var/log/ml-init.log
nvidia-smi -c EXCLUSIVE_PROCESS
python -m torch.distributed.launch --nproc_per_node=${gpu_count} train.py
```

### 3.2 `try()` and `can()` — Graceful Error Handling

`try(val1, val2, ..., fallback)` returns the first non-error value. Critical when optional attributes may not exist in a data source:

```hcl
# ami data source might not have tags — try() prevents a plan-time crash
owner_email = try(data.aws_ami.ubuntu.tags["OwnerEmail"], "unknown@company.com")
```

`can(expression)` returns `true` if the expression evaluates without error. Use in validation and conditional logic:

```hcl
validation {
  condition     = can(regex("^ami-", var.custom_ami))
  error_message = "AMI ID must start with 'ami-'."
}
```

### 3.3 Collection Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `flatten()` | Flatten nested lists | `flatten([[1,2],[3,4]])` → `[1,2,3,4]` |
| `merge()` | Merge maps (right wins) | `merge({a=1}, {b=2})` → `{a=1, b=2}` |
| `jsonencode()` | Serialize to JSON | `jsonencode({port = 8080})` → `"{\"port\":8080}"` |
| `base64encode()` | Base64 encode string | `base64encode("user_data")` → `"dXNlcl9kYXRh"` |
| `lookup()` | Map access with default | `lookup(var.tags, "env", "dev")` |
| `element()` | List access with wrap | `element(["a","b","c"], 5)` → `"c"` (wraps) |
| `one()` | Extract single from list | `one(aws_subnet.main[*].id)` — errors if >1 |

---

## 4. Lifecycle Rules — Controlling Update Behavior

The `lifecycle` meta-argument inside a `resource` block controls what happens during updates (in-place, destroy-create, or ignore):

```hcl
resource "aws_instance" "training" {

  lifecycle {
    create_before_destroy = true   # Create new instance before destroying old
    prevent_destroy       = true   # Block `terraform destroy` entirely
    ignore_changes        = [ami, tags["LastDeployed"]]  # Ignore external changes
  }
}
```

### 4.1 `create_before_destroy = true`

Default behavior: Terraform destroys the old resource, then creates the new one. If the resource must change in a way the provider can't do in-place (e.g., changing an EC2 instance's AMI), you get downtime.

With `create_before_destroy`, Terraform creates the replacement first, then destroys the old one. This enables zero-downtime deployments of ALBs, RDS instances, and GPU training nodes:

```hcl
resource "aws_launch_template" "ml_workers" {
  image_id = data.aws_ami.deep_learning.id
  # ...

  lifecycle {
    create_before_destroy = true  # New ASG instances launch before old ones terminate
  }
}
```

⚠️ `create_before_destroy` can fail if the old and new resources have conflicting unique constraints (e.g., two RDS instances with the same identifier). You may need to temporarily rename the resource.

### 4.2 `prevent_destroy = true`

Blocks accidental `terraform destroy`. Use on production databases, S3 buckets with irreplaceable data, and KMS keys:

```hcl
resource "aws_db_instance" "production" {
  identifier = "ml-prod-db"

  lifecycle {
    prevent_destroy = true   # ⚠️ Must remove this line before intentional destroy
  }
}
```

If you need to destroy the resource, remove the `prevent_destroy` line, run `terraform apply` (which updates the lifecycle in state without destroying), then `terraform destroy`.

### 4.3 `ignore_changes`

Tells Terraform to ignore external modifications to specified attributes. Essential when other tools modify resources Terraform manages:

```hcl
resource "aws_autoscaling_group" "ml_workers" {
  desired_capacity = var.min_workers
  # When auto-scaling policy changes desired_capacity, don't revert it

  lifecycle {
    ignore_changes = [desired_capacity]  # Let auto-scaling do its job
  }
}
```

**Caso real: Spotify** runs Terraform-managed Kubernetes clusters where the cluster autoscaler continuously adjusts node counts. Their ASG definitions use `ignore_changes = [desired_capacity, max_size, min_size]` because the autoscaler (not Terraform) is the source of truth for cluster size. Without `ignore_changes`, Terraform would reset the cluster to minimum size on every apply.

---

## 5. Conditional Resource Creation

### 5.1 `count` with Conditionals

```hcl
resource "aws_nat_gateway" "optional" {
  count         = var.enable_nat ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
}
```

When `var.enable_nat = false`, `count = 0` — no NAT gateway is created. `aws_nat_gateway.optional` resolves to an empty list `[]`. References handle this with `one()`: `nat_ip = one(aws_nat_gateway.optional[*].public_ip)`.

### 5.2 `for_each` with Conditionals

```hcl
resource "aws_s3_bucket" "environments" {
  for_each = {
    for env, config in var.environments :
    env => config
    if config.create_bucket  # ⚠️ Filters at plan time
  }
  bucket = "${each.key}-ml-models"
}
```

For a single conditional resource, use the empty-map trick:

```hcl
resource "aws_backup_plan" "optional" {
  for_each = var.enable_backups ? { "backup" = true } : {}
  name     = "ml-backup-plan"
  # ...
}
```

When `var.enable_backups = false`, `for_each` iterates over `{}` — zero resources. When `true`, one resource with key `"backup"`. No `count.index` ambiguity.

¡Sorpresa! `for_each = var.enable_backups ? toset(["backup"]) : toset([])` also works, but the empty map `{}` is more idiomatic because the key "backup" provides self-documentation.

---

## 6. Provisioners — The Last Resort

Provisioners (`local-exec`, `remote-exec`, `file`) run scripts on resource creation or destruction. They break Terraform's declarative model: Terraform cannot track what the script does, cannot diff its effects, and cannot roll it back.

```hcl
resource "aws_instance" "ml_node" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "g4dn.xlarge"

  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y nvidia-driver-535",
    ]
    connection {
      type        = "ssh"
      host        = self.public_ip
      user        = "ubuntu"
      private_key = file("~/.ssh/ml-key.pem")
    }
  }
}
```

⚠️ Provisioners are a **last resort**. Alternatives in order of preference:
1. **Cloud-init scripts** in `user_data`: first-boot initialization, idempotent, no SSH required
2. **Packer**: pre-bake AMIs with all software installed — the instance boots ready
3. **Ansible**: if you truly need post-boot configuration, use Ansible with a Terraform provisioner that only invokes `ansible-playbook` (and even then, prefer Ansible's own dynamic inventory)

💡 The only defensible use of `local-exec` is `terraform output` post-processing: `command = "echo '${aws_instance.ml_node.public_ip}' > inventory.ini"`.

---

## 🎯 Key Takeaways

- `for_each` (keyed) is strictly superior to `count` (indexed) for resource creation: it survives reordering, adding, and removing elements without destroying unrelated resources
- `for` expressions transform collections at plan time: map, filter, group, and flatten — think of them as functional pipelines for Terraform data
- Dynamic blocks generate N nested blocks from a single collection definition, eliminating DRY violations in security groups, IAM policies, and load balancer rules
- Lifecycle rules (`create_before_destroy`, `prevent_destroy`, `ignore_changes`) give you fine-grained control over destructive operations — essential for production database and networking resources
- `templatefile()` separates configuration logic (HCL) from runtime scripts (bash, cloud-init, IAM JSON), keeping your Terraform modules clean and testable
- `try()` and `can()` provide graceful fallbacks for missing attributes — without them, a single `null` value crashes the entire plan
- Provisioners break idempotency; prefer cloud-init, Packer, or Ansible for post-provisioning configuration

## 📦 Código de Compresión

```hcl
variable "training_nodes" {
  type = map(object({
    instance_type = string
    zone          = string
    ports         = list(number)
  }))
}

locals {
  node_ami = data.aws_ami.ubuntu.id
  userdata = templatefile("${path.module}/scripts/init.sh.tpl", {
    model_bucket = aws_s3_bucket.models.id
  })
}

resource "aws_security_group" "ml" {
  name = "ml-training-sg"

  dynamic "ingress" {
    for_each = flatten([for n in var.training_nodes : [
      for p in n.ports : { from_port = p, to_port = p, protocol = "tcp", cidr = "10.0.0.0/8" }
    ]])
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = [ingress.value.cidr]
    }
  }
}

resource "aws_instance" "trainer" {
  for_each      = { for k, v in var.training_nodes : k => v if v.instance_type != "t2.micro" }
  ami           = local.node_ami
  instance_type = each.value.instance_type

  user_data = local.userdata
  vpc_security_group_ids = [aws_security_group.ml.id]

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "trainer-${each.key}", Zone = each.value.zone }
}

output "trainer_ips" {
  value = {for k, inst in aws_instance.trainer : k => inst.private_ip}
}
```

💡 The `dynamic "ingress"` with `flatten` and nested `for` expressions is a one-liner that generates exactly the right set of ingress rules from the `training_nodes` data — no copy-paste, no manual count.

---

## References

- HashiCorp. (2024). *Terraform Language: Expressions*. https://developer.hashicorp.com/terraform/language/expressions — `for`, `for_each`, `count`, `dynamic`, and all built-in functions.
- HashiCorp. (2024). *Terraform Language: Lifecycle Meta-Argument*. https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle — `create_before_destroy`, `prevent_destroy`, `ignore_changes`.
- Brikman, Y. (2022). *Terraform: Up & Running*, 3rd ed. O'Reilly, Ch. 5: "Terraform Tips and Tricks: Loops, If-Statements, Deployment, and Gotchas."
- HashiCorp. (2024). *Terraform Registry*. https://registry.terraform.io/ — Reference implementations using `for_each` and `dynamic` in production modules.
- [[01 - Terraform Fundamentals - HCL, State and Resource Graph|Note 01 — HCL and State]]
- [[03 - Terraform Modules - Workspaces and Multi-Environment Patterns|Note 03 — Modules]]
- [[09/29 - CI-CD for ML]]
- [[10 - Cloud, Infra y Backend/22 - Cloud Computing/01 - Fundamentos de Cloud y Modelos de Servicio|Cloud Fundamentals]]
