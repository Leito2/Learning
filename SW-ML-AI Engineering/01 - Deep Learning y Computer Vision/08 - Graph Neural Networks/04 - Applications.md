# 🧪 GNN Applications: Molecules, Recommendations, and Fraud

## Introduction

GNNs excel wherever data has relational structure. This case-study note covers three production domains — molecular property prediction, recommendation systems, and fraud detection — that showcase GNNs' unique ability to leverage graph topology for predictions impossible with traditional ML.

---

## 1. 🧬 Molecular Property Prediction (Drug Discovery)

### The Graph Representation

Molecules are natural graphs — atoms are nodes, bonds are edges:

```
Ethanol (C₂H₅OH):
    H   H
    |   |
  H-C-C-O-H
    |   |
    H   H

Graph:
  Nodes: {C₁, C₂, O, H₁, H₂, H₃, H₄, H₅, H₆}
  Edges: {(C₁,C₂), (C₁,H₁), (C₁,H₂), ...}
  Node features: atom type, valence, charge, hybridization
  Edge features: bond type (single, double, aromatic)
```

### Architecture: Message-Passing with Edge Features

Molecular GNNs incorporate edge features into message-passing:

$$
h_v^{(l+1)} = \sigma\left( W^{(l)} h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \text{MLP}\left( h_u^{(l)}, e_{uv} \right) \right)
$$

Where $e_{uv}$ encodes bond type, distance, and stereochemistry.

### Key Results

| Task | GNN Architecture | Performance |
|---|---|---|
| **Molecular solubility prediction** | GCN with edge features | 0.90 R² |
| **Drug-target binding affinity** | GAT + 3D coordinates | State-of-the-art |
| **Protein structure prediction** | SE(3)-equivariant GNN | AlphaFold |
| **Retrosynthesis planning** | GNN + Transformer | >90% top-1 accuracy |

---

## 2. 🛒 Recommendation Systems (PinSage, UberEats)

### The Graph Construction

Recommendation graphs are bipartite — users connected to items through interactions:

```
User-Item Graph:
┌──────────┐     ┌──────────┐
│  User A  │────▶│  Item X  │
│          │     └────┬─────┘
└──────────┘          │
     │                │ Co-purchased
     │ View           │
     ▼                ▼
┌──────────┐     ┌──────────┐
│  Item Y  │◀────│  User B  │
└──────────┘     └──────────┘
```

### GNN Embedding for Recommendation

GraphSAGE generates embeddings for users and items that capture their relational context:

- **User embedding** = f(items they interacted with, users with similar taste)
- **Item embedding** = f(users who interacted with it, similar items)

The recommendation score is simply: $\text{score}(u, i) = \text{embed}(u)^T \text{embed}(i)$

### Why GNNs Beat Collaborative Filtering

| Method | Uses Graph Structure? | Handles Cold Start? | Captures Higher-Order Relationships? |
|---|---|---|---|
| **Matrix Factorization (ALS)** | ❌ No | ❌ Poor | ❌ No |
| **GNN (GraphSAGE)** | ✅ Yes | ✅ Via neighbor features | ✅ Multi-hop |

---

## 3. 💰 Fraud Detection (PayPal, Alipay)

### The Graph Construction

Financial transactions form a heterogeneous graph:

```
Transaction Graph:
┌──────────┐  transfer  ┌──────────┐
│ Account A│───────────▶│ Account B│
└──────────┘            └──────────┘
     │                       │
     │ shares device         │ shares IP
     ▼                       ▼
┌──────────┐            ┌──────────┐
│ Account C│            │ Account D│
└──────────┘            └──────────┘
```

Fraud rings exhibit graph patterns invisible to feature-based models:
- Dense subgraphs of accounts sharing devices/IPs
- Rapid fund transfers through intermediary accounts
- Structural similarity to known fraud patterns

### GNN Fraud Detection Pipeline

1. **Construct graph:** Accounts = nodes, transactions = edges
2. **Node features:** Account age, transaction velocity, balance history
3. **GNN:** GAT or GraphSAGE with heterogeneous message types (transfer, device-sharing, IP-sharing)
4. **Prediction:** Per-node classification (fraudulent vs legitimate)

### Why GNNs Beat Traditional Fraud Detection

| Method | Detects | Misses |
|---|---|---|
| **Rules (amount > $10K)** | Obvious fraud | Sophisticated rings |
| **Feature-based ML (XGBoost)** | Individual suspicious accounts | Coordinated rings (features look normal per-account) |
| **GNN** | Behavioral patterns across the graph | Requires graph construction and engineering |

---

## ⚠️ Production GNN Challenges

- **Dynamic graphs:** New nodes and edges arrive continuously. GraphSAGE handles this; GCN/GAT require retraining.
- **Scale:** Billion-node graphs require neighborhood sampling (GraphSAGE) or distributed training.
- **Heterogeneity:** Real graphs have multiple node and edge types. Heterogeneous GNNs (RGCN, HGT) handle this but are more complex.
- **Feature engineering:** Node features matter as much as architecture. Invest time in feature design before tuning GNN hyperparameters.

---

## References

- Stokes et al., "A Deep Learning Approach to Antibiotic Discovery" (Cell, 2020) — GNNs for drug discovery
- Ying et al., "PinSage" (KDD, 2018) — GraphSAGE at Pinterest scale
- Liu et al., "GeniePath: Graph Neural Networks with Adaptive Receptive Paths" (AAAI, 2019) — GNNs for fraud detection at Alipay
