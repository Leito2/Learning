# 🔷 Graph Convolutional Networks (GCN)

## Introduction

GCN is the foundational GNN architecture — it generalizes the convolution operation from grid-structured data (images) to graph-structured data (nodes + edges). A GCN layer updates each node's representation by aggregating features from its neighbors, creating a local receptive field analogous to a CNN's kernel but adapted to arbitrary graph topology.

This module covers the message-passing framework, the GCN propagation rule, and how stacking GCN layers builds increasingly global node representations.

---

## Message-Passing Framework

All GNNs follow the message-passing paradigm:

```
For each node v:
  1. Message: Each neighbor u sends a message m_uv = MSG(h_u, h_v, e_uv)
  2. Aggregate: v aggregates all messages: a_v = AGG({m_uv : u ∈ N(v)})
  3. Update: v updates its embedding: h_v' = UPDATE(h_v, a_v)
```

### GCN Propagation Rule

The GCN layer implements message-passing with a specific normalization:

$$
H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)
$$

Where:
- $H^{(l)}$: Node features at layer $l$ (N×F matrix)
- $\tilde{A} = A + I$: Adjacency matrix with self-loops
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$: Degree matrix
- $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$: Symmetric normalization (prevents feature explosion for high-degree nodes)
- $W^{(l)}$: Learnable weight matrix
- $\sigma$: Non-linearity (ReLU)

### Intuition

Each GCN layer averages neighbor features with self-features, then applies a learned linear transformation + non-linearity. Stacking K layers = each node sees its K-hop neighborhood:

```
Layer 1: Node sees immediate neighbors (1-hop)
Layer 2: Node sees neighbors-of-neighbors (2-hop)
Layer 3: Node sees 3-hop neighborhood
```

---

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, X, A_norm):
        """
        X: Node features (N, in_features)
        A_norm: Normalized adjacency matrix (N, N)
        """
        # Aggregate: A_norm @ X averages neighbor features
        support = A_norm @ X
        # Update: linear transformation + activation
        return F.relu(support @ self.W)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, out_features))

    def forward(self, X, A_norm):
        for i, layer in enumerate(self.layers):
            X = layer(X, A_norm)
        return F.log_softmax(X, dim=1)  # Node classification

def normalize_adj(A):
    """Compute D^{-1/2} (A+I) D^{-1/2}"""
    A_hat = A + torch.eye(A.size(0))  # Add self-loops
    D_hat_inv_sqrt = torch.diag(A_hat.sum(1).pow(-0.5))
    return D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
```

---

## GCN Properties

| Property | GCN | CNN (for comparison) |
|---|---|---|
| **Receptive field** | K-hop neighborhood | K×K pixel window |
| **Weight sharing** | Same W across all nodes | Same kernel across all positions |
| **Permutation invariance** | Yes (node order doesn't matter) | No (pixel order matters) |
| **Transductive** | Yes (requires full graph at training) | No (trained on any image) |
| **Scalability** | Full-batch (all nodes in memory) | Mini-batch (samples) |

---

## ⚠️ Limitations

- **Transductive only:** GCN requires the full graph at training time. Cannot predict on unseen nodes — this is why GraphSAGE was developed.
- **Equal neighbor weighting:** All neighbors contribute equally (modulo degree normalization). GAT fixes this with attention.
- **Oversmoothing:** 3+ layers cause all node embeddings to converge — losing local distinctions. Remedies: residual connections, fewer layers.

---

## References

- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR, 2017)
