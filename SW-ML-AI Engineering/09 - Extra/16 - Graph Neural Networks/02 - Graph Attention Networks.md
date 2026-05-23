# 🔶 Graph Attention Networks (GAT)

## Introduction

GCN treats all neighbors equally — each neighbor's message is weighted solely by the graph structure (degree normalization). GAT introduces **attention** to GNNs: each node learns which neighbors are most relevant for its task, dynamically weighting messages based on both nodes' features. This is the same principle that made Transformers dominant in NLP, applied to graph edges.

---

## How GAT Works

### The Attention Mechanism

For a node $i$ and neighbor $j$:

1. **Attention coefficients** are computed from node features:

   $$
   e_{ij} = \text{LeakyReLU}\left( a^T [W h_i \parallel W h_j] \right)
   $$

2. **Softmax normalization** over neighbors:

   $$
   \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
   $$

3. **Weighted aggregation**:

   $$
   h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)
   $$

Where:
- $W$: Shared linear transformation
- $a$: Attention vector (learnable)
- $\parallel$: Concatenation
- $\alpha_{ij}$: Attention weight — how much node $i$ attends to node $j$

### Multi-Head Attention

Like Transformers, GAT uses multiple attention heads for stability:

$$
h_i' = \mathop{\|}_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j \right)
$$

Each head learns a different attention pattern — some heads may focus on structural similarity, others on feature similarity.

---

## GAT vs GCN

| Property | GCN | GAT |
|---|---|---|
| **Neighbor weighting** | Fixed (degree normalization) | Learned (attention) |
| **Dynamic weights** | No (same for all inputs) | Yes (depends on features) |
| **Interpretability** | None | Attention weights show which neighbors matter |
| **Computational cost** | O(E × F) | O(E × F × heads) — more expensive |
| **Performance** | Baseline | Typically 1-5% better |

---

## Implementation

```python
import torch
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # One linear projection per head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        # Attention vector per head
        self.a = nn.Parameter(torch.randn(1, num_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, edge_index):
        """
        X: Node features (N, in_features)
        edge_index: Edge list (2, E)
        """
        N = X.size(0)

        # Linear transformation
        h = self.W(X).view(N, self.num_heads, self.out_features)  # (N, H, F')

        # Source and target node features
        src, dst = edge_index  # (E,), (E,)
        h_src = h[src]  # (E, H, F')
        h_dst = h[dst]

        # Concatenate and compute attention
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # (E, H, 2F')
        e = self.leaky_relu((h_cat * self.a).sum(-1))  # (E, H)

        # Softmax per source node
        from torch_scatter import scatter_softmax
        alpha = scatter_softmax(e, dst, dim=0)  # (E, H)

        # Weighted aggregation
        alpha = self.dropout(alpha)
        out = torch.zeros(N, self.num_heads, self.out_features, device=X.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_src)

        return out.mean(dim=1)  # Average heads for output
```

---

## References

- Veličković et al., "Graph Attention Networks" (ICLR, 2018)
- [PyTorch Geometric: GAT](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html)
