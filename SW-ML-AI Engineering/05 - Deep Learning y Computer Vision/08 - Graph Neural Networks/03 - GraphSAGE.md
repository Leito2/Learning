# 🔷 GraphSAGE: Inductive Representation Learning

## Introduction

GCN and GAT are transductive — they require the entire graph at training time and cannot generate embeddings for nodes unseen during training. GraphSAGE (Graph SAmple and aggreGatE) solves this by learning a function that maps node features to embeddings, enabling inference on completely new nodes and new subgraphs without retraining. This makes GraphSAGE the GNN of choice for production systems where the graph constantly changes — new users join Pinterest, new proteins are discovered, new transactions appear in fraud detection.

---

## How GraphSAGE Works

Instead of the full-graph GCN propagation, GraphSAGE samples a fixed-size neighborhood and applies a learned aggregation function:

```
For each node v:
  1. Sample k neighbors from N(v)
  2. Aggregate their features: h_N(v) = AGG({h_u : u ∈ sampled neighbors})
  3. Update: h_v' = σ(W · CONCAT(h_v, h_N(v)))
```

### Aggregation Functions

| Aggregator | Formula | Properties |
|---|---|---|
| **Mean** | $h_{N(v)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u} h_u$ | Simple, effective baseline |
| **LSTM** | LSTM over random permutation of neighbors | Expressive, but not permutation invariant |
| **Pooling** | $h_{N(v)} = \max(\{ \sigma(W h_u + b) : u \in \mathcal{N}(v) \})$ | Element-wise max over transformed neighbor features |

### Sampling Strategy

GraphSAGE samples neighbors at each layer:

```
Layer 1: Sample S₁ neighbors per node  → Approximates 1-hop neighborhood
Layer 2: Sample S₂ neighbors from each sampled node → Approximates 2-hop
```

This bounds computation per batch to O(S₁ × S₂) regardless of graph size — enabling GNNs on billion-node graphs.

---

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggr="mean"):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features)
        self.W_neigh = nn.Linear(in_features, out_features)
        self.aggr = aggr

    def forward(self, X, edge_index):
        """X: (N, F), edge_index: (2, E)"""
        row, col = edge_index  # row = target, col = source

        # Aggregate neighbor features
        if self.aggr == "mean":
            from torch_scatter import scatter_mean
            neigh_features = scatter_mean(X[col], row, dim=0)
        elif self.aggr == "pool":
            from torch_scatter import scatter_max
            neigh_features, _ = scatter_max(F.relu(self.W_neigh(X[col])), row, dim=0)

        # Update: combine self and neighbor features
        h_self = self.W_self(X)
        h_neigh = self.W_neigh(neigh_features)
        return F.relu(h_self + h_neigh)

class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(in_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        self.layers.append(GraphSAGELayer(hidden_dim, out_features))

    def forward(self, X, edge_index):
        for layer in self.layers:
            X = layer(X, edge_index)
        return F.log_softmax(X, dim=1)
```

---

## GraphSAGE vs GCN vs GAT

| Property | GCN | GAT | GraphSAGE |
|---|---|---|---|
| **Training** | Full graph required | Full graph required | Mini-batch (sampled neighbors) |
| **New nodes** | Cannot embed | Cannot embed | Can embed (inductive) |
| **Scalability** | Memory-bound (full adjacency) | Memory-bound | Computation-bound (sampling) |
| **Graph size** | Millions of nodes | Millions | Billions of nodes |
| **Production** | Rare | Rare | Pinterest, UberEats, PayPal |

---

## 🌍 Production Use: Pinterest PinSage

Pinterest's PinSage is the most famous GraphSAGE production deployment:
- **Graph:** 3 billion nodes (pins + boards), 18 billion edges
- **Task:** Generate embeddings for related pin recommendation
- **Architecture:** GraphSAGE with importance-based neighbor sampling (random walks to identify "important" neighbors)
- **Scale:** Trains on billions of nodes, serves embeddings for millions of requests

---

## References

- Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS, 2017)
- Ying et al., "PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (KDD, 2018)
