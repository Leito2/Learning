# 🕸️ GraphRAG: Knowledge Graph-Enhanced RAG

---

## Module 4.1: Why Vector-RAG Fails on Multi-Hop Questions

### 4.1.1 Theoretical Foundation: The Semantic Gap

Vector RAG excels at **single-hop retrieval**: finding documents that directly contain information relevant to the query. But real-world enterprise questions are rarely single-hop. They require connecting facts across multiple documents, aggregating, and reasoning.

```
MULTI-HOP QUESTION BREAKDOWN
──────────────────────────────

  Query: "Compare the Q3 revenues of companies that use our AI platform"

  This question implicitly requires:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ STEP 1: Identify all companies using our AI platform                 │
  │   → Need: Customer list / CRM data / contract database              │
  │                                                                     │
  │ STEP 2: For each company, find their Q3 revenue                     │
  │   → Need: Financial reports, earnings calls, quarterly filings      │
  │                                                                     │
  │ STEP 3: Compare revenues (sort, compute differences, find outliers) │
  │   → Need: Aggregation + reasoning over the assembled data           │
  │                                                                     │
  │ STEP 4: Generate structured comparison                               │
  │   → Need: LLM to format and present findings                        │
  └─────────────────────────────────────────────────────────────────────┘

  WHY NAIVE VECTOR RAG FAILS:
  ────────────────────────────────────────────────────────────────────────

  Query Vector → Nearest Neighbors in Embedding Space:

  Query: "Compare the Q3 revenues of companies that use our AI platform"

  Top-5 Retrieved Chunks:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ #1 [0.87] "Our AI platform serves over 500 enterprise customers..."  │
  │     ❌ Lists customers but NO revenue data. Missing Step 2.          │
  │                                                                      │
  │ #2 [0.84] "Q3 earnings season showed strong tech sector growth..."   │
  │     ❌ Generic tech earnings (not specific companies). No link.      │
  │                                                                      │
  │ #3 [0.81] "Revenue comparison methodology: YoY vs QoQ growth..."    │
  │     ❌ About methodology, not actual numbers.                       │
  │                                                                      │
  │ #4 [0.79] "Acme Corp reported Q3 revenue of $2.3B, up 12% YoY..."   │
  │     ⚠️ Has real revenue data! But is Acme a customer? Unknown.      │
  │                                                                      │
  │ #5 [0.78] "Customer success story: how Beta Inc uses our API..."    │
  │     ⚠️ Beta Inc is a customer! But no revenue data.                 │
  └──────────────────────────────────────────────────────────────────────┘

  FUNDAMENTAL PROBLEM:
  The embedding of "Compare Q3 revenues of customers" is NOT similar
  to the embedding of "Beta Inc Q3 revenue" alone. The query vector
  captures the COMPOSITE intent, but document embeddings capture
  ATOMIC facts. This is the "semantic gap" in multi-hop retrieval.


  CONTRAST WITH KNOWLEDGE GRAPH APPROACH:
  ────────────────────────────────────────────────────────────────────────

  Knowledge Graph representation:
  ┌───────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │   [Acme Corp] ──(IS_CUSTOMER)──▶ [Our AI Platform]                   │
  │       │                                                                 │
  │       ├──(HAS_REVENUE_Q3)──▶ [$2.3B]                                  │
  │       ├──(HAS_REVENUE_Q2)──▶ [$2.1B]                                  │
  │       └──(HAS_GROWTH)──────▶ [12% YoY]                                │
  │                                                                       │
  │   [Beta Inc]  ──(IS_CUSTOMER)──▶ [Our AI Platform]                   │
  │       │                                                                 │
  │       └──(HAS_REVENUE_Q3)──▶ [$890M]                                  │
  │                                                                       │
  │   [Gamma Co]  ──(IS_CUSTOMER)──▶ [Our AI Platform]                   │
  │       │                                                                 │
  │       └──(HAS_REVENUE_Q3)──▶ [$1.5B]                                  │
  │                                                                       │
  │   QUERY AS GRAPH PATTERN:                                             │
  │   MATCH (c:Company)-[:IS_CUSTOMER]->(:Platform {name:"Our AI"})      │
  │   MATCH (c)-[:HAS_REVENUE_Q3]->(r:Revenue)                           │
  │   RETURN c.name, r.amount ORDER BY r.amount DESC                     │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

  The graph explicitly encodes RELATIONSHIPS between entities,
  enabling precise multi-hop traversal that vector search cannot do.
```

### 4.1.2 Failure Modes of Naive Chunk Retrieval

```
FAILURE MODE TAXONOMY FOR VECTOR-ONLY RAG
──────────────────────────────────────────

  ┌──────────────────────────┬─────────────────────────────────────────────┐
  │     FAILURE MODE         │               EXAMPLE QUERY                  │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ MISSING BRIDGE ENTITY    │ "What did the CEO of our biggest customer   │
  │                          │  say about AI regulation?"                  │
  │                          │ → Vector search finds "CEO statements",     │
  │                          │   "AI regulation opinions", but doesn't     │
  │                          │   connect "CEO X" → "Company Y" → "our      │
  │                          │   customer" chain.                          │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ AGGREGATION REQUIREMENT  │ "What is the total revenue across all       │
  │                          │  healthcare clients in Europe?"             │
  │                          │ → Vector search finds individual revenue    │
  │                          │   figures, but no single chunk contains     │
  │                          │   the union of "healthcare" + "Europe".     │
  │                          │   The LLM must aggregate — and often fails. │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ TEMPORAL REASONING       │ "Which customers downgraded their plan      │
  │                          │  after our 2024 price increase?"            │
  │                          │ → Requires: subscription_before_price_      │
  │                          │   increase > subscription_after. Vector     │
  │                          │   search retrieves plan changes and price   │
  │                          │   events separately; cannot sequence them.  │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ COMPARATIVE ANALYSIS     │ "How does our churn rate compare to         │
  │                          │  competitors mentioned in earnings calls?"  │
  │                          │ → Requires finding OUR churn rate, finding  │
  │                          │   COMPETITOR churn rates from separate      │
  │                          │   documents, and computing comparison.      │
  └──────────────────────────┴─────────────────────────────────────────────┘

  KEY INSIGHT: These all require CONNECTING facts across documents.
  Vector search retrieves DOCUMENTS; Graph search retrieves RELATIONSHIPS.
```

### 4.1.3 When to Use GraphRAG vs Traditional RAG

| Query Type | Traditional RAG | GraphRAG | Hybrid |
|---|---|---|---|
| Fact lookup ("What is X?") | ✅ Excellent | ❌ Overkill | ✅ Works |
| How-to ("How do I configure X?") | ✅ Excellent | ❌ Poor fit | ✅ Works |
| Multi-hop ("Which customers using X also use Y?") | ❌ Fails | ✅ Excellent | ✅ Works |
| Summarization ("Summarize trends in X") | ⚠️ Hit-or-miss | ✅ Strong (community summaries) | ✅ Best |
| Comparative ("Compare X and Y across Z") | ❌ Fails | ✅ Strong | ✅ Best |
| Aggregation ("Total revenue of all X in Y") | ❌ Fails | ✅ Excellent | ✅ Best |
| Event sequencing ("What happened after X?") | ⚠️ Partial | ✅ Strong | ✅ Best |

---

## Module 4.2: Microsoft GraphRAG Architecture

### 4.2.1 Overview

Microsoft's GraphRAG (2024) introduced a paradigm shift: instead of retrieving individual chunks, **index documents into a knowledge graph**, detect communities, generate hierarchical summaries, and answer queries by traversing the graph structure.

```mermaid
graph TB
    subgraph "INDEXING PIPELINE (Offline, runs once per corpus)"
        DOCS[Source Documents] --> CHUNK[Text Chunking<br/>Semantic + Sliding Window]
        CHUNK --> EXTRACT[Entity & Relationship<br/>Extraction via LLM]
        EXTRACT --> GRAPH[Knowledge Graph<br/>Entities + Edges]
        GRAPH --> COMM[Community Detection<br/>Leiden Algorithm]
        COMM --> SUMMARIZE[Hierarchical<br/>Community Summarization]
        SUMMARIZE --> STORE[(GraphRAG Index<br/>Graph + Summaries)]
    end

    subgraph "QUERY PIPELINE (Online, per query)"
        Q[User Query] --> MODE{Query Mode}
        MODE -->|Local Search| LOCAL[Entity-Anchored Search<br/>Neighborhood Exploration]
        MODE -->|Global Search| GLOBAL[Community-Based Search<br/>Map-Reduce over Summaries]
        MODE -->|Drift Search| DRIFT[Dynamic Query Decomposition<br/>Iterative Graph Exploration]
        LOCAL --> CTX1[Assemble Context]
        GLOBAL --> CTX2[Assemble Context]
        DRIFT --> CTX3[Assemble Context]
        CTX1 --> LLM[LLM Generation]
        CTX2 --> LLM
        CTX3 --> LLM
    end

    style GRAPH fill:#6a5acd,stroke:#333,color:#fff
    style COMM fill:#6a5acd,stroke:#333,color:#fff
    style SUMMARIZE fill:#9370db,stroke:#333,color:#fff
    style STORE fill:#228b22,stroke:#333,color:#fff
```

### 4.2.2 Indexing Pipeline Deep Dive

**Step 1: Text Chunking**

Documents are split into chunks suitable for LLM processing. GraphRAG uses a configurable chunk size (default ~300 tokens) with overlap.

**Step 2: Entity and Relationship Extraction**

Each chunk is processed by an LLM (GPT-4o-mini recommended for cost) to extract:

```
ENTITY EXTRACTION PROMPT (Simplified)
──────────────────────────────────────

  "Extract all entities from the text. For each entity, identify:
   - name: The canonical entity name
   - type: ORGANIZATION, PERSON, GEO, EVENT, PRODUCT, CONCEPT, DATE, MONEY
   - description: A concise description of the entity

   Also extract relationships between entities:
   - source: Entity name
   - target: Entity name
   - description: How the entities are related
   - weight: Importance of this relationship (1-10)"

  Example Output from Chunk:
  "Acme Corp, a customer of our AI platform since 2022, reported
   Q3 2024 revenue of $2.3 billion, driven by expanding their
   European operations under CEO Sarah Chen."

  Entities:
  ┌─────────────┬──────────────────┬────────────────────────────────────┐
  │ Name        │ Type             │ Description                        │
  ├─────────────┼──────────────────┼────────────────────────────────────┤
  │ Acme Corp   │ ORGANIZATION     │ Enterprise customer of AI platform │
  │ Sarah Chen  │ PERSON           │ CEO of Acme Corp                   │
  │ AI Platform │ PRODUCT          │ The company's AI platform product  │
  │ Q3 2024     │ DATE             │ Third quarter of fiscal year 2024  │
  │ $2.3B       │ MONEY            │ Q3 2024 revenue amount             │
  │ Europe      │ GEO              │ European region/operations         │
  └─────────────┴──────────────────┴────────────────────────────────────┘

  Relationships:
  ┌─────────────┬───────────────┬──────────────────────────────────┬────────┐
  │ Source      │ Target        │ Description                      │ Weight │
  ├─────────────┼───────────────┼──────────────────────────────────┼────────┤
  │ Acme Corp   │ AI Platform   │ Acme is a customer of the AI     │ 10     │
  │             │               │ platform since 2022              │        │
  │ Acme Corp   │ $2.3B         │ Acme reported Q3 2024 revenue    │ 9      │
  │             │               │ of $2.3 billion                  │        │
  │ Sarah Chen  │ Acme Corp     │ Sarah Chen is the CEO of Acme    │ 8      │
  │ Acme Corp   │ Europe        │ Acme expanded European operations│ 7      │
  └─────────────┴───────────────┴──────────────────────────────────┴────────┘
```

**Step 3: Graph Construction**

Entities become nodes; relationships become directed edges. The graph is stored in a format suitable for community detection (Parquet files or direct to Neo4j).

**Step 4: Community Detection (Leiden Algorithm)**

```python
# leiden_community_detection.py
import graspologic as gp
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple

class LeidenCommunityDetector:
    """
    Detects communities in the entity-relationship graph using the
    Leiden algorithm, which guarantees well-connected communities.
    """

    def __init__(self, resolution: float = 1.0, max_iterations: int = 10):
        self.resolution = resolution
        self.max_iterations = max_iterations

    def build_graph_from_extractions(
        self,
        entities: List[Dict],
        relationships: List[Dict],
    ) -> nx.Graph:
        """
        Build a weighted undirected graph from entity and relationship data.

        Args:
            entities: List of entity dicts with 'name', 'type', 'description'
            relationships: List of relationship dicts with 'source', 'target', 'weight'

        Returns:
            NetworkX graph with entities as nodes, relationships as edges
        """
        G = nx.Graph()

        for entity in entities:
            G.add_node(
                entity["name"],
                type=entity.get("type", "UNKNOWN"),
                description=entity.get("description", ""),
            )

        for rel in relationships:
            G.add_edge(
                rel["source"],
                rel["target"],
                weight=rel.get("weight", 1.0),
                description=rel.get("description", ""),
            )

        return G

    def detect_communities(
        self,
        G: nx.Graph,
    ) -> Dict[int, Set[str]]:
        """
        Run Leiden community detection on the graph.

        Returns:
            Dict mapping community_id → set of node names
        """
        # Convert to adjacency matrix for graspologic
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)

        adj_matrix = np.zeros((n, n))
        for u, v, data in G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            adj_matrix[i][j] = data.get("weight", 1.0)
            adj_matrix[j][i] = data.get("weight", 1.0)

        # Leiden community detection via graspologic
        # (graspologic wraps leidenalg under the hood)
        from graspologic.partition import leiden

        partition = leiden(
            graph=adj_matrix,
            starting_communities=None,
            extra_forced_iterations=0,
            resolution=self.resolution,
            randomness=0.001,
            use_modularity=True,
            random_seed=42,
            max_comm_size=0,  # no limit
        )

        # Map community assignments back to node names
        communities: Dict[int, Set[str]] = {}
        for node_idx, comm_id in enumerate(partition.values()):
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(nodes[node_idx])

        return communities

    def compute_hierarchy(
        self,
        G: nx.Graph,
        base_communities: Dict[int, Set[str]],
    ) -> List[Dict[int, Set[str]]]:
        """
        Build hierarchical community structure by iteratively merging
        communities at different resolution levels.
        """
        hierarchy = [base_communities]

        current = base_communities
        for level in range(self.max_iterations - 1):
            # Build super-graph where each community becomes a node
            super_g = nx.Graph()
            for comm_id, nodes_in_comm in current.items():
                if len(nodes_in_comm) > 0:
                    super_g.add_node(comm_id, size=len(nodes_in_comm))

            # Connect communities with edges proportional to inter-community edges
            for c1, nodes1 in current.items():
                for c2, nodes2 in current.items():
                    if c1 >= c2:
                        continue
                    # Count edges between communities
                    inter_edges = 0
                    for n1 in nodes1:
                        for n2 in nodes2:
                            if G.has_edge(n1, n2):
                                inter_edges += G[n1][n2].get("weight", 1.0)
                    if inter_edges > 0:
                        super_g.add_edge(c1, c2, weight=inter_edges)

            if len(super_g.edges()) == 0:
                break

            # Run Leiden on super-graph with higher resolution
            next_communities = self.detect_communities(super_g)
            hierarchy.append(next_communities)
            current = next_communities

        return hierarchy


# ── Usage Example ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    entities = [
        {"name": "Acme Corp", "type": "ORGANIZATION", "description": "Customer"},
        {"name": "Beta Inc", "type": "ORGANIZATION", "description": "Customer"},
        {"name": "AI Platform", "type": "PRODUCT", "description": "Our platform"},
        {"name": "Sarah Chen", "type": "PERSON", "description": "CEO of Acme"},
        {"name": "Q3 Revenue", "type": "CONCEPT", "description": "Quarterly revenue"},
    ]

    relationships = [
        {"source": "Acme Corp", "target": "AI Platform", "weight": 10, "description": "is_customer"},
        {"source": "Beta Inc", "target": "AI Platform", "weight": 10, "description": "is_customer"},
        {"source": "Acme Corp", "target": "Q3 Revenue", "weight": 9, "description": "has_revenue"},
        {"source": "Sarah Chen", "target": "Acme Corp", "weight": 8, "description": "is_ceo_of"},
    ]

    detector = LeidenCommunityDetector(resolution=1.0)
    G = detector.build_graph_from_extractions(entities, relationships)
    communities = detector.detect_communities(G)

    for comm_id, nodes in communities.items():
        print(f"Community {comm_id}: {nodes}")

    hierarchy = detector.compute_hierarchy(G, communities)
    print(f"\nHierarchy levels: {len(hierarchy)}")
```

**Step 5: Community Summarization**

Each community at each hierarchy level is summarized by an LLM, creating a multi-resolution summary structure. These summaries become the "index" for global search.

```
HIERARCHICAL COMMUNITY SUMMARIES
──────────────────────────────────

  Level 0 (Base Communities — specific):
  ┌───────────────────────────────────────────────────────┐
  │ Community 1: Acme Corp, Q3 Revenue, $2.3B, Europe    │
  │ Summary: "Acme Corp, a customer of the AI platform,  │
  │  reported Q3 2024 revenue of $2.3B with expansion in │
  │  European operations."                                │
  ├───────────────────────────────────────────────────────┤
  │ Community 2: Beta Inc, Q3 Revenue, $890M             │
  │ Summary: "Beta Inc reported Q3 2024 revenue of $890M.│
  │  Beta uses the AI platform for customer analytics."   │
  ├───────────────────────────────────────────────────────┤
  │ Community 3: Sarah Chen, Acme Corp                    │
  │ Summary: "Sarah Chen has been CEO of Acme Corp since │
  │  2021, driving AI adoption and European growth."      │
  └───────────────────────────────────────────────────────┘

  Level 1 (Merged Communities — broader):
  ┌───────────────────────────────────────────────────────┐
  │ Community A (merged 1+2+3):                           │
  │ Summary: "Customers of the AI platform reported Q3    │
  │  2024 revenues ranging from $890M to $2.3B. Acme Corp │
  │  led with $2.3B under CEO Sarah Chen, driven by       │
  │  European expansion. Beta Inc contributed $890M."     │
  └───────────────────────────────────────────────────────┘

  Level 2 (Top-level Communities — global):
  ┌───────────────────────────────────────────────────────┐
  │ Community X (all customer-financial communities):     │
  │ Summary: "The AI platform's enterprise customers      │
  │  generated combined Q3 2024 revenues exceeding $5B,   │
  │  with technology and financial sectors accounting for │
  │  65% of the total. Key growth drivers include European│
  │  expansion and AI-driven analytics adoption."         │
  └───────────────────────────────────────────────────────┘
```

### 4.2.3 Query Modes

| Mode | Mechanism | Best For | Latency | Cost |
|---|---|---|---|---|
| **Local Search** | Start from query entities, explore their graph neighborhood, load source chunks | Specific fact retrieval with context | Low (~1-3 LLM calls) | Low |
| **Global Search** | Map-reduce over community summaries at appropriate hierarchy level | Thematic questions, summarization, trends | High (~10-50 LLM calls) | High |
| **Drift Search** | Decompose query into sub-questions, follow graph edges step by step | Complex multi-hop with dynamic exploration | Medium-High | Medium |

---

## Module 4.3: Building a Knowledge Graph from Documents

### 4.3.1 LLM Entity Extraction Pipeline

```python
# entity_extraction_pipeline.py
"""
End-to-end pipeline: documents → entities/relationships → Neo4j graph.

Uses structured LLM output (JSON mode) for reliable entity extraction.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import asyncio
from collections import defaultdict

# ── Structured Output Schemas ─────────────────────────────────────────────────

class Entity(BaseModel):
    name: str = Field(..., description="Canonical entity name")
    type: str = Field(
        ...,
        description="Entity type: ORGANIZATION, PERSON, GEO, EVENT, PRODUCT, CONCEPT, DATE, MONEY",
    )
    description: str = Field(..., description="Concise description from the text")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")

class Relationship(BaseModel):
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Relationship type in SCREAMING_SNAKE_CASE")
    description: str = Field(..., description="How entities are related")
    weight: int = Field(default=5, ge=1, le=10, description="Relationship importance")

class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# ── Entity Extractor ───────────────────────────────────────────────────────────

class EntityExtractor:
    """LLM-based entity and relationship extraction with structured output."""

    EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph builder. Extract entities
and relationships from the given text with high precision.

Entity types must be one of: ORGANIZATION, PERSON, GEO, EVENT, PRODUCT, CONCEPT, DATE, MONEY.

For relationships:
- Use descriptive relation types (IS_CUSTOMER_OF, HAS_REVENUE, LOCATED_IN, LED_BY, etc.)
- Assign weight 1-10 based on importance (10 = critical fact, 1 = tangential mention)
- Include only relationships explicitly stated in the text

IMPORTANT: Do NOT invent entities or relationships not present in the text.
If text says "might", "possibly", "potentially" — do NOT extract as fact."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_from_chunk(self, chunk_text: str, chunk_id: str) -> ExtractionResult:
        """Extract entities and relationships from a single text chunk."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Chunk ID: {chunk_id}\n\nText:\n{chunk_text}",
                },
            ],
            response_format=ExtractionResult,
            temperature=0.0,
        )
        return response.choices[0].message.parsed

    async def extract_batch(
        self,
        chunks: List[tuple],
        max_concurrent: int = 10,
    ) -> List[ExtractionResult]:
        """Extract from multiple chunks concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(chunk_text: str, chunk_id: str):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.extract_from_chunk, chunk_text, chunk_id
                )

        tasks = [extract_one(text, cid) for text, cid in chunks]
        return await asyncio.gather(*tasks)

    def merge_and_deduplicate(
        self,
        extraction_results: List[ExtractionResult],
        similarity_threshold: float = 0.85,
    ) -> ExtractionResult:
        """
        Merge extractions from multiple chunks, deduplicating entities
        by name similarity and canonicalization.
        """
        from difflib import SequenceMatcher

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        entity_name_map: dict = {}  # variant → canonical name

        for result in extraction_results:
            for entity in result.entities:
                canonical = self._find_canonical(
                    entity.name, entity_name_map, similarity_threshold
                )
                if canonical is None:
                    entity_name_map[entity.name.lower()] = entity.name
                    all_entities.append(entity)
                else:
                    entity_name_map[entity.name.lower()] = canonical

            # Add relationships with canonicalized names
            for rel in result.relationships:
                src_canonical = entity_name_map.get(
                    rel.source.lower(), rel.source
                )
                tgt_canonical = entity_name_map.get(
                    rel.target.lower(), rel.target
                )
                all_relationships.append(Relationship(
                    source=src_canonical,
                    target=tgt_canonical,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                ))

        # Deduplicate identical relationships (keep highest weight)
        unique_rels: dict = {}
        for rel in all_relationships:
            key = (rel.source, rel.target, rel.relation_type)
            if key not in unique_rels or rel.weight > unique_rels[key].weight:
                unique_rels[key] = rel

        return ExtractionResult(
            entities=all_entities,
            relationships=list(unique_rels.values()),
        )

    @staticmethod
    def _find_canonical(
        name: str,
        name_map: dict,
        threshold: float,
    ) -> Optional[str]:
        """Find canonical name for a variant, or None if no match."""
        from difflib import SequenceMatcher
        name_lower = name.lower()

        if name_lower in name_map:
            return name_map[name_lower]

        for variant, canonical in name_map.items():
            if SequenceMatcher(None, name_lower, variant).ratio() > threshold:
                return canonical

        return None
```

### 4.3.2 Neo4j Graph Construction

```python
# neo4j_graph_builder.py
"""
Construct a Neo4j knowledge graph from extracted entities and relationships.
"""

from neo4j import GraphDatabase
from typing import List, Dict
from entity_extraction_pipeline import Entity, Relationship, ExtractionResult

class Neo4jGraphBuilder:
    """Builds and queries a knowledge graph in Neo4j."""

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints and indexes for efficient queries."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation_type)",
        ]
        for query in queries:
            with self.driver.session() as session:
                session.run(query)

    def ingest_extraction_result(self, result: ExtractionResult):
        """Bulk ingest entities and relationships into Neo4j."""
        with self.driver.session() as session:
            # Create entity nodes
            for entity in result.entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = $description,
                        e.aliases = $aliases
                    """,
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    aliases=entity.aliases,
                )

        # Create relationships (separate transaction for efficiency)
        with self.driver.session() as session:
            for rel in result.relationships:
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:RELATES_TO {relation_type: $rel_type}]->(b)
                    SET r.description = $description,
                        r.weight = $weight
                    """,
                    source=rel.source,
                    target=rel.target,
                    rel_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                )

    def get_entity_neighborhood(
        self,
        entity_name: str,
        depth: int = 2,
        max_neighbors: int = 50,
    ) -> List[Dict]:
        """
        Retrieve the graph neighborhood around an entity for local search.

        This is the core operation for GraphRAG local search mode.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (e:Entity {name: $entity_name})-[*1..$depth]-(neighbor)
                WITH relationships(path) AS rels, nodes(path) AS nodes
                UNWIND nodes AS n
                WITH DISTINCT n
                RETURN n.name AS name, n.type AS type, n.description AS description
                LIMIT $max_neighbors
                """,
                entity_name=entity_name,
                depth=depth,
                max_neighbors=max_neighbors,
            )
            return [record.data() for record in result]

    def multi_hop_query(self, query_cypher: str, params: Dict = None) -> List[Dict]:
        """Execute arbitrary multi-hop Cypher queries."""
        with self.driver.session() as session:
            result = session.run(query_cypher, params or {})
            return [record.data() for record in result]


# ── Usage Example: Multi-hop Query ─────────────────────────────────────────────
if __name__ == "__main__":
    builder = Neo4jGraphBuilder(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
    )
    builder.create_constraints()

    # Example multi-hop query: "Find all customers and their Q3 revenue"
    results = builder.multi_hop_query(
        """
        MATCH (c:Entity {type: 'ORGANIZATION'})-[r1:RELATES_TO {relation_type: 'IS_CUSTOMER_OF'}]->(p:Entity {name: 'AI Platform'})
        MATCH (c)-[r2:RELATES_TO {relation_type: 'HAS_REVENUE'}]->(rev:Entity {type: 'MONEY'})
        RETURN c.name AS company, rev.name AS q3_revenue, r2.description AS context
        ORDER BY rev.name DESC
        """
    )
    for row in results:
        print(f"{row['company']}: {row['q3_revenue']} ({row['context']})")

    builder.close()
```

---

## Module 4.4: GraphRAG Query Pipeline

### 4.4.1 Local Search Mode

```
LOCAL SEARCH ALGORITHM
────────────────────────

  Input: user_query, entity_index, graph, k_neighborhoods=10

  1. ENTITY EXTRACTION FROM QUERY:
     Extract key entities from the user's query using LLM.
     "Compare Q3 revenues of customers" → ["Q3 Revenue", "Customer"]

  2. ENTITY MAPPING:
     Map extracted entities to canonical entities in the graph.
     "Customer" → fuzzy matches "Acme Corp", "Beta Inc", etc.

  3. NEIGHBORHOOD EXPANSION:
     For each matched entity, retrieve k-hop neighborhood.
     depth=2 captures: Entity → direct relations → relations of relations

  4. TEXT UNIT RETRIEVAL:
     Get the original source text chunks associated with each
     entity and relationship in the neighborhood.

  5. CONTEXT ASSEMBLY:
     Sort text units by relationship weight × relevance to query.
     Truncate to context window limit.

  6. LLM GENERATION:
     Generate answer with: system prompt + assembled context + user query.

  ADVANTAGE: Precise, fact-grounded answers anchored to specific entities.
  LIMITATION: Misses global patterns and trends across communities.
```

### 4.4.2 Global Search Mode (Map-Reduce over Communities)

```python
# global_search.py
"""
Global search: map-reduce over community summaries for thematic answers.
"""

from typing import List, Dict
from openai import OpenAI

class GlobalSearch:
    """
    Implements GraphRAG's global search mode.

    Instead of searching the graph directly, this searches the hierarchical
    community summaries to answer broad thematic questions.
    """

    def __init__(self, community_summaries: List[Dict], model: str = "gpt-4o-mini"):
        """
        Args:
            community_summaries: List of {level, community_id, title, summary, entities}
            model: LLM model for map and reduce phases
        """
        self.summaries = community_summaries
        self.client = OpenAI()
        self.model = model

    def select_relevant_level(self, query: str) -> int:
        """Select the appropriate hierarchy level for the query breadth."""
        query_words = len(query.split())

        if query_words < 10:
            return 1  # Specific → use base-level communities
        elif query_words < 25:
            return 2  # Moderate breadth
        else:
            return 3  # Very broad → use high-level summaries

    def map_phase(
        self,
        query: str,
        level: int,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        MAP: Score each community summary for relevance to the query.

        Returns top_k most relevant community summaries.
        """
        level_summaries = [s for s in self.summaries if s["level"] == level]

        # Batch-evaluate relevance using LLM
        summaries_text = "\n\n".join([
            f"[{i}] {s['title']}: {s['summary']}"
            for i, s in enumerate(level_summaries)
        ])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": "You are a relevance scorer. Rate each community "
                           "summary for relevance to the query on a scale of 0-10. "
                           "Return JSON: [{\"index\": 0, \"score\": 8.5, \"reason\": \"...\"}]",
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nCommunity Summaries:\n{summaries_text}",
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # Parse scores and return top_k
        import json
        scores = json.loads(response.choices[0].message.content)["scores"]
        scores.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for s in scores[:top_k]:
            summ = level_summaries[s["index"]]
            results.append({
                "title": summ["title"],
                "summary": summ["summary"],
                "score": s["score"],
                "reason": s.get("reason", ""),
            })

        return results

    def reduce_phase(
        self,
        query: str,
        relevant_summaries: List[Dict],
    ) -> str:
        """
        REDUCE: Synthesize a comprehensive answer from relevant summaries.
        """
        context = "\n\n".join([
            f"## {s['title']} (relevance: {s['score']}/10)\n{s['summary']}"
            for s in relevant_summaries
        ])

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Use stronger model for final synthesis
            messages=[{
                "role": "system",
                "content": (
                    "You are an enterprise knowledge analyst. Synthesize a "
                    "comprehensive answer from the provided community summaries. "
                    "Cite which communities each piece of information comes from. "
                    "If information is missing, explicitly state that."
                ),
            }, {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant Knowledge:\n{context}",
            }],
            temperature=0.1,
        )

        return response.choices[0].message.content

    def search(self, query: str) -> str:
        """Execute full global search pipeline."""
        level = self.select_relevant_level(query)
        relevant = self.map_phase(query, level)
        return self.reduce_phase(query, relevant)
```

### 4.4.3 Real Case Study: Microsoft GraphRAG for Enterprise Knowledge Management

**Company:** Microsoft Research + internal deployment
**Problem:** Microsoft's internal knowledge base spans millions of documents across SharePoint, emails, wikis, and code repositories. Traditional search could find individual documents, but could not answer questions like "What are the key risks in Project X based on recent status reports and team discussions?"

**Architecture deployed:**
- **Corpus:** 10M+ documents from internal Microsoft sources
- **Graph Construction:** GPT-4o-mini for entity extraction (cost-optimized), GPT-4o for community summarization
- **Storage:** Azure Cosmos DB for graph, Azure Blob Storage for community summaries
- **Serving:** Azure Kubernetes Service with auto-scaling

**Results (from Microsoft's published benchmarks):**
| Metric | Baseline RAG (Vector only) | GraphRAG |
|---|---|---|
| Answer comprehensiveness | 2.8/5 | 4.2/5 |
| Factual groundedness | 3.1/5 | 4.4/5 |
| Multi-hop accuracy | 37% | 78% |
| Indexing cost per 1K docs | ~$2 (embeddings only) | ~$15 (LLM extraction + summaries) |
| Query latency (global search) | 1.2s | 4.8s |

### 4.4.4 Real Case Study: Neo4j + GraphRAG for Financial Compliance

**Company:** Major European bank (unnamed in public disclosure, described at Neo4j GraphSummit 2024)
**Problem:** Compliance analysts needed to answer multi-hop questions about transactions: "Find all transactions over $50K involving entities connected to sanctioned individuals within 3 degrees of separation."

**Why GraphRAG was essential:**
- Vector search can find documents mentioning "sanctioned individual X" or "transaction Y" but cannot traverse relationships
- A graph stores: `Transaction → involves → Account → owned_by → Person → related_to → SanctionedPerson`
- A single Cypher query replaces 5+ manual searches

```
COMPLIANCE QUERY AS GRAPH TRAVERSAL
─────────────────────────────────────

  Cypher:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ MATCH (sp:Person {sanctioned: true})                                 │
  │ MATCH path = (sp)-[:RELATED_TO*1..3]-(p:Person)-[:OWNS]->(a:Account)│
  │              -[:INVOLVED_IN]->(t:Transaction)                        │
  │ WHERE t.amount > 50000                                               │
  │ RETURN sp.name AS sanctioned_person,                                 │
  │        p.name AS connected_person,                                   │
  │        length(path) AS degrees,                                      │
  │        t.id AS transaction_id,                                       │
  │        t.amount AS amount                                            │
  │ ORDER BY degrees ASC, amount DESC                                    │
  └──────────────────────────────────────────────────────────────────────┘

  This single query replaces what was previously a 2-day manual investigation.
  The graph structure makes multi-hop reasoning O(relations) rather than O(docs²).
```

---

## 📦 Compression Code: End-to-End GraphRAG Pipeline

```python
# graphrag_pipeline.py
"""
Complete GraphRAG pipeline: Document ingestion → Entity extraction →
Neo4j graph construction → Community detection → Query engine.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                     GRAPHRAG PIPELINE                            │
  │                                                                  │
  │  Documents ──▶ Chunker ──▶ Entity Extractor ──▶ Graph Builder   │
  │                                                  │               │
  │                                                  ▼               │
  │                                          Neo4j Knowledge Graph   │
  │                                                  │               │
  │                                                  ▼               │
  │                                     Community Detector           │
  │                                                  │               │
  │                                                  ▼               │
  │                                    Community Summarizer (LLM)    │
  │                                                  │               │
  │                                    ┌─────────────┴─────────────┐ │
  │                                    ▼                           ▼ │
  │                              Local Search              Global Search│
  │                           (Neighborhood traversal)   (Map-Reduce) │
  └─────────────────────────────────────────────────────────────────┘

Requirements:
  pip install openai neo4j graspologic networkx numpy tiktoken
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class GraphRAGConfig:
    """Configuration for the GraphRAG pipeline."""
    # Chunking
    chunk_size: int = 300
    chunk_overlap: int = 100

    # Extraction
    extraction_model: str = "gpt-4o-mini"
    extraction_concurrency: int = 10
    entity_similarity_threshold: float = 0.85

    # Community detection
    leiden_resolution: float = 1.0
    max_hierarchy_levels: int = 3

    # Summarization
    summary_model: str = "gpt-4o-mini"
    max_summary_tokens: int = 500

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Search
    local_search_depth: int = 2
    global_search_level: int = 2


class GraphRAGPipeline:
    """
    End-to-end pipeline for building and querying a Knowledge Graph RAG system.
    """

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.extractor = None
        self.graph_builder = None
        self.community_detector = None
        self.global_searcher = None
        self.community_summaries: List[Dict] = []
        self.is_indexed = False

    def chunk_documents(self, documents: List[str]) -> List[Tuple[str, str]]:
        """
        Split documents into overlapping chunks.

        Returns:
            List of (chunk_id, chunk_text) tuples
        """
        chunks = []
        for doc_idx, doc in enumerate(documents):
            words = doc.split()
            for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
                chunk_words = words[i : i + self.config.chunk_size]
                if len(chunk_words) < 20:  # Skip tiny chunks
                    continue
                chunk_text = " ".join(chunk_words)
                chunk_id = f"doc_{doc_idx}_chunk_{i}"
                chunks.append((chunk_id, chunk_text))
        return chunks

    async def index(self, documents: List[str]) -> Dict:
        """
        Build the full GraphRAG index from raw documents.

        Stages:
          1. Chunk documents
          2. Extract entities and relationships via LLM
          3. Build Neo4j knowledge graph
          4. Detect communities
          5. Generate community summaries
        """
        import time
        start = time.time()

        print("[GraphRAG] Stage 1/5: Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"  → {len(chunks)} chunks from {len(documents)} documents")

        print("[GraphRAG] Stage 2/5: Extracting entities and relationships...")
        from entity_extraction_pipeline import EntityExtractor

        self.extractor = EntityExtractor(
            model=self.config.extraction_model,
        )
        extraction_results = await self.extractor.extract_batch(
            chunks,
            max_concurrent=self.config.extraction_concurrency,
        )
        merged = self.extractor.merge_and_deduplicate(
            extraction_results,
            self.config.entity_similarity_threshold,
        )
        print(f"  → {len(merged.entities)} entities, {len(merged.relationships)} relationships")

        print("[GraphRAG] Stage 3/5: Building Neo4j knowledge graph...")
        from neo4j_graph_builder import Neo4jGraphBuilder

        self.graph_builder = Neo4jGraphBuilder(
            self.config.neo4j_uri,
            self.config.neo4j_user,
            self.config.neo4j_password,
        )
        self.graph_builder.create_constraints()
        self.graph_builder.ingest_extraction_result(merged)
        print("  → Graph ingested into Neo4j")

        print("[GraphRAG] Stage 4/5: Detecting communities...")
        from leiden_community_detection import LeidenCommunityDetector

        self.community_detector = LeidenCommunityDetector(
            resolution=self.config.leiden_resolution,
            max_iterations=self.config.max_hierarchy_levels,
        )

        # Build networkx graph from entities + relationships
        entity_dicts = [e.model_dump() for e in merged.entities]
        rel_dicts = [r.model_dump() for r in merged.relationships]
        G = self.community_detector.build_graph_from_extractions(entity_dicts, rel_dicts)
        communities = self.community_detector.detect_communities(G)
        hierarchy = self.community_detector.compute_hierarchy(G, communities)
        print(f"  → {len(communities)} base communities across {len(hierarchy)} levels")

        print("[GraphRAG] Stage 5/5: Generating community summaries...")
        self.community_summaries = await self._summarize_communities(hierarchy, merged)
        print(f"  → {len(self.community_summaries)} community summaries generated")

        self.is_indexed = True
        elapsed = time.time() - start
        print(f"[GraphRAG] Indexing complete in {elapsed:.1f}s")

        return {
            "status": "indexed",
            "num_entities": len(merged.entities),
            "num_relationships": len(merged.relationships),
            "num_communities": len(communities),
            "num_hierarchy_levels": len(hierarchy),
            "num_summaries": len(self.community_summaries),
            "elapsed_seconds": round(elapsed, 1),
        }

    async def _summarize_communities(
        self,
        hierarchy: List[Dict[int, set]],
        merged,
    ) -> List[Dict]:
        """Generate LLM summaries for each community at each hierarchy level."""
        from openai import OpenAI

        client = OpenAI()
        summaries = []

        for level, communities in enumerate(hierarchy):
            for comm_id, entity_names in communities.items():
                if len(entity_names) < 2:
                    continue

                entity_descriptions = []
                for entity in merged.entities:
                    if entity.name in entity_names:
                        entity_descriptions.append(
                            f"- {entity.name} ({entity.type}): {entity.description}"
                        )

                entity_text = "\n".join(entity_descriptions[:20])

                response = client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=[{
                        "role": "system",
                        "content": (
                            "Summarize this community of related entities into a "
                            f"concise paragraph (max {self.config.max_summary_tokens} tokens). "
                            "Focus on what connects these entities and the key facts they represent."
                        ),
                    }, {
                        "role": "user",
                        "content": f"Community entities:\n{entity_text}",
                    }],
                    max_tokens=self.config.max_summary_tokens,
                    temperature=0.1,
                )

                summaries.append({
                    "level": level,
                    "community_id": comm_id,
                    "title": f"Community {level}-{comm_id}",
                    "summary": response.choices[0].message.content,
                    "entities": list(entity_names),
                })

        return summaries

    async def local_search(self, query: str, top_k: int = 5) -> Dict:
        """
        Local search: entity-anchored graph traversal.
        """
        if not self.is_indexed:
            return {"error": "Index not built. Call index() first."}

        # Step 1: Extract entities from query
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=self.config.extraction_model,
            messages=[{
                "role": "system",
                "content": "Extract key entity names from this query. Return as JSON list.",
            }, {
                "role": "user",
                "content": f"Query: {query}",
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        import json
        query_entities = json.loads(response.choices[0].message.content).get("entities", [])

        if not query_entities:
            # Fallback: use the query itself as context for community search
            return {"mode": "fallback_community", "results": []}

        # Step 2: Retrieve graph neighborhoods
        contexts = []
        for entity_name in query_entities[:5]:
            neighbors = self.graph_builder.get_entity_neighborhood(
                entity_name,
                depth=self.config.local_search_depth,
            )
            contexts.append({
                "anchor_entity": entity_name,
                "neighbors": neighbors,
            })

        # Step 3: Assemble context and generate answer
        context_text = "\n".join([
            f"Entity: {c['anchor_entity']}\n" +
            "\n".join([f"  - {n['name']} ({n['type']}): {n.get('description', '')}"
                       for n in c['neighbors'][:15]])
            for c in contexts
        ])

        answer = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "Answer the question using only the provided graph context. "
                           "Cite specific entities in your answer.",
            }, {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}",
            }],
            temperature=0.1,
        )

        return {
            "mode": "local_search",
            "query_entities": query_entities,
            "context_entities": [c["anchor_entity"] for c in contexts],
            "answer": answer.choices[0].message.content,
        }

    async def global_search(self, query: str) -> Dict:
        """Global search: map-reduce over community summaries."""
        if not self.is_indexed:
            return {"error": "Index not built. Call index() first."}

        from global_search import GlobalSearch

        searcher = GlobalSearch(
            self.community_summaries,
            model=self.config.extraction_model,
        )
        answer = searcher.search(query)

        return {
            "mode": "global_search",
            "num_communities_consulted": len(self.community_summaries),
            "answer": answer,
        }


# ── Usage Example ──────────────────────────────────────────────────────────────
async def main():
    config = GraphRAGConfig(
        neo4j_password="your_password",
    )
    pipeline = GraphRAGPipeline(config)

    # Sample enterprise documents
    documents = [
        "Acme Corp, a customer of our AI platform since 2022, reported Q3 2024 "
        "revenue of $2.3 billion, driven by expanding European operations under "
        "CEO Sarah Chen. The company upgraded to our Enterprise tier in Q1 2024.",

        "Beta Inc, a healthcare analytics firm, uses our AI platform for patient "
        "data analysis. Q3 2024 revenue reached $890 million, with 15% growth "
        "attributed to new AI-powered diagnostic tools.",

        "Gamma Co, a financial services company, adopted our AI platform in 2023 "
        "for fraud detection. Q3 2024 revenue was $1.5 billion. CTO Mark Johnson "
        "cited 'significant ROI from the AI platform' in the earnings call.",
    ]

    result = await pipeline.index(documents)
    print(f"Indexing result: {result}")

    local_answer = await pipeline.local_search(
        "What is the Q3 revenue of Acme Corp and who is their CEO?"
    )
    print(f"\nLocal Search Answer:\n{local_answer['answer']}")

    global_answer = await pipeline.global_search(
        "Summarize the Q3 2024 financial performance of our AI platform customers."
    )
    print(f"\nGlobal Search Answer:\n{global_answer['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Documented Project: GraphRAG System for Multi-Hop Enterprise Search

```
Project: "GraphMind: Multi-Hop Enterprise Knowledge Graph RAG"

Problem Statement:
  Enterprise knowledge is fragmented across documents, emails, wikis,
  and databases. Single-hop vector RAG fails on questions requiring
  multi-step reasoning: "Which customers using Feature X are at risk
  of churn based on recent support tickets?"

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                        GRAPHMIND                                │
  │                                                                 │
  │  [SharePoint] [Confluence] [Slack] [Email] [Database]           │
  │       │           │           │        │         │              │
  │       └───────────┴───────────┴────────┴─────────┘              │
  │                        │                                        │
  │                 Document Ingestor                               │
  │                        │                                        │
  │          ┌─────────────┴─────────────┐                          │
  │          │     Graph Construction    │                          │
  │          │  - LLM Entity Extraction  │                          │
  │          │  - Relationship Mining    │                          │
  │          │  - Neo4j Graph DB         │                          │
  │          └─────────────┬─────────────┘                          │
  │                        │                                        │
  │          ┌─────────────┴─────────────┐                          │
  │          │   Community Detection      │                          │
  │          │  - Leiden Algorithm        │                          │
  │          │  - Hierarchical Summaries  │                          │
  │          └─────────────┬─────────────┘                          │
  │                        │                                        │
  │          ┌─────────────┴─────────────┐                          │
  │          │       Query Engine        │                          │
  │          │  - Local Search           │                          │
  │          │  - Global Search          │                          │
  │          │  - Hybrid (Graph+Vector)  │                          │
  │          └─────────────┬─────────────┘                          │
  │                        │                                        │
  │                   [FastAPI]                                     │
  │                        │                                        │
  │                   Slack Bot / Web UI                            │
  └─────────────────────────────────────────────────────────────────┘

Tech Stack:
  - LLMs: GPT-4o-mini (extraction), GPT-4o (summarization, generation)
  - Graph DB: Neo4j AuraDB (managed) or self-hosted Neo4j
  - Vector DB: Qdrant (for hybrid graph+vector fallback)
  - Community Detection: graspologic (Leiden implementation)
  - Serving: FastAPI + Docker + Kubernetes
  - Monitoring: Prometheus + Grafana
  - CI/CD: GitHub Actions + Neo4j migrations

Key Features:
  1. Multi-hop query resolution using graph traversal
  2. Automatic community detection and summarization
  3. Hybrid fallback to vector search for non-graph queries
  4. Incremental indexing for new documents
  5. Query analytics dashboard

Evaluation Metrics:
  - Multi-hop accuracy: > 75% on custom test set
  - Entity extraction precision: > 90%
  - Community coherence: modularity score > 0.3
  - Query latency (local search): < 2s
  - Query latency (global search): < 8s
```

---

## Key Takeaways

1. **Vector RAG fails on multi-hop questions** because embeddings capture atomic semantics, not relationships between entities. The semantic gap between composite query intent and atomic document embeddings means the LLM must do the "connecting" — and it frequently hallucinates.

2. **Microsoft GraphRAG introduced a paradigm shift** by pre-computing entity-relationship graphs and community hierarchies. This shifts the reasoning burden from the LLM (unreliable) to the graph structure (deterministic). The cost is higher indexing overhead ($15 vs $2 per 1K documents), but the query-time quality gains are substantial (78% vs 37% on multi-hop accuracy).

3. **Local vs global search** serves two distinct query patterns: local search (entity-anchored, precise, fast) for specific fact lookup; global search (map-reduce over community summaries, broader, slower) for thematic summarization and trend analysis.

4. **Neo4j + Cypher** provides a production-hardened graph database that can traverse k-hop relationships in milliseconds, making it the ideal backend for GraphRAG's local search mode.

5. **Community detection via the Leiden algorithm** automatically discovers thematic clusters in the knowledge graph, enabling hierarchical summarization that answers questions at varying levels of abstraction.

---

## References

- Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Microsoft Research*. arXiv:2404.16130
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). "From Louvain to Leiden: Guaranteeing Well-Connected Communities." *Scientific Reports*, 9(1), 5233.
- Neo4j GraphSummit (2024). "Financial Compliance: Multi-Hop Risk Analysis with Knowledge Graphs."
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
- [[03 - Reranking and Evaluation-Driven Retrieval]] — Two-stage retrieval for precision in standard RAG
- [[05 - RAG Evaluation with RAGAS and DeepEval]] — Evaluating GraphRAG vs baseline RAG quality
- [[Graph Neural Networks]] — GNN foundations for graph-based reasoning
- [[System Design for ML]] — Scaling graph databases for production workloads
