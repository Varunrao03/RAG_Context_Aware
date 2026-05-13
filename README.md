# Overview

A fully self-contained Retrieval-Augmented Generation (RAG) system built in Python,
comparing two retrieval strategies against the same document corpus. The system runs
entirely in-memory with no external vector database, and mocks the Vertex AI SDK so
it works without GCP credentials while preserving the exact API surface needed to go
live in production.

# Architecture 

1. RAW Vector Search
   
   Query text
    │
    ▼ EmbeddingService.embed_one()          [TextEmbeddingModel — mocked]
    │
    ▼ VectorStore.search()                  [NumPy dot product, in-memory]
    │
    ▼ List[SearchResult]

2. Vertex AI Vector Search

   Query text
    │
    ▼ EmbeddingService.embed_one()          [Real TextEmbeddingModel on GCP]
    │
    ▼ VertexVectorStore.search()            [Vertex AI Vector Search index]
    │
    ▼ List[SearchResult]


# Result

Three complex queries were run through both strategies. All scores are cosine
similarity (range −1 to 1, higher = more relevant).

Q1 — "How does the system handle peak load"
*Vague infrastructure query — no technical keywords in the raw phrasing*

| Rank | Strategy A (Raw)        | Score  | Strategy B (Expanded)  | Score |
|------|-------------------------|--------|------------------------|--------|
| 1    | `peak_load_management`  | 0.7396 | `peak_load_management` | 0.7626 |
| 2    | `scalability_overview`  | 0.4739 | `scalability_overview` | 0.5899 |
| 3    | `scalability_overview`  | 0.4631 | `scalability_overview` | 0.5013 |

Expanded query:*added "autoscaling load balancing horizontal scaling throughput
latency capacity planning rate limiting queue architecture high concurrency"

Outcome: Both strategies retrieve the same sources. Strategy B scores higher
across all 3 ranks because the expansion adds precise infrastructure vocabulary
that aligns tightly with the corpus. Score delta at rank-1: *+0.023*.



Q2 — "What techniques reduce errors in AI-generated answers"
*Indirect RAG/hallucination query — no "RAG", "retrieval", or "hallucination" in the raw query*

| Rank | Strategy A (Raw)            | Score  | Strategy B (Expanded)    | Score  |
|------|-----------------------------|--------|--------------------------|--------|
| 1    | `rag_overview`              | 0.4291 | `rag_overview`           | 0.6845 |
| 2    | `machine_learning_overview` | 0.2623 | `rag_overview` (chunk 2) | 0.5173 |
| 3    | `rag_overview` (chunk 2)    | 0.2292 | `vector_databases`       | 0.3321 |

Expanded query: added "reducing hallucinations in large language model outputs
retrieval augmented generation RAG grounding factual accuracy knowledge base
anchoring faithfulness citation verification"

Outcome: Strategy B correctly identifies `rag_overview` at rank 1 with a
dramatically higher score, and surfaces a second RAG chunk at rank 2 that Strategy A
missed entirely. Score delta at rank-1: *0.255 (+60%)*.
This is the clearest win for query expansion — the raw query uses indirect phrasing
("errors in AI-generated answers") that doesn't match corpus vocabulary, while the
expansion bridges that gap with precise RAG terminology.

---

Q3 — "How do modern language models process sequential text efficiently"
*Multi-concept transformer/NLP query — could match several documents*

|Rank|    Strategy A (Raw)     | Score  |       Strategy B (Expanded)       | Score      |
|----|-------------------------|--------|-----------------------------------|------------|
| 1  | `transformers_overview` | 0.4708 | `transformers_overview`           | **0.7458** |
| 2  | `nlp_overview`          | 0.4108 | `transformers_overview` (chunk 2) | **0.5473** |
| 3  | `rag_overview`          | 0.4102 | `nlp_overview`                    | **0.4930** |

Expanded query: added "large language model transformer architecture
self-attention BERT GPT T5 sequential text processing parallel computation NLP
natural language understanding generation"

Outcome: Strategy B locks `transformers_overview` into both rank-1 and rank-2
with high confidence, and removes the irrelevant `rag_overview` from the results
entirely. Strategy A's top-3 scores are clustered tightly (0.47 / 0.41 / 0.41),
indicating low confidence. Strategy B's spread (0.75 / 0.55 / 0.49) shows clear
signal. Score delta at rank-1: *+0.275 (+58%)*.
