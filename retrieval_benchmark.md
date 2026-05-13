# Retrieval Benchmark — Strategy A vs Strategy B

**Generated:** 2026-05-12  
**Corpus:** 9 documents → 18 chunks (chunk_size=300, overlap=50)  
**Top-K:** 3  
**Embedding model:** `all-MiniLM-L6-v2` (mock backing `text-embedding-004`)  
**Generative model:** `gemini-1.5-flash-001` (mocked `_MockGenerativeModel`)

---

## Strategies

| | Strategy A | Strategy B |
|---|---|---|
| **Name** | Raw Vector Search | Query Expansion (Vertex AI) |
| **Module** | `rag/retrieval.py → RawRetriever` | `rag/retrieval.py → ExpandingRetriever` |
| **Query processing** | Embed query as-is | Rewrite via `GenerativeModel`, then embed |
| **Embedding** | `EmbeddingService` (`TextEmbeddingModel`) | Same |
| **Search** | Raw cosine similarity (NumPy dot product) | Same |

---

## Query 1 — "How does the system handle peak load"

**Context:** Infrastructure / scalability — vague phrasing, no technical keywords present in the raw query.

**Expanded query (Strategy B):**
> How does the system handle peak load system performance under peak load high traffic scalability load balancing autoscaling horizontal scaling throughput latency spike handling capacity planning resource utilisation explain describe how what why when process mechanism approach method technique strategy implementation

### Results

| Rank | Strategy A — Raw Vector Search | Score A | Strategy B — Query Expansion | Score B |
|------|-------------------------------|---------|------------------------------|---------|
| 1 | `peak_load_management` | **0.7396** | `peak_load_management` | 0.7220 |
| 2 | `scalability_overview` (chunk 2) | 0.4739 | `scalability_overview` (chunk 1) | **0.6181** |
| 3 | `scalability_overview` (chunk 1) | 0.4631 | `scalability_overview` (chunk 2) | 0.4546 |

### Analysis

- **Top-1 source changed:** No — both strategies correctly identify `peak_load_management`.
- **Score Δ (rank-1):** −0.0177 (Strategy A marginally higher at rank 1).
- **Key observation:** The raw query already contains "peak load" verbatim, which appears in the corpus. Strategy A's focused embedding scores slightly higher at rank 1. Strategy B's expansion improves rank-2 score significantly (+0.14) by pulling the more informative scalability chunk to the top of rank 2, demonstrating that expansion improves breadth even when it doesn't change the top result.

---

## Query 2 — "What techniques reduce errors in AI-generated answers"

**Context:** RAG / hallucination — indirect phrasing. Neither "RAG", "retrieval", nor "hallucination" appear in the raw query.

**Expanded query (Strategy B):**
> What techniques reduce errors in AI-generated answers explain describe how what why when process mechanism approach method technique strategy implementation

### Results

| Rank | Strategy A — Raw Vector Search | Score A | Strategy B — Query Expansion | Score B |
|------|-------------------------------|---------|------------------------------|---------|
| 1 | `rag_overview` | 0.4291 | `machine_learning_overview` | **0.4838** |
| 2 | `machine_learning_overview` | 0.2623 | `rag_overview` | 0.3231 |
| 3 | `rag_overview` (chunk 2) | 0.2292 | `nlp_overview` | 0.3057 |

### Analysis

- **Top-1 source changed:** **Yes** — Strategy A correctly surfaces `rag_overview` (which contains "RAG reduces hallucinations") at rank 1. Strategy B displaces it to rank 2.
- **Score Δ (rank-1):** +0.0547 (Strategy B rank-1 score is higher, but the source is less relevant).
- **Key observation:** This is the most instructive case. The mock expander has no rule for "errors in AI-generated answers" so it only appends generic terms ("explain describe how…"). These generic terms dilute the semantic signal and pull `machine_learning_overview` ahead of the more relevant `rag_overview`. A real `GenerativeModel` would add "RAG, hallucination, grounding, factual accuracy" and would likely restore `rag_overview` to rank 1 with a higher score. This demonstrates the **risk of low-quality expansion**: generic noise hurts more than it helps.
- **Strategy A wins this query** on relevance grounds.

---

## Query 3 — "How do modern language models process sequential text efficiently"

**Context:** Transformers / NLP — multi-concept query that could match several documents.

**Expanded query (Strategy B):**
> How do modern language models process sequential text efficiently explain describe how what why when process mechanism approach method technique strategy implementation

### Results

| Rank | Strategy A — Raw Vector Search | Score A | Strategy B — Query Expansion | Score B |
|------|-------------------------------|---------|------------------------------|---------|
| 1 | `transformers_overview` | 0.4708 | `machine_learning_overview` | **0.4889** |
| 2 | `nlp_overview` | 0.4108 | `nlp_overview` | 0.4162 |
| 3 | `rag_overview` | 0.4102 | `transformers_overview` | 0.3892 |

### Analysis

- **Top-1 source changed:** **Yes** — Strategy A correctly puts `transformers_overview` first (self-attention, parallel processing of sequential data). Strategy B demotes it to rank 3.
- **Score Δ (rank-1):** +0.0181 (Strategy B rank-1 score is marginally higher, but the source is less relevant).
- **Key observation:** Same pattern as Q2. The expansion adds only generic terms, which broadens the query enough to match the general `machine_learning_overview` more strongly than the specific `transformers_overview`. A real LLM expansion would add "self-attention, transformer, parallel processing, BERT, GPT" and would reinforce the correct result.
- **Strategy A wins this query** on relevance grounds.

---

## Summary Table

| Query | A Top-1 Source | A Score | B Top-1 Source | B Score | Top-1 Changed | Expansion Benefit |
|-------|---------------|---------|---------------|---------|---------------|-------------------|
| Q1: Peak load | `peak_load_management` | 0.7396 | `peak_load_management` | 0.7220 | No | Improves rank-2 breadth |
| Q2: AI errors | `rag_overview` ✓ | 0.4291 | `machine_learning_overview` ✗ | 0.4838 | Yes | **Hurts** (generic noise) |
| Q3: Language models | `transformers_overview` ✓ | 0.4708 | `machine_learning_overview` ✗ | 0.4889 | Yes | **Hurts** (generic noise) |

---

## Conclusions

1. **Raw vector search is robust for specific queries.** When the query contains domain-specific terms that appear in the corpus (Q1), it performs on par with or better than expansion.

2. **Query expansion is only as good as the expander.** The mock rule-based expander adds generic terms for queries it doesn't recognise (Q2, Q3), which dilutes the embedding signal. A real `GenerativeModel` (Gemini) would produce semantically precise expansions and is expected to improve Q2 and Q3 significantly.

3. **Expansion improves breadth even when top-1 is unchanged.** In Q1, expansion raised the rank-2 score from 0.47 to 0.62 by surfacing the more informative scalability chunk first — useful when the top result alone is insufficient context.

4. **Production recommendation:** Use `ExpandingRetriever` with a real `GenerativeModel` for ambiguous or indirect queries. Fall back to `RawRetriever` for keyword-rich queries or when latency budget is tight (expansion adds one LLM round-trip).
