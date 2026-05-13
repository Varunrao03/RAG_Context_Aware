"""
compare_strategies.py
---------------------
Runs 3 complex queries through both RAG strategies and produces:
  1. A structured JSON file  → comparison_results.json
  2. A formatted table       → printed to stdout

Strategies compared
~~~~~~~~~~~~~~~~~~~
  A) Raw Vector Search   (rag.RawRetriever)
     Query is embedded as-is and searched via cosine similarity.

  B) Query Expansion RAG (rag.ExpandingRetriever)
     Query is first rewritten/expanded by GenerativeModel (mocked Vertex AI),
     then embedded via TextEmbeddingModel and searched via cosine similarity.

Usage
~~~~~
    python compare_strategies.py
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List

# ── Silence all sub-module logs; only WARNING+ will surface ─────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Import from the rag package ──────────────────────────────────────────────
from rag import EmbeddingService, ExpandingRetriever, RawRetriever

logger = logging.getLogger(__name__)

# Suppress INFO from noisy sub-modules
for _noisy in ("rag.embedding", "rag.retrieval", "rag.storage", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Shared corpus (superset of both demos) ───────────────────────────────────
DOCUMENTS = [
    (
        "machine_learning_overview",
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly "
        "programmed. It focuses on developing computer programs that can access "
        "data and use it to learn for themselves. The process begins with "
        "observations or data, such as examples, direct experience, or instruction, "
        "to look for patterns in data and make better decisions in the future.",
    ),
    (
        "deep_learning_overview",
        "Deep learning is part of a broader family of machine learning methods "
        "based on artificial neural networks with representation learning. Learning "
        "can be supervised, semi-supervised or unsupervised. Deep learning "
        "architectures such as deep neural networks, recurrent neural networks, "
        "convolutional neural networks and transformers have been applied to fields "
        "including computer vision, speech recognition, natural language processing, "
        "and bioinformatics.",
    ),
    (
        "nlp_overview",
        "Natural language processing (NLP) is a subfield of linguistics, computer "
        "science, and artificial intelligence concerned with the interactions between "
        "computers and human language, in particular how to program computers to "
        "process and analyze large amounts of natural language data. The goal is a "
        "computer capable of understanding the contents of documents, including the "
        "contextual nuances of the language within them.",
    ),
    (
        "rag_overview",
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines the "
        "strengths of retrieval-based and generative models. It retrieves relevant "
        "documents from a knowledge base and uses them as context for a language "
        "model to generate accurate, grounded responses. RAG reduces hallucinations "
        "and keeps responses factual by anchoring generation in retrieved evidence.",
    ),
    (
        "vector_databases",
        "Vector databases store data as high-dimensional vectors and enable fast "
        "similarity search. They are commonly used in semantic search, recommendation "
        "systems, and RAG pipelines. Popular vector databases include Pinecone, "
        "Weaviate, Chroma, and FAISS. Unlike traditional databases, they retrieve "
        "results based on semantic closeness rather than exact keyword matches.",
    ),
    (
        "transformers_overview",
        "Transformers are a type of neural network architecture introduced in the "
        "paper 'Attention Is All You Need' (2017). They rely on self-attention "
        "mechanisms to process sequential data in parallel, making them highly "
        "efficient for NLP tasks. Models like BERT, GPT, and T5 are all based on "
        "the transformer architecture and have achieved state-of-the-art results "
        "across a wide range of benchmarks.",
    ),
    (
        "scalability_overview",
        "System scalability refers to the ability of a system to handle increasing "
        "amounts of work by adding resources. Horizontal scaling adds more machines "
        "to a pool, while vertical scaling increases the capacity of existing machines. "
        "Load balancers distribute incoming traffic across multiple servers to prevent "
        "any single server from becoming a bottleneck during peak load periods.",
    ),
    (
        "peak_load_management",
        "Peak load management involves strategies to handle sudden spikes in traffic "
        "or demand. Autoscaling automatically provisions additional compute resources "
        "when demand rises and releases them when demand falls. Rate limiting, caching, "
        "and queue-based architectures are common techniques to maintain system "
        "stability and low latency under high concurrency and peak traffic conditions.",
    ),
    (
        "cloud_infrastructure",
        "Cloud infrastructure enables elastic resource allocation, allowing systems "
        "to scale up during high-demand periods and scale down during off-peak hours. "
        "Kubernetes orchestrates containerised workloads and supports horizontal pod "
        "autoscaling based on CPU or custom metrics. CDN caching reduces origin server "
        "load by serving static assets from edge nodes closer to end users.",
    ),
]

SOURCE_IDS = [d[0] for d in DOCUMENTS]
TEXTS      = [d[1] for d in DOCUMENTS]

# ── Three complex, intentionally ambiguous / multi-concept queries ────────────
QUERIES = [
    (
        "Q1",
        "How does the system handle peak load",
        "Infrastructure / scalability — vague phrasing, no technical keywords",
    ),
    (
        "Q2",
        "What techniques reduce errors in AI-generated answers",
        "RAG / hallucination — indirect phrasing, no 'RAG' or 'retrieval' keyword",
    ),
    (
        "Q3",
        "How do modern language models process sequential text efficiently",
        "Transformers / NLP — multi-concept, could match several documents",
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text for display."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def run_comparison() -> List[Dict[str, Any]]:
    """
    Build both RAG systems, run all queries, and return a list of comparison
    records — one per query.
    """
    print("Building RAG systems and ingesting corpus…")

    # Shared embedder — one model load for both strategies
    embedder = EmbeddingService()

    # Strategy A — Raw Vector Search
    raw_rag = RawRetriever(embedding_service=embedder, top_k=3)
    raw_rag.ingest(TEXTS, SOURCE_IDS)

    # Strategy B — Query Expansion (Vertex AI mocked)
    exp_rag = ExpandingRetriever(embedding_service=embedder, top_k=3)
    exp_rag.ingest(TEXTS, SOURCE_IDS)

    print(f"Corpus: {len(TEXTS)} documents → {len(raw_rag.store)} chunks\n")

    records: List[Dict[str, Any]] = []

    for qid, query_text, description in QUERIES:
        # Strategy A — returns List[SearchResult] directly
        raw_results = raw_rag.query(query_text)

        # Strategy B — returns (expanded_query, List[SearchResult])
        expanded_query, exp_results = exp_rag.query(query_text)

        def fmt_results(results) -> List[Dict[str, Any]]:
            return [
                {
                    "rank":   r.rank,
                    "score":  round(r.score, 4),
                    "source": r.chunk.source,
                    "text":   r.chunk.text,
                }
                for r in results
            ]

        record = {
            "query_id":          qid,
            "query":             query_text,
            "description":       description,
            "expanded_query":    expanded_query,
            "strategy_A_raw_vector_search": fmt_results(raw_results),
            "strategy_B_query_expansion":   fmt_results(exp_results),
            "analysis": {
                "top1_source_changed": (
                    raw_results[0].chunk.source != exp_results[0].chunk.source
                    if raw_results and exp_results else None
                ),
                "score_delta_rank1": round(
                    exp_results[0].score - raw_results[0].score, 4
                ) if raw_results and exp_results else None,
                "unique_sources_A": list({r.chunk.source for r in raw_results}),
                "unique_sources_B": list({r.chunk.source for r in exp_results}),
            },
        }
        records.append(record)

    return records


def print_table(records: List[Dict[str, Any]]) -> None:
    """Print a human-readable comparison table to stdout."""
    W = 100
    THICK = "═" * W
    THIN  = "─" * W
    MID   = "·" * W

    print("\n" + THICK)
    print("  RAG STRATEGY COMPARISON  —  Raw Vector Search  vs  Query Expansion (Vertex AI)")
    print(THICK)

    for rec in records:
        qid   = rec["query_id"]
        query = rec["query"]
        desc  = rec["description"]
        exp   = rec["expanded_query"]
        a_res = rec["strategy_A_raw_vector_search"]
        b_res = rec["strategy_B_query_expansion"]
        ana   = rec["analysis"]

        print(f"\n  {qid}: {query}")
        print(f"  Context : {desc}")
        print(THIN)

        # Expanded query (wrapped)
        exp_wrapped = textwrap.fill(exp, width=W - 14, subsequent_indent=" " * 14)
        print(f"  Expanded: {exp_wrapped}")
        print(THIN)

        # Side-by-side results header
        col = (W - 4) // 2
        print(f"  {'STRATEGY A — Raw Vector Search':<{col}}  {'STRATEGY B — Query Expansion':<{col}}")
        print(f"  {'-'*col}  {'-'*col}")

        for i in range(3):
            a = a_res[i] if i < len(a_res) else None
            b = b_res[i] if i < len(b_res) else None

            a_line = (
                f"#{a['rank']} [{a['score']:.4f}] {a['source']}"
                if a else "—"
            )
            b_line = (
                f"#{b['rank']} [{b['score']:.4f}] {b['source']}"
                if b else "—"
            )
            print(f"  {a_line:<{col}}  {b_line:<{col}}")

            # Text snippet on next line, indented
            a_txt = _truncate(a["text"], 90) if a else ""
            b_txt = _truncate(b["text"], 90) if b else ""
            a_txt_w = textwrap.fill(a_txt, width=col - 2, subsequent_indent="    ")
            b_txt_w = textwrap.fill(b_txt, width=col - 2, subsequent_indent="    ")
            # Zip lines for alignment
            a_lines = a_txt_w.splitlines() or [""]
            b_lines = b_txt_w.splitlines() or [""]
            max_lines = max(len(a_lines), len(b_lines))
            for j in range(max_lines):
                al = a_lines[j] if j < len(a_lines) else ""
                bl = b_lines[j] if j < len(b_lines) else ""
                print(f"  {al:<{col}}  {bl:<{col}}")

            if i < 2:
                print(f"  {'·'*col}  {'·'*col}")

        # Analysis summary
        print(THIN)
        changed = "YES ✓" if ana["top1_source_changed"] else "NO"
        delta   = ana["score_delta_rank1"]
        delta_s = f"+{delta:.4f}" if delta and delta >= 0 else f"{delta:.4f}"
        print(
            f"  Analysis │ Top-1 source changed: {changed:<8} "
            f"│ Score Δ (rank-1): {delta_s}  "
            f"│ Expansion benefit: {'HIGH' if delta and delta > 0.15 else 'MODERATE' if delta and delta > 0 else 'NONE'}"
        )
        print(THICK)


def save_json(records: List[Dict[str, Any]], path: str = "comparison_results.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = run_comparison()
    print_table(records)
    save_json(records)
