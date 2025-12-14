# Introduction

## Motivation

Online social networks naturally form graphs: users are nodes, and relationships between users are edges. Studying these graphs helps answer questions such as:

- How connected is the network?
- Do we observe “small-world” behavior (short paths between random users)?
- Are there clear communities (clusters) of users?
- Which nodes appear to be central or influential from a topological perspective?

For a student project, a realistic dataset is important because toy graphs often hide the practical challenges of data size and computational cost. The **Pokec social network** (soc-Pokec) from SNAP is a strong choice because it is large, real, and well-documented. The dataset includes **1.6M+ nodes** and **30M+ directed edges**, making it large enough to require careful engineering.

## Problem statement

The central problem of this project is:

> **How can we perform a meaningful, well-explained topology-focused analysis of the Pokec social network under realistic memory constraints, while producing outputs that are reproducible, interpretable, and easy to explore?**

This problem includes two types of challenges:

1. **Scientific/analytical challenges**: selecting the right metrics, designing experiments (especially for “six degrees”), and interpreting results responsibly when we rely on sampling.
2. **Engineering challenges**: handling memory limits, choosing graph representations, avoiding expensive intermediate structures, designing a workflow that does not freeze the machine, and building interactive outputs that always correspond to the latest run.

## Key design idea: topology-first and memory-aware

The project deliberately focuses on **topology** (structure) rather than user attributes. Although Pokec provides a large profile table, using attributes at full scale is expensive and not strictly required to study path lengths, reciprocity, density, clustering, or communities. Keeping the scope focused allows the project to go deeper on graph algorithms and sampling.

Because the full Pokec graph is extremely large, several common patterns are not feasible on a typical laptop:

- Computing all-pairs shortest paths.
- Running full-graph community detection or betweenness centrality.
- Materializing huge Python sets such as `set(G.edges())` for 30M edges.
- Attaching per-node attributes through slow per-node DataFrame lookups.

Instead, this project uses a **pipeline of “full graph only when necessary”** operations (e.g., loading edges, extracting the largest component) and **sampling-based computations** for expensive tasks (shortest paths, community detection, centrality, visualization).

## Research questions

To keep the report focused while still deep and complete, the analysis is organized around the following research questions:

1. **Connectivity and components**: How much of the graph lies in the largest connected component, especially after enforcing reciprocity (mutual friendships)?
2. **Small-world behavior / six degrees**: What is the typical shortest-path distance between two users (estimated by sampling), and does it match the “six degrees” intuition?
3. **Community structure**: Do we observe strong community structure (via Louvain modularity) when analyzing a representative sample subgraph?
4. **Centrality and hubs**: Which nodes become central in sampled subgraphs, and what do heavy-tailed degrees imply for connectivity?
5. **Practical ML add-on**: Can classical link prediction features provide a strong baseline on a sampled subgraph, and what can we learn from these predictions?

## Contributions

This project contributes both analysis results and a practical tool:

- A **Streamlit** application that runs the analysis pipeline and supports interactive exploration.
- A memory-aware method to construct a **mutual friendship graph** without building huge intermediate edge sets.
- A consistent approach to **sampling** for shortest paths, clustering, community detection, and centrality.
- A **network explorer** visualization with community coloring and shortest-path highlighting.
- A clean separation between **live session results** and **files on disk**, avoiding confusion caused by stale artifacts.
- An optional ML module for **link prediction** that stays small and fast.

## Report organization

The remainder of this report follows the required structure:

- **Related Works** reviews key background on social network analysis, community detection, and link prediction.
- **Methods** describes the dataset, graph construction choices, sampling strategies, and system implementation.
- **Experimental Results** reports measured values and provides interpretation.
- **Conclusion** summarizes findings, strengths, weaknesses, and future work.
