# Conclusion

## Summary of what was done

This project analyzed the **Pokec social network** dataset from SNAP with a focus on **topology**. Because the dataset is very large (1.63M nodes and 30.6M directed edges), a major part of the work was to design an analysis pipeline that can run on a student machine without crashing.

To do this, the project:

- Implemented two graph modes:
  - **mutual friendships** (reciprocal edges only)
  - **all connections** (undirected view of directed edges)
- Extracted the **largest connected component (LCC)** and computed basic metrics.
- Estimated **degrees of separation** using sampled shortest paths.
- Estimated clustering coefficient using sampled nodes.
- Performed Louvain **community detection** on a BFS-sampled subgraph.
- Computed degree and approximate betweenness **centrality** on a sampled subgraph.
- Built a Streamlit dashboard with interactive plots and a Network Explorer with shortest-path highlighting.
- Added a lightweight **machine learning** extension (link prediction) to make the project richer.

## Main experimental conclusions (mutual mode)

Based on the exported mutual-mode run:

- The directed graph has 1,632,803 nodes and 30,622,564 edges.
- The mutual graph contains 8,320,600 reciprocal edges.
- The mutual LCC contains 1,198,274 nodes (73.39% of all nodes) and 8,312,834 edges.
- The estimated average shortest path length is ~5.67 (median 6), supporting a small-world interpretation.
- Louvain on a 50k-node BFS sample found 37 communities with modularity ~0.7855.
- Link prediction on the 1,000-node visualization subgraph achieved ROC-AUC ~0.936.

These results show that even with strict constraints, we can still obtain meaningful social-network conclusions.

## Strengths

- **Memory-aware design:** avoided heavy intermediate structures (especially for mutual edge extraction).
- **Clear reproducibility:** configuration saved with results; exports make reporting easier.
- **Interactive exploration:** visualization with community coloring and path finding helps interpretation.
- **Good sanity checks:** results are consistent with dataset documentation (e.g., max observed shortest path 11).
- **Practical ML extension:** link prediction demonstrates how graph features can be used in a simple ML model.

## Weaknesses

- **Sampling bias:** several results rely on BFS-based sampling, which can bias toward dense regions.
- **Not full-graph community/centrality:** due to scale, the project cannot claim exact global community structure.
- **Limited cross-mode comparison so far:** the exported artifacts currently focus on mutual mode; the all-connections experiment should be added for a stronger comparative report.
- **NetworkX scalability limits:** NetworkX is excellent for clarity but not optimized for graphs of this size.

## Future work

To improve the project further, the following steps are recommended:

1. **Complete mutual vs all comparison**

   - Run the full pipeline for `all` mode and export results.

2. **Reduce sampling bias**

   - Repeat BFS sampling from multiple seeds.
   - Compare BFS sampling with uniform node sampling or random-walk sampling.

3. **Use more scalable graph tooling**

   - Evaluate igraph or graph-tool for faster community detection and centrality.

4. **More advanced ML for link prediction**

   - Use time-based splits if temporal data is available.
   - Try more models (e.g., gradient boosting) and more negative sampling strategies.

5. **(Optional) Add profile attributes**
   - If the scope expands, integrate profile features for node classification or community interpretation.

## Final remark

A key lesson from this project is that large-scale graph analysis is as much about **engineering** as it is about **algorithms**. The best analysis is not the most complex one, but the one that is **correct, reproducible, and feasible** under real constraints.
