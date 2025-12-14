# Methods (Part B): Sampling Strategy and Algorithms

This project studies a very large network (over 1.6M nodes). Many standard algorithms that work on small graphs are not feasible at this scale in a typical student environment. Therefore, the core method is:

> **Use the full graph only for operations that are linear or near-linear in the number of edges/nodes, and use sampling for operations that scale super-linearly (or require many shortest-path computations).**

Below I explain each major step, what was chosen, and what alternatives were considered.

## Overall pipeline

For a chosen graph mode (`mutual` or `all`), the analysis pipeline is:

1. **Load or build the graph**
   - `all`: load an undirected graph from the edge list.
   - `mutual`: load a directed graph, compute reciprocity, then build the mutual undirected graph.
2. **Extract the largest connected component (LCC)**
3. **Compute degrees and basic metrics**
4. **Estimate shortest-path length distribution** by sampling random node pairs
5. **Estimate clustering coefficient** by sampling nodes
6. **Community detection (Louvain)** on a BFS-sampled subgraph
7. **Centrality analysis** on a BFS-sampled subgraph
8. **Visualization sampling** (BFS-based) and compute a spring layout

Each step was chosen to balance correctness, interpretability, and feasibility.

## Step 1: Graph loading and memory constraints

### Why NetworkX (and why not a faster library)

NetworkX is not the fastest library for huge graphs, but it is widely used in education, easy to understand, and has a large set of algorithms. This project targets **university student-level** code readability.

**Why not igraph / graph-tool?**

- They are faster and more memory efficient for very large graphs.
- However, they add installation complexity (especially graph-tool) and reduce portability.
- The project requirement is mainly about a clear report and correct reasoning, not squeezing maximum performance.

### Directed vs undirected

The Pokec edges are directed. In many social platforms, “friendship” is undirected, but Pokec is documented as oriented.

This project uses two interpretations:

- `all`: undirected interpretation of any directed link.
- `mutual`: undirected interpretation of only reciprocated links.

This choice is important because directed graphs require different definitions for connectivity and path length. For example:

- In a directed graph, you must distinguish between weakly connected components and strongly connected components.
- In an undirected graph, connectivity and shortest paths are simpler and closer to the “degrees of separation” intuition.

The project focuses on undirected reachability and distances (degrees of separation), so the main analysis runs on an undirected LCC.

## Step 2: Extracting the largest connected component (LCC)

### What we compute

We compute:

- number of connected components
- size of the largest component
- top component sizes (top 5)
- percentage of nodes in the largest component

### Why the LCC is necessary

Shortest paths between random nodes only make sense if a path exists. If a graph has many isolated components, the “average distance” becomes dominated by “no path” cases.

By focusing on the LCC we:

- reduce the chance of “no path” events
- analyze the main part of the social network
- match the dataset’s known property that most nodes belong to a large connected region (especially in the original directed WCC)

### Why not keep multiple components?

We considered analyzing several largest components separately. However:

- the second-largest components are tiny compared to the main LCC (in the mutual graph, the second-largest component has only dozens of nodes)
- a component-by-component analysis would make the report longer without adding much insight

Therefore, the report focuses on the LCC for nearly all metrics.

## Step 3: Basic metrics and degree statistics

On the LCC we compute:

- number of nodes and edges
- average degree
- density
- degree summary: min / median / max

### Why degree distribution matters

Real social networks often show **heavy-tailed** degree distributions:

- most nodes have small degree
- a small number of nodes have very large degree (hubs)

This affects many other properties:

- hubs reduce average path length
- hubs can increase vulnerability to targeted removal
- visualization becomes “hairball-like” because hubs create many edges

### Practical plotting choice: outlier handling

A full degree histogram can be dominated by extreme degrees. In the app, the plot uses:

- random sampling of degrees for responsiveness
- a configurable percentile cap (default 99%) to zoom the x-axis

This is a plotting decision, not a data manipulation decision. It improves readability without changing computed metrics.

## Step 4: Shortest paths (degrees of separation)

### Why not compute exact average path length

Exact average shortest path length requires many shortest-path computations. On a graph with over a million nodes, this is not feasible.

### Sampling approach used

We estimate the shortest path length distribution by:

- sampling `num_pairs` random pairs of nodes from the LCC (with replacement)
- computing `nx.shortest_path_length(G, u, v)` for each pair
- collecting lengths, and computing mean/median/std/min/max

This produces an estimate of “degrees of separation.”

### Why this is reasonable

The dataset readme provides reference values (diameter 11, effective diameter 5.3). Sampling does not guarantee capturing all rare cases, but with a reasonable number of pairs it often provides a stable estimate for the typical distance.

### Why not BFS from many sources instead

An alternative is to pick many source nodes and do BFS trees. That would provide many distances at once and may be more efficient.

However:

- it is harder to explain and implement cleanly
- the pair-sampling method is simple to describe and matches the “random pair” interpretation

For a student report, explainability is important, so the random-pair approach was selected.

## Step 5: Clustering coefficient (sampled)

### Why not compute clustering for all nodes

Computing clustering for every node is heavy on very large graphs, and it is also sensitive to graph representation.

### Sampling approach used

We randomly sample `clustering_sample` nodes from the LCC and compute `nx.clustering(G, nodes=sample_nodes)`, then average the values.

This gives a stable estimate of the network’s local clustering tendency.

### Interpretation caution

Because the clustering is sampled, we must be careful:

- it estimates the average clustering, not the exact value
- different random seeds can change the estimate slightly

Therefore, the report treats it as an estimate and reports the sampling size.

## Step 6: Community detection (Louvain) on a sampled subgraph

### Why not run Louvain on the full LCC

Louvain on a 1.2M-node graph can be very expensive in Python, especially inside NetworkX.

### Sampling method used

We construct a BFS-sampled subgraph:

- pick a high-degree start node
- run BFS until we collect `community_sample` nodes (e.g., 50,000)
- run Louvain on the induced subgraph

This sampling has a motivation:

- starting from a hub tends to reach a dense region quickly
- the subgraph is more “representative” of the main network than a small random induced subgraph

### Why BFS sampling (and why not uniform node sampling)

Uniform random sampling can produce a disconnected subgraph or miss important dense regions.

BFS sampling tends to preserve local structure and connectivity, which is useful for community detection and visualization.

### Limitations

The main limitation is bias:

- BFS sampling over-represents nodes near the start node
- communities found may reflect that region more than the entire network

The report addresses this explicitly. This is a reasonable tradeoff for a student-scale project.

## Step 7: Centrality analysis on a sampled subgraph

### Degree centrality

Degree centrality is computed directly on a sampled connected subgraph. It is easy to interpret and identifies hubs.

### Approximate betweenness

Betweenness centrality is expensive. The project uses NetworkX’s approximation by sampling `k` sources:

- this approximates betweenness while limiting cost

### Why not PageRank

PageRank is meaningful for directed graphs and information flow. Since the main analysis is undirected and topology-focused, PageRank was not emphasized. It can be included as future work.

## Step 8: Visualization subgraph and layout

Visualizing millions of nodes is impossible in a browser. The project uses a 1,000-node sampled graph for the network visualization.

### Sampling method

The visualization subgraph is built by:

1. selecting a candidate pool of nodes using reservoir sampling
2. choosing the maximum-degree node from that pool
3. BFS sampling from that node until `viz_nodes` are collected

This approach avoids scanning degree for all nodes in Python lists and provides a dense, interesting region.

### Layout

We use `spring_layout` with tunable parameters `k` and `iterations`. The report discusses how:

- higher `k` spreads nodes out (less clutter)
- more iterations stabilizes the layout but costs more time

### Community coloring

Communities for visualization are computed on the visualization subgraph. This makes it possible to color nodes by community and provide a clear interactive plot.

## Summary of “why this over that”

The main theme is:

- **Exact algorithms** were replaced by **sampling-based approximations**.
- Graph construction was written to avoid memory-heavy intermediate structures.
- Visualization and ML were performed on manageable subgraphs.

These design choices are not only engineering decisions; they also affect the interpretation of results, so the report includes clear limitations and justification.
