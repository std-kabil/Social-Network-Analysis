# Experimental Results

This section reports the results obtained from the implemented pipeline, using the saved run artifacts in `outputs/`. The main goal is to connect each metric to an interpretation, and also to discuss limitations and “why this was done this way.”

## 5.1 Experimental setup

### Hardware/software environment

The analysis was performed in a typical student environment (local machine) using Python tooling and Streamlit. The final runtime and memory usage are recorded in the saved JSON output:

- **Total runtime (mutual run):** ~216.66 seconds
- **End-of-run RAM usage:** ~8.18 GB

These numbers show that a full-graph analysis of Pokec is possible on a non-server machine if we are careful with memory and rely on sampling for heavy tasks.

### Data and configuration

All results in this section come from:

- `outputs/analysis_results_mutual.json`
- `outputs/network_summary_mutual.csv`
- centrality/community CSVs
- ML metrics in `outputs/link_prediction_metrics_mutual.json`

The key configuration values were:

- graph mode: **mutual**
- shortest path samples: **10,000** random pairs
- clustering sample: **10,000** nodes
- Louvain sample: **50,000** nodes (BFS sampled)
- centrality sample: **10,000** nodes (BFS sampled)
- betweenness approximation: **k = 500**
- visualization subgraph: **1,000 nodes**

## 5.2 Dataset sanity check (readme vs our run)

The official Pokec readme provides reference statistics:

- Nodes: 1,632,803
- Directed edges: 30,622,564
- Diameter: 11
- Effective diameter (90%): 5.3

Our run loaded exactly the same node and edge counts in the directed graph:

- directed nodes: **1,632,803**
- directed edges: **30,622,564**

This consistency is important because it increases confidence that the dataset was parsed correctly.

## 5.3 Mutual friendships graph construction

### Reciprocity

The directed Pokec graph has a measured reciprocity of:

- **Reciprocity ≈ 0.5434**

This means that more than half of directed ties are reciprocated. In social-network terms, this suggests that Pokec has a substantial amount of mutual “friendship” behavior, not only one-way links.

### Mutual edges

After extracting mutual (reciprocal) relationships, the mutual undirected graph has:

- **mutual edges:** 8,320,600

A very important engineering observation is that this step can easily crash a machine if implemented naïvely. The memory-safe approach (streaming through edges and checking for reverse edges) enabled this graph to be constructed without building massive intermediate sets.

### “Undirected derived” edges

The summary table also includes a derived value:

- **undirected derived edges ≈ 22,301,964**

This value helps illustrate how many directed edges are _not_ part of mutual friendships. In other words, a large number of edges represent non-reciprocated ties.

## 5.4 Connectivity: connected components and the LCC

Once the mutual graph was built, we extracted connected components and the largest connected component (LCC). The results are:

- number of components: **426,901**
- LCC nodes: **1,198,274**
- LCC percentage: **73.39%**
- LCC edges: **8,312,834**

A key observation is that enforcing mutual friendships significantly fragments the graph:

- The total mutual graph still has all nodes, but it breaks into hundreds of thousands of components.
- The LCC contains most edges and a majority of nodes, but not all nodes.

### Interpretation

This result is very common in social networks:

- Many users have very few reciprocal ties, creating isolates or tiny components.
- The “social core” remains strongly connected.

The top component sizes show how extreme the gap is:

- Largest: 1,198,274
- Second-largest: 32
- Others: 12, 9, 8

So, after the LCC, the remaining components are extremely small. This supports the decision to focus on the LCC for most metrics.

## 5.5 Basic network metrics (mutual LCC)

The mutual LCC has:

- nodes: **1,198,274**
- edges: **8,312,834**
- average degree: **13.87**
- density: **1.16 × 10^-5**
- sampled average clustering coefficient: **0.1050**

### Density

The density is extremely small because the graph has over a million nodes. In a complete graph, the number of possible edges grows as `n(n-1)/2`, which is enormous at this scale. Real social networks are sparse: each user connects to only a tiny fraction of all users.

### Clustering coefficient

The clustering coefficient around **0.105** is quite meaningful:

- It suggests local “triadic closure” (friends of friends are often friends).
- It is close to the readme’s reported clustering coefficient (0.1094), even though our measure was sampled and on the mutual-LCC representation.

We do not claim it is identical because:

- the readme value is for the directed network’s WCC, not our mutual LCC
- our clustering is computed on a sample of nodes

But it is still a useful consistency check.

### Degree statistics

The mutual LCC degree summary is:

- minimum degree: **1** (within the LCC)
- median degree: **7**
- maximum degree: **7,266**

This heavy-tailed behavior (very large max degree compared to median) is typical and explains why the degree histogram needs a “zoom” option. Without zooming, a few hubs stretch the x-axis and hide the main mass of the distribution.

## 5.6 Degrees of separation (shortest path sampling)

The project estimates shortest path lengths by sampling random node pairs within the LCC.

### Results

For 10,000 random pairs:

- found paths: **10,000**
- failed: **0**
- average distance: **5.6749**
- median distance: **6**
- standard deviation: **1.0610**
- min observed: **2**
- max observed: **11**

### Interpretation

These results support a small-world structure:

- The average distance is close to 6, which matches the “six degrees” idea.
- The maximum observed path length (11) matches the dataset readme’s diameter (11). This is a strong sanity check.

However, we must be careful:

- We did not compute the true diameter; we only observed a max of 11 in sampled pairs.
- The readme’s diameter is based on the directed graph’s reported statistics; our graph is mutual and undirected within the LCC.

Still, the agreement is reassuring.

### Why sampling is acceptable here

At this size, an exact all-pairs approach is impossible. Sampling gives a good estimate of the typical distance. The report treats the average distance as an estimate and explicitly reports the number of sampled pairs.

## 5.7 Community detection (Louvain on BFS sample)

Community detection was performed on a BFS-sampled subgraph of 50,000 nodes.

### Results

- sample nodes: **50,000**
- sample edges: **223,265**
- communities found: **37**
- modularity: **0.7855**

### Interpretation

A modularity of ~0.79 is quite high and indicates strong community structure in the sampled region.

The top communities (by size) show that the sample is divided into several large communities instead of one huge community, which is consistent with social network behavior.

### Limitations and bias

This result is for a BFS region. BFS can bias results toward the neighborhood of a hub. Therefore:

- we interpret the result as “community structure exists and is strong” rather than claiming an exact community count for the full graph
- for future work, multiple BFS starting points or uniform sampling could be compared

## 5.8 Centrality analysis (sampled)

Centrality was computed on a BFS sample of 10,000 nodes.

### Degree centrality (top nodes)

From the saved CSV (`top_degree_centrality_mutual.csv`), the top node in the sample is:

- node **5867**, degree-in-sample **7266**, degree-centrality ≈ **0.7267**

This illustrates hub behavior: one node connects to a large fraction of the sampled subgraph.

### Betweenness centrality (approximate)

From `top_betweenness_mutual.csv`, the top node is also:

- node **5867**, betweenness ≈ **0.7692**

This is expected: hubs often lie on many shortest paths, especially in networks with a star-like structure around important connectors.

### What these rankings mean (and what they do not mean)

Because this is based on a sampled subgraph:

- we can confidently say “node 5867 is central in the sampled region”
- we should not claim “node 5867 is the most central node in the entire Pokec network”

The report emphasizes this difference to avoid over-claiming.

## 5.9 Visualization and interactive exploration

A major deliverable of the project is an interactive visualization:

- The Network Explorer renders a 1,000-node sampled subgraph.
- Nodes are colored by community.
- Hover shows node ID, degree, and community.
- The user can select a source and target node and highlight the shortest path.

### Practical visualization challenges

Large networks create “hairball” plots. This is not a bug; it is a reality of dense regions. The project provides controls to improve clarity:

- adjust spring-layout `k` and `iterations`
- adjust visualization node count
- improve community color contrast (categorical remapping + high-contrast colors)

The visualization is saved as HTML (`network_visualization_mutual.html`) so it can be used in the report or presentation.

## 5.10 Machine learning extension: link prediction

To make the project more modern and “cooler,” a link prediction module was added.

### Dataset construction

On the 1,000-node visualization subgraph:

- positive edges: existing edges (sampled)
- negative edges: randomly sampled non-edges

The dataset was balanced:

- positives: 2,485
- negatives: 2,485

### Features

For each candidate pair `(u, v)`, the model uses:

- common neighbors
- Jaccard coefficient
- Adamic–Adar
- preferential attachment
- degree features (deg_u, deg_v, sum, product)

### Model

A simple logistic regression model was trained with feature scaling. The goal is not to beat research-level models, but to provide a strong baseline and a clear explanation.

### Results

From `link_prediction_metrics_mutual.json`:

- ROC-AUC ≈ **0.9361**
- Average Precision ≈ **0.9398**

These are strong results for a simple model, suggesting that the local topology contains strong signals for predicting missing edges.

### Interpretation and caution

We must be careful when interpreting this:

- The dataset is constructed from the same static snapshot; it does not represent “future links.”
- Random negative sampling can be easier than real-world negatives.
- Performance may drop if we evaluate on a more realistic split.

Still, as an educational extension, it demonstrates a successful integration of ML with graph features.

## 5.11 Summary of key findings

For the mutual-friendships configuration:

- Pokec shows **small-world behavior** with average distance ~**5.67** and observed max **11**.
- The mutual graph becomes highly fragmented, but a large LCC still contains **~73%** of nodes.
- The network is sparse but shows meaningful **clustering** (~0.105) and strong sampled **community structure** (modularity ~0.786).
- A few nodes behave as hubs, which strongly affects paths and visualization.
- Link prediction on a sampled subgraph achieves strong baseline metrics (ROC-AUC ~0.936).

## 5.12 Missing experiment: full “all connections” comparison

The Streamlit app supports an “all connections” mode, but the current repository exports only include the mutual-mode full results.

To complete the comparison section, the next step is to:

- run the app in `all` mode
- click **Save results to outputs/**
- include the resulting `analysis_results_all.json` and plots

In the report’s future work, this is described as a required extension. The methods are already implemented; what remains is running and exporting the second set of results.
