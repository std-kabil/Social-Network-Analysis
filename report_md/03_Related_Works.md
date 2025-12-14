# Related Works

This section briefly reviews the main ideas that support the methods used in this project. The goal is not to survey everything, but to explain why the selected approaches are reasonable for a large social network dataset.

## Social network datasets and graph analysis

Large online social networks have been widely used as benchmarks for studying connectivity, user interaction patterns, and community structure. The Pokec dataset is distributed through SNAP and has become a common dataset for graph mining research.

In network science, core concepts such as **degree distribution**, **clustering**, **components**, and **shortest paths** are standard tools for describing the topology of a graph [1], [2]. For very large graphs, the main practical challenge is that several “exact” computations are too expensive, so researchers rely on approximations and sampling.

## Small-world phenomenon and “six degrees”

The “six degrees of separation” idea is commonly associated with Milgram’s social experiment [3]. In graph terms, the concept is related to the distribution of shortest path lengths between nodes.

Watts and Strogatz popularized the **small-world** model, showing that networks can have both high clustering and short average path lengths [4]. Newman also studied path lengths and clustering in large networks and discussed how these properties can be measured and interpreted [1].

In practice, computing all-pairs shortest paths is impossible for graphs with millions of nodes. Therefore, many studies use **random sampling of node pairs** to estimate path-length distributions. This is the approach adopted in this project.

## Community detection and modularity

Community detection aims to split a graph into groups of nodes with dense internal connections and sparse external connections. A widely used approach for large graphs is the **Louvain method**, which optimizes **modularity** in a greedy, hierarchical manner [5].

The Louvain method is attractive because:

- It is relatively fast compared to many alternatives.
- It often produces meaningful partitions on social networks.
- It produces a single scalar (modularity) that helps quantify “how strong” the community structure is.

However, community detection on the full Pokec graph is heavy. Therefore, this project applies Louvain on a **sampled subgraph**, with a clear explanation of the tradeoff: the result reflects the sampled region and not necessarily the entire network.

## Centrality measures and scalability

Centrality measures quantify which nodes appear “important” in a network.

- **Degree centrality** is simple and scalable; it depends only on local connections.
- **Betweenness centrality** is more informative but expensive because it relies on shortest paths. NetworkX supports an approximation by sampling `k` source nodes, which provides a usable ranking for large graphs when `k` is small.

This project uses degree centrality and approximate betweenness centrality on a sampled subgraph for practicality.

## Link prediction: classical similarity features and ML baselines

Link prediction asks whether a pair of nodes is likely to have a link (or to form a link in the future). A classical approach is to compute similarity scores:

- **Common Neighbors**: number of shared neighbors
- **Jaccard coefficient**: normalized overlap
- **Adamic–Adar**: weighted common neighbors emphasizing rare shared neighbors
- **Preferential attachment**: degree-based heuristic

These methods are standard baselines in link prediction [6], [7]. A common next step is to use these scores as features for a simple classifier such as logistic regression.

In this project, link prediction is not the main focus, but it is used as a small ML extension to make the project richer while staying computationally lightweight.

## Tools and reproducibility

Several tools make modern network analysis practical:

- **NetworkX** provides a flexible Python interface for graphs, with many standard algorithms.
- **Streamlit** supports interactive dashboards for data analysis.
- **Plotly** supports interactive graphs and HTML exports.

The project’s engineering work emphasizes reproducibility and clarity: results shown in the Streamlit UI are generated from the current run and can be exported to files for later use.

---

## References used in Related Works (preview)

A full reference list is provided in the **References** section. Here the citations are included to show what is being referenced:

- [1] M. E. J. Newman, _Networks: An Introduction_, 2010.
- [2] D. Easley and J. Kleinberg, _Networks, Crowds, and Markets_, 2010.
- [3] S. Milgram, “The Small World Problem,” _Psychology Today_, 1967.
- [4] D. J. Watts and S. H. Strogatz, “Collective dynamics of ‘small-world’ networks,” _Nature_, 1998.
- [5] V. D. Blondel et al., “Fast unfolding of communities in large networks,” _J. Stat. Mech._, 2008.
- [6] D. Liben-Nowell and J. Kleinberg, “The link-prediction problem for social networks,” _JASIST_, 2007.
- [7] L. A. Adamic and E. Adar, “Friends and neighbors on the Web,” _Social Networks_, 2003.
