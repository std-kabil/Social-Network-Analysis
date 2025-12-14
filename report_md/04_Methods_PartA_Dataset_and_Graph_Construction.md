# Methods (Part A): Dataset and Graph Construction

This section describes the dataset, the graph representations used in the project, and the design decisions made to handle scale and memory constraints.

## Dataset: Pokec social network (soc-Pokec)

### Data source and meaning

The dataset used is the **Pokec social network**, provided by SNAP. Pokec is a Slovak online social network. The dataset includes:

- `soc-pokec-relationships.txt`: directed friendship relations, one relation per line.
- `soc-pokec-profiles.txt`: user profile attributes in Slovak (not used in this project because the focus is topology).

The dataset readme states that the friendships are **oriented (directed)**, meaning that an entry `u v` indicates user `u` lists user `v` as a friend.

### Dataset statistics (from the official readme)

From `soc-pokec-readme.txt`:

- Nodes: **1,632,803**
- Directed edges: **30,622,564**
- Nodes in largest WCC: **1,632,803 (1.000)**
- Edges in largest WCC: **30,622,564 (1.000)**
- Nodes in largest SCC: **1,304,537 (0.799)**
- Edges in largest SCC: **29,183,655 (0.953)**
- Average clustering coefficient: **0.1094**
- Diameter: **11**
- 90-percentile effective diameter: **5.3**

These numbers are useful as a sanity check for our own computations. In particular, the reported diameter of 11 provides a reference point for sampled shortest-path experiments.

## Why topology-only (and why not profiles)

The dataset contains a very wide profile table with many attributes. In early notebook attempts, attaching profile attributes to every node caused major performance issues:

- A naïve implementation iterated over all nodes and repeatedly accessed a Pandas DataFrame using `.loc` for each node.
- At this scale (over 1.6M nodes), this approach is too slow and causes memory pressure.

Because the project objective is **graph topology**, the profile table was excluded from the main analysis. This is an example of an important “why not” decision:

- **Why not profiles?** They are interesting, but they introduce a second very large dataset and push the project toward attribute analysis (demographics, text processing), which is outside the scope.
- **Why topology?** Topology supports clear metrics (paths, clustering, communities), and we can explain tradeoffs and sampling while staying within practical constraints.

## Graph representations and modes

The Pokec relationships are directed. There are multiple valid ways to interpret this, and each choice affects results. This project implements two graph modes:

### 1) Mutual friendships only (reciprocal edges)

A mutual friendship is a pair `(u, v)` such that **both** directed edges exist: `u -> v` and `v -> u`. The mutual graph is undirected and contains only reciprocal edges.

**Motivation:** Mutual links may represent stronger social ties than one-way links. They often create a denser “core” of reliable relationships.

**Engineering challenge:** Building the mutual graph naïvely by materializing a Python set of all edges is memory heavy:

- `set(G_directed.edges())` for ~30M edges is extremely large (tens of millions of tuples).
- This can exceed memory and freeze the machine.

**Memory-aware solution used:** iterate through directed edges and check whether the reverse edge exists:

- For each edge `(u, v)`, if `u < v` and the reverse edge `(v, u)` exists, add `(u, v)` to the mutual undirected graph.
- This avoids storing a separate set of edges.

This was implemented in the Streamlit app as `build_mutual_graph`.

### 2) All connections (undirected view)

In the “all connections” mode, each directed edge is treated as an undirected connection. This produces an undirected graph that represents “there is some relationship between u and v,” without requiring reciprocity.

**Motivation:** This mode is closer to the raw dataset size and may better represent information flow in a directed network.

**Tradeoff:** Converting directed edges into undirected edges may create connections that are not “mutual friendships,” which changes clustering and community structure.

### Why keep both modes?

The project is designed to allow a comparison:

- Mutual mode emphasizes strong ties but removes many edges.
- All-connections mode includes weak/unreciprocated ties and may increase connectivity.

In practice, mutual mode is often more memory-friendly after construction because it contains fewer edges. Also, some metrics (like reciprocity) are meaningful only when starting from the directed graph.

## Largest Connected Component (LCC) selection

Many real-world networks contain many small disconnected components (isolates, tiny groups). Several algorithms assume connectivity (or become hard to interpret when the graph is fragmented). Therefore, the analysis extracts the **largest connected component** (LCC) and performs most measurements on it.

### Why analyze the LCC?

- The LCC contains the majority of nodes that participate in the main network.
- Shortest-path sampling is meaningful when paths exist between nodes.
- Many metrics become more stable when applied to the largest connected region.

### Why not analyze all components?

- The graph contains many small components; analyzing each component separately would be complex and not very informative.
- Many node pairs across components have no path, which would dominate shortest-path statistics.

### Connectivity vs directed structure

Because the original network is directed, one could consider:

- Weakly connected components (WCC) on the directed graph.
- Strongly connected components (SCC) on the directed graph.

The dataset readme reports both WCC and SCC sizes. In this project, after we convert to an undirected representation (mutual or all-connections), we extract the LCC of the undirected graph. This is aligned with the goal of measuring “degrees of separation” in the sense of undirected reachability.

## Reproducibility and configuration

All experiments are controlled by a configuration object that includes:

- graph mode
- random seed
- sampling sizes (shortest path pairs, clustering nodes, community nodes, centrality nodes)
- visualization sampling parameters

This ensures that results can be re-run and compared using consistent settings.
