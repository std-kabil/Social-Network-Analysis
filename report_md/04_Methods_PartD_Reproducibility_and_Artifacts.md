# Methods (Part D): Reproducibility, Parameters, and Output Artifacts

This part documents how to reproduce the experiments and what artifacts are produced. This is important for a course report because it shows that the results are not “random screenshots,” but come from a controlled process.

## Reproducibility strategy

### Fixed random seed

The analysis uses a fixed random seed (default **42**) for:

- random node-pair selection for shortest paths
- node sampling for clustering coefficient
- reservoir sampling for visualization candidate nodes
- sampling for link prediction (positive edge subset and negative non-edges)

A fixed seed makes the results repeatable. If the seed changes, results can change slightly (especially sampled estimates).

### Configuration saved in the results JSON

Each analysis run stores its configuration inside the saved JSON file:

- `outputs/analysis_results_{mode}.json`

This configuration includes:

- graph mode (`mutual` or `all`)
- sample sizes (`num_pairs`, `clustering_sample`, `community_sample`, `centrality_sample`)
- approximation parameter (`betweenness_k`)
- visualization settings (`viz_nodes`, `k`, `iterations`)
- degree histogram plotting cap percentile

Because the configuration is saved together with results, the report can always trace a number back to the exact run settings.

## Experimental setup used in the saved mutual run

The saved results in this repository include a full run for **mutual** mode. The key settings (from `analysis_results_mutual.json`) are:

- **Graph mode:** mutual
- **Shortest path pairs:** 10,000
- **Clustering sample nodes:** 10,000
- **Community sample nodes (Louvain BFS sample):** 50,000
- **Centrality sample nodes (BFS sample):** 10,000
- **Betweenness approximation:** `k = 500`
- **Degree histogram cap:** 99th percentile (plot-only)
- **Visualization subgraph:** 1,000 nodes, spring layout `k=0.5`, `iterations=50`

## What “Save results to outputs/” generates

A common source of confusion during development was that “files in `outputs/`” can include old artifacts created by notebooks. The Streamlit app was updated so that clicking **Save results to outputs/** always generates artifacts from the current session.

### Core analysis artifacts

When you run the analysis and click save, the app generates:

- `analysis_results_{mode}.json`  
  Full metrics payload.

- `network_summary_{mode}.csv` and `network_summary_{mode}.png`  
  A clean, report-ready metric summary table.

- Plotly HTML plots:

  - `degree_distribution_{mode}.html`
  - `path_length_distribution_{mode}.html`
  - `network_visualization_{mode}.html` (if the visualization subgraph is available)

- Extra plots added to make the project more complete:

  - `top_degree_centrality_{mode}.html`
  - `top_betweenness_{mode}.html`
  - `community_sizes_{mode}.html`
  - `component_sizes_{mode}.html`

- CSV tables for report tables:
  - `top_degree_centrality_{mode}.csv`
  - `top_betweenness_{mode}.csv`
  - `top_communities_{mode}.csv`
  - `component_sizes_{mode}.csv`

### ML (link prediction) artifacts

If the ML tab is executed before saving, additional files are saved:

- `link_prediction_metrics_{mode}.json`
- `link_prediction_roc_{mode}.html`
- `link_prediction_pr_{mode}.html`
- `link_prediction_top_edges_{mode}.csv`

These artifacts make it easier to write a strong Experimental Results section with real plots and real tables.

## How to reproduce the run step-by-step

1. **Prepare data**

   - Ensure `data/soc-pokec-relationships.txt` exists.

2. **Start the Streamlit app**

   - Run: `streamlit run app.py`

3. **Choose graph mode**

   - Mutual friendships only (recommended first).

4. **Set parameters**

   - Use the default settings (or the same ones as listed above).

5. **Run analysis**

   - Click **Run analysis** and wait for completion.

6. **(Optional) Run ML**

   - Go to **ML (Link Prediction)** and click **Run link prediction**.

7. **Save artifacts**

   - Click **Save results to outputs/**.

8. **Use the saved files in the report**
   - Include screenshots of the Plotly plots, or link them as figure references.

## Practical note on Streamlit warnings

During one run, Streamlit produced warnings such as:

- “Please replace `use_container_width` with `width` …”

These warnings indicate future deprecation but do not invalidate the results. The codebase was updated to use `width='stretch'` where applicable.
