# Methods (Part C): System Implementation and Engineering Challenges

This project is not only about network metrics; it also includes a practical system that can be used to explore results. This section describes the implementation and explains the main engineering challenges and how they were solved.

## Development approach: notebooks vs Streamlit app

### Early notebook workflow

The initial analysis was performed in a Jupyter notebook. This is a natural starting point because notebooks allow quick experimentation.

However, notebooks have challenges for a very large dataset:

- Long-running cells can freeze the kernel.
- It is easy to accidentally create huge intermediate objects.
- Output artifacts (images/HTML) can remain in the `outputs/` folder and become confused with later results.

### Motivation for Streamlit

Streamlit was introduced to provide:

- a repeatable analysis workflow
- interactive parameter control
- clear separation between _running the analysis_ and _viewing results_
- exportable outputs (JSON, CSV, HTML, PNG)

The Streamlit app became the “main interface” of the project.

## Memory-driven engineering decisions

The biggest practical constraint was memory. Several common patterns caused crashes or extreme slowness:

### 1) Mutual edge construction

A typical approach to build a mutual graph is:

- store all edges in a set
- for each edge check if reverse exists

At 30M edges, storing a set of edges is huge and can freeze a student machine.

**Solution used:** a streaming mutual-edge algorithm:

- iterate through directed edges
- if reverse edge exists, add to undirected mutual graph
- never build a giant edge set

### 2) Avoiding expensive per-node attribute attachment

Even simple loops over all nodes can be too slow. A naïve approach to attach profile attributes was:

- for each node, do `profiles.loc[node, ...]`

This is slow because it performs many index lookups.

**Decision:** focus on topology only; keep the pipeline small and clear.

### 3) Avoiding full-graph heavy algorithms

Algorithms like full betweenness centrality and full community detection are not feasible on the full graph. The app uses sampling and approximations.

## UI design and state management

### Why state matters

In Streamlit, the script can rerun frequently (for example when you interact with widgets). If the app does not store results in session state, it might:

- recompute heavy tasks unintentionally
- display stale plots from a previous run
- mix outputs from different runs

### Session state design

The app stores a compact “analysis payload” in `st.session_state`, including:

- the summary analysis dictionary (`analysis`)
- `degrees` (as a compact NumPy array for plotting)
- `path_lengths` (list from sampled shortest paths)
- `viz` (nodes/edges/positions for visualization)

The full LCC graph object is deleted after extracting the required arrays, to free memory.

## A key challenge: stale outputs vs latest run

### Problem

When the app displays files from `outputs/`, it is easy to accidentally show artifacts produced by an older notebook run. This confused the interpretation of results.

### Solution

The app’s **Outputs** tab was redesigned to show:

1. **Latest run (from Streamlit session)**

   - always generated from session state
   - summary table as a dataframe
   - plots regenerated from current session data

2. **Files on disk (outputs/)**
   - displayed in an expander
   - clearly labeled as potentially legacy
   - optional image preview

This design prevents a user from thinking that an old PNG is “the result of the current analysis.”

## Duplicate element IDs and stable chart rendering

Streamlit requires unique IDs for widgets and charts. When similar plots appear in multiple tabs, a `StreamlitDuplicateElementId` error can occur.

**Fix:** provide unique `key=` values for each chart.

Additionally, Streamlit sometimes reuses elements between reruns. To guarantee that the Outputs tab always represents the latest run, the app tracks a `run_id` that increments after each analysis, and uses it to generate unique keys.

## Exporting results

A project report is easier to write when results are exportable.

### What gets saved

The app can save:

- `analysis_results_{mode}.json`: full analysis payload
- `network_summary_{mode}.csv`: metric-value table
- `network_summary_{mode}.png`: rendered summary table (Matplotlib)
- plot HTML files:
  - degree distribution
  - shortest path length distribution
  - network visualization
  - centrality bar charts
  - community size chart
  - component sizes chart
- link prediction artifacts (if ML ran):
  - metrics JSON
  - ROC/PR curve HTML
  - top predicted edges CSV

### Why HTML

Plotly HTML exports are convenient because they keep interactivity and can be opened without Streamlit.

## Handling Streamlit deprecations

During development, Streamlit emitted warnings about `use_container_width`. The app was updated to use the newer `width='stretch'` option for charts and images.

This type of warning is not part of network science, but it matters for a stable deliverable.

## Summary

The main engineering outcome is a reproducible tool that:

- can run a meaningful analysis under limited memory
- avoids common performance traps
- produces exports that support a well-documented report
- provides interactive exploration (visualization + pathfinder + link prediction)

This system-level work is essential because, without it, many of the analyses would not be practical on a student machine.
