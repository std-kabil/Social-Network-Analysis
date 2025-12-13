### Full Project Plan: Social Network Analysis for Six Degrees of Separation Using soc-Pokec Dataset

This is a comprehensive, step-by-step project plan for implementing a data mining course project focused on the "six degrees of separation" theory. The project will use graph mining techniques on the soc-Pokec dataset to compute and visualize degrees of separation, identify network patterns, and provide insights into social connectivity. The plan is designed to be handed off to an AI agent (e.g., another Grok instance or coding AI) for execution. It assumes the agent has access to a Python environment with necessary libraries and can handle file downloads/processing.

The plan is structured for clarity: **Objectives**, **Requirements**, **Detailed Steps**, **Deliverables**, and **Testing/Evaluation**. The agent should follow it sequentially, documenting code in a Jupyter Notebook for reproducibility. If issues arise (e.g., memory constraints), the agent should note adaptations.

#### 1. Project Objectives

- **Primary Goal**: Empirically test and visualize the six degrees of separation theory on a real social network dataset. Compute average shortest path lengths (degrees of separation) between users, analyze network properties (e.g., clustering, centrality), and perform basic mining tasks like community detection and link prediction.
- **Key Insights to Extract**:
  - Average degrees of separation (expect ~4-6 based on dataset stats).
  - Identification of influential "hubs" (high centrality nodes).
  - Community structures and how they affect connectivity.
  - Potential new connections via link prediction.
- **Why Impressive for a Course Project**: Handles large-scale graph data (1.6M nodes, 30M edges), integrates multiple data mining techniques, and ties to sociological theory with real-world applications (e.g., social media analysis).
- **Scope Limitations**: Focus on sampling for efficiency (full all-pairs paths are computationally infeasible). Use undirected graph for separation calculations unless directed analysis is added as an extension.

#### 2. Requirements

- **Environment**: Python 3.10+ (use virtualenv or conda for isolation).
- **Libraries** (install via pip if not present):
  - `networkx` (for graph operations).
  - `pandas` (for attribute handling).
  - `matplotlib` and `seaborn` (for basic visualizations).
  - `plotly` (for interactive dashboards).
  - `community` (python-louvain for community detection; install via `pip install python-louvain`).
  - `scikit-learn` (for any ML-based link prediction).
  - Optional: `igraph` or `graph-tool` if NetworkX is too slow (for performance on large graphs).
- **Hardware**: At least 16GB RAM (32GB recommended, as per user's config). Use sampling if memory issues occur.
- **Dataset**:
  - Download from Stanford SNAP: https://snap.stanford.edu/data/soc-Pokec.html.
  - Files: `soc-pokec-relationships.txt.gz` (edges), `soc-pokec-profiles.txt.gz` (node attributes), and `soc-pokec-readme.txt` (for attribute descriptions).
  - Ethical Note: Data is anonymized; do not attempt de-anonymization.
- **Tools for Development**: Jupyter Notebook for code/documentation. Git for version control (create a repo on GitHub).

#### 3. Detailed Steps

Follow these steps in order. For each, write clean, commented Python code in a notebook section. Handle errors gracefully (e.g., try-except for file I/O). Test incrementally.

**Step 1: Setup and Data Acquisition (1-2 hours)**

- Create a project directory (e.g., `six_degrees_project`).
- Initialize a Git repo: `git init`.
- Create a Jupyter Notebook: `main_analysis.ipynb`.
- Download dataset files using code (e.g., via `requests` library) or manually, then unzip:

  ```python
  import requests
  import gzip
  import shutil

  urls = {
      'edges': 'https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz',
      'profiles': 'https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz',
      'readme': 'https://snap.stanford.edu/data/soc-pokec-readme.txt'
  }

  for name, url in urls.items():
      response = requests.get(url, stream=True)
      with open(f'{name}.gz' if 'gz' in url else name, 'wb') as f:
          shutil.copyfileobj(response.raw, f)

  # Unzip
  for gz_file in ['edges.gz', 'profiles.gz']:
      with gzip.open(f'soc-pokec-{gz_file}', 'rb') as f_in:
          with open(f'soc-pokec-{gz_file.replace(".gz", ".txt")}', 'wb') as f_out:
              shutil.copyfileobj(f_in, f_out)
  ```

- Verify files: Check line counts (edges: ~30M, profiles: ~1.6M) using `wc -l`.

**Step 2: Data Loading and Preprocessing (2-3 hours)**

- Load edges into a NetworkX graph:

  ```python
  import networkx as nx

  G = nx.read_edgelist('soc-pokec-relationships.txt', create_using=nx.DiGraph(), nodetype=int)
  G_undirected = G.to_undirected()  # For separation calculations
  print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
  ```

- Load profiles with Pandas:

  ```python
  import pandas as pd

  # Define columns from readme (59 fields; list them fully here based on readme content)
  columns = ['user_id', 'public', 'gender', 'dob', 'age', 'height', 'weight', 'body_type', 'hair_color', 'eyes',
             'relationship', 'looking_for', 'children', 'smoker', 'drinker', 'education', 'spoken_languages',
             # ... complete the list up to 59, e.g., 'region', 'registration_timestamp', 'last_login_timestamp'
             ]

  profiles = pd.read_csv('soc-pokec-profiles.txt', sep='\t', header=None, names=columns, dtype=str, low_memory=False)
  profiles.set_index('user_id', inplace=True)
  profiles.fillna('unknown', inplace=True)  # Handle missing values

  # Attach key attributes to graph nodes (e.g., gender, age)
  for node in G.nodes():
      if node in profiles.index:
          G.nodes[node]['gender'] = profiles.loc[node, 'gender']
          G.nodes[node]['age'] = profiles.loc[node, 'age']
          # Add more as needed
  ```

- Preprocess attributes: Convert numerical (e.g., age to int where possible), encode categoricals if needed for mining.

**Step 3: Core Analysis - Compute Degrees of Separation (3-4 hours)**

- Sample and compute shortest paths:

  ```python
  import random
  from networkx.algorithms.shortest_paths import shortest_path_length

  num_samples = 10000  # Adjustable; balance time vs. accuracy
  path_lengths = []
  for _ in range(num_samples):
      src, tgt = random.sample(list(G_undirected.nodes()), 2)
      try:
          length = shortest_path_length(G_undirected, src, tgt)
          path_lengths.append(length)
      except nx.NetworkXNoPath:
          pass

  avg_degree = sum(path_lengths) / len(path_lengths)
  print(f'Average degrees of separation: {avg_degree}')
  ```

- Compute network metrics: Average clustering coefficient, degree distribution.
  ```python
  clustering_coeff = nx.average_clustering(G_undirected, count_zeros=True)
  degrees = [d for n, d in G_undirected.degree()]
  ```
- Extension: Compute reciprocity for directed graph.

**Step 4: Advanced Mining Tasks (4-5 hours)**

- Community Detection:

  ```python
  import community as community_louvain

  partition = community_louvain.best_partition(G_undirected)
  # Analyze: Size of communities, modularity
  modularity = community_louvain.modularity(partition, G_undirected)
  ```

- Centrality Measures (identify hubs):
  ```python
  centrality = nx.betweenness_centrality(G_undirected, k=1000)  # Sample k for efficiency
  top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
  ```
- Link Prediction (e.g., predict potential friendships):
  ```python
  from sklearn.metrics import roc_auc_score
  # Use NetworkX predictors (e.g., Jaccard)
  preds = nx.jaccard_coefficient(G_undirected, ebunch=None)  # On subsample
  # Or simple ML: Train on features like common neighbors
  ```

**Step 5: Visualization and Insights (2-3 hours)**

- Plot path length distribution:

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.histplot(path_lengths, bins=20)
  plt.title('Distribution of Degrees of Separation')
  plt.savefig('path_lengths_hist.png')
  ```

- Interactive graph viz (subsample 10k nodes):
  ```python
  import plotly.graph_objects as go
  sub_G = G_undirected.subgraph(random.sample(list(G_undirected.nodes()), 10000))
  # Use nx.spring_layout, then plot with Plotly
  ```
- Dashboard: Use Streamlit for an interactive app (e.g., input two nodes, show path).

**Step 6: Reporting and Deployment (1-2 hours)**

- Compile insights into a Markdown report: Summarize findings (e.g., "Average separation: 4.8, confirming small-world theory").
- Create a GitHub repo: Push notebook, code, visuals.
- Optional Deployment: Host Streamlit app on a free service (e.g., Streamlit Sharing).

#### 4. Deliverables

- Jupyter Notebook (`main_analysis.ipynb`) with all code, outputs, and explanations.
- Visuals: PNG/PDF files (histograms, graphs).
- Report: `project_report.md` with objectives, methods, results, discussion (tie back to theory), and limitations.
- GitHub Repo URL: For sharing.
- Zip of entire project folder.

#### 5. Testing/Evaluation

- Test: Run end-to-end; verify avg_degree ~4-6, no crashes.
- Metrics: Report computation time, memory usage (use `psutil`).
- Edge Cases: Handle disconnected components, missing attributes.
- Improvements: If time, add temporal analysis using timestamps.

Total Estimated Time: 15-20 hours. Agent: Start with Step 1 and provide progress updates if interactive. If clarifications needed, note them. This plan ensures a complete, impressive project!
