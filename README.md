# Social Network Analysis: Six Degrees of Separation

A comprehensive data mining project analyzing the **soc-Pokec** dataset to empirically test the "six degrees of separation" theory using graph mining techniques.

## Project Overview

This project implements social network analysis on the soc-Pokec dataset (Slovak social network with 1.6M users and 30M relationships) to:

- Compute average shortest path lengths (degrees of separation)
- Analyze network properties (clustering, centrality, density)
- Perform community detection using Louvain algorithm
- Identify influential hub nodes
- Predict potential new connections via link prediction

## Dataset

**soc-Pokec** from [Stanford SNAP](https://snap.stanford.edu/data/soc-Pokec.html):

- **Nodes**: ~1.6 million users
- **Edges**: ~30 million directed friendships
- **Attributes**: 59 profile fields (age, gender, region, etc.)

## Project Structure

```
Social-Network-Analysis/
├── main_analysis.ipynb    # Main Jupyter notebook with all analysis
├── requirements.txt       # Python dependencies
├── project-plan.md        # Detailed project plan
├── README.md              # This file
├── data/                  # Dataset files (downloaded automatically)
│   ├── soc-pokec-relationships.txt
│   ├── soc-pokec-profiles.txt
│   └── soc-pokec-readme.txt
└── outputs/               # Generated visualizations and results
    ├── path_lengths_distribution.png
    ├── degree_distribution.png
    ├── community_size_distribution.png
    ├── clustering_distribution.png
    ├── network_summary.png
    ├── network_visualization.html
    └── analysis_results.json
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Social-Network-Analysis
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook main_analysis.ipynb
   ```

2. Run cells sequentially. The notebook will:
   - Download the dataset automatically (~500MB compressed)
   - Load and preprocess the graph
   - Compute degrees of separation
   - Perform community detection and centrality analysis
   - Generate visualizations

**Note**: The full analysis requires at least 16GB RAM. Sampling is used for computationally intensive operations.

## Key Findings

Expected results based on the analysis:

| Metric                         | Value    |
| ------------------------------ | -------- |
| Average Degrees of Separation  | ~4-5     |
| Network Density                | ~0.00002 |
| Average Clustering Coefficient | ~0.1     |
| Number of Communities          | ~100-500 |

The analysis confirms the "small-world" property of social networks, with most users connected within 6 degrees.

## Techniques Used

- **Graph Mining**: NetworkX for graph operations
- **Community Detection**: Louvain algorithm (python-louvain)
- **Centrality Measures**: Degree, Betweenness, PageRank
- **Link Prediction**: Jaccard coefficient, Adamic-Adar index
- **Visualization**: Matplotlib, Seaborn, Plotly

## Requirements

- Python 3.10+
- 16GB+ RAM (32GB recommended)
- ~2GB disk space for dataset

## License

This project is for educational purposes. The soc-Pokec dataset is provided by Stanford SNAP under their terms of use.

## References

- [Six Degrees of Separation (Wikipedia)](https://en.wikipedia.org/wiki/Six_degrees_of_separation)
- [soc-Pokec Dataset](https://snap.stanford.edu/data/soc-Pokec.html)
- [NetworkX Documentation](https://networkx.org/)
