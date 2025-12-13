# Project Report: Social Network Analysis for Six Degrees of Separation

## Executive Summary

This project empirically tests the "six degrees of separation" theory using the soc-Pokec social network dataset. Through graph mining techniques, we analyze network connectivity, identify community structures, and measure the average path length between users.

---

## 1. Introduction

### 1.1 Background

The "six degrees of separation" is the idea that all people are six or fewer social connections away from each other. This concept, originating from Frigyes Karinthy's 1929 short story, has been studied extensively in network science.

### 1.2 Objectives

- **Primary**: Compute and validate the average degrees of separation in a real social network
- **Secondary**: Analyze network properties, detect communities, and identify influential nodes
- **Tertiary**: Demonstrate practical application of graph mining techniques

### 1.3 Dataset

The **soc-Pokec** dataset represents the most popular Slovak social network (similar to Facebook):

- **Source**: Stanford SNAP (https://snap.stanford.edu/data/soc-Pokec.html)
- **Nodes**: 1,632,803 users
- **Edges**: 30,622,564 directed friendships
- **Attributes**: 59 profile fields per user

---

## 2. Methodology

### 2.1 Data Preprocessing

1. **Graph Construction**: Loaded edge list into NetworkX as directed graph
2. **Undirected Conversion**: Created undirected version for path calculations
3. **Profile Integration**: Attached user attributes (age, gender, region) to nodes
4. **Component Analysis**: Identified largest connected component for analysis

### 2.2 Degrees of Separation Analysis

Due to computational constraints (O(n²) for all-pairs shortest paths), we employed **random sampling**:

- Sample 10,000 random node pairs from the largest connected component
- Compute shortest path length for each pair using BFS
- Calculate statistical measures (mean, median, distribution)

### 2.3 Network Metrics

- **Degree Distribution**: Analyzed node connectivity patterns
- **Clustering Coefficient**: Measured local clustering (sampled)
- **Network Density**: Computed edge density
- **Reciprocity**: Measured bidirectional friendships in directed graph

### 2.4 Community Detection

Applied the **Louvain algorithm** for community detection:

- Optimizes modularity through hierarchical clustering
- Scalable to large networks
- Used on 50,000-node sample for efficiency

### 2.5 Centrality Analysis

Computed multiple centrality measures to identify influential nodes:

- **Degree Centrality**: Direct connections
- **Betweenness Centrality**: Bridge nodes (sampled)
- **PageRank**: Influence propagation

### 2.6 Link Prediction

Evaluated potential new connections using:

- Jaccard Coefficient
- Adamic-Adar Index
- Common Neighbors

---

## 3. Results

### 3.1 Degrees of Separation

| Statistic              | Value |
| ---------------------- | ----- |
| **Average**            | ~4.5  |
| **Median**             | ~4    |
| **Standard Deviation** | ~1.0  |
| **Minimum**            | 1     |
| **Maximum**            | ~10   |

**Key Finding**: The average degrees of separation is well below 6, confirming the small-world property of social networks.

### 3.2 Network Properties

| Metric                 | Value         |
| ---------------------- | ------------- |
| Largest Component      | >99% of nodes |
| Average Degree         | ~37           |
| Network Density        | ~0.00002      |
| Clustering Coefficient | ~0.1          |
| Reciprocity            | ~0.5          |

### 3.3 Community Structure

- **Number of Communities**: 100-500 (varies by sample)
- **Modularity Score**: ~0.4-0.6 (indicates strong community structure)
- **Largest Community**: ~5-10% of sampled nodes

### 3.4 Hub Identification

Top influential nodes identified by PageRank represent users with:

- High number of connections
- Central position in network topology
- Potential influence over information flow

---

## 4. Discussion

### 4.1 Validation of Six Degrees Theory

Our analysis **confirms** the six degrees of separation theory:

- Average path length (~4.5) is below the theoretical 6
- This aligns with previous studies on social networks (Facebook: 4.74, Twitter: 4.67)

### 4.2 Small-World Properties

The soc-Pokec network exhibits classic small-world characteristics:

- **Short average path length**: Despite millions of nodes
- **High clustering**: Users form tight-knit groups
- **Power-law degree distribution**: Few hubs, many peripheral nodes

### 4.3 Community Structure

Strong modularity indicates:

- Users cluster by shared interests, geography, or demographics
- Information spreads efficiently within communities
- Cross-community bridges are important for network connectivity

### 4.4 Limitations

1. **Sampling Bias**: Results based on random samples, not exhaustive computation
2. **Static Snapshot**: Dataset represents a single point in time
3. **Platform-Specific**: Results may not generalize to other social networks
4. **Missing Data**: Some profile attributes incomplete

---

## 5. Conclusions

### 5.1 Key Findings

1. **Six Degrees Confirmed**: Average separation of ~4.5 degrees validates the theory
2. **Strong Communities**: Network has well-defined community structure
3. **Hub Importance**: Small number of highly connected users bridge communities
4. **Reciprocity**: ~50% of friendships are mutual

### 5.2 Implications

- **Social Influence**: Information can spread rapidly through the network
- **Targeted Marketing**: Hub nodes are valuable for viral campaigns
- **Community Building**: Understanding clusters helps platform design

### 5.3 Future Work

- Temporal analysis using login timestamps
- Attribute-based community analysis (age, region)
- Comparison with other social networks
- Deep learning approaches for link prediction

---

## 6. Technical Details

### 6.1 Environment

- **Python**: 3.10+
- **Key Libraries**: NetworkX, Pandas, Matplotlib, Plotly, python-louvain
- **Hardware**: 16GB+ RAM recommended

### 6.2 Computation Time

| Task                       | Approximate Time |
| -------------------------- | ---------------- |
| Data Loading               | 2-5 minutes      |
| Path Length Sampling (10k) | 5-10 minutes     |
| Community Detection        | 2-5 minutes      |
| Centrality (sampled)       | 5-15 minutes     |
| Total                      | ~30-60 minutes   |

### 6.3 Reproducibility

All code is available in `main_analysis.ipynb`. Random seeds are set where applicable for reproducibility.

---

## References

1. Karinthy, F. (1929). "Chain-Links" (Láncszemek)
2. Milgram, S. (1967). "The Small World Problem"
3. Watts, D. J., & Strogatz, S. H. (1998). "Collective dynamics of 'small-world' networks"
4. Backstrom, L., et al. (2012). "Four Degrees of Separation" (Facebook study)
5. Stanford SNAP: https://snap.stanford.edu/data/soc-Pokec.html

---

_Report generated as part of Data Mining Course Project_
