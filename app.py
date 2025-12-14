import csv
import gc
import io
import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import psutil
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


DATA_DIR_DEFAULT = Path("data")
OUTPUTS_DIR_DEFAULT = Path("outputs")


def get_memory_usage_gb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


@dataclass(frozen=True)
class AnalysisConfig:
    graph_mode: str  # "mutual" | "all"
    num_pairs: int
    random_seed: int
    clustering_sample: int
    community_sample: int
    centrality_sample: int
    betweenness_k: int
    degree_hist_percentile: float


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_results_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_graph_all_connections(edges_file: Path) -> nx.Graph:
    # Treat each line (u v) as an undirected edge.
    # This avoids storing a full directed graph in memory.
    return nx.read_edgelist(str(edges_file), create_using=nx.Graph(), nodetype=int)


def load_graph_directed(edges_file: Path) -> nx.DiGraph:
    return nx.read_edgelist(str(edges_file), create_using=nx.DiGraph(), nodetype=int)


def build_mutual_graph(G_directed: nx.DiGraph, progress_cb=None) -> nx.Graph:
    # Memory-friendly mutual edge extraction:
    # iterate edges and check reverse existence; no `set(G.edges())`.
    G_mutual = nx.Graph()
    G_mutual.add_nodes_from(G_directed.nodes())

    total_edges = G_directed.number_of_edges()
    last_update = 0

    for idx, (u, v) in enumerate(G_directed.edges()):
        if u < v and G_directed.has_edge(v, u):
            G_mutual.add_edge(u, v)

        if progress_cb is not None:
            # Update at most ~200 times
            if total_edges > 0:
                step = max(1, total_edges // 200)
                if idx - last_update >= step:
                    last_update = idx
                    progress_cb(min(0.999, idx / total_edges))

    if progress_cb is not None:
        progress_cb(1.0)

    return G_mutual


def connected_components_summary(G: nx.Graph) -> Tuple[int, set[int], List[int]]:
    # Returns (num_components, largest_component_nodes, sizes_top5)
    num = 0
    largest: set[int] = set()
    top_sizes: List[int] = []

    for comp in nx.connected_components(G):
        num += 1
        comp_size = len(comp)
        if comp_size > len(largest):
            largest = comp

        top_sizes.append(comp_size)
        if len(top_sizes) > 50:
            top_sizes.sort(reverse=True)
            top_sizes = top_sizes[:5]

    top_sizes.sort(reverse=True)
    top5 = top_sizes[:5]
    return num, largest, top5


def lcc_subgraph(G: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
    t0 = time.time()
    num_components, largest_nodes, top5 = connected_components_summary(G)
    G_lcc = G.subgraph(largest_nodes).copy()
    elapsed = time.time() - t0

    info = {
        "num_components": num_components,
        "largest_component_nodes": len(largest_nodes),
        "largest_component_pct": 100.0 * len(largest_nodes) / max(1, G.number_of_nodes()),
        "top_component_sizes": top5,
        "elapsed_seconds": elapsed,
    }
    return G_lcc, info


def max_degree_node(G: nx.Graph) -> int:
    node, _deg = max(G.degree(), key=lambda x: x[1])
    return int(node)


def reservoir_sample_nodes(G: nx.Graph, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    sample: List[int] = []
    for i, n in enumerate(G.nodes(), start=1):
        if i <= k:
            sample.append(int(n))
        else:
            j = int(rng.integers(1, i + 1))
            if j <= k:
                sample[j - 1] = int(n)
    return sample


def build_viz_graph(G: nx.Graph, max_nodes: int, seed: int, candidate_pool: int = 100) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return nx.Graph()

    pool = int(min(candidate_pool, G.number_of_nodes()))
    candidates = reservoir_sample_nodes(G, pool, seed)
    if not candidates:
        start_node = max_degree_node(G)
    else:
        start_node = max(candidates, key=lambda n: G.degree(n))

    sample_nodes = bfs_sample_nodes(G, int(start_node), min(max_nodes, G.number_of_nodes()))
    return G.subgraph(sample_nodes).copy()


def sample_shortest_paths(
    G: nx.Graph,
    num_pairs: int,
    seed: int,
    progress_cb=None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    # Numpy array is much smaller than list[int]
    nodes = np.fromiter(G.nodes(), dtype=np.int64, count=G.number_of_nodes())
    if nodes.size == 0:
        return {
            "samples": 0,
            "found": 0,
            "failed": 0,
            "lengths": [],
        }

    lengths: List[int] = []
    failed = 0

    # Stream pairs in chunks to avoid huge intermediate arrays
    chunk = 500
    done = 0

    while done < num_pairs:
        take = min(chunk, num_pairs - done)
        src = rng.choice(nodes, size=take, replace=True)
        tgt = rng.choice(nodes, size=take, replace=True)

        for u, v in zip(src.tolist(), tgt.tolist()):
            if u == v:
                continue
            try:
                lengths.append(nx.shortest_path_length(G, u, v))
            except nx.NetworkXNoPath:
                failed += 1

        done += take
        if progress_cb is not None:
            progress_cb(done / num_pairs)

    if progress_cb is not None:
        progress_cb(1.0)

    if lengths:
        arr = np.asarray(lengths, dtype=np.int32)
        stats = {
            "samples": num_pairs,
            "found": int(arr.size),
            "failed": int(failed),
            "average": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }
    else:
        stats = {
            "samples": num_pairs,
            "found": 0,
            "failed": int(failed),
            "average": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {"stats": stats, "lengths": lengths}


def compute_basic_metrics(G: nx.Graph, clustering_sample: int, seed: int) -> Dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Degrees as numpy array to reduce overhead
    degrees = np.fromiter((d for _, d in G.degree()), dtype=np.int32, count=n)
    avg_degree = float(degrees.mean()) if n else 0.0
    density = float(nx.density(G)) if n else 0.0

    rng = np.random.default_rng(seed)
    if n > 0:
        sample_size = int(min(clustering_sample, n))
        # Sample nodes without building a full python list
        nodes = np.fromiter(G.nodes(), dtype=np.int64, count=n)
        sample_nodes = rng.choice(nodes, size=sample_size, replace=False).tolist()
        clustering = nx.clustering(G, nodes=sample_nodes)
        avg_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0
    else:
        avg_clustering = 0.0

    return {
        "nodes": int(n),
        "edges": int(m),
        "average_degree": avg_degree,
        "density": density,
        "average_clustering_sampled": avg_clustering,
        "degree_summary": {
            "min": int(degrees.min()) if n else 0,
            "max": int(degrees.max()) if n else 0,
            "median": float(np.median(degrees)) if n else 0.0,
        },
        "degrees_for_plot": degrees,
    }


def bfs_sample_nodes(G: nx.Graph, start_node: int, max_nodes: int) -> List[int]:
    visited = set([start_node])
    q = deque([start_node])

    while q and len(visited) < max_nodes:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
                if len(visited) >= max_nodes:
                    break

    return list(visited)


def community_detection_louvain(G: nx.Graph, sample_size: int, seed: int) -> Dict[str, Any]:
    import community as community_louvain

    if G.number_of_nodes() == 0:
        return {"num_communities": 0, "modularity": None, "top_communities": []}

    start_node = max_degree_node(G)

    sample_nodes = bfs_sample_nodes(G, start_node, min(sample_size, G.number_of_nodes()))
    G_sample = G.subgraph(sample_nodes).copy()

    partition = community_louvain.best_partition(G_sample, random_state=seed)
    modularity = community_louvain.modularity(partition, G_sample)

    sizes = Counter(partition.values())
    top = [(int(cid), int(sz)) for cid, sz in sizes.most_common(10)]

    return {
        "sample_nodes": int(G_sample.number_of_nodes()),
        "sample_edges": int(G_sample.number_of_edges()),
        "num_communities": int(len(sizes)),
        "modularity": float(modularity),
        "top_communities": top,
    }


def centrality_analysis(G: nx.Graph, sample_size: int, betweenness_k: int, seed: int) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"degree": [], "betweenness": []}

    start_node = max_degree_node(G)

    sample_nodes = bfs_sample_nodes(G, start_node, min(sample_size, G.number_of_nodes()))
    Gc = G.subgraph(sample_nodes).copy()

    degree_centrality = nx.degree_centrality(Gc)
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    bw_k = min(betweenness_k, Gc.number_of_nodes())
    betweenness = nx.betweenness_centrality(Gc, k=bw_k, seed=seed)
    top_bw = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "sample_nodes": int(Gc.number_of_nodes()),
        "sample_edges": int(Gc.number_of_edges()),
        "degree": [(int(n), float(v), int(Gc.degree(n))) for n, v in top_degree],
        "betweenness": [(int(n), float(v)) for n, v in top_bw],
    }


def plot_degree_hist(degrees: np.ndarray, color: str, title: str, cap_percentile: float = 99.0) -> go.Figure:
    if degrees.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Plot only a sample for responsiveness
    sample_n = int(min(200_000, degrees.size))
    rng = np.random.default_rng(42)
    idx = rng.choice(degrees.size, size=sample_n, replace=False)
    deg_s = degrees[idx]

    cap_p = float(cap_percentile)
    if cap_p < 1.0:
        cap_p = 1.0
    if cap_p > 100.0:
        cap_p = 100.0
    cap_val = int(np.ceil(float(np.percentile(deg_s, cap_p))))
    cap_val = max(1, cap_val)
    deg_plot = deg_s[deg_s <= cap_val]

    nbins = int(min(120, max(20, cap_val)))

    fig = go.Figure(data=[go.Histogram(x=deg_plot, nbinsx=nbins, marker_color=color, opacity=0.8)])
    fig.update_layout(title=title, xaxis_title="Degree", yaxis_title="Count")
    fig.update_xaxes(range=[0, cap_val])
    return fig


def plot_network_graph(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    pos: Dict[int, Tuple[float, float]],
    degrees: Dict[int, int],
    communities: Optional[Dict[int, int]] = None,
    path_nodes: Optional[List[int]] = None,
) -> go.Figure:
    # edges
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.3, color="#888"),
        hoverinfo="none",
    )

    # nodes
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_deg = [degrees.get(n, 0) for n in nodes]

    node_size = [min(5.0 + (d / 2.0), 20.0) for d in node_deg]

    if communities is not None:
        comm_raw = [communities.get(n, -1) for n in nodes]
        uniq = sorted({int(c) for c in comm_raw if c is not None and int(c) >= 0})
        if uniq:
            perm = np.random.default_rng(42).permutation(len(uniq))
            remap = {cid: int(perm[i]) for i, cid in enumerate(uniq)}
            node_color = [remap.get(int(c), -1) if c is not None else -1 for c in comm_raw]
            cmin = 0
            cmax = max(1, len(uniq) - 1)
        else:
            node_color = comm_raw
            cmin = None
            cmax = None

        hover = [
            f"Node: {n}<br>Degree: {degrees.get(n, 0)}<br>Community: {communities.get(n, 'N/A')}"
            for n in nodes
        ]
        colorbar_title = "Community"
        colorscale = "Turbo"
    else:
        node_color = node_deg
        hover = [f"Node: {n}<br>Degree: {degrees.get(n, 0)}" for n in nodes]
        colorbar_title = "Degree"
        colorscale = "Viridis"
        cmin = None
        cmax = None

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=hover,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=node_color,
            size=node_size,
            colorbar=dict(thickness=15, title=colorbar_title, xanchor="left"),
            cmin=cmin,
            cmax=cmax,
            line=dict(width=0.9, color="rgba(0,0,0,0.55)"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # highlight path
    if path_nodes and len(path_nodes) >= 2:
        path_edges = list(zip(path_nodes, path_nodes[1:]))
        px: List[float] = []
        py: List[float] = []
        for u, v in path_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            px.extend([x0, x1, None])
            py.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(width=3, color="red"),
                hoverinfo="none",
                name="Shortest path",
            )
        )

        # mark endpoints
        fig.add_trace(
            go.Scatter(
                x=[pos[path_nodes[0]][0], pos[path_nodes[-1]][0]],
                y=[pos[path_nodes[0]][1], pos[path_nodes[-1]][1]],
                mode="markers",
                marker=dict(size=12, color=["limegreen", "red"], line=dict(width=1, color="black")),
                hovertext=[f"source: {path_nodes[0]}", f"target: {path_nodes[-1]}"] ,
                hoverinfo="text",
                name="Endpoints",
            )
        )

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        title=dict(text=f"soc-Pokec Network Sample ({len(nodes)} nodes)", font=dict(size=16)),
    )
    return fig


def plot_path_length_hist(lengths: List[int], color: str, title: str) -> go.Figure:
    fig = go.Figure()
    if lengths:
        fig.add_trace(go.Histogram(x=lengths, nbinsx=20, marker_color=color, opacity=0.8))
    fig.update_layout(title=title, xaxis_title="Shortest path length", yaxis_title="Count")
    return fig


def plot_top_degree_centrality(centrality: Dict[str, Any], color: str, title: str) -> go.Figure:
    items = centrality.get("degree", []) or []
    nodes = [str(t[0]) for t in items]
    vals = [float(t[1]) for t in items]
    fig = go.Figure(data=[go.Bar(x=vals, y=nodes, orientation="h", marker_color=color)])
    fig.update_layout(title=title, xaxis_title="Degree centrality", yaxis_title="Node")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_top_betweenness(centrality: Dict[str, Any], color: str, title: str) -> go.Figure:
    items = centrality.get("betweenness", []) or []
    nodes = [str(t[0]) for t in items]
    vals = [float(t[1]) for t in items]
    fig = go.Figure(data=[go.Bar(x=vals, y=nodes, orientation="h", marker_color=color)])
    fig.update_layout(title=title, xaxis_title="Betweenness centrality", yaxis_title="Node")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_top_communities(comm: Dict[str, Any], title: str) -> go.Figure:
    items = comm.get("top_communities", []) or []
    cids = [str(t[0]) for t in items]
    sizes = [int(t[1]) for t in items]
    fig = go.Figure(data=[go.Bar(x=cids, y=sizes, marker_color="#6c757d")])
    fig.update_layout(title=title, xaxis_title="Community", yaxis_title="Size (nodes)")
    return fig


def plot_top_component_sizes(lcc: Dict[str, Any], title: str) -> go.Figure:
    sizes = lcc.get("top_component_sizes", []) or []
    x = [f"#{i + 1}" for i in range(len(sizes))]
    fig = go.Figure(data=[go.Bar(x=x, y=[int(s) for s in sizes], marker_color="#6c757d")])
    fig.update_layout(title=title, xaxis_title="Component rank", yaxis_title="Size (nodes)")
    return fig


def build_graph_from_viz(viz: Dict[str, Any]) -> nx.Graph:
    Gs = nx.Graph()
    nodes = viz.get("nodes", []) or []
    edges = viz.get("edges", []) or []
    Gs.add_nodes_from([int(n) for n in nodes])
    Gs.add_edges_from([(int(u), int(v)) for u, v in edges])
    return Gs


def sample_negative_edges(G: nx.Graph, n: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []

    neg: List[Tuple[int, int]] = []
    tries = 0
    max_tries = max(10_000, 50 * n)

    while len(neg) < n and tries < max_tries:
        tries += 1
        u = int(rng.choice(nodes))
        v = int(rng.choice(nodes))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if G.has_edge(a, b):
            continue
        neg.append((a, b))

    return neg


def compute_link_features(G: nx.Graph, pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[str]]:
    if not pairs:
        return np.zeros((0, 0), dtype=float), []

    deg = dict(G.degree())
    cn_map = {(int(u), int(v)): int(sum(1 for _ in nx.common_neighbors(G, int(u), int(v)))) for u, v in pairs}
    jac_map = {(int(u), int(v)): float(p) for u, v, p in nx.jaccard_coefficient(G, pairs)}
    aa_map = {(int(u), int(v)): float(p) for u, v, p in nx.adamic_adar_index(G, pairs)}
    pa_map = {(int(u), int(v)): float(p) for u, v, p in nx.preferential_attachment(G, pairs)}

    X: List[List[float]] = []
    for u, v in pairs:
        u = int(u)
        v = int(v)
        du = float(deg.get(u, 0))
        dv = float(deg.get(v, 0))
        X.append(
            [
                float(cn_map.get((u, v), 0.0)),
                float(jac_map.get((u, v), 0.0)),
                float(aa_map.get((u, v), 0.0)),
                float(pa_map.get((u, v), 0.0)),
                du,
                dv,
                du + dv,
                du * dv,
            ]
        )

    feats = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "deg_u",
        "deg_v",
        "deg_sum",
        "deg_product",
    ]
    return np.asarray(X, dtype=float), feats


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, title: str) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, title: str) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
    return fig


def write_csv_table(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for r in rows:
        w.writerow(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def save_results(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_summary_rows(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    def fmt_int(v: Any) -> Any:
        if v is None:
            return None
        try:
            return f"{int(v):,}"
        except Exception:
            return v

    def fmt_float(v: Any, decimals: int) -> Any:
        if v is None:
            return None
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return v

    meta = analysis["graph"].get("meta", {})
    lcc = analysis["graph"]["lcc"]
    comm_info = analysis.get("community_detection", {})
    dos = analysis.get("degrees_of_separation", {})

    rows: List[Dict[str, Any]] = []
    if analysis["graph"]["mode"] == "mutual":
        rows.append({"Metric": "Total Nodes (Directed loaded)", "Value": fmt_int(meta.get("directed_nodes"))})
        rows.append({"Metric": "Total Edges (Directed loaded)", "Value": fmt_int(meta.get("directed_edges"))})
        if meta.get("directed_edges") is not None and meta.get("mutual_edges") is not None:
            rows.append(
                {
                    "Metric": "Total Edges (Undirected derived)",
                    "Value": fmt_int(int(meta.get("directed_edges")) - int(meta.get("mutual_edges"))),
                }
            )
        rows.append({"Metric": "Total Edges (Mutual)", "Value": fmt_int(meta.get("mutual_edges"))})
    else:
        rows.append({"Metric": "Total Nodes (Undirected loaded)", "Value": fmt_int(meta.get("loaded_nodes"))})
        rows.append({"Metric": "Total Edges (Undirected loaded)", "Value": fmt_int(meta.get("loaded_edges"))})

    rows.extend(
        [
            {"Metric": "Largest Component Size (nodes)", "Value": fmt_int(lcc.get("nodes"))},
            {"Metric": "Largest Component Edges", "Value": fmt_int(lcc.get("edges"))},
            {"Metric": "Average Degree", "Value": fmt_float(analysis["network_metrics"].get("average_degree"), 2)},
            {"Metric": "Network Density", "Value": fmt_float(analysis["network_metrics"].get("density"), 6)},
            {"Metric": "Average Clustering Coeff. (sampled)", "Value": fmt_float(analysis["network_metrics"].get("average_clustering_sampled"), 4)},
            {"Metric": "Reciprocity (Directed)", "Value": fmt_float(analysis["graph"].get("reciprocity"), 4)},
            {"Metric": "Avg. Degrees of Separation", "Value": fmt_float(dos.get("average"), 2)},
            {"Metric": "Number of Communities (sampled)", "Value": fmt_int(comm_info.get("num_communities"))},
            {"Metric": "Modularity (sampled)", "Value": fmt_float(comm_info.get("modularity"), 4)},
        ]
    )
    return rows


def save_summary_png(path: Path, rows: List[Dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    metrics = [str(r.get("Metric", "")) for r in rows]
    values = ["" if r.get("Value") is None else str(r.get("Value")) for r in rows]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * (len(rows) + 1))))
    ax.axis("off")

    data = list(zip(metrics, values))
    table = ax.table(cellText=data, colLabels=["Metric", "Value"], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def clear_memory() -> None:
    gc.collect()


def main() -> None:
    st.set_page_config(page_title="Social Network Analysis (Pokec)", layout="wide")

    st.title("Social Network Analysis (soc-Pokec)")

    with st.sidebar:
        st.header("Configuration")

        graph_mode_label = st.selectbox(
            "Graph mode",
            options=["Mutual friendships only", "All connections"],
            index=0,
        )
        graph_mode = "mutual" if graph_mode_label.startswith("Mutual") else "all"

        data_dir = Path(st.text_input("Data directory", value=str(DATA_DIR_DEFAULT)))
        outputs_dir = Path(st.text_input("Outputs directory", value=str(OUTPUTS_DIR_DEFAULT)))

        edges_file = data_dir / "soc-pokec-relationships.txt"
        st.caption(f"Edges file: `{edges_file}`")

        num_pairs = st.number_input("Shortest-path samples (pairs)", min_value=100, max_value=100_000, value=10_000, step=100)
        clustering_sample = st.number_input("Clustering sample (nodes)", min_value=100, max_value=200_000, value=10_000, step=100)
        community_sample = st.number_input("Community detection sample (nodes)", min_value=1_000, max_value=200_000, value=50_000, step=1_000)
        centrality_sample = st.number_input("Centrality sample (nodes)", min_value=1_000, max_value=200_000, value=10_000, step=1_000)
        betweenness_k = st.number_input("Betweenness k (sampled)", min_value=50, max_value=5_000, value=500, step=50)
        random_seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

        st.divider()
        st.subheader("Network Explorer")
        viz_nodes = st.number_input("Visualizer subgraph nodes", min_value=200, max_value=10_000, value=1000, step=100)
        viz_k = st.number_input("Spring layout k", min_value=0.01, max_value=5.0, value=0.5, step=0.05)
        viz_iterations = st.number_input("Spring layout iterations", min_value=10, max_value=500, value=50, step=10)

        st.divider()
        degree_hist_percentile = st.slider(
            "Degree histogram x-axis cap (percentile)",
            min_value=90.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
        )

        cfg = AnalysisConfig(
            graph_mode=graph_mode,
            num_pairs=int(num_pairs),
            random_seed=int(random_seed),
            clustering_sample=int(clustering_sample),
            community_sample=int(community_sample),
            centrality_sample=int(centrality_sample),
            betweenness_k=int(betweenness_k),
            degree_hist_percentile=float(degree_hist_percentile),
        )

        run_btn = st.button("Run analysis", type="primary")
        save_btn = st.button("Save results to outputs/")
        clear_btn = st.button("Clear session results")
        clear_cache_btn = st.button("Clear Streamlit cache")

        st.divider()
        st.metric("RAM used (GB)", f"{get_memory_usage_gb():.2f}")

    if clear_cache_btn:
        st.cache_data.clear()
        st.cache_resource.clear()
        clear_memory()
        st.success("Cleared Streamlit cache.")

    if clear_btn:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_run_id", None)
        st.session_state.pop("analysis_run_at", None)
        clear_memory()
        st.success("Cleared session results.")

    tabs = st.tabs(["Overview", "Run / Results", "Network Explorer", "Outputs", "ML (Link Prediction)"])

    with tabs[0]:
        st.subheader("Overview")
        st.write(
            "This app runs a sampled analysis of the **soc-Pokec** social network. "
            "To avoid memory issues, it loads and analyzes **only the selected graph mode** at a time."
        )

        legacy = read_results_json(outputs_dir / "analysis_results.json")
        if legacy is not None:
            st.caption("Found existing `outputs/analysis_results.json` (previous run).")
            st.json(legacy)
        else:
            st.caption("No precomputed `outputs/analysis_results.json` found.")

        st.write("Existing output artifacts:")
        if outputs_dir.exists():
            for p in sorted(outputs_dir.glob("*")):
                if p.is_file() and p.name != ".gitkeep":
                    st.write(f"- `{p.name}`")
        else:
            st.write(f"`{outputs_dir}` does not exist yet.")

    with tabs[1]:
        st.subheader("Run / Results")

        if run_btn:
            try:
                ensure_file_exists(edges_file, "edge list")
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            status = st.status("Running analysis...", expanded=True)
            status.write(f"Config: `{cfg}`")
            status.write(f"Initial RAM: {get_memory_usage_gb():.2f} GB")

            t0 = time.time()

            progress = st.progress(0.0)

            reciprocity_val: Optional[float] = None
            graph_meta: Dict[str, Any] = {}
            if cfg.graph_mode == "all":
                status.write("Loading FULL undirected graph (all connections)...")
                G = load_graph_all_connections(edges_file)
                graph_meta = {
                    "loaded_graph_type": "undirected",
                    "loaded_nodes": int(G.number_of_nodes()),
                    "loaded_edges": int(G.number_of_edges()),
                }
                progress.progress(0.15)
            else:
                status.write("Loading directed graph (for reciprocity + mutual edge extraction)...")
                Gd = load_graph_directed(edges_file)
                reciprocity_val = float(nx.reciprocity(Gd))
                status.write(f"Reciprocity: {reciprocity_val:.4f}")

                graph_meta = {
                    "loaded_graph_type": "directed",
                    "directed_nodes": int(Gd.number_of_nodes()),
                    "directed_edges": int(Gd.number_of_edges()),
                }

                status.write("Building mutual (reciprocal) undirected graph...")

                def mutual_progress(x: float) -> None:
                    progress.progress(0.15 + 0.35 * x)

                G = build_mutual_graph(Gd, progress_cb=mutual_progress)
                graph_meta["mutual_nodes"] = int(G.number_of_nodes())
                graph_meta["mutual_edges"] = int(G.number_of_edges())
                del Gd
                clear_memory()

            status.write(f"Graph loaded/built. RAM: {get_memory_usage_gb():.2f} GB")

            status.write("Extracting largest connected component (LCC)...")
            G_lcc, cc_info = lcc_subgraph(G)
            del G
            clear_memory()
            progress.progress(0.55)

            status.write(f"LCC nodes: {G_lcc.number_of_nodes():,}, edges: {G_lcc.number_of_edges():,}")
            status.write(f"Connected components: {cc_info['num_components']:,}")
            status.write(f"RAM after LCC: {get_memory_usage_gb():.2f} GB")

            status.write("Sampling shortest paths...")

            def sp_progress(x: float) -> None:
                progress.progress(0.55 + 0.20 * x)

            sp = sample_shortest_paths(G_lcc, cfg.num_pairs, cfg.random_seed, progress_cb=sp_progress)

            status.write("Computing network metrics...")
            metrics = compute_basic_metrics(G_lcc, cfg.clustering_sample, cfg.random_seed)
            progress.progress(0.80)

            status.write("Community detection (Louvain) on BFS sample...")
            comm = community_detection_louvain(G_lcc, cfg.community_sample, cfg.random_seed)
            progress.progress(0.90)

            status.write("Centrality analysis on BFS sample...")
            cent = centrality_analysis(G_lcc, cfg.centrality_sample, cfg.betweenness_k, cfg.random_seed)
            progress.progress(0.97)

            status.write("Building visualization subgraph...")
            G_viz = build_viz_graph(G_lcc, int(viz_nodes), seed=cfg.random_seed, candidate_pool=100)
            viz_nodes_list = [int(n) for n in G_viz.nodes()]
            viz_edges_list = [(int(u), int(v)) for u, v in G_viz.edges()]
            viz_degrees = {int(n): int(d) for n, d in G_viz.degree()}

            status.write("Detecting communities for visualization subgraph...")
            import community as community_louvain

            viz_partition = community_louvain.best_partition(G_viz)
            viz_communities = {int(n): int(cid) for n, cid in viz_partition.items()}

            status.write("Computing graph layout...")
            pos_raw = nx.spring_layout(G_viz, k=float(viz_k), iterations=int(viz_iterations), seed=42)
            viz_pos = {int(n): (float(xy[0]), float(xy[1])) for n, xy in pos_raw.items()}
            del G_viz
            clear_memory()

            # Keep only compact results in session state
            elapsed = time.time() - t0
            analysis = {
                "config": asdict(cfg),
                "graph": {
                    "mode": cfg.graph_mode,
                    "reciprocity": reciprocity_val,
                    "meta": graph_meta,
                    "lcc": {
                        "nodes": int(G_lcc.number_of_nodes()),
                        "edges": int(G_lcc.number_of_edges()),
                        **cc_info,
                    },
                },
                "degrees_of_separation": sp["stats"],
                "network_metrics": {
                    "nodes": metrics["nodes"],
                    "edges": metrics["edges"],
                    "average_degree": metrics["average_degree"],
                    "density": metrics["density"],
                    "average_clustering_sampled": metrics["average_clustering_sampled"],
                    **metrics["degree_summary"],
                },
                "community_detection": comm,
                "centrality": cent,
                "visualization": {
                    "nodes": int(len(viz_nodes_list)),
                    "edges": int(len(viz_edges_list)),
                    "layout": "spring",
                    "k": float(viz_k),
                    "iterations": int(viz_iterations),
                },
                "runtime_seconds": float(elapsed),
                "ram_gb_end": float(get_memory_usage_gb()),
            }

            st.session_state["analysis"] = {
                "analysis": analysis,
                "degrees": metrics["degrees_for_plot"],
                "path_lengths": sp.get("lengths", []),
                "viz": {
                    "nodes": viz_nodes_list,
                    "edges": viz_edges_list,
                    "pos": viz_pos,
                    "degrees": viz_degrees,
                    "communities": viz_communities,
                },
            }

            st.session_state["analysis_run_id"] = int(st.session_state.get("analysis_run_id", 0)) + 1
            st.session_state["analysis_run_at"] = float(time.time())

            progress.progress(1.0)
            status.update(label="Analysis completed", state="complete", expanded=False)

            # Free the big LCC graph now that we have degrees + path lengths
            del G_lcc
            clear_memory()

        sess = st.session_state.get("analysis")
        if sess is None:
            st.info("Click **Run analysis** in the sidebar to compute results.")
        else:
            analysis = sess["analysis"]
            degrees = sess["degrees"]
            path_lengths = sess["path_lengths"]

            run_id = int(st.session_state.get("analysis_run_id", 0))
            cap_p = float(analysis.get("config", {}).get("degree_hist_percentile", 99.0))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Graph mode", analysis["graph"]["mode"])
            c2.metric("LCC nodes", f"{analysis['graph']['lcc']['nodes']:,}")
            c3.metric("LCC edges", f"{analysis['graph']['lcc']['edges']:,}")
            c4.metric("Runtime (s)", f"{analysis['runtime_seconds']:.1f}")

            st.subheader("Key metrics")
            st.json(analysis)

            st.subheader("Plots")
            if analysis["graph"]["mode"] == "mutual":
                color = "steelblue"
            else:
                color = "coral"

            st.plotly_chart(
                plot_degree_hist(degrees, color=color, title="Degree distribution (sampled for plotting)", cap_percentile=cap_p),
                width='stretch',
                key=f"run_results_degree_hist_{run_id}",
            )
            st.plotly_chart(
                plot_path_length_hist(path_lengths, color=color, title="Shortest path length distribution (sampled pairs)"),
                width='stretch',
                key=f"run_results_path_hist_{run_id}",
            )

            st.caption(f"RAM used now: {get_memory_usage_gb():.2f} GB")

        if save_btn:
            sess = st.session_state.get("analysis")
            if sess is None:
                st.warning("Nothing to save yet. Run the analysis first.")
            else:
                payload = sess["analysis"]
                out_path = outputs_dir / f"analysis_results_{payload['graph']['mode']}.json"
                save_results(out_path, payload)
                mode = payload["graph"]["mode"]

                outputs_dir.mkdir(parents=True, exist_ok=True)

                rows = build_summary_rows(payload)
                summary_png = outputs_dir / f"network_summary_{mode}.png"
                summary_ok = save_summary_png(summary_png, rows)

                if mode == "mutual":
                    color = "steelblue"
                else:
                    color = "coral"

                cap_p = float(payload.get("config", {}).get("degree_hist_percentile", 99.0))
                fig_deg = plot_degree_hist(
                    sess["degrees"],
                    color=color,
                    title="Degree distribution (sampled for plotting)",
                    cap_percentile=cap_p,
                )
                (outputs_dir / f"degree_distribution_{mode}.html").write_bytes(
                    fig_deg.to_html(include_plotlyjs=True).encode("utf-8")
                )

                fig_pl = plot_path_length_hist(sess["path_lengths"], color=color, title="Shortest path length distribution (sampled pairs)")
                (outputs_dir / f"path_length_distribution_{mode}.html").write_bytes(
                    fig_pl.to_html(include_plotlyjs=True).encode("utf-8")
                )

                if sess.get("viz") is not None:
                    viz = sess["viz"]
                    fig_net = plot_network_graph(
                        nodes=viz["nodes"],
                        edges=viz["edges"],
                        pos=viz["pos"],
                        degrees=viz["degrees"],
                        communities=viz.get("communities"),
                        path_nodes=None,
                    )
                    (outputs_dir / f"network_visualization_{mode}.html").write_bytes(
                        fig_net.to_html(include_plotlyjs=True).encode("utf-8")
                    )

                cent = payload.get("centrality", {})
                comm = payload.get("community_detection", {})
                lcc = payload.get("graph", {}).get("lcc", {})

                fig_cent_deg = plot_top_degree_centrality(cent, color=color, title="Top nodes by degree centrality (sampled)")
                (outputs_dir / f"top_degree_centrality_{mode}.html").write_bytes(
                    fig_cent_deg.to_html(include_plotlyjs=True).encode("utf-8")
                )

                fig_cent_bw = plot_top_betweenness(cent, color=color, title="Top nodes by betweenness centrality (sampled)")
                (outputs_dir / f"top_betweenness_{mode}.html").write_bytes(
                    fig_cent_bw.to_html(include_plotlyjs=True).encode("utf-8")
                )

                fig_comm = plot_top_communities(comm, title="Top community sizes (sampled)")
                (outputs_dir / f"community_sizes_{mode}.html").write_bytes(
                    fig_comm.to_html(include_plotlyjs=True).encode("utf-8")
                )

                fig_cc = plot_top_component_sizes(lcc, title="Largest connected components (top sizes)")
                (outputs_dir / f"component_sizes_{mode}.html").write_bytes(
                    fig_cc.to_html(include_plotlyjs=True).encode("utf-8")
                )

                write_csv_table(
                    outputs_dir / f"network_summary_{mode}.csv",
                    headers=["Metric", "Value"],
                    rows=[[r.get("Metric"), r.get("Value")] for r in rows],
                )

                write_csv_table(
                    outputs_dir / f"top_degree_centrality_{mode}.csv",
                    headers=["node", "degree_centrality", "degree_in_sample"],
                    rows=[[t[0], t[1], t[2]] for t in (cent.get("degree", []) or [])],
                )
                write_csv_table(
                    outputs_dir / f"top_betweenness_{mode}.csv",
                    headers=["node", "betweenness"],
                    rows=[[t[0], t[1]] for t in (cent.get("betweenness", []) or [])],
                )
                write_csv_table(
                    outputs_dir / f"top_communities_{mode}.csv",
                    headers=["community_id", "size"],
                    rows=[[t[0], t[1]] for t in (comm.get("top_communities", []) or [])],
                )
                write_csv_table(
                    outputs_dir / f"component_sizes_{mode}.csv",
                    headers=["rank", "size"],
                    rows=[[i + 1, int(s)] for i, s in enumerate(lcc.get("top_component_sizes", []) or [])],
                )

                msg = [f"Saved: {out_path}"]
                msg.append(f"Saved: {outputs_dir / f'degree_distribution_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'path_length_distribution_{mode}.html'}")
                if sess.get("viz") is not None:
                    msg.append(f"Saved: {outputs_dir / f'network_visualization_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'top_degree_centrality_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'top_betweenness_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'community_sizes_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'component_sizes_{mode}.html'}")
                msg.append(f"Saved: {outputs_dir / f'network_summary_{mode}.csv'}")
                msg.append(f"Saved: {outputs_dir / f'top_degree_centrality_{mode}.csv'}")
                msg.append(f"Saved: {outputs_dir / f'top_betweenness_{mode}.csv'}")
                msg.append(f"Saved: {outputs_dir / f'top_communities_{mode}.csv'}")
                msg.append(f"Saved: {outputs_dir / f'component_sizes_{mode}.csv'}")
                if summary_ok:
                    msg.append(f"Saved: {summary_png}")
                else:
                    msg.append("Summary PNG not saved (matplotlib missing).")

                ml = st.session_state.get("ml")
                if ml is not None and ml.get("mode") == mode:
                    (outputs_dir / f"link_prediction_metrics_{mode}.json").write_text(
                        json.dumps(ml.get("metrics", {}), indent=2),
                        encoding="utf-8",
                    )
                    if ml.get("roc_fig") is not None:
                        (outputs_dir / f"link_prediction_roc_{mode}.html").write_bytes(
                            ml["roc_fig"].to_html(include_plotlyjs=True).encode("utf-8")
                        )
                    if ml.get("pr_fig") is not None:
                        (outputs_dir / f"link_prediction_pr_{mode}.html").write_bytes(
                            ml["pr_fig"].to_html(include_plotlyjs=True).encode("utf-8")
                        )
                    preds = ml.get("top_predictions", []) or []
                    write_csv_table(
                        outputs_dir / f"link_prediction_top_edges_{mode}.csv",
                        headers=["u", "v", "probability"],
                        rows=[[p[0], p[1], p[2]] for p in preds],
                    )

                    msg.append(f"Saved: {outputs_dir / f'link_prediction_metrics_{mode}.json'}")
                    msg.append(f"Saved: {outputs_dir / f'link_prediction_top_edges_{mode}.csv'}")
                    if (outputs_dir / f"link_prediction_roc_{mode}.html").exists():
                        msg.append(f"Saved: {outputs_dir / f'link_prediction_roc_{mode}.html'}")
                    if (outputs_dir / f"link_prediction_pr_{mode}.html").exists():
                        msg.append(f"Saved: {outputs_dir / f'link_prediction_pr_{mode}.html'}")

                st.success("\n".join(msg))

    with tabs[2]:
        st.subheader("Network Explorer")

        sess = st.session_state.get("analysis")
        if sess is None or "viz" not in sess:
            st.info("Run the analysis first to generate a visualization subgraph.")
        else:
            viz = sess["viz"]
            nodes = viz["nodes"]
            edges = viz["edges"]
            pos = viz["pos"]
            deg = viz["degrees"]
            comm = viz.get("communities")

            st.caption(f"Subgraph: {len(nodes):,} nodes, {len(edges):,} edges")

            with st.form("path_finder"):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    src = st.number_input("Source node", min_value=0, value=int(nodes[0]) if nodes else 0, step=1)
                with c2:
                    tgt = st.number_input("Target node", min_value=0, value=int(nodes[-1]) if nodes else 0, step=1)
                with c3:
                    submitted = st.form_submit_button("Find shortest path")

            path_nodes: Optional[List[int]] = None
            path_msg = None
            if submitted:
                Gs = nx.Graph()
                Gs.add_nodes_from(nodes)
                Gs.add_edges_from(edges)
                if src not in Gs:
                    path_msg = "Source node is not in the visualized subgraph. Try another node." 
                elif tgt not in Gs:
                    path_msg = "Target node is not in the visualized subgraph. Try another node." 
                else:
                    try:
                        path_nodes = [int(n) for n in nx.shortest_path(Gs, int(src), int(tgt))]
                        path_msg = f"Shortest path length: {len(path_nodes) - 1} | Path: {' -> '.join(map(str, path_nodes[:20]))}{' ...' if len(path_nodes) > 20 else ''}"
                    except nx.NetworkXNoPath:
                        path_msg = "No path found between these nodes within the subgraph." 

            if path_msg:
                st.write(path_msg)

            fig = plot_network_graph(nodes=nodes, edges=edges, pos=pos, degrees=deg, communities=comm, path_nodes=path_nodes)
            st.plotly_chart(fig, width='stretch', key="network_explorer_graph")

            html_bytes = fig.to_html(include_plotlyjs=True).encode("utf-8")
            st.download_button("Download visualization as HTML", data=html_bytes, file_name="network_explorer.html")

            if st.button("Save visualization to outputs/network_visualization.html"):
                out_html = outputs_dir / "network_visualization.html"
                out_html.parent.mkdir(parents=True, exist_ok=True)
                out_html.write_bytes(html_bytes)
                st.success(f"Saved: {out_html}")

    with tabs[3]:
        st.subheader("Outputs")

        sess = st.session_state.get("analysis")
        if sess is not None:
            analysis = sess["analysis"]
            degrees = sess["degrees"]
            path_lengths = sess["path_lengths"]
            viz = sess.get("viz")

            run_id = int(st.session_state.get("analysis_run_id", 0))
            cap_p = float(analysis.get("config", {}).get("degree_hist_percentile", 99.0))

            if analysis["graph"]["mode"] == "mutual":
                color = "steelblue"
            else:
                color = "coral"

            st.subheader("Latest run (from Streamlit session)")
            rows = build_summary_rows(analysis)
            st.dataframe(rows, use_container_width=True, hide_index=True)

            summary_png = outputs_dir / f"network_summary_{analysis['graph']['mode']}.png"
            csum1, csum2 = st.columns([1, 1])
            with csum1:
                if st.button("Generate summary PNG in outputs/", key=f"outputs_generate_summary_png_{run_id}"):
                    outputs_dir.mkdir(parents=True, exist_ok=True)
                    ok = save_summary_png(summary_png, rows)
                    if ok:
                        st.success(f"Saved: {summary_png}")
                    else:
                        st.warning("Summary PNG not saved (matplotlib missing).")
            with csum2:
                if summary_png.exists():
                    st.download_button(
                        "Download summary PNG",
                        data=summary_png.read_bytes(),
                        file_name=summary_png.name,
                        key=f"outputs_download_summary_png_{run_id}",
                    )

            fig_deg = plot_degree_hist(degrees, color=color, title="Degree distribution (sampled for plotting)", cap_percentile=cap_p)
            st.plotly_chart(fig_deg, width='stretch', key=f"outputs_degree_hist_{run_id}")
            st.download_button(
                "Download degree distribution (HTML)",
                data=fig_deg.to_html(include_plotlyjs=True).encode("utf-8"),
                file_name=f"degree_distribution_{analysis['graph']['mode']}.html",
                key=f"outputs_degree_hist_dl_{run_id}",
            )

            fig_pl = plot_path_length_hist(path_lengths, color=color, title="Shortest path length distribution (sampled pairs)")
            st.plotly_chart(fig_pl, width='stretch', key=f"outputs_path_hist_{run_id}")
            st.download_button(
                "Download path length distribution (HTML)",
                data=fig_pl.to_html(include_plotlyjs=True).encode("utf-8"),
                file_name=f"path_length_distribution_{analysis['graph']['mode']}.html",
                key=f"outputs_path_hist_dl_{run_id}",
            )

            if viz is not None:
                fig_net = plot_network_graph(
                    nodes=viz["nodes"],
                    edges=viz["edges"],
                    pos=viz["pos"],
                    degrees=viz["degrees"],
                    communities=viz.get("communities"),
                    path_nodes=None,
                )
                st.plotly_chart(fig_net, width='stretch', key=f"outputs_network_graph_{run_id}")
                st.download_button(
                    "Download network visualization (HTML)",
                    data=fig_net.to_html(include_plotlyjs=True).encode("utf-8"),
                    file_name=f"network_visualization_{analysis['graph']['mode']}.html",
                    key=f"outputs_network_graph_dl_{run_id}",
                )
        else:
            st.info("Run the analysis first to generate outputs.")

        st.divider()
        st.subheader("Files on disk (outputs/)")
        if not outputs_dir.exists():
            st.info(f"No outputs directory found at `{outputs_dir}`")
        else:
            with st.expander("Show files in outputs/"):
                preview = st.checkbox("Preview images", value=False, key="outputs_preview_images")
                images = sorted(outputs_dir.glob("*.png"))
                if images:
                    st.caption("Some files may be legacy notebook outputs and won’t update unless re-saved from the app.")
                    if preview:
                        cols = st.columns(2)
                        for idx, img in enumerate(images):
                            with cols[idx % 2]:
                                st.image(str(img), caption=img.name, width='stretch')
                    else:
                        for img in images:
                            st.write(f"- `{img.name}`")
                else:
                    st.caption("No .png outputs found.")

                html = outputs_dir / "network_visualization.html"
                if html.exists():
                    st.caption("`network_visualization.html` exists. Streamlit can’t inline-render it safely by default.")
                    st.download_button("Download network_visualization.html", data=html.read_bytes(), file_name=html.name)

    with tabs[4]:
        st.subheader("ML (Link Prediction)")

        sess = st.session_state.get("analysis")
        if sess is None or sess.get("viz") is None:
            st.info("Run the analysis first to generate a visualization subgraph. The ML module uses that subgraph.")
        elif not SKLEARN_AVAILABLE:
            st.warning("scikit-learn is not available in this environment. Install `scikit-learn` to use this tab.")
        else:
            analysis = sess["analysis"]
            viz = sess["viz"]
            run_id = int(st.session_state.get("analysis_run_id", 0))

            if analysis["graph"]["mode"] == "mutual":
                color = "steelblue"
            else:
                color = "coral"

            Gs = build_graph_from_viz(viz)
            st.caption(f"Training/evaluating on sampled subgraph: {Gs.number_of_nodes():,} nodes, {Gs.number_of_edges():,} edges")

            cml1, cml2, cml3 = st.columns([1, 1, 1])
            with cml1:
                ml_pos_edges = st.number_input(
                    "Positive edges for dataset",
                    min_value=200,
                    max_value=50_000,
                    value=5_000,
                    step=200,
                    key=f"ml_pos_edges_{run_id}",
                )
            with cml2:
                ml_test_size = st.slider(
                    "Test split",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.25,
                    step=0.05,
                    key=f"ml_test_size_{run_id}",
                )
            with cml3:
                ml_topk = st.number_input(
                    "Top predicted missing links",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    key=f"ml_topk_{run_id}",
                )

            ml_btn = st.button("Run link prediction", key=f"ml_run_btn_{run_id}")

            if ml_btn:
                if Gs.number_of_edges() < 10 or Gs.number_of_nodes() < 10:
                    st.warning("Visualization subgraph is too small for ML.")
                else:
                    rng = np.random.default_rng(int(analysis.get("config", {}).get("random_seed", 42)))
                    edges_all = list(Gs.edges())
                    pos_n = int(min(int(ml_pos_edges), len(edges_all)))
                    pos_idx = rng.choice(len(edges_all), size=pos_n, replace=False)
                    pos = [(int(edges_all[i][0]), int(edges_all[i][1])) for i in pos_idx.tolist()]
                    neg = sample_negative_edges(Gs, pos_n, seed=int(analysis.get("config", {}).get("random_seed", 42)) + 1)

                    X_pos, feat_names = compute_link_features(Gs, pos)
                    X_neg, _ = compute_link_features(Gs, neg)
                    X = np.vstack([X_pos, X_neg])
                    y = np.concatenate([np.ones(X_pos.shape[0], dtype=int), np.zeros(X_neg.shape[0], dtype=int)])

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=float(ml_test_size),
                        random_state=int(analysis.get("config", {}).get("random_seed", 42)),
                        stratify=y,
                    )

                    model = Pipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(
                                    max_iter=2000,
                                    class_weight="balanced",
                                    solver="lbfgs",
                                ),
                            ),
                        ]
                    )
                    model.fit(X_train, y_train)
                    y_score = model.predict_proba(X_test)[:, 1]

                    auc = float(roc_auc_score(y_test, y_score))
                    ap = float(average_precision_score(y_test, y_score))

                    roc_fig = plot_roc_curve(y_test, y_score, title=f"Link prediction ROC (AUC={auc:.3f})")
                    pr_fig = plot_pr_curve(y_test, y_score, title=f"Link prediction Precision-Recall (AP={ap:.3f})")

                    # Predict missing links (top-k) on random candidate non-edges
                    cand_n = int(max(2000, 40 * int(ml_topk)))
                    candidates = sample_negative_edges(Gs, cand_n, seed=int(analysis.get("config", {}).get("random_seed", 42)) + 2)
                    X_cand, _ = compute_link_features(Gs, candidates)
                    if X_cand.size > 0:
                        p_cand = model.predict_proba(X_cand)[:, 1]
                        order = np.argsort(-p_cand)[: int(ml_topk)]
                        top_preds = [(int(candidates[i][0]), int(candidates[i][1]), float(p_cand[i])) for i in order.tolist()]
                    else:
                        top_preds = []

                    metrics = {
                        "mode": analysis["graph"]["mode"],
                        "nodes": int(Gs.number_of_nodes()),
                        "edges": int(Gs.number_of_edges()),
                        "dataset_pos": int(X_pos.shape[0]),
                        "dataset_neg": int(X_neg.shape[0]),
                        "test_size": float(ml_test_size),
                        "roc_auc": auc,
                        "average_precision": ap,
                        "features": feat_names,
                    }

                    st.session_state["ml"] = {
                        "mode": analysis["graph"]["mode"],
                        "metrics": metrics,
                        "top_predictions": top_preds,
                        "roc_fig": roc_fig,
                        "pr_fig": pr_fig,
                    }
                    st.success("ML run complete.")

            ml = st.session_state.get("ml")
            if ml is not None and ml.get("mode") == analysis["graph"]["mode"]:
                m = ml.get("metrics", {})
                c1, c2 = st.columns(2)
                c1.metric("ROC AUC", f"{m.get('roc_auc', 0.0):.3f}")
                c2.metric("Avg Precision", f"{m.get('average_precision', 0.0):.3f}")

                st.plotly_chart(ml["roc_fig"], width='stretch', key=f"ml_roc_{run_id}")
                st.plotly_chart(ml["pr_fig"], width='stretch', key=f"ml_pr_{run_id}")

                st.subheader("Top predicted missing links")
                preds = ml.get("top_predictions", []) or []
                if preds:
                    st.dataframe(
                        [{"u": p[0], "v": p[1], "probability": p[2]} for p in preds],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No predictions available.")

                ml_json = json.dumps(ml.get("metrics", {}), indent=2).encode("utf-8")
                st.download_button(
                    "Download ML metrics (JSON)",
                    data=ml_json,
                    file_name=f"link_prediction_metrics_{analysis['graph']['mode']}.json",
                    key=f"ml_download_metrics_{run_id}",
                )



if __name__ == "__main__":
    main()
