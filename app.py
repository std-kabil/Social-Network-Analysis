import gc
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


def connected_components_summary(G: nx.Graph) -> Tuple[int, List[int], List[int]]:
    # Returns (num_components, largest_component_nodes, sizes_top5)
    num = 0
    largest: List[int] = []
    top_sizes: List[int] = []

    for comp in nx.connected_components(G):
        num += 1
        comp_size = len(comp)
        if comp_size > len(largest):
            largest = list(comp)

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

    degrees_dict = dict(G.degree())
    start_node = max(degrees_dict, key=degrees_dict.get)

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

    degrees_dict = dict(G.degree())
    start_node = max(degrees_dict, key=degrees_dict.get)

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


def plot_degree_hist(degrees: np.ndarray, color: str, title: str) -> go.Figure:
    if degrees.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Plot only a sample for responsiveness
    sample_n = int(min(200_000, degrees.size))
    rng = np.random.default_rng(42)
    idx = rng.choice(degrees.size, size=sample_n, replace=False)
    deg_s = degrees[idx]

    fig = go.Figure(data=[go.Histogram(x=deg_s, nbinsx=120, marker_color=color, opacity=0.8)])
    fig.update_layout(title=title, xaxis_title="Degree", yaxis_title="Count")
    return fig


def plot_path_length_hist(lengths: List[int], color: str, title: str) -> go.Figure:
    fig = go.Figure()
    if lengths:
        fig.add_trace(go.Histogram(x=lengths, nbinsx=20, marker_color=color, opacity=0.8))
    fig.update_layout(title=title, xaxis_title="Shortest path length", yaxis_title="Count")
    return fig


def save_results(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

        cfg = AnalysisConfig(
            graph_mode=graph_mode,
            num_pairs=int(num_pairs),
            random_seed=int(random_seed),
            clustering_sample=int(clustering_sample),
            community_sample=int(community_sample),
            centrality_sample=int(centrality_sample),
            betweenness_k=int(betweenness_k),
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
        clear_memory()
        st.success("Cleared session results.")

    tabs = st.tabs(["Overview", "Run / Results", "Outputs"])

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
            if cfg.graph_mode == "all":
                status.write("Loading FULL undirected graph (all connections)...")
                G = load_graph_all_connections(edges_file)
                progress.progress(0.15)
            else:
                status.write("Loading directed graph (for reciprocity + mutual edge extraction)...")
                Gd = load_graph_directed(edges_file)
                reciprocity_val = float(nx.reciprocity(Gd))
                status.write(f"Reciprocity: {reciprocity_val:.4f}")

                status.write("Building mutual (reciprocal) undirected graph...")

                def mutual_progress(x: float) -> None:
                    progress.progress(0.15 + 0.35 * x)

                G = build_mutual_graph(Gd, progress_cb=mutual_progress)
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

            # Keep only compact results in session state
            elapsed = time.time() - t0
            analysis = {
                "config": asdict(cfg),
                "graph": {
                    "mode": cfg.graph_mode,
                    "reciprocity": reciprocity_val,
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
                "runtime_seconds": float(elapsed),
                "ram_gb_end": float(get_memory_usage_gb()),
            }

            st.session_state["analysis"] = {
                "analysis": analysis,
                "degrees": metrics["degrees_for_plot"],
                "path_lengths": sp.get("lengths", []),
            }

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

            st.plotly_chart(plot_degree_hist(degrees, color=color, title="Degree distribution (sampled for plotting)"), use_container_width=True)
            st.plotly_chart(plot_path_length_hist(path_lengths, color=color, title="Shortest path length distribution (sampled pairs)"), use_container_width=True)

            st.caption(f"RAM used now: {get_memory_usage_gb():.2f} GB")

        if save_btn:
            sess = st.session_state.get("analysis")
            if sess is None:
                st.warning("Nothing to save yet. Run the analysis first.")
            else:
                payload = sess["analysis"]
                out_path = outputs_dir / f"analysis_results_{payload['graph']['mode']}.json"
                save_results(out_path, payload)
                st.success(f"Saved: {out_path}")

    with tabs[2]:
        st.subheader("Outputs")
        if not outputs_dir.exists():
            st.info(f"No outputs directory found at `{outputs_dir}`")
        else:
            images = sorted(outputs_dir.glob("*.png"))
            if images:
                cols = st.columns(2)
                for idx, img in enumerate(images):
                    with cols[idx % 2]:
                        st.image(str(img), caption=img.name, use_container_width=True)
            else:
                st.caption("No .png outputs found.")

            html = outputs_dir / "network_visualization.html"
            if html.exists():
                st.caption("`network_visualization.html` exists. Streamlit can’t inline-render it safely by default.")
                st.download_button("Download network_visualization.html", data=html.read_bytes(), file_name=html.name)


if __name__ == "__main__":
    main()
