from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

CATEGORY_COLORS = {
    "Grains and Pulses": "#2f6f73",
    "Meat and Dairy": "#8c4b4b",
    "Vegetables and Fruits": "#6f8f45",
    "Edible Oil and Condiments": "#b8792f",
    "Energy and Utilities": "#4c6f9f",
    "Clothing and Footwear": "#7a5a9e",
    "Household and Services": "#6b7280",
}


def create_visualizations(
    output_dir: str | Path,
    graphs: dict[str, nx.Graph],
    graph_summary_df: pd.DataFrame,
    top_items_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    weighted_top_items_df: pd.DataFrame | None = None,
    categories_df: pd.DataFrame | None = None,
    temporal_summary_df: pd.DataFrame | None = None,
    category_summary_df: pd.DataFrame | None = None,
    component_alignment_summary_df: pd.DataFrame | None = None,
    aggregated_df: pd.DataFrame | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    saved_files.extend(_draw_yearly_graphs(output_path, graphs, categories_df))
    saved_files.append(_draw_edge_count_chart(output_path, graph_summary_df))
    saved_files.append(_draw_degree_centrality_chart(output_path, top_items_df))
    saved_files.append(_draw_sensitivity_chart(output_path, sensitivity_df))
    saved_files.append(_draw_density_sensitivity_chart(output_path, sensitivity_df))
    if weighted_top_items_df is not None:
        saved_files.append(_draw_weighted_degree_chart(output_path, weighted_top_items_df))
    if temporal_summary_df is not None:
        saved_files.append(_draw_temporal_status_chart(output_path, temporal_summary_df))
    if category_summary_df is not None:
        saved_files.append(_draw_category_edge_chart(output_path, category_summary_df))
    if component_alignment_summary_df is not None:
        saved_files.append(_draw_component_alignment_chart(output_path, component_alignment_summary_df))
    if aggregated_df is not None:
        saved_files.extend(_draw_similarity_heatmaps(output_path, aggregated_df))
    return saved_files


def _draw_yearly_graphs(
    output_path: Path,
    graphs: dict[str, nx.Graph],
    categories_df: pd.DataFrame | None,
) -> list[Path]:
    saved_files: list[Path] = []
    category_by_item = _category_lookup(categories_df)
    stable_positions = _stable_graph_positions(list(graphs.values()))

    for window, graph in sorted(graphs.items()):
        fig, ax = plt.subplots(figsize=(16, 11))
        positions = stable_positions
        degrees = dict(graph.degree())
        active_nodes = [node for node in graph.nodes() if degrees.get(node, 0) > 0]
        isolated_nodes = [node for node in graph.nodes() if degrees.get(node, 0) == 0]
        active_graph = graph.subgraph(active_nodes).copy()
        isolated_graph = graph.subgraph(isolated_nodes).copy()
        top_nodes = _top_degree_nodes(graph, limit=5)

        node_sizes = [150 + (degrees[node] * 36) for node in active_graph.nodes()]
        node_colors = [
            "#f59e0b" if node in top_nodes else CATEGORY_COLORS.get(category_by_item.get(node, ""), "#4f9d8f")
            for node in active_graph.nodes()
        ]
        edge_widths = [
            max(0.65, active_graph.edges[edge].get("support_cities", 1) / 3.3)
            for edge in active_graph.edges()
        ]

        nx.draw_networkx_edges(
            active_graph,
            positions,
            width=edge_widths,
            alpha=0.22 if active_graph.number_of_edges() > 60 else 0.34,
            edge_color="#374151",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            active_graph,
            positions,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="#111827",
            linewidths=0.9,
            ax=ax,
        )
        if isolated_nodes:
            nx.draw_networkx_nodes(
                isolated_graph,
                positions,
                node_size=85,
                node_color="#e2e8f0",
                edgecolors="#94a3b8",
                linewidths=0.65,
                alpha=0.62,
                ax=ax,
            )

        labels = {node: str(index + 1) for index, node in enumerate(top_nodes)}
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=8,
            font_weight="bold",
            font_color="white",
            ax=ax,
        )

        ax.set_title(
            f"{window} Item Co-movement Graph",
            fontsize=19,
            weight="bold",
            pad=16,
        )
        ax.text(
            0.5,
            0.955,
            "Numbered orange nodes are the top connected items; grey nodes are isolated items.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#334155",
        )
        ax.axis("off")
        ax.set_xlim(-3.6, 4.85)
        ax.set_ylim(-2.95, 2.65)
        _add_clean_graph_side_panel(ax, graph, top_nodes, color="#2f6f73")
        _add_category_legend(ax, category_by_item, graph)
        target = output_path / f"graph_{window}.png"
        fig.tight_layout()
        fig.savefig(target, dpi=200)
        plt.close(fig)
        saved_files.append(target)

    return saved_files


def _stable_graph_positions(graphs: list[nx.Graph]) -> dict[str, tuple[float, float]]:
    union = nx.Graph()
    for graph in graphs:
        union.add_nodes_from(graph.nodes())
        union.add_edges_from(graph.edges())

    main_nodes = [node for node, degree in union.degree() if degree > 0]
    isolated_nodes = [node for node, degree in union.degree() if degree == 0]
    main_graph = union.subgraph(main_nodes).copy()
    main_pos = nx.spring_layout(main_graph, seed=42, k=1.65, iterations=350)
    main_pos = _normalize_positions(main_pos, x_span=(-2.7, 2.7), y_span=(-2.25, 1.55))

    isolated_pos: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(sorted(isolated_nodes)):
        row, col = divmod(index, 8)
        isolated_pos[node] = (-3.25 + col * 0.42, -2.45 - row * 0.32)

    return {**main_pos, **isolated_pos}


def _normalize_positions(
    positions: dict[str, tuple[float, float]],
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    if not positions:
        return {}
    xs = [value[0] for value in positions.values()]
    ys = [value[1] for value in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    return {
        node: (
            x_span[0] + (value[0] - min_x) / width * (x_span[1] - x_span[0]),
            y_span[0] + (value[1] - min_y) / height * (y_span[1] - y_span[0]),
        )
        for node, value in positions.items()
    }


def _top_degree_nodes(graph: nx.Graph, limit: int) -> list[str]:
    if graph.number_of_edges() == 0:
        return []
    return [node for node, degree in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:limit] if degree > 0]


def _add_clean_graph_side_panel(ax: plt.Axes, graph: nx.Graph, top_nodes: list[str], color: str) -> None:
    possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
    density = 0 if possible_edges == 0 else graph.number_of_edges() / possible_edges * 100
    top_lines = [f"{index}. {_short_label(node, max_len=26)}" for index, node in enumerate(top_nodes, start=1)]
    top_text = "\n".join(top_lines) if top_lines else "No connected item"
    ax.text(
        0.70,
        0.78,
        "How to read this\n"
        "Blue/category nodes = connected items\n"
        "Grey nodes = isolated items\n"
        "Numbered orange nodes = top items",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#0f172a",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#ffffff", "edgecolor": "#dbe3ef"},
    )
    ax.text(
        0.70,
        0.53,
        "Top connected items\n" + top_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#0f172a",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#ffffff", "edgecolor": "#dbe3ef"},
    )
    ax.text(
        0.50,
        0.055,
        f"Nodes: {graph.number_of_nodes()}   |   Edges: {graph.number_of_edges()}   |   "
        f"Density: {density:.1f}%   |   Components: {nx.number_connected_components(graph)}",
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#ffffff", "edgecolor": color, "linewidth": 2},
    )


def _short_label(text: str, max_len: int) -> str:
    clean = (
        str(text)
        .replace("(Average Quality)", "")
        .replace("Average", "Avg")
        .replace("Quality", "Q")
        .replace("Cooking Oil", "Oil")
        .replace("Vegetable Ghee", "Ghee")
        .replace("Hi-Speed Diesel", "Diesel")
        .strip()
    )
    return clean if len(clean) <= max_len else clean[: max_len - 1] + "."


def _draw_edge_count_chart(output_path: Path, graph_summary_df: pd.DataFrame) -> Path:
    target = output_path / "edge_count_by_window.png"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(graph_summary_df["Window"], graph_summary_df["Edges"], color="#2f6f73")
    ax.set_title("Number of Item Relationships in Each Window", weight="bold")
    ax.set_xlabel("Window")
    ax.set_ylabel("Edges")
    ax.bar_label(bars, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)

    return target


def _draw_degree_centrality_chart(output_path: Path, top_items_df: pd.DataFrame) -> Path:
    target = output_path / "top_degree_centrality.png"
    degree_df = top_items_df.loc[top_items_df["Metric"] == "degree"].copy()
    windows = sorted(degree_df["Window"].unique())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(windows),
        figsize=(18, 7),
        sharex=True,
    )
    if len(windows) == 1:
        axes = [axes]

    for ax, window in zip(axes, windows, strict=True):
        window_df = degree_df.loc[degree_df["Window"] == window].sort_values("Rank", ascending=False)
        labels = [_wrap_label(item, width=20, max_words=6) for item in window_df["Item"]]
        bars = ax.barh(labels, window_df["Score"], color="#806443")
        ax.set_title(window, weight="bold", fontsize=14)
        ax.set_xlabel("Degree Centrality")
        ax.bar_label(bars, labels=[f"{score:.2f}" for score in window_df["Score"]], padding=4, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.16)

    fig.suptitle("Top Degree Centrality Items", fontsize=18, weight="bold", y=0.98)
    fig.supxlabel("Higher value means the item has more direct graph connections", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.55)
    fig.savefig(target, dpi=200)
    plt.close(fig)

    return target


def _draw_sensitivity_chart(output_path: Path, sensitivity_df: pd.DataFrame) -> Path:
    target = output_path / "threshold_sensitivity_edges.png"
    chart_df = sensitivity_df.copy()
    chart_df["Setting"] = chart_df.apply(lambda row: f"tau={row['Tau']}, K={row['K']}", axis=1)

    pivot = chart_df.pivot_table(index="Setting", columns="Window", values="Edges", aggfunc="sum")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=ax, color=["#2f6f73", "#806443", "#5b6472"])
    ax.set_title("Threshold Sensitivity by Edge Count", weight="bold")
    ax.set_xlabel("Threshold Setting")
    ax.set_ylabel("Edges")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.legend(title="Window", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)

    return target


def _draw_density_sensitivity_chart(output_path: Path, sensitivity_df: pd.DataFrame) -> Path:
    target = output_path / "threshold_density_curve.png"
    chart_df = sensitivity_df.copy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5.5), sharey=True)
    windows = sorted(chart_df["Window"].unique())
    colors = {
        2: "#2f6f73",
        4: "#4c6f9f",
        6: "#806443",
        8: "#8c4b4b",
    }

    for ax, window in zip(axes, windows, strict=True):
        window_df = chart_df.loc[chart_df["Window"] == window]
        for k_value, k_df in sorted(window_df.groupby("K")):
            ordered = k_df.sort_values("Tau")
            ax.plot(
                ordered["Tau"],
                ordered["DensityPercent"],
                marker="o",
                linewidth=2,
                color=colors.get(int(k_value), "#6b7280"),
                label=f"K={k_value}",
            )

        ax.set_title(window, weight="bold")
        ax.set_xlabel("Similarity threshold τ")
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Graph density (%)")
    fig.suptitle("Graph Density vs Similarity Threshold", fontsize=17, weight="bold", y=1.02)
    fig.legend(title="City support", frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.84, wspace=0.2)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return target


def _draw_weighted_degree_chart(output_path: Path, weighted_top_items_df: pd.DataFrame) -> Path:
    target = output_path / "weighted_degree_comparison.png"
    degree_df = weighted_top_items_df.loc[
        weighted_top_items_df["Metric"] == "weighted_degree"
    ].copy()
    top_rank_df = degree_df.loc[degree_df["Rank"] <= 3].copy()
    windows = sorted(top_rank_df["Window"].unique())

    fig, axes = plt.subplots(nrows=1, ncols=len(windows), figsize=(19, 7), sharex=False)
    if len(windows) == 1:
        axes = [axes]

    for ax, window in zip(axes, windows, strict=True):
        window_df = top_rank_df.loc[top_rank_df["Window"] == window].copy()
        window_df["Label"] = window_df.apply(
            lambda row: f"{_scheme_label(row['WeightScheme'])}\n{_wrap_label(row['Item'], width=18, max_words=5)}",
            axis=1,
        )
        window_df = window_df.iloc[::-1]
        colors = window_df["WeightScheme"].map(
            {"support_cities": "#2f6f73", "average_similarity": "#b8792f"}
        )
        bars = ax.barh(window_df["Label"], window_df["Score"], color=colors)
        ax.set_title(window, weight="bold", fontsize=14)
        ax.set_xlabel("Weighted Degree")
        ax.bar_label(bars, labels=[f"{score:.2f}" for score in window_df["Score"]], padding=4, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.18)

    fig.suptitle("Weighted Degree Centrality Under Two Weighting Schemes", fontsize=18, weight="bold", y=0.98)
    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="Support cities", markerfacecolor="#2f6f73", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Average similarity", markerfacecolor="#b8792f", markersize=10),
    ]
    fig.legend(handles=legend_items, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.65)
    fig.savefig(target, dpi=200)
    plt.close(fig)

    return target


def _draw_temporal_status_chart(output_path: Path, temporal_summary_df: pd.DataFrame) -> Path:
    target = output_path / "temporal_edge_status.png"
    ordered = temporal_summary_df.sort_values("EdgeCount", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    labels = [_wrap_label(status, width=18, max_words=4) for status in ordered["Status"]]
    bars = ax.barh(labels, ordered["EdgeCount"], color="#4c6f9f")
    ax.set_title("Temporal Edge Status Across Three Windows", weight="bold")
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Edge Status")
    ax.bar_label(bars, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _draw_category_edge_chart(output_path: Path, category_summary_df: pd.DataFrame) -> Path:
    target = output_path / "category_edge_split.png"
    pivot = category_summary_df.pivot(index="Window", columns="EdgeType", values="EdgeCount").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color={"Within Category": "#2f6f73", "Between Categories": "#806443"},
    )
    ax.set_title("Within-Category vs Between-Category Edges", weight="bold")
    ax.set_xlabel("Window")
    ax.set_ylabel("Edges")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Edge Type", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _draw_component_alignment_chart(output_path: Path, component_alignment_summary_df: pd.DataFrame) -> Path:
    target = output_path / "component_category_alignment.png"
    pivot = component_alignment_summary_df.pivot(index="Window", columns="Alignment", values="ComponentCount").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color={"Mostly One Category": "#6f8f45", "Mixed Categories": "#8c4b4b"},
    )
    ax.set_title("Connected Component Alignment With Categories", weight="bold")
    ax.set_xlabel("Window")
    ax.set_ylabel("Connected Components")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Alignment", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _draw_similarity_heatmaps(output_path: Path, aggregated_df: pd.DataFrame) -> list[Path]:
    saved_files: list[Path] = []

    for window, window_df in sorted(aggregated_df.groupby("Window")):
        top_edges = window_df.sort_values("AverageSimilarity", ascending=False).head(20)
        heatmap_items = sorted(set(top_edges["ItemA"]).union(top_edges["ItemB"]))
        matrix = pd.DataFrame(0.0, index=heatmap_items, columns=heatmap_items)

        for _, row in window_df.iterrows():
            item_a = row["ItemA"]
            item_b = row["ItemB"]
            if item_a in matrix.index and item_b in matrix.columns:
                matrix.loc[item_a, item_b] = row["AverageSimilarity"]
                matrix.loc[item_b, item_a] = row["AverageSimilarity"]

        short_labels = {item: _wrap_label(item, width=14, max_words=4) for item in heatmap_items}
        matrix = matrix.rename(index=short_labels, columns=short_labels)

        fig, ax = plt.subplots(figsize=(15, 13))
        sns.heatmap(
            matrix,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            linewidths=0.3,
            linecolor="#e5e7eb",
            square=True,
            cbar_kws={"label": "Average cosine similarity"},
            ax=ax,
        )
        ax.set_title(f"{window} Strongest Average Similarities", weight="bold")
        ax.set_xlabel("Item")
        ax.set_ylabel("Item")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)
        fig.tight_layout()

        target = output_path / f"similarity_heatmap_{window}.png"
        fig.savefig(target, dpi=200)
        plt.close(fig)
        saved_files.append(target)

    return saved_files


def _category_lookup(categories_df: pd.DataFrame | None) -> dict[str, str]:
    if categories_df is None:
        return {}
    return dict(zip(categories_df["Item"], categories_df["Category"], strict=True))


def _add_category_legend(ax: plt.Axes, category_by_item: dict[str, str], graph: nx.Graph) -> None:
    if not category_by_item:
        return

    graph_categories = sorted({category_by_item.get(node, "") for node in graph.nodes() if category_by_item.get(node, "")})
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=category,
            markerfacecolor=CATEGORY_COLORS.get(category, "#4f9d8f"),
            markeredgecolor="#111827",
            markersize=8,
        )
        for category in graph_categories
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8, ncol=2)


def _short_multiline(text: str) -> str:
    short_text = shorten(str(text), width=42, placeholder="...")
    return "\n".join(wrap(short_text, width=22))


def _wrap_label(text: str, width: int, max_words: int) -> str:
    words = str(text).split()
    trimmed = " ".join(words[:max_words])
    if len(words) > max_words:
        trimmed += "..."
    return "\n".join(wrap(trimmed, width=width))


def _scheme_label(weight_scheme: str) -> str:
    if weight_scheme == "support_cities":
        return "Support"
    if weight_scheme == "average_similarity":
        return "Similarity"
    return str(weight_scheme)
