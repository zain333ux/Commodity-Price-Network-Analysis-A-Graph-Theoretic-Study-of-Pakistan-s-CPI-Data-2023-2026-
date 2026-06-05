from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_interactive_visualizations(
    output_dir: str | Path,
    graphs: dict[str, nx.Graph],
    graph_summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    temporal_summary_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    saved_files.extend(_draw_interactive_graphs(output_path, graphs, categories_df))
    saved_files.append(_draw_interactive_edge_counts(output_path, graph_summary_df))
    saved_files.append(_draw_interactive_threshold_sensitivity(output_path, sensitivity_df))
    saved_files.append(_draw_interactive_temporal_status(output_path, temporal_summary_df))
    saved_files.append(_draw_interactive_category_split(output_path, category_summary_df))
    return saved_files


def _draw_interactive_graphs(
    output_path: Path,
    graphs: dict[str, nx.Graph],
    categories_df: pd.DataFrame,
) -> list[Path]:
    saved_files: list[Path] = []
    category_by_item = dict(zip(categories_df["Item"], categories_df["Category"], strict=True))

    for window, graph in sorted(graphs.items()):
        positions = nx.spring_layout(graph, seed=42, k=0.9, iterations=120, weight="support_cities")

        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_text: list[str] = []
        for item_a, item_b, data in graph.edges(data=True):
            x0, y0 = positions[item_a]
            x1, y1 = positions[item_b]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(
                f"{item_a} - {item_b}<br>"
                f"Support cities: {data.get('support_cities', 0)}<br>"
                f"Average similarity: {data.get('average_similarity', 0):.3f}"
            )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={"width": 1.2, "color": "rgba(75, 85, 99, 0.45)"},
            hoverinfo="skip",
            mode="lines",
            name="Relationships",
        )

        node_x: list[float] = []
        node_y: list[float] = []
        node_text: list[str] = []
        node_category: list[str] = []
        node_size: list[int] = []
        degrees = dict(graph.degree())

        for node in graph.nodes():
            x, y = positions[node]
            node_x.append(float(x))
            node_y.append(float(y))
            category = category_by_item.get(node, "Unknown")
            node_category.append(category)
            node_size.append(12 + degrees[node] * 3)
            node_text.append(
                f"Item: {node}<br>"
                f"Category: {category}<br>"
                f"Degree: {degrees[node]}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hovertext=node_text,
            hoverinfo="text",
            marker={
                "size": node_size,
                "color": pd.Categorical(node_category).codes,
                "colorscale": "Viridis",
                "line": {"width": 1, "color": "#111827"},
                "showscale": False,
            },
            text=list(graph.nodes()),
            name="Items",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"{window} Interactive Item Co-movement Graph",
            showlegend=False,
            hovermode="closest",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=760,
        )

        target = output_path / f"interactive_graph_{window}.html"
        fig.write_html(target)
        saved_files.append(target)

    return saved_files


def _draw_interactive_edge_counts(output_path: Path, graph_summary_df: pd.DataFrame) -> Path:
    fig = px.bar(
        graph_summary_df,
        x="Window",
        y="Edges",
        text="Edges",
        title="Interactive Edge Count by Window",
    )
    fig.update_traces(marker_color="#2f6f73", textposition="outside")
    fig.update_layout(height=520)
    target = output_path / "interactive_edge_count_by_window.html"
    fig.write_html(target)
    return target


def _draw_interactive_threshold_sensitivity(output_path: Path, sensitivity_df: pd.DataFrame) -> Path:
    chart_df = sensitivity_df.copy()
    chart_df["Setting"] = chart_df.apply(lambda row: f"tau={row['Tau']}, K={row['K']}", axis=1)
    fig = px.bar(
        chart_df,
        x="Setting",
        y="Edges",
        color="Window",
        barmode="group",
        hover_data=["Nodes", "ConnectedComponents"],
        title="Interactive Threshold Sensitivity",
    )
    fig.update_layout(height=620, xaxis_tickangle=-35)
    target = output_path / "interactive_threshold_sensitivity.html"
    fig.write_html(target)
    return target


def _draw_interactive_temporal_status(output_path: Path, temporal_summary_df: pd.DataFrame) -> Path:
    fig = px.bar(
        temporal_summary_df,
        x="Status",
        y="EdgeCount",
        text="EdgeCount",
        title="Interactive Temporal Edge Status",
    )
    fig.update_traces(marker_color="#4c6f9f", textposition="outside")
    fig.update_layout(height=540, xaxis_tickangle=-25)
    target = output_path / "interactive_temporal_edge_status.html"
    fig.write_html(target)
    return target


def _draw_interactive_category_split(output_path: Path, category_summary_df: pd.DataFrame) -> Path:
    fig = px.bar(
        category_summary_df,
        x="Window",
        y="EdgeCount",
        color="EdgeType",
        barmode="stack",
        hover_data=["PercentOfWindowEdges"],
        title="Interactive Within-Category vs Between-Category Edges",
    )
    fig.update_layout(height=540)
    target = output_path / "interactive_category_edge_split.html"
    fig.write_html(target)
    return target
