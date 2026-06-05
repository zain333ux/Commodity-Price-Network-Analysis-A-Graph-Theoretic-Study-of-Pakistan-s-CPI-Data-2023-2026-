from __future__ import annotations

import networkx as nx
import pandas as pd


WINDOW_ORDER = ["W1", "W2", "W3"]


def summarize_temporal_edges(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    edge_sets = _edge_sets_by_window(graphs)
    all_edges = set().union(*edge_sets.values())

    rows: list[dict[str, object]] = []
    for edge in sorted(all_edges):
        present_windows = [window for window in WINDOW_ORDER if edge in edge_sets.get(window, set())]
        rows.append(
            {
                "ItemA": edge[0],
                "ItemB": edge[1],
                "PresentIn": ",".join(present_windows),
                "WindowCount": len(present_windows),
                "Status": _edge_status(present_windows),
            }
        )

    return pd.DataFrame(rows).sort_values(["Status", "WindowCount", "ItemA", "ItemB"]).reset_index(drop=True)


def temporal_summary_table(temporal_edges_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        temporal_edges_df.groupby("Status")
        .size()
        .reset_index(name="EdgeCount")
        .sort_values("Status")
        .reset_index(drop=True)
    )
    return summary


def window_to_window_changes(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    edge_sets = _edge_sets_by_window(graphs)
    rows: list[dict[str, object]] = []

    for previous_window, next_window in [("W1", "W2"), ("W2", "W3")]:
        previous_edges = edge_sets.get(previous_window, set())
        next_edges = edge_sets.get(next_window, set())

        rows.append(
            {
                "Comparison": f"{previous_window} to {next_window}",
                "Stayed": len(previous_edges.intersection(next_edges)),
                "Appeared": len(next_edges.difference(previous_edges)),
                "Disappeared": len(previous_edges.difference(next_edges)),
            }
        )

    return pd.DataFrame(rows)


def persistent_edge_strength(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    edge_sets = _edge_sets_by_window(graphs)
    persistent_edges = set.intersection(*(edge_sets[window] for window in WINDOW_ORDER))

    rows: list[dict[str, object]] = []
    for item_a, item_b in sorted(persistent_edges):
        row: dict[str, object] = {
            "ItemA": item_a,
            "ItemB": item_b,
        }

        for window in WINDOW_ORDER:
            edge_data = graphs[window].get_edge_data(item_a, item_b, default={})
            row[f"{window}_SupportCities"] = edge_data.get("support_cities", 0)
            row[f"{window}_SupportRatio"] = edge_data.get("support_ratio", 0)
            row[f"{window}_AverageSimilarity"] = round(edge_data.get("average_similarity", 0), 6)

        rows.append(row)

    return pd.DataFrame(rows)


def top_temporal_edges(temporal_edges_df: pd.DataFrame, status: str, limit: int = 10) -> pd.DataFrame:
    return (
        temporal_edges_df.loc[temporal_edges_df["Status"] == status]
        .head(limit)
        .reset_index(drop=True)
    )


def _edge_sets_by_window(graphs: dict[str, nx.Graph]) -> dict[str, set[tuple[str, str]]]:
    edge_sets: dict[str, set[tuple[str, str]]] = {}

    for window, graph in graphs.items():
        edge_sets[window] = {_normalize_edge(item_a, item_b) for item_a, item_b in graph.edges()}

    return edge_sets


def _normalize_edge(item_a: str, item_b: str) -> tuple[str, str]:
    return tuple(sorted((item_a, item_b)))


def _edge_status(present_windows: list[str]) -> str:
    if present_windows == WINDOW_ORDER:
        return "Persistent"
    if present_windows == ["W1"]:
        return "Only W1"
    if present_windows == ["W2"]:
        return "Only W2"
    if present_windows == ["W3"]:
        return "Only W3"
    if present_windows == ["W1", "W2"]:
        return "Disappeared in W3"
    if present_windows == ["W2", "W3"]:
        return "Appeared after W1"
    if present_windows == ["W1", "W3"]:
        return "Returned in W3"
    return "Other"
