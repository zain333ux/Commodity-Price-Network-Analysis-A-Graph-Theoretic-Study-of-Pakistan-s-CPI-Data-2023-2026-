from __future__ import annotations

import networkx as nx
import pandas as pd


DEFAULT_TAU = 0.6
DEFAULT_K = 5


def aggregate_similarity_across_cities(
    similarity_df: pd.DataFrame,
    tau: float = DEFAULT_TAU,
) -> pd.DataFrame:
    flagged = similarity_df.copy()
    flagged["SupportedByCity"] = flagged["CosineSimilarity"] >= tau

    aggregated = (
        flagged.groupby(["Window", "ItemA", "ItemB"])
        .agg(
            SupportCities=("SupportedByCity", "sum"),
            AverageSimilarity=("CosineSimilarity", "mean"),
            MaxSimilarity=("CosineSimilarity", "max"),
            CityComparisons=("City", "nunique"),
        )
        .reset_index()
    )
    aggregated["SupportCities"] = aggregated["SupportCities"].astype(int)
    aggregated["SupportRatio"] = (
        aggregated["SupportCities"] / aggregated["CityComparisons"]
    ).round(4)
    aggregated["Tau"] = tau
    return aggregated.sort_values(["Window", "ItemA", "ItemB"]).reset_index(drop=True)


def build_yearly_graphs(
    aggregated_df: pd.DataFrame,
    vectors_df: pd.DataFrame,
    k_threshold: int = DEFAULT_K,
) -> dict[str, nx.Graph]:
    graphs: dict[str, nx.Graph] = {}
    items_by_window = _items_by_window(vectors_df)

    for window, window_items in items_by_window.items():
        graph = nx.Graph()
        graph.add_nodes_from(window_items)

        window_edges = aggregated_df.loc[aggregated_df["Window"] == window].copy()
        # Edge rule from the project: an item pair is connected only if enough cities support it.
        window_edges = window_edges.loc[window_edges["SupportCities"] >= k_threshold]

        for _, row in window_edges.iterrows():
            graph.add_edge(
                row["ItemA"],
                row["ItemB"],
                support_cities=int(row["SupportCities"]),
                support_ratio=float(row["SupportRatio"]),
                average_similarity=float(row["AverageSimilarity"]),
                max_similarity=float(row["MaxSimilarity"]),
                support_weight=int(row["SupportCities"]),
                similarity_weight=float(row["AverageSimilarity"]),
            )

        graph.graph["k_threshold"] = k_threshold
        graphs[window] = graph

    return graphs


def summarize_aggregated_similarity(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        aggregated_df.groupby("Window")
        .agg(
            ItemPairs=("SupportCities", "size"),
            SupportedPairs=("SupportCities", lambda s: int((s > 0).sum())),
            MaxSupportCities=("SupportCities", "max"),
            AverageSupportCities=("SupportCities", "mean"),
        )
        .reset_index()
        .sort_values("Window")
        .reset_index(drop=True)
    )
    return summary


def summarize_graphs(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window, graph in graphs.items():
        nodes = graph.number_of_nodes()
        possible_edges = nodes * (nodes - 1) // 2
        density = graph.number_of_edges() / possible_edges if possible_edges else 0.0
        rows.append(
            {
                "Window": window,
                "Nodes": nodes,
                "Edges": graph.number_of_edges(),
                "PossibleEdges": possible_edges,
                "Density": round(density, 4),
                "DensityPercent": round(density * 100, 2),
                "ConnectedComponents": nx.number_connected_components(graph),
            }
        )

    return pd.DataFrame(rows).sort_values("Window").reset_index(drop=True)


def run_threshold_sensitivity(
    similarity_df: pd.DataFrame,
    vectors_df: pd.DataFrame,
    tau_values: list[float],
    k_values: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for tau in tau_values:
        aggregated_df = aggregate_similarity_across_cities(similarity_df, tau=tau)

        for k_value in k_values:
            graphs = build_yearly_graphs(aggregated_df, vectors_df, k_threshold=k_value)

            for window, graph in graphs.items():
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()
                possible_edges = nodes * (nodes - 1) // 2
                density = edges / possible_edges if possible_edges else 0.0
                top_item, top_score = _top_degree_item(graph)
                # These numbers show how strict or relaxed a threshold setting is.
                rows.append(
                    {
                        "Tau": tau,
                        "K": k_value,
                        "Window": window,
                        "Nodes": nodes,
                        "Edges": edges,
                        "PossibleEdges": possible_edges,
                        "Density": round(density, 4),
                        "DensityPercent": round(density * 100, 2),
                        "IsolatedNodes": nx.number_of_isolates(graph),
                        "ConnectedComponents": nx.number_connected_components(graph),
                        "TopDegreeItem": top_item,
                        "TopDegreeScore": round(top_score, 6),
                    }
                )

    return pd.DataFrame(rows).sort_values(["Tau", "K", "Window"]).reset_index(drop=True)


def _items_by_window(vectors_df: pd.DataFrame) -> dict[str, list[str]]:
    grouped = (
        vectors_df.groupby("Window")["Item"]
        .agg(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )
    return grouped


def _top_degree_item(graph: nx.Graph) -> tuple[str, float]:
    if graph.number_of_edges() == 0:
        return "No connected item", 0.0

    degree_centrality = nx.degree_centrality(graph)
    return max(degree_centrality.items(), key=lambda item: item[1])
