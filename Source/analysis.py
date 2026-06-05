from __future__ import annotations

import networkx as nx
import pandas as pd


def analyze_graphs(graphs: dict[str, nx.Graph]) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}

    for window, graph in graphs.items():
        results[window] = {
            "degree": nx.degree_centrality(graph),
            "closeness": nx.closeness_centrality(graph),
            "betweenness": nx.betweenness_centrality(graph),
        }

    return results


def analyze_weighted_graphs(
    graphs: dict[str, nx.Graph],
    weight_attribute: str,
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}

    for window, graph in graphs.items():
        distance_graph = _graph_with_distance_weights(graph, weight_attribute)
        results[window] = {
            "weighted_degree": _weighted_degree_centrality(graph, weight_attribute),
            "weighted_closeness": nx.closeness_centrality(distance_graph, distance="distance"),
            "weighted_betweenness": nx.betweenness_centrality(distance_graph, weight="distance"),
        }

    return results


def top_pagerank_items(
    graphs: dict[str, nx.Graph],
    weight_attribute: str = "support_weight",
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, graph in graphs.items():
        pagerank_scores = nx.pagerank(graph, weight=weight_attribute)

        for rank, (item_name, score) in _top_ranked_items(pagerank_scores, top_n):
            rows.append(
                {
                    "Window": window,
                    "Rank": rank,
                    "Item": item_name,
                    "PageRankScore": round(score, 6),
                    "WeightAttribute": weight_attribute,
                }
            )

    return pd.DataFrame(rows).sort_values(["Window", "Rank"]).reset_index(drop=True)


def largest_component_centrality(
    graphs: dict[str, nx.Graph],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, graph in graphs.items():
        if graph.number_of_nodes() == 0:
            continue

        largest_nodes = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_nodes).copy()
        metrics = {
            "degree": nx.degree_centrality(largest_graph),
            "closeness": nx.closeness_centrality(largest_graph),
            "betweenness": nx.betweenness_centrality(largest_graph),
        }

        for metric_name, values in metrics.items():
            for rank, (item_name, score) in _top_ranked_items(values, top_n):
                rows.append(
                    {
                        "Window": window,
                        "LargestComponentSize": largest_graph.number_of_nodes(),
                        "Metric": metric_name,
                        "Rank": rank,
                        "Item": item_name,
                        "Score": round(score, 6),
                    }
                )

    return pd.DataFrame(rows).sort_values(["Window", "Metric", "Rank"]).reset_index(drop=True)


def compare_degree_top_items(
    absolute_top_items_df: pd.DataFrame,
    percentage_top_items_df: pd.DataFrame,
) -> pd.DataFrame:
    absolute_degree = absolute_top_items_df.loc[absolute_top_items_df["Metric"] == "degree"]
    percentage_degree = percentage_top_items_df.loc[percentage_top_items_df["Metric"] == "degree"]

    rows: list[dict[str, object]] = []
    for window in sorted(absolute_degree["Window"].unique()):
        absolute_items = absolute_degree.loc[absolute_degree["Window"] == window, "Item"].tolist()
        percentage_items = percentage_degree.loc[percentage_degree["Window"] == window, "Item"].tolist()
        common_items = sorted(set(absolute_items).intersection(percentage_items))
        rows.append(
            {
                "Window": window,
                "TopAbsoluteChangeItems": " | ".join(absolute_items),
                "TopPercentageChangeItems": " | ".join(percentage_items),
                "CommonTopItems": len(common_items),
                "CommonItems": " | ".join(common_items),
            }
        )

    return pd.DataFrame(rows)


def top_weighted_central_items(
    weighted_results_by_scheme: dict[str, dict[str, dict[str, dict[str, float]]]],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scheme_name, analysis_results in weighted_results_by_scheme.items():
        for window, metrics in analysis_results.items():
            for metric_name, values in metrics.items():
                for rank, (item_name, score) in _top_ranked_items(values, top_n):
                    rows.append(
                        {
                            "WeightScheme": scheme_name,
                            "Window": window,
                            "Metric": metric_name,
                            "Rank": rank,
                            "Item": item_name,
                            "Score": round(score, 6),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["WeightScheme", "Window", "Metric", "Rank"]).reset_index(drop=True)


def compare_weighted_degree_rankings(weighted_top_items_df: pd.DataFrame) -> pd.DataFrame:
    degree_rows = weighted_top_items_df.loc[weighted_top_items_df["Metric"] == "weighted_degree"].copy()
    support_rows = degree_rows.loc[degree_rows["WeightScheme"] == "support_cities"]
    similarity_rows = degree_rows.loc[degree_rows["WeightScheme"] == "average_similarity"]

    rows: list[dict[str, object]] = []
    for window in sorted(degree_rows["Window"].unique()):
        support_items = support_rows.loc[support_rows["Window"] == window, "Item"].tolist()
        similarity_items = similarity_rows.loc[similarity_rows["Window"] == window, "Item"].tolist()
        rows.append(
            {
                "Window": window,
                "TopSupportWeightItems": " | ".join(support_items),
                "TopSimilarityWeightItems": " | ".join(similarity_items),
                "CommonTopItems": len(set(support_items).intersection(similarity_items)),
            }
        )

    return pd.DataFrame(rows)


def top_central_items(
    analysis_results: dict[str, dict[str, dict[str, float]]],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, metrics in analysis_results.items():
        for metric_name, values in metrics.items():
            for rank, (item_name, score) in _top_ranked_items(values, top_n):
                rows.append(
                    {
                        "Window": window,
                        "Metric": metric_name,
                        "Rank": rank,
                        "Item": item_name,
                        "Score": round(score, 6),
                    }
                )

    return pd.DataFrame(rows).sort_values(["Window", "Metric", "Rank"]).reset_index(drop=True)


def _weighted_degree_centrality(graph: nx.Graph, weight_attribute: str) -> dict[str, float]:
    total_possible_nodes = max(graph.number_of_nodes() - 1, 1)
    weighted_scores: dict[str, float] = {}

    for node in graph.nodes():
        total_weight = sum(
            edge_data.get(weight_attribute, 0.0)
            for _, _, edge_data in graph.edges(node, data=True)
        )
        weighted_scores[node] = float(total_weight) / total_possible_nodes

    return weighted_scores


def _graph_with_distance_weights(graph: nx.Graph, weight_attribute: str) -> nx.Graph:
    distance_graph = graph.copy()

    for item_a, item_b, edge_data in distance_graph.edges(data=True):
        similarity_strength = max(float(edge_data.get(weight_attribute, 0.0)), 0.000001)
        # Stronger relationships should be treated as shorter paths for closeness and betweenness.
        edge_data["distance"] = 1 / similarity_strength

    return distance_graph


def _top_ranked_items(scores: dict[str, float], top_n: int) -> list[tuple[int, tuple[str, float]]]:
    ranked_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return list(enumerate(ranked_items, start=1))
