from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def load_item_categories(file_path: str | Path, expected_items: list[str]) -> pd.DataFrame:
    categories_df = pd.read_csv(file_path)
    required_columns = {"Item", "Category"}
    missing_columns = required_columns.difference(categories_df.columns)
    if missing_columns:
        raise ValueError("Category file must contain Item and Category columns.")

    categories_df = categories_df.loc[:, ["Item", "Category"]].copy()
    categories_df["Item"] = categories_df["Item"].astype("string").str.strip()
    categories_df["Category"] = categories_df["Category"].astype("string").str.strip()

    missing_items = sorted(set(expected_items).difference(set(categories_df["Item"].tolist())))
    if missing_items:
        raise ValueError(f"Missing category mapping for {len(missing_items)} items: {missing_items}")

    return categories_df


def build_category_edge_table(
    graphs: dict[str, nx.Graph],
    categories_df: pd.DataFrame,
) -> pd.DataFrame:
    category_by_item = dict(zip(categories_df["Item"], categories_df["Category"], strict=True))
    rows: list[dict[str, object]] = []

    for window, graph in sorted(graphs.items()):
        for item_a, item_b, edge_data in graph.edges(data=True):
            category_a = category_by_item[item_a]
            category_b = category_by_item[item_b]
            edge_type = "Within Category" if category_a == category_b else "Between Categories"

            rows.append(
                {
                    "Window": window,
                    "ItemA": item_a,
                    "CategoryA": category_a,
                    "ItemB": item_b,
                    "CategoryB": category_b,
                    "EdgeType": edge_type,
                    "SupportCities": edge_data.get("support_cities", 0),
                    "AverageSimilarity": edge_data.get("average_similarity", 0),
                }
            )

    return pd.DataFrame(rows).sort_values(["Window", "EdgeType", "CategoryA", "CategoryB"]).reset_index(drop=True)


def summarize_category_edges(category_edges_df: pd.DataFrame) -> pd.DataFrame:
    total_edges = category_edges_df.groupby("Window").size().rename("TotalEdges").reset_index()
    edge_type_counts = (
        category_edges_df.groupby(["Window", "EdgeType"])
        .size()
        .reset_index(name="EdgeCount")
    )

    summary = edge_type_counts.merge(total_edges, on="Window", how="left")
    summary["PercentOfWindowEdges"] = (summary["EdgeCount"] / summary["TotalEdges"] * 100).round(2)
    return summary.sort_values(["Window", "EdgeType"]).reset_index(drop=True)


def summarize_category_pairs(category_edges_df: pd.DataFrame) -> pd.DataFrame:
    prepared = category_edges_df.copy()
    prepared["CategoryPair"] = prepared.apply(_category_pair_label, axis=1)

    summary = (
        prepared.groupby(["Window", "CategoryPair"])
        .size()
        .reset_index(name="EdgeCount")
        .sort_values(["Window", "EdgeCount"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def analyze_component_category_alignment(
    graphs: dict[str, nx.Graph],
    categories_df: pd.DataFrame,
) -> pd.DataFrame:
    category_by_item = dict(zip(categories_df["Item"], categories_df["Category"], strict=True))
    rows: list[dict[str, object]] = []

    for window, graph in sorted(graphs.items()):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)

        for component_number, component_items in enumerate(components, start=1):
            component_list = sorted(component_items)
            category_counts: dict[str, int] = {}

            for item in component_list:
                category = category_by_item[item]
                category_counts[category] = category_counts.get(category, 0) + 1

            main_category = max(category_counts, key=category_counts.get)
            main_category_count = category_counts[main_category]
            component_size = len(component_list)
            purity_percent = round((main_category_count / component_size) * 100, 2)

            rows.append(
                {
                    "Window": window,
                    "Component": component_number,
                    "ComponentSize": component_size,
                    "CategoryCount": len(category_counts),
                    "MainCategory": main_category,
                    "MainCategoryItems": main_category_count,
                    "PurityPercent": purity_percent,
                    "Alignment": "Mostly One Category" if purity_percent >= 60 else "Mixed Categories",
                    "Items": " | ".join(component_list),
                }
            )

    return pd.DataFrame(rows).sort_values(["Window", "Component"]).reset_index(drop=True)


def summarize_component_alignment(component_alignment_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        component_alignment_df.groupby(["Window", "Alignment"])
        .size()
        .reset_index(name="ComponentCount")
        .sort_values(["Window", "Alignment"])
        .reset_index(drop=True)
    )
    return summary


def summarize_component_alignment_size2plus(component_alignment_df: pd.DataFrame) -> pd.DataFrame:
    filtered = component_alignment_df.loc[component_alignment_df["ComponentSize"] >= 2].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["Window", "Alignment", "ComponentCount"])

    summary = (
        filtered.groupby(["Window", "Alignment"])
        .size()
        .reset_index(name="ComponentCount")
        .sort_values(["Window", "Alignment"])
        .reset_index(drop=True)
    )
    return summary


def _category_pair_label(row: pd.Series) -> str:
    first, second = sorted([str(row["CategoryA"]), str(row["CategoryB"])])
    return f"{first} - {second}"
