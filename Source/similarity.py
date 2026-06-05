from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def compute_citywise_similarity(
    vectors_df: pd.DataFrame,
    vector_column: str = "PriceChangeVector",
) -> pd.DataFrame:
    similarity_rows: list[dict[str, object]] = []

    grouped = vectors_df.groupby(["Window", "City"], sort=True)
    for (window, city), group in grouped:
        city_rows = group.sort_values("Item").reset_index(drop=True)

        # We compare every pair of items inside the same city and the same time window.
        for left_index, right_index in combinations(range(len(city_rows)), 2):
            left_row = city_rows.iloc[left_index]
            right_row = city_rows.iloc[right_index]

            similarity_rows.append(
                {
                    "Window": window,
                    "City": city,
                    "ItemA": left_row["Item"],
                    "ItemB": right_row["Item"],
                    "CosineSimilarity": round(
                        _cosine_similarity(left_row[vector_column], right_row[vector_column]),
                        6,
                    ),
                }
            )

    similarity_df = pd.DataFrame(similarity_rows)
    return similarity_df.sort_values(["Window", "City", "ItemA", "ItemB"]).reset_index(drop=True)


def summarize_similarity(similarity_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        similarity_df.groupby("Window")
        .agg(
            Cities=("City", "nunique"),
            ItemPairs=("CosineSimilarity", "size"),
            AverageSimilarity=("CosineSimilarity", "mean"),
            MinimumSimilarity=("CosineSimilarity", "min"),
            MaximumSimilarity=("CosineSimilarity", "max"),
        )
        .reset_index()
        .sort_values("Window")
        .reset_index(drop=True)
    )
    return summary


def _cosine_similarity(left_vector: list[float], right_vector: list[float]) -> float:
    left = np.array(left_vector, dtype=float)
    right = np.array(right_vector, dtype=float)

    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)

    # If a vector has no movement at all, we return 0 so it does not create a fake strong match.
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return float(np.dot(left, right) / (left_norm * right_norm))
