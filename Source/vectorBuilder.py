from __future__ import annotations

import pandas as pd


def build_price_vectors(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(["Window", "City", "Item", "WindowMonthIndex"]).copy()
    vectors = (
        sorted_df.groupby(["Window", "City", "Item"])
        .agg(
            PriceVector=("Price", lambda s: [float(value) for value in s.tolist()]),
            Months=("WindowMonthIndex", lambda s: [int(value) for value in s.tolist()]),
        )
        .reset_index()
    )
    vectors["PriceChangeVector"] = vectors["PriceVector"].apply(_compute_price_changes)
    vectors["PercentageChangeVector"] = vectors["PriceVector"].apply(_compute_percentage_changes)
    vectors["VectorLength"] = vectors["PriceChangeVector"].apply(len)
    return vectors.sort_values(["Window", "City", "Item"]).reset_index(drop=True)


def summarize_vectors(vectors_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        vectors_df.groupby("Window")
        .agg(
            ItemCityVectors=("Item", "size"),
            Cities=("City", "nunique"),
            Items=("Item", "nunique"),
            PricePoints=("PriceVector", lambda s: len(s.iloc[0]) if not s.empty else 0),
            ChangeSteps=("VectorLength", "max"),
        )
        .reset_index()
        .sort_values("Window")
        .reset_index(drop=True)
    )
    return summary


def _compute_price_changes(price_vector: list[float]) -> list[float]:
    return [
        round(price_vector[index] - price_vector[index - 1], 4)
        for index in range(1, len(price_vector))
    ]


def _compute_percentage_changes(price_vector: list[float]) -> list[float]:
    changes: list[float] = []

    for index in range(1, len(price_vector)):
        previous_price = price_vector[index - 1]
        current_price = price_vector[index]

        if previous_price == 0:
            changes.append(0.0)
        else:
            changes.append(round((current_price - previous_price) / previous_price, 6))

    return changes
