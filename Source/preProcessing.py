from __future__ import annotations

import pandas as pd


WINDOWS = {
    "W1": ((2023, 3), (2024, 2)),
    "W2": ((2024, 3), (2025, 2)),
    "W3": ((2025, 3), (2026, 2)),
}
COMMON_WINDOW_MONTH_COUNT = 12
EXPECTED_MONTH_INDEXES = tuple(range(1, COMMON_WINDOW_MONTH_COUNT + 1))


def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Window"] = prepared.apply(_pick_window_label, axis=1)
    prepared = prepared.dropna(subset=["Window"]).reset_index(drop=True)
    return prepared


def add_window_month_index(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["WindowMonthIndex"] = prepared.apply(_window_month_index, axis=1)
    prepared = prepared.dropna(subset=["WindowMonthIndex"]).copy()
    prepared["WindowMonthIndex"] = prepared["WindowMonthIndex"].astype(int)
    return prepared


def summarize_windows(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Window")
        .agg(
            Records=("Price", "size"),
            Cities=("City", "nunique"),
            Items=("Item", "nunique"),
            Months=("WindowMonthIndex", "nunique"),
        )
        .reset_index()
    )
    return summary.sort_values("Window").reset_index(drop=True)


def filter_common_window_months(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.loc[df["WindowMonthIndex"] <= COMMON_WINDOW_MONTH_COUNT].copy()
    return filtered.sort_values(["Window", "City", "Item", "WindowMonthIndex"]).reset_index(drop=True)


def month_coverage(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby(["Window", "WindowMonthIndex"])
        .size()
        .reset_index(name="Records")
        .sort_values(["Window", "WindowMonthIndex"])
        .reset_index(drop=True)
    )
    return coverage


def complete_sequence_summary(df: pd.DataFrame) -> pd.DataFrame:
    sequence_counts = _build_sequence_status(df)
    summary = (
        sequence_counts.groupby("Window")
        .agg(
            CompleteSequences=("IsComplete", lambda s: int(s.sum())),
            IncompleteSequences=("IsComplete", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values("Window")
        .reset_index(drop=True)
    )
    return summary


def filter_complete_sequences(df: pd.DataFrame) -> pd.DataFrame:
    sequence_counts = _build_sequence_status(df)
    complete_keys = sequence_counts.loc[
        sequence_counts["IsComplete"],
        ["Window", "City", "Item"],
    ]
    filtered = df.merge(complete_keys, on=["Window", "City", "Item"], how="inner")
    return filtered.sort_values(["Window", "City", "Item", "WindowMonthIndex"]).reset_index(drop=True)


def _build_sequence_status(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Window", "City", "Item"])["WindowMonthIndex"]
        .agg(lambda s: tuple(sorted(set(int(value) for value in s.tolist()))))
        .reset_index(name="ObservedIndexes")
    )
    # A sequence is complete only if it has exactly the full March-to-February month indexes.
    grouped["IsComplete"] = grouped["ObservedIndexes"].apply(lambda indexes: indexes == EXPECTED_MONTH_INDEXES)
    return grouped


def _pick_window_label(row: pd.Series) -> str | None:
    year = int(row["Year"])
    month = int(row["Month"])

    for label, ((start_year, start_month), (end_year, end_month)) in WINDOWS.items():
        if _is_within_range(year, month, start_year, start_month, end_year, end_month):
            return label
    return None


def _window_month_index(row: pd.Series) -> int | None:
    label = row["Window"]
    year = int(row["Year"])
    month = int(row["Month"])
    start_year, start_month = WINDOWS[label][0]

    return ((year - start_year) * 12) + (month - start_month) + 1


def _is_within_range(
    year: int,
    month: int,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> bool:
    current = (year, month)
    start = (start_year, start_month)
    end = (end_year, end_month)
    return start <= current <= end
