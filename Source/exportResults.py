from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_result_tables(
    output_dir: str | Path,
    tables: dict[str, pd.DataFrame],
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for file_name, table in tables.items():
        target = output_path / file_name
        table.to_csv(target, index=False)
        saved_files.append(target)

    return saved_files
