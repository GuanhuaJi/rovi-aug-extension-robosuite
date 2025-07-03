#!/usr/bin/env python3
"""
inspect_parquet.py
------------------
Print a Parquet file’s schema, metadata, and per-row-group column stats.

Usage
-----
    python inspect_parquet.py /path/to/file.parquet
"""

from pathlib import Path
import sys
import pyarrow.parquet as pq


def inspect(fp: Path) -> None:
    pf = pq.ParquetFile(fp)

    # ─── Schema ───────────────────────────────────────────────────────────
    print("\n=== FILE SCHEMA ===")
    print(pf.schema)

    # ─── File-level metadata ──────────────────────────────────────────────
    meta = pf.metadata
    print("\n=== FILE METADATA ===")
    print(f"Rows            : {meta.num_rows}")
    print(f"Row-groups       : {meta.num_row_groups}")
    print(f"Created by       : {meta.created_by}")
    print(f"Serialized size  : {meta.serialized_size} bytes")

    # ─── Row-groups & column chunks ───────────────────────────────────────
    print("\n=== ROW-GROUPS ===")
    for rg_idx in range(meta.num_row_groups):
        rg = meta.row_group(rg_idx)
        print(f"\nRow-group {rg_idx}: "
              f"{rg.num_rows} rows, {rg.total_byte_size} bytes")

        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)

            # PyArrow ≥10: .path_in_schema (str)   |  ≤9: .path (ColumnPath)
            path = (col.path_in_schema
                    if hasattr(col, "path_in_schema")
                    else col.path.to_string())

            stats = col.statistics
            stats_str = (f"min={stats.min} max={stats.max}"
                         if stats and stats.has_min_max
                         else "no stats")

            print(f"  • {path:<30} | "
                  f"enc:{col.compression:<8} | "
                  f"{col.total_compressed_size:>7} B | {stats_str}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_parquet.py <file.parquet>", file=sys.stderr)
        sys.exit(1)

    inspect(Path(sys.argv[1]).expanduser())

'''
python /home/guanhuaji/mirage/robot2robot/rendering/read_parquet.py /home/abrashid/lerobot_datasets/autolab_ur5_0_100/data/chunk-000/episode_000000.parquet


'''
