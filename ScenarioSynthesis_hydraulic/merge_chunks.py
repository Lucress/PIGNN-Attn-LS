"""Merge deterministic Parquet chunks and verify their aggregate row count."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def _chunk_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Chunk name must end in an integer: {path.name}") from exc


def merge_chunks(input_dir: Path, output: Path, expected_chunks: int, expected_rows: int, overwrite: bool) -> None:
    chunks = sorted(input_dir.glob("chunk_*.parquet"), key=_chunk_number)
    if len(chunks) != expected_chunks:
        raise RuntimeError(f"Expected {expected_chunks} chunks, found {len(chunks)} in {input_dir}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = None
    rows = 0
    try:
        for chunk in chunks:
            parquet = pq.ParquetFile(chunk)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                elif table.schema != writer.schema:
                    raise RuntimeError(f"Schema mismatch in {chunk}")
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {expected_rows} rows, merged {rows}")
    temporary.replace(output)
    print(f"Merged {len(chunks)} chunks and {rows} rows into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-chunks", type=int, default=18)
    parser.add_argument("--expected-rows", type=int, default=36000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    merge_chunks(args.input_dir, args.output, args.expected_chunks, args.expected_rows, args.overwrite)


if __name__ == "__main__":
    main()
