#!/usr/bin/env python
"""Re-lay-out a parquet with small row groups.

Training shuffles, so a batch of 4 drawn at random touches 4 different row
groups. Parquet decodes a whole row group to serve any row in it, so a file
written with 6000-row groups decodes 6000 rows to deliver 4. Measured on the OPF
files: 3191 ms/batch at ~2000 rows/group against 176 ms/batch at 20, an 18x
penalty that is pure waste.

The fix is to rewrite with small row groups. This costs a little file size --
compression works over a smaller window -- and buys back the decode.

    python rewrite_rowgroups.py IN.parquet OUT.parquet --row-group-size 20
    python rewrite_rowgroups.py --scan DIR            # report layouts, write nothing

Streams row group by row group, so peak memory is one input group, not the file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


def describe(path: str) -> dict:
    f = pq.ParquetFile(path)
    md = f.metadata
    sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    return {
        "rows": md.num_rows,
        "groups": md.num_row_groups,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "bytes": os.path.getsize(path),
    }


def rewrite(src: str, dst: str, row_group_size: int, compression: str) -> None:
    f = pq.ParquetFile(src)
    schema = f.schema_arrow
    tmp = dst + ".partial"
    t0 = time.time()
    rows = 0
    writer = pq.ParquetWriter(tmp, schema, compression=compression)
    try:
        # iter_batches respects the requested batch size regardless of the
        # input's own group layout, so this works for any source file.
        for batch in f.iter_batches(batch_size=row_group_size):
            writer.write_table(pa.Table.from_batches([batch], schema=schema))
            rows += batch.num_rows
    finally:
        writer.close()
    if rows != f.metadata.num_rows:
        os.remove(tmp)
        raise RuntimeError(
            f"{src}: wrote {rows} rows, source has {f.metadata.num_rows}")
    os.replace(tmp, dst)
    print(f"  {rows} rows in {time.time()-t0:.0f}s -> {dst}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?")
    p.add_argument("dst", nargs="?")
    p.add_argument("--row-group-size", type=int, default=20)
    p.add_argument("--compression", default="snappy")
    p.add_argument("--scan", metavar="DIR",
                   help="report row-group layout of every parquet under DIR")
    p.add_argument("--pattern", default="_NR_branchrows_directSI.parquet",
                   help="filename filter used with --scan")
    args = p.parse_args()

    if args.scan:
        rows = []
        for name in sorted(os.listdir(args.scan)):
            if not name.endswith(".parquet") or args.pattern not in name:
                continue
            path = os.path.join(args.scan, name)
            if os.path.getsize(path) == 0:
                rows.append((name, None))
                continue
            try:
                rows.append((name, describe(path)))
            except Exception as e:  # a truncated file should not stop the scan
                print(f"{name}: UNREADABLE ({type(e).__name__})")
                continue
        for name, d in rows:
            if d is None:
                print(f"{name[:78]:80s} EMPTY (0 bytes)")
            else:
                flag = "ok" if d["max"] <= 64 else "NEEDS REWRITE"
                print(f"{name[:78]:80s} rows={d['rows']:7d} groups={d['groups']:6d} "
                      f"max/group={d['max']:6d} {d['bytes']/2**30:7.2f} GB  {flag}")
        return 0

    if not args.src or not args.dst:
        p.error("src and dst are required unless --scan is given")
    if os.path.getsize(args.src) == 0:
        print(f"{args.src}: empty, skipped", file=sys.stderr)
        return 1
    before = describe(args.src)
    print(f"{os.path.basename(args.src)}: {before['rows']} rows, "
          f"{before['groups']} groups, max {before['max']}/group, "
          f"{before['bytes']/2**30:.2f} GB", flush=True)
    if before["max"] <= args.row_group_size:
        print("  already small enough, skipped")
        return 0
    rewrite(args.src, args.dst, args.row_group_size, args.compression)
    after = describe(args.dst)
    print(f"  now {after['groups']} groups, max {after['max']}/group, "
          f"{after['bytes']/2**30:.2f} GB "
          f"({100*(after['bytes']/before['bytes']-1):+.1f}% size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
