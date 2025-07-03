#!/usr/bin/env python3
"""
subdir_mtime_timeline.py
------------------------
Create a one-line timeline of sub-directory last-modified times
(rounded to minutes) and save it as a PDF.

Usage
-----
    python /home/guanhuaji/mirage/robot2robot/rendering/get_folder_modification_time.py
"""

import argparse
import datetime as dt
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")          # headless / non-interactive
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PACIFIC = ZoneInfo("America/Los_Angeles")

def gather_dirs(root: Path, recursive: bool = False):
    """Return a list of Path objects for sub-dirs under *root*."""
    src = root.rglob("*") if recursive else root.iterdir()
    return [p for p in src if p.is_dir()]

from dataset_pair_location import dataset_path

def main():
    ap = argparse.ArgumentParser(
        description="Save a one-line PDF timeline of sub-directory mtimes (minute precision)"
    )
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Include nested sub-directories")
    ap.add_argument("-o", "--output", default="subdir_mtime_timeline.jpg",
                    help="PDF filename (default: subdir_mtime_timeline.jpg)")
    args = ap.parse_args()
    dataset = "austin_sailor"
    dir = Path(f"{dataset_path[dataset]}/{dataset}/Jaco_rgb")
    

    subdirs = gather_dirs(dir, args.recursive)
    if not subdirs:
        sys.exit("No sub-directories found")

    # Build (Path, mtime-in-Pacific) pairs, rounded to the minute, then sort
    pairs = sorted(
        (
            (
                p,
                dt.datetime
                  .fromtimestamp(p.stat().st_mtime, PACIFIC)   # convert to PT
                  .replace(second=0, microsecond=0)
            )
            for p in subdirs
        ),
        key=lambda x: x[1],
    )

    names = [p.relative_to(dir).as_posix() for p, _ in pairs]
    times = [t for _, t in pairs]
    y = [0] * len(times)           # single horizontal line

    # --- size guard -------------------------------------------------
    dpi = 150
    fig_width = len(times) * 0.25          # your original rule (inches)
    max_px = 65000                         # Pillow limit

    # shrink dpi if the rasterised width would exceed 65 500 px
    if fig_width * dpi > max_px:
        dpi = int(max_px // fig_width)
        print(f"[info] Down-sampling to {dpi} dpi to stay within {max_px}px width")

    # also hard-cap the physical width so the PDF isn’t gigantic
    fig_width = min(fig_width, 40)
    # ----------------------------------------------------------------

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, len(times) * 0.25), 2.5))

    ax.scatter(times, y, marker="|", s=300)        # vertical ticks
    ax.set_ylim(-1, 1)
    ax.get_yaxis().set_visible(False)

    # Show timezone abbreviation in the tick labels
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M %Z", tz=PACIFIC)
    )
    ax.set_xlabel("Last modified (rounded to minute, Pacific Time)")
    ax.set_title(f"Sub-directory modification timeline in {dir}")

    # Optional annotations (comment out for lots of dirs)
    for xi, yi, label in zip(times, y, names):
        ax.annotate(label, (xi, yi), rotation=90, va="bottom", ha="center",
                    fontsize=7, xytext=(0, 5), textcoords="offset points")

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    plt.savefig(args.output, dpi=dpi, bbox_inches="tight")
    print(f"Saved timeline to {args.output}")


if __name__ == "__main__":
    main()
