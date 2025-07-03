#!/usr/bin/env python3
"""
save_numeric_files.py – List numeric file names, sort them, and save as JSON.

Examples
--------
# Non-recursive, write whitelist.json in the same dir
python /home/guanhuaji/mirage/robot2robot/rendering/update_whitelist.py /home/abinayadinesh/paired_images/taco_play/source_robot_states/Kinova3/offsets
python /home/guanhuaji/mirage/robot2robot/rendering/update_whitelist.py /home/abinayadinesh/paired_images/taco_play/source_robot_states/Kinova3/offsets

# Recursive, pick a different output file and key
python save_numeric_files.py /path/to/dir -r -o my_list.json --key MyRobot
"""
from pathlib import Path
import argparse, json, sys

RANGE_START, RANGE_END = 0, 3241            # inclusive

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build whitelist & blacklist for numeric file names."
    )
    ap.add_argument("directory", help="Directory to scan")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into sub-directories")
    ap.add_argument("-O", "--outdir", default=".", metavar="DIR",
                    help="Where to put the JSON files (default: current dir)")
    ap.add_argument("--key", default="Kinova3",
                    help='Dictionary key to use (default: "Kinova3")')
    return ap.parse_args()

def collect_numbers(root: Path, recursive: bool) -> set[int]:
    walker = root.rglob("*") if recursive else root.iterdir()
    found = set()
    for p in walker:
        if p.is_file():
            try:
                n = int(p.stem)
            except ValueError:
                continue                         # ignore non-numeric stems
            if RANGE_START <= n <= RANGE_END:
                found.add(n)
    return found

def write_json(path: Path, key: str, data: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({key: data}, f, indent=2)

def main() -> None:
    args = parse_args()
    root = Path(args.directory).expanduser().resolve()

    if not root.is_dir():
        print(f"❌  {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    present = collect_numbers(root, args.recursive)
    expected = set(range(RANGE_START, RANGE_END + 1))

    whitelist = sorted(present)
    blacklist = sorted(expected - present)

    outdir = Path(args.outdir).expanduser().resolve()
    write_json(outdir / "whitelist.json", args.key, whitelist)
    write_json(outdir / "blacklist.json", args.key, blacklist)

    print(f"✅  {len(whitelist)} present, {len(blacklist)} missing.\n"
          f"   whitelist.json → {outdir / 'whitelist.json'}\n"
          f"   blacklist.json → {outdir / 'blacklist.json'}")

if __name__ == "__main__":
    main()