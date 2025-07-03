import json
from pathlib import Path

whitelist_path = "/home/guanhuaji/mirage/robot2robot/rendering/paired_images"
dataset        = "kaist"
robots         = ["Panda", "IIWA", "Sawyer", "Jaco", "Kinova3"]

whitelists = []

for robot in robots:
    whitelist_file = Path(whitelist_path) / dataset / robot / "whitelist.json"
    try:
        with open(whitelist_file, "r") as f:
            data = json.load(f)

        # Each file’s key *should* match the robot name.
        # Fallback to “first value found” if the key is missing.
        frames = data.get(robot)
        if frames is None:
            # e.g. {"frames": [...] }  or  {"IIWA": [...] } for wrong robot
            frames = next(iter(data.values()))  

        whitelists.append(set(frames))

    except FileNotFoundError:
        print(f"[WARN] {whitelist_file} not found – skipping.")
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {whitelist_file} – skipping.")

# ---- Intersection across all robots ----------------------------------------
if whitelists:
    common_indices = sorted(set.intersection(*whitelists))
    print(f"Common indices ({len(common_indices)} total): {common_indices}")
else:
    common_indices = []
    print("No whitelists loaded; nothing to intersect.")

# ---- (Optional) save result -------------------------------------------------
out_file = Path(whitelist_path) / dataset / "intersection_whitelist.json"
out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, "w") as f:
    json.dump({"intersection": common_indices}, f, indent=2)
print(f"Saved intersection to {out_file}")
