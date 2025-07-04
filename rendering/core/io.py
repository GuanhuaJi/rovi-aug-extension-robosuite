from contextlib import contextmanager
import json
import os
import pathlib
import portalocker

@contextmanager
def locked_json(path: pathlib.Path, mode="r+", default=lambda: {}):
    # 1️⃣ acquire an **exclusive lock**
    with portalocker.Lock(str(path), mode, timeout=30) as fp:    # ← blocks here
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            data = default()
        yield data                       # 🔒  work with the dict while locked
        fp.seek(0), fp.truncate()        # 2️⃣ rewind
        json.dump(data, fp, indent=2)    # 3️⃣ write
        fp.flush(), os.fsync(fp.fileno())  # 4️⃣ durability
    # ➡ lock is released automatically

def atomic_write_json(obj, path: pathlib.Path):
    with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, mode="w") as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())           # make sure it’s on disk
    os.replace(tmp.name, path) 