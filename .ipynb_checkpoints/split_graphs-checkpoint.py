"""
Split graphs.json into per-sample .pkl files under aag/samples/
This avoids loading the entire 5.7G file into memory at once.
"""
import json
import pickle
import pathlib
import ijson

GRAPHS_PATH = "/root/autodl-tmp/project/myDatasets/aag/graphs.json"
OUT_DIR = pathlib.Path("/root/autodl-tmp/project/myDatasets/aag/samples")
OUT_DIR.mkdir(exist_ok=True)

count = 0
skipped = 0

print("Splitting graphs.json into per-sample pkl files...")
with open(GRAPHS_PATH, "rb") as f:
    parser = ijson.items(f, "item")
    try:
        for item in parser:
            fn, data = item[0], item[1]
            out_path = OUT_DIR / (fn + ".pkl")
            if not out_path.exists():
                with open(out_path, "wb") as pf:
                    pickle.dump(data, pf, protocol=4)
            count += 1
            if count % 2000 == 0:
                print(f"  {count} done...")
    except ijson.common.IncompleteJSONError:
        print(f"  Warning: truncated JSON, processed {count} records.")

print(f"Done. {count} files written to {OUT_DIR}")
