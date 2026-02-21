"""
Generate attr_stat.json for Custom27 dataset using streaming JSON parse.
Reads graphs.json in chunks to avoid OOM on large files.
"""
import json
import numpy as np
import pathlib

GRAPHS_PATH = "/root/autodl-tmp/project/myDatasets/aag/graphs.json"
OUT_PATH = "/root/autodl-tmp/project/myDatasets/aag/attr_stat.json"
TRAIN_SPLIT = "/root/autodl-tmp/project/myDatasets/train.txt"

# Load train split filenames
with open(TRAIN_SPLIT, "r") as f:
    train_set = set(line.strip() for line in f if line.strip())

print(f"Train set size: {len(train_set)}")

face_attrs = []
edge_attrs = []
count = 0

print("Streaming graphs.json ...")
with open(GRAPHS_PATH, "r") as f:
    # The file is a JSON array: [ [fn, data], [fn, data], ... ]
    # Use ijson if available, else manual bracket parsing
    try:
        import ijson
        parser = ijson.items(f, "item")
        try:
            for item in parser:
                fn, data = item[0], item[1]
                if fn not in train_set:
                    continue
                fa = np.array(data["graph_face_attr"], dtype=np.float32)
                ea = np.array(data["graph_edge_attr"], dtype=np.float32)
                face_attrs.append(fa)
                edge_attrs.append(ea)
                count += 1
                if count % 1000 == 0:
                    print(f"  processed {count} train graphs...")
        except ijson.common.IncompleteJSONError:
            print(f"  Warning: graphs.json is truncated, collected {count} train graphs so far.")
    except ImportError:
        print("ijson not found, falling back to full load (may be slow)...")
        f.seek(0)
        dataset = json.load(f)
        for item in dataset:
            fn, data = item[0], item[1]
            if fn not in train_set:
                continue
            fa = np.array(data["graph_face_attr"], dtype=np.float32)
            ea = np.array(data["graph_edge_attr"], dtype=np.float32)
            face_attrs.append(fa)
            edge_attrs.append(ea)
            count += 1

print(f"Collected {count} train graphs. Computing statistics...")

all_face = np.concatenate(face_attrs, axis=0)  # (N_faces, D_face)
all_edge = np.concatenate(edge_attrs, axis=0)  # (N_edges, D_edge)

stat = {
    "mean_face_attr": all_face.mean(axis=0).tolist(),
    "std_face_attr": all_face.std(axis=0).tolist(),
    "mean_edge_attr": all_edge.mean(axis=0).tolist(),
    "std_edge_attr": all_edge.std(axis=0).tolist(),
}

with open(OUT_PATH, "w") as f:
    json.dump(stat, f)

print(f"Saved attr_stat.json to {OUT_PATH}")
print(f"face_attr dim: {all_face.shape[1]}, edge_attr dim: {all_edge.shape[1]}")
