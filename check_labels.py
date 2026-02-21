"""
检查数据集中面数量和标签数量是否匹配
用法: python check_labels.py [--dataset PATH]
"""
import os
import json
import pickle
import argparse
from pathlib import Path


def check_dataset(dataset_path):
    path = Path(dataset_path)
    samples_dir = path / "aag" / "samples"
    labels_dir = path / "labels"

    # Collect all file IDs from split files
    all_ids = set()
    for split in ["train", "val", "test"]:
        split_file = path / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            all_ids.update(ids)
            print(f"  {split}.txt: {len(ids)} samples")

    print(f"\nTotal unique samples: {len(all_ids)}")

    match = 0
    mismatch = 0
    missing_label = 0
    missing_graph = 0
    mismatch_details = []

    for fid in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x):
        pkl_path = samples_dir / (fid + ".pkl")
        if not pkl_path.exists():
            missing_graph += 1
            continue

        with open(pkl_path, "rb") as pf:
            data = pickle.load(pf)

        num_faces = len(data["graph_face_attr"])

        label_file = labels_dir / (fid + ".json")
        if not label_file.exists():
            missing_label += 1
            continue

        with open(label_file) as f:
            labels_data = json.load(f)

        num_labels = len(labels_data["cls"])

        if num_labels == num_faces:
            match += 1
        else:
            mismatch += 1
            mismatch_details.append((fid, num_faces, num_labels))

    total_checked = match + mismatch
    print(f"\nResults:")
    print(f"  Checked                  : {total_checked}")
    print(f"  Match   (faces == labels): {match}")
    print(f"  Mismatch(faces != labels): {mismatch}")
    print(f"  Missing label file       : {missing_label}")
    print(f"  Missing graph file       : {missing_graph}")

    if mismatch_details:
        print(f"\nMismatch details (first 50):")
        print(f"  {'id':<15} {'num_faces':>10} {'num_labels':>10} {'diff':>8}")
        print(f"  {'-'*45}")
        for fid, nf, nl in mismatch_details[:50]:
            print(f"  {fid:<15} {nf:>10} {nl:>10} {nl-nf:>+8}")
        if len(mismatch_details) > 50:
            print(f"  ... and {len(mismatch_details)-50} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/root/autodl-tmp/project/myDatasets")
    args = parser.parse_args()
    check_dataset(args.dataset)
