import pathlib
import json
import os

import torch
import dgl
import numpy as np

from .base import BaseDataset
from utils.data_utils import load_one_graph


class Custom27Dataset(BaseDataset):
    """
    Custom dataset with 30 classes:
    - 27 machining features (0-26)
    - 3 additional classes (27-29)
    
    Label format from JSON:
    {
        "cls": {"0": 27, "1": 27, "2": 26, ...},  # face_id -> class_id
        "seg": [[7], [9, 11], [10], ...],          # instance segmentation (optional)
        "bottom": {"0": 0, "1": 0, ...}            # bottom face labels (optional)
    }
    """
    
    @staticmethod
    def num_classes():
        return 30  # 0-29
    
    @staticmethod
    def get_feature_names():
        return [
            'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', 
            '6sides_passage', 'triangular_through_slot', 'rectangular_through_slot', 
            'circular_through_slot', 'rectangular_through_step', '2sides_through_step', 
            'slanted_through_step', 'Oring', 'blind_hole', 'triangular_pocket', 
            'rectangular_pocket', '6sides_pocket', 'circular_end_pocket', 
            'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
            'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 
            'round', 'plane', 'cylinder', 'cone',
            'class_27', 'class_28', 'class_29'
        ]
    
    def __init__(self, 
                 root_dir, 
                 graphs=None, 
                 split="train", 
                 normalize=True, 
                 center_and_scale=True, 
                 random_rotate=False, 
                 num_train_data=-1, 
                 transform=None, 
                 num_threads=0):
        """
        Load the Custom 30-class Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            graphs (list, optional): List of graph data from graphs.json.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
            num_train_data (int, optional): Number of training examples to use. Defaults to -1 (all training examples will be used).
            transform (callable, optional): Transformation to apply to the data.
            num_threads (int, optional): Number of threads to use for data loading. Defaults to 0.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        self.transform = transform
        self.random_rotate = random_rotate
        assert split in ("train", "val", "test")

        # Load data partition from train.txt, val.txt, test.txt file
        split_file = path.joinpath(f"{split}.txt")
        if split_file.exists():
            with open(str(split_file), 'r') as f:
                split_filelist = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")

        if split == "train" and num_train_data > 0:
            split_filelist = split_filelist[:num_train_data]

        # Load graphs
        print(f"Loading {split} data...")
        split_filelist = set(split_filelist)
        graph_path = path.joinpath("aag")
        self.load_graphs(graph_path, graphs, split_filelist, center_and_scale, normalize, num_threads)
        print("Done loading {} files".format(len(self.data)))

    def _collate(self, batch):
        """
        Collate a batch of data samples together into a single batch.

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data.
        """
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph,
                "filename": batched_filenames}
    
    def load_one_graph(self, fn, data):
        """
        Load the data for a single file.

        Args:
            fn (str): Filename.
            data (dict): Data for the file from graphs.json.

        Returns:
            dict: Data for the file.
        """
        # Load the graph using base class method
        sample = load_one_graph(fn, data)
        num_faces = sample['graph'].num_nodes()
        
        # Load the label file
        label_file = self.path.joinpath("labels").joinpath(fn + ".json")
        if not label_file.exists():
            print(f"Warning: Label file not found for {fn}, skipping...")
            return None
            
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        
        # Parse label format: {"cls": {"0": 27, "1": 27, ...}, "seg": [...], "bottom": {...}}
        if not isinstance(labels_data, dict) or "cls" not in labels_data:
            print(f"Warning: Invalid label format for {fn}, skipping...")
            return None
        
        cls_labels = labels_data["cls"]
        
        # Convert cls labels to array
        face_labels = np.zeros(num_faces, dtype=np.int32)
        for face_id_str, class_id in cls_labels.items():
            face_id = int(face_id_str)
            if face_id < num_faces:
                face_labels[face_id] = class_id
            else:
                print(f"Warning: Face ID {face_id} out of range for {fn} (num_faces={num_faces})")
        
        # Verify number of labeled faces matches
        if len(cls_labels) != num_faces:
            print(f"Warning: Number of labels ({len(cls_labels)}) != number of faces ({num_faces}) for {fn}")
        
        sample["graph"].ndata["y"] = torch.tensor(face_labels).long()
        return sample


if __name__ == '__main__':
    dataset = Custom27Dataset(root_dir='../your_dataset_path', split='train', 
                               center_and_scale=True, normalize=False)
    print(dataset[0]["graph"].ndata["y"])
