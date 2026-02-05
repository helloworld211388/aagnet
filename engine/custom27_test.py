import os
import time
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassJaccardIndex)
import wandb

from dataloader.custom27 import Custom27Dataset
from models.segmentors import AAGNetSegmentor
from utils.misc import seed_torch, init_logger, print_num_params


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_API_KEY"] = '##################'
    os.environ["WANDB_MODE"] = "offline"
    
    # start a new wandb run to track this script
    dataset_name = "Custom27"
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
            project="aagnet_" + dataset_name + "_test",
            config={
                "edge_attr_dim": 12,
                "node_attr_dim": 10,
                "edge_attr_emb": 64,
                "node_attr_emb": 64,
                "edge_grid_dim": 0, 
                "node_grid_dim": 7,
                "edge_grid_emb": 0, 
                "node_grid_emb": 64,
                "num_layers": 3,
                "delta": 2,
                "mlp_ratio": 2,
                "drop": 0.25, 
                "drop_path": 0.25,
                "head_hidden_dim": 64,
                "conv_on_edge": False,
                "use_uv_gird": True,
                "use_edge_attr": True,
                "use_face_attr": True,

                "seed": 42,
                "device": 'cuda',
                "architecture": "AAGNetGraphEncoder",
                "dataset": "../your_dataset_path",  # CHANGE THIS to your dataset path
                "batch_size": 256,
                "weight_path": "output/your_weight.pth",  # CHANGE THIS to your trained weight path
                }
        )
    
    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    dataset = wandb.config['dataset']
    
    n_classes = Custom27Dataset.num_classes()
    print(f"Number of classes: {n_classes}")
    print(f"Feature names: {Custom27Dataset.get_feature_names()}")

    # Build model
    model = AAGNetSegmentor(num_classes=n_classes,
                            arch=wandb.config['architecture'],
                            edge_attr_dim=wandb.config['edge_attr_dim'], 
                            node_attr_dim=wandb.config['node_attr_dim'], 
                            edge_attr_emb=wandb.config['edge_attr_emb'], 
                            node_attr_emb=wandb.config['node_attr_emb'],
                            edge_grid_dim=wandb.config['edge_grid_dim'], 
                            node_grid_dim=wandb.config['node_grid_dim'], 
                            edge_grid_emb=wandb.config['edge_grid_emb'], 
                            node_grid_emb=wandb.config['node_grid_emb'], 
                            num_layers=wandb.config['num_layers'], 
                            delta=wandb.config['delta'], 
                            mlp_ratio=wandb.config['mlp_ratio'], 
                            drop=wandb.config['drop'], 
                            drop_path=wandb.config['drop_path'], 
                            head_hidden_dim=wandb.config['head_hidden_dim'],
                            conv_on_edge=wandb.config['conv_on_edge'],
                            use_uv_gird=wandb.config['use_uv_gird'],
                            use_edge_attr=wandb.config['use_edge_attr'],
                            use_face_attr=wandb.config['use_face_attr'],)
    model = model.to(device)
    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    # Load trained weights
    print(f"Loading weights from {wandb.config['weight_path']}")
    model_param = torch.load(wandb.config['weight_path'], map_location=device)
    model.load_state_dict(model_param)
    print("Weights loaded successfully!")

    # Load test dataset
    test_dataset = Custom27Dataset(root_dir=dataset, split='test', 
                                    center_and_scale=False, normalize=True, random_rotate=False,
                                    num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], 
                                               shuffle=False, drop_last=False, pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    save_path = 'output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, 'test_' + time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'test_log.txt'))

    with torch.no_grad():
        logger.info(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []
        
        # For per-class metrics
        per_class_correct = torch.zeros(n_classes).to(device)
        per_class_total = torch.zeros(n_classes).to(device)
        
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            seg_label = graphs.ndata["y"]
            
            # Forward pass
            seg_pred = model(graphs)
            loss = seg_loss(seg_pred, seg_label)
            test_losses.append(loss.item())
            
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            
            # Calculate per-class accuracy
            pred_labels = seg_pred.argmax(dim=1)
            for c in range(n_classes):
                mask = seg_label == c
                per_class_correct[c] += (pred_labels[mask] == seg_label[mask]).sum()
                per_class_total[c] += mask.sum()
        
        # Overall metrics
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        
        logger.info(f'========== Overall Test Results ==========')
        logger.info(f'test_loss: {mean_test_loss:.4f}')
        logger.info(f'test_seg_acc: {mean_test_seg_acc:.4f}')
        logger.info(f'test_seg_iou: {mean_test_seg_iou:.4f}')
        
        # Per-class metrics
        logger.info(f'\n========== Per-Class Accuracy ==========')
        feature_names = Custom27Dataset.get_feature_names()
        for c in range(n_classes):
            if per_class_total[c] > 0:
                class_acc = (per_class_correct[c] / per_class_total[c]).item()
                logger.info(f'{feature_names[c]:30s} (class {c:2d}): {class_acc:.4f} ({int(per_class_total[c])} samples)')
            else:
                logger.info(f'{feature_names[c]:30s} (class {c:2d}): N/A (0 samples)')
        
        # Log to wandb
        wandb.log({
            'test_loss': mean_test_loss, 
            'test_seg_acc': mean_test_seg_acc, 
            'test_seg_iou': mean_test_seg_iou
        })
        
        print(f"\n========== Test Complete ==========")
        print(f"Overall Accuracy: {mean_test_seg_acc:.4f}")
        print(f"Overall IoU: {mean_test_seg_iou:.4f}")
        print(f"Results saved to: {save_path}")
