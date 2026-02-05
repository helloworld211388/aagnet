"""
验证Custom27数据集格式的工具脚本
使用方法: python -m utils.validate_custom27_dataset --dataset_path ../your_dataset_path
"""

import argparse
import pathlib
import json
import os
from collections import Counter


def validate_dataset(dataset_path):
    """验证数据集格式是否正确"""
    
    path = pathlib.Path(dataset_path)
    print(f"正在验证数据集: {path}")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. 检查必需的文件夹
    print("\n[1] 检查目录结构...")
    required_dirs = ['aag', 'labels']
    optional_dirs = ['steps']
    
    for dir_name in required_dirs:
        dir_path = path / dir_name
        if not dir_path.exists():
            errors.append(f"缺少必需文件夹: {dir_name}/")
        else:
            print(f"  ✓ 找到文件夹: {dir_name}/")
    
    for dir_name in optional_dirs:
        dir_path = path / dir_name
        if dir_path.exists():
            print(f"  ✓ 找到可选文件夹: {dir_name}/")
        else:
            warnings.append(f"未找到可选文件夹: {dir_name}/")
    
    # 2. 检查数据分割文件
    print("\n[2] 检查数据分割文件...")
    required_files = ['train.txt', 'val.txt', 'test.txt']
    split_files = {}
    
    for file_name in required_files:
        file_path = path / file_name
        if not file_path.exists():
            errors.append(f"缺少必需文件: {file_name}")
        else:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                split_files[file_name] = lines
                print(f"  ✓ {file_name}: {len(lines)} 个文件")
    
    if len(errors) > 0:
        print("\n" + "=" * 60)
        print("发现严重错误，无法继续验证:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    # 3. 检查文件一致性
    print("\n[3] 检查文件一致性...")
    all_files = set()
    for split_name, files in split_files.items():
        all_files.update(files)
    
    print(f"  总共 {len(all_files)} 个唯一文件")
    
    # 检查重复
    all_files_list = []
    for files in split_files.values():
        all_files_list.extend(files)
    duplicates = [item for item, count in Counter(all_files_list).items() if count > 1]
    if duplicates:
        warnings.append(f"发现 {len(duplicates)} 个重复文件在不同分割中")
        print(f"  ⚠ 警告: {len(duplicates)} 个文件在多个分割中出现")
    else:
        print(f"  ✓ 没有重复文件")
    
    # 4. 检查AAG文件 (graphs.json)
    print("\n[4] 检查AAG文件...")
    aag_dir = path / 'aag'
    graphs_file = aag_dir / 'graphs.json'
    
    if not graphs_file.exists():
        errors.append("缺少 aag/graphs.json 文件")
        print(f"  ✗ 缺少 aag/graphs.json 文件")
    else:
        print(f"  ✓ 找到 aag/graphs.json 文件")
        try:
            with open(graphs_file, 'r') as f:
                graphs_data = json.load(f)
            print(f"  ✓ graphs.json 包含 {len(graphs_data)} 个图")
            
            # 检查graphs.json中的文件是否与split文件匹配
            graph_files = set([item[0] for item in graphs_data])
            missing_in_graphs = all_files - graph_files
            if missing_in_graphs:
                warnings.append(f"{len(missing_in_graphs)} 个文件在graphs.json中缺失")
                print(f"  ⚠ {len(missing_in_graphs)} 个文件在graphs.json中缺失")
                if len(missing_in_graphs) <= 5:
                    for fn in list(missing_in_graphs)[:5]:
                        print(f"      - {fn}")
            else:
                print(f"  ✓ 所有文件都在graphs.json中")
        except Exception as e:
            errors.append(f"读取graphs.json时出错: {str(e)}")
            print(f"  ✗ 读取graphs.json时出错: {str(e)}")
    
    # 5. 检查标签文件
    print("\n[5] 检查标签文件...")
    labels_dir = path / 'labels'
    label_files = set([f.stem for f in labels_dir.glob('*.json')])
    print(f"  找到 {len(label_files)} 个标签文件")
    
    missing_labels = all_files - label_files
    if missing_labels:
        errors.append(f"{len(missing_labels)} 个文件缺少标签数据")
        print(f"  ✗ {len(missing_labels)} 个文件缺少标签数据")
        if len(missing_labels) <= 5:
            for fn in list(missing_labels)[:5]:
                print(f"      - {fn}")
    else:
        print(f"  ✓ 所有文件都有标签数据")
    
    # 6. 验证标签格式和类别范围
    print("\n[6] 验证标签格式...")
    sample_files = list(all_files)[:5]  # 检查前5个文件
    label_stats = {
        'min_label': float('inf'),
        'max_label': float('-inf'),
        'total_faces': 0,
        'class_counts': Counter()
    }
    
    for fn in sample_files:
        label_file = labels_dir / f"{fn}.json"
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    labels_data = json.load(f)
                
                # 新格式: {"cls": {"0": 27, "1": 27, ...}, "seg": [...], "bottom": {...}}
                if isinstance(labels_data, dict) and 'cls' in labels_data:
                    cls_labels = labels_data['cls']
                    num_faces = len(cls_labels)
                    
                    # 统计标签
                    for face_id_str, class_id in cls_labels.items():
                        label_stats['min_label'] = min(label_stats['min_label'], class_id)
                        label_stats['max_label'] = max(label_stats['max_label'], class_id)
                        label_stats['class_counts'][class_id] += 1
                        label_stats['total_faces'] += 1
                    
                    print(f"  ✓ {fn}: {num_faces} 个面")
                    
                    # 检查是否有seg和bottom字段
                    if 'seg' in labels_data:
                        print(f"      - 包含实例分割信息: {len(labels_data['seg'])} 个实例")
                    if 'bottom' in labels_data:
                        print(f"      - 包含底面标签")
                else:
                    warnings.append(f"文件 {fn} 的标签格式不符合预期 (应包含'cls'字段)")
                    print(f"  ⚠ {fn}: 标签格式不符合预期")
                    continue
                
            except Exception as e:
                errors.append(f"读取标签文件 {fn} 时出错: {str(e)}")
                print(f"  ✗ {fn}: 读取失败 - {str(e)}")
    
    if label_stats['total_faces'] > 0:
        print(f"\n  标签统计 (基于 {len(sample_files)} 个样本):")
        print(f"    - 最小标签值: {label_stats['min_label']}")
        print(f"    - 最大标签值: {label_stats['max_label']}")
        print(f"    - 总面数: {label_stats['total_faces']}")
        
        if label_stats['min_label'] < 0 or label_stats['max_label'] > 29:
            warnings.append(f"标签值超出常见范围 [0, 29]: [{label_stats['min_label']}, {label_stats['max_label']}]")
            print(f"  ⚠ 标签值范围: [{label_stats['min_label']}, {label_stats['max_label']}]")
            print(f"    (如果你的类别数不是30，请修改dataloader/custom27.py中的num_classes)")
        else:
            print(f"  ✓ 标签值在范围内 [0, {label_stats['max_label']}]")
        
        print(f"\n  类别分布 (前10个):")
        for label, count in label_stats['class_counts'].most_common(10):
            print(f"    类别 {label:2d}: {count:5d} 个面")
    
    # 7. 验证graphs.json格式
    print("\n[7] 验证graphs.json格式...")
    if graphs_file.exists():
        try:
            with open(graphs_file, 'r') as f:
                graphs_data = json.load(f)
            
            # 检查前3个样本
            for i, (fn, graph_data) in enumerate(graphs_data[:3]):
                # 检查必需的键
                required_keys = ['graph', 'graph_face_attr', 'graph_edge_attr']
                missing_keys = [key for key in required_keys if key not in graph_data]
                
                if missing_keys:
                    errors.append(f"图数据 {fn} 缺少键: {missing_keys}")
                    print(f"  ✗ {fn}: 缺少键 {missing_keys}")
                else:
                    num_nodes = graph_data['graph']['num_nodes']
                    num_faces = len(graph_data['graph_face_attr'])
                    print(f"  ✓ {fn}: {num_nodes} 个节点, {num_faces} 个面属性")
                    
        except Exception as e:
            errors.append(f"验证graphs.json格式时出错: {str(e)}")
            print(f"  ✗ 验证失败 - {str(e)}")
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结:")
    print("=" * 60)
    
    if len(errors) == 0 and len(warnings) == 0:
        print("✓ 数据集格式完全正确！可以开始训练。")
        return True
    else:
        if len(errors) > 0:
            print(f"\n发现 {len(errors)} 个错误:")
            for error in errors:
                print(f"  ✗ {error}")
        
        if len(warnings) > 0:
            print(f"\n发现 {len(warnings)} 个警告:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        
        if len(errors) > 0:
            print("\n请修复错误后再开始训练。")
            return False
        else:
            print("\n警告不影响训练，但建议检查。")
            return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证Custom27数据集格式')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集根目录路径')
    args = parser.parse_args()
    
    success = validate_dataset(args.dataset_path)
    
    if success:
        print("\n" + "=" * 60)
        print("下一步:")
        print("  1. 修改 engine/custom27_trainer.py 中的数据集路径")
        print("  2. 运行训练: python -m engine.custom27_trainer")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("请先修复数据集问题")
        print("=" * 60)
