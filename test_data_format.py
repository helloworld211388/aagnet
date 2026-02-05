"""
测试数据格式是否正确
使用方法: python test_data_format.py --dataset_path /path/to/dataset
"""

import argparse
import json
import pathlib
from collections import Counter


def test_graphs_json(dataset_path):
    """测试graphs.json格式"""
    print("=" * 60)
    print("测试 graphs.json 格式")
    print("=" * 60)
    
    graphs_file = pathlib.Path(dataset_path) / 'aag' / 'graphs.json'
    
    if not graphs_file.exists():
        print(f"✗ 文件不存在: {graphs_file}")
        return False
    
    try:
        with open(graphs_file, 'r') as f:
            graphs_data = json.load(f)
        
        print(f"✓ 成功加载 graphs.json")
        print(f"  - 总共 {len(graphs_data)} 个图")
        
        # 检查前3个
        for i, item in enumerate(graphs_data[:3]):
            if len(item) != 2:
                print(f"✗ 格式错误: 应该是 [filename, data] 格式")
                return False
            
            filename, data = item
            print(f"\n  样本 {i+1}: {filename}")
            
            # 检查必需的键
            required_keys = ['graph', 'graph_face_attr', 'graph_edge_attr']
            for key in required_keys:
                if key not in data:
                    print(f"    ✗ 缺少键: {key}")
                    return False
                else:
                    if key == 'graph':
                        num_nodes = data[key].get('num_nodes', 0)
                        print(f"    ✓ {key}: {num_nodes} 个节点")
                    else:
                        print(f"    ✓ {key}: {len(data[key])} 个元素")
        
        print(f"\n✓ graphs.json 格式正确")
        return True
        
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_format(dataset_path):
    """测试标签文件格式"""
    print("\n" + "=" * 60)
    print("测试标签文件格式")
    print("=" * 60)
    
    labels_dir = pathlib.Path(dataset_path) / 'labels'
    
    if not labels_dir.exists():
        print(f"✗ 目录不存在: {labels_dir}")
        return False
    
    # 找到所有标签文件
    label_files = list(labels_dir.glob('*.json'))
    
    if len(label_files) == 0:
        print(f"✗ 没有找到标签文件")
        return False
    
    print(f"✓ 找到 {len(label_files)} 个标签文件")
    
    # 检查前3个
    class_stats = Counter()
    
    for i, label_file in enumerate(label_files[:3]):
        print(f"\n  文件 {i+1}: {label_file.name}")
        
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            
            # 检查格式
            if not isinstance(label_data, dict):
                print(f"    ✗ 格式错误: 应该是字典格式")
                return False
            
            if 'cls' not in label_data:
                print(f"    ✗ 缺少 'cls' 字段")
                return False
            
            cls_data = label_data['cls']
            print(f"    ✓ cls: {len(cls_data)} 个面")
            
            # 统计类别
            for face_id, class_id in cls_data.items():
                class_stats[class_id] += 1
            
            # 检查可选字段
            if 'seg' in label_data:
                print(f"    ✓ seg: {len(label_data['seg'])} 个实例")
            
            if 'bottom' in label_data:
                print(f"    ✓ bottom: {len(label_data['bottom'])} 个面")
            
        except Exception as e:
            print(f"    ✗ 读取失败: {e}")
            return False
    
    # 显示类别统计
    print(f"\n  类别统计 (基于前3个文件):")
    print(f"    - 最小类别: {min(class_stats.keys())}")
    print(f"    - 最大类别: {max(class_stats.keys())}")
    print(f"    - 类别分布:")
    for class_id, count in sorted(class_stats.items()):
        print(f"      类别 {class_id:2d}: {count:3d} 个面")
    
    print(f"\n✓ 标签文件格式正确")
    return True


def test_split_files(dataset_path):
    """测试数据分割文件"""
    print("\n" + "=" * 60)
    print("测试数据分割文件")
    print("=" * 60)
    
    path = pathlib.Path(dataset_path)
    split_files = ['train.txt', 'val.txt', 'test.txt']
    
    all_files = set()
    
    for split_file in split_files:
        file_path = path / split_file
        
        if not file_path.exists():
            print(f"✗ 文件不存在: {split_file}")
            return False
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"✓ {split_file}: {len(lines)} 个文件")
        all_files.update(lines)
    
    print(f"\n  总共 {len(all_files)} 个唯一文件")
    
    return True


def test_consistency(dataset_path):
    """测试数据一致性"""
    print("\n" + "=" * 60)
    print("测试数据一致性")
    print("=" * 60)
    
    path = pathlib.Path(dataset_path)
    
    # 加载graphs.json
    graphs_file = path / 'aag' / 'graphs.json'
    with open(graphs_file, 'r') as f:
        graphs_data = json.load(f)
    graph_files = set([item[0] for item in graphs_data])
    print(f"  graphs.json: {len(graph_files)} 个文件")
    
    # 加载标签文件
    labels_dir = path / 'labels'
    label_files = set([f.stem for f in labels_dir.glob('*.json')])
    print(f"  labels/: {len(label_files)} 个文件")
    
    # 加载分割文件
    split_files = set()
    for split in ['train.txt', 'val.txt', 'test.txt']:
        with open(path / split, 'r') as f:
            split_files.update([line.strip() for line in f if line.strip()])
    print(f"  split files: {len(split_files)} 个文件")
    
    # 检查一致性
    print(f"\n  一致性检查:")
    
    # 检查split文件中的文件是否都在graphs.json中
    missing_in_graphs = split_files - graph_files
    if missing_in_graphs:
        print(f"    ✗ {len(missing_in_graphs)} 个文件在graphs.json中缺失")
        for fn in list(missing_in_graphs)[:5]:
            print(f"        - {fn}")
    else:
        print(f"    ✓ 所有split文件都在graphs.json中")
    
    # 检查split文件中的文件是否都有标签
    missing_labels = split_files - label_files
    if missing_labels:
        print(f"    ✗ {len(missing_labels)} 个文件缺少标签")
        for fn in list(missing_labels)[:5]:
            print(f"        - {fn}")
    else:
        print(f"    ✓ 所有split文件都有标签")
    
    if not missing_in_graphs and not missing_labels:
        print(f"\n✓ 数据一致性检查通过")
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='测试数据格式')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='数据集路径')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("AAGNet 数据格式测试")
    print("=" * 60)
    print(f"数据集路径: {args.dataset_path}\n")
    
    results = []
    
    # 运行所有测试
    results.append(("graphs.json格式", test_graphs_json(args.dataset_path)))
    results.append(("标签文件格式", test_label_format(args.dataset_path)))
    results.append(("数据分割文件", test_split_files(args.dataset_path)))
    results.append(("数据一致性", test_consistency(args.dataset_path)))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！数据格式正确。")
        print("\n下一步:")
        print("  1. 运行完整验证:")
        print(f"     python -m utils.validate_custom27_dataset --dataset_path {args.dataset_path}")
        print("  2. 开始训练:")
        print("     python -m engine.custom27_trainer")
    else:
        print("✗ 部分测试失败，请检查数据格式。")
        print("\n参考文档:")
        print("  - 完整使用指南.md")
        print("  - QUICK_START_27CLASS.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
