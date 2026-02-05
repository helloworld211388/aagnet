"""
快速测试脚本：验证环境和配置是否正确
使用方法: python test_setup.py --dataset_path /path/to/your/dataset
"""

import argparse
import sys
import os

def test_imports():
    """测试必需的包是否已安装"""
    print("=" * 60)
    print("测试1: 检查Python包")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('dgl', 'DGL'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('torchmetrics', 'TorchMetrics'),
        ('torch_ema', 'torch-ema'),
        ('wandb', 'wandb'),
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name} 已安装")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        print(f"  conda install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ 所有必需的包都已安装")
        return True


def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "=" * 60)
    print("测试2: 检查CUDA")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            return True
        else:
            print(f"  ⚠ CUDA 不可用，将使用CPU训练（速度会很慢）")
            return False
    except Exception as e:
        print(f"  ✗ 检查CUDA时出错: {e}")
        return False


def test_custom_files():
    """测试自定义文件是否存在"""
    print("\n" + "=" * 60)
    print("测试3: 检查自定义文件")
    print("=" * 60)
    
    required_files = [
        'dataloader/custom27.py',
        'engine/custom27_trainer.py',
        'engine/custom27_test.py',
        'utils/validate_custom27_dataset.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} 不存在")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n缺少以下文件: {', '.join(missing_files)}")
        return False
    else:
        print("\n✓ 所有自定义文件都存在")
        return True


def test_dataset(dataset_path):
    """测试数据集路径和基本结构"""
    print("\n" + "=" * 60)
    print("测试4: 检查数据集")
    print("=" * 60)
    
    if not dataset_path:
        print("  ⚠ 未提供数据集路径，跳过此测试")
        print("  使用 --dataset_path 参数指定数据集路径")
        return None
    
    if not os.path.exists(dataset_path):
        print(f"  ✗ 数据集路径不存在: {dataset_path}")
        return False
    
    print(f"  ✓ 数据集路径存在: {dataset_path}")
    
    # 检查必需的子目录
    required_dirs = ['aag', 'labels']
    required_files = ['train.txt', 'val.txt', 'test.txt']
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_name}/ 存在")
        else:
            print(f"  ✗ {dir_name}/ 不存在")
            all_exist = False
    
    for file_name in required_files:
        file_path = os.path.join(dataset_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                num_lines = len([line for line in f if line.strip()])
            print(f"  ✓ {file_name} 存在 ({num_lines} 个文件)")
        else:
            print(f"  ✗ {file_name} 不存在")
            all_exist = False
    
    if all_exist:
        print("\n✓ 数据集基本结构正确")
        print("\n建议运行完整验证:")
        print(f"  python -m utils.validate_custom27_dataset --dataset_path {dataset_path}")
        return True
    else:
        print("\n✗ 数据集结构不完整")
        return False


def test_dataloader(dataset_path):
    """测试数据加载器是否能正常工作"""
    print("\n" + "=" * 60)
    print("测试5: 测试数据加载器")
    print("=" * 60)
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("  ⚠ 数据集路径无效，跳过此测试")
        return None
    
    try:
        from dataloader.custom27 import Custom27Dataset
        
        print("  正在加载数据集（这可能需要一些时间）...")
        dataset = Custom27Dataset(
            root_dir=dataset_path,
            split='train',
            center_and_scale=False,
            normalize=True,
            random_rotate=False,
            num_train_data=10,  # 只加载10个样本进行测试
            num_threads=1
        )
        
        print(f"  ✓ 成功加载 {len(dataset)} 个样本")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"  ✓ 成功获取样本")
        print(f"    - 文件名: {sample['filename']}")
        print(f"    - 节点数: {sample['graph'].num_nodes()}")
        print(f"    - 边数: {sample['graph'].num_edges()}")
        
        # 测试dataloader
        dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"  ✓ 成功创建DataLoader")
        print(f"    - 批次大小: {len(batch['filename'])}")
        
        print("\n✓ 数据加载器工作正常")
        return True
        
    except Exception as e:
        print(f"  ✗ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='测试AAGNet 27类配置')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='数据集路径（可选）')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("AAGNet 27类配置测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("Python包", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("自定义文件", test_custom_files()))
    results.append(("数据集结构", test_dataset(args.dataset_path)))
    results.append(("数据加载器", test_dataloader(args.dataset_path)))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        else:
            status = "⚠ 跳过"
        print(f"  {name:20s}: {status}")
    
    # 给出建议
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    
    all_passed = all(r in [True, None] for _, r in results)
    critical_passed = results[0][1] and results[2][1]  # Python包和自定义文件
    
    if all_passed and results[3][1] is True:
        print("✓ 所有测试通过！你可以开始训练了。")
        print("\n下一步:")
        print("  1. 运行完整数据集验证:")
        print(f"     python -m utils.validate_custom27_dataset --dataset_path {args.dataset_path}")
        print("  2. 修改训练配置:")
        print("     编辑 engine/custom27_trainer.py 第47行")
        print("  3. 开始训练:")
        print("     python -m engine.custom27_trainer")
    elif critical_passed:
        print("⚠ 基本环境配置正确，但需要准备数据集。")
        print("\n下一步:")
        print("  1. 准备数据集（参考 CUSTOM27_USAGE.md）")
        print("  2. 运行数据集验证:")
        print("     python -m utils.validate_custom27_dataset --dataset_path /path/to/dataset")
        print("  3. 重新运行此测试:")
        print("     python test_setup.py --dataset_path /path/to/dataset")
    else:
        print("✗ 发现问题，请先解决上述失败的测试。")
        print("\n参考文档:")
        print("  - 快速开始: QUICK_START_27CLASS.md")
        print("  - 详细指南: CUSTOM27_USAGE.md")
        print("  - 中文文档: README_27CLASS_CN.md")


if __name__ == '__main__':
    main()
