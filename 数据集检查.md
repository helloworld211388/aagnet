# 数据集检查记录

## 项目路径
- 项目：`/root/autodl-tmp/project/AAGNet-main`
- 数据集：`/root/autodl-tmp/project/myDatasets`

## 数据集结构
- `aag/samples/*.pkl` — 图数据，每个样本一个文件，key 包括 `graph_face_attr` 等
- `labels/*.json` — 标签文件，结构为 `{"cls": {...}}` （dict，key 为面 id）
- `train.txt` / `val.txt` / `test.txt` — 数据集划分文件
- `aag/graphs.json` — 原始大文件（5.7G），由 `split_graphs.py` 拆分为 samples/

## 检查脚本
`/root/autodl-tmp/project/AAGNet-main/check_labels.py`

用法：
```bash
python3 check_labels.py --dataset /root/autodl-tmp/project/myDatasets
```

检查逻辑：
- 面数量 = `data["graph_face_attr"]` 的长度
- 标签数量 = `labels["cls"]` 的长度

## 检查结果

| 项目 | 数量 |
|------|------|
| train.txt | 20372 |
| val.txt | 4365 |
| test.txt | 4366 |
| 总唯一样本 | 29103 |
| 已检查（有 pkl 文件） | 21876 |
| 面数 == 标签数（匹配） | 21772 |
| 面数 != 标签数（不匹配） | 104 |
| 缺少 label 文件 | 0 |
| 缺少 pkl 文件 | 7227 |

## 不匹配样本（104 个，部分列举）

大多数差值为 ±1，少数差 2~4。

| id | num_faces | num_labels | diff |
|----|-----------|------------|------|
| 375 | 29 | 28 | -1 |
| 557 | 24 | 22 | -2 |
| 1145 | 15 | 13 | -2 |
| 3304 | 14 | 17 | +3 |
| 3953 | 24 | 26 | +2 |
| 6834 | 15 | 11 | -4 |
| ... | ... | ... | ... |

## 缺失 pkl 的根本原因

`graphs.json` 本身被截断，只有 21876 条记录，而 split 文件共有 29103 个 ID。
运行 `split_graphs.py` 输出：
```
Warning: truncated JSON, processed 21876 records.
Done. 21876 files written to ...
```

缺失的 7227 个样本在 graphs.json 中不存在，无法从该文件生成。

## 待解决问题

1. **7227 个缺失 pkl**：graphs.json 截断，需要原始 STEP 文件重新运行 `dataset/AAGExtractor.py` 生成，或从 split 文件中删除这些 ID。
2. **104 个面/标签不匹配**：可在 dataloader 中跳过，或修复标签文件。

## 数据生成流程

```
STEP 文件 (CAD 模型)
    ↓
dataset/AAGExtractor.py  （需要 pythonocc + occwl）
    ↓
aag/graphs.json  （大文件，当前截断）
    ↓
split_graphs.py
    ↓
aag/samples/*.pkl
    ↓
dataloader/base.py  （训练时加载）
```

## 相关脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| split_graphs.py | `AAGNet-main/split_graphs.py` | graphs.json 拆分为 pkl |
| AAGExtractor.py | `AAGNet-main/dataset/AAGExtractor.py` | STEP 文件 → graphs.json |
| main.py | `AAGNet-main/dataset/main.py` | 生成合成 STEP 文件 |
| check_labels.py | `AAGNet-main/check_labels.py` | 验证 pkl/label 一致性 |
| base.py | `AAGNet-main/dataloader/base.py` | 训练时加载 pkl |
