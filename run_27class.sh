#!/bin/bash
# AAGNet 27类数据集训练脚本
# 使用方法: bash run_27class.sh [command] [options]

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "AAGNet 27类数据集训练工具"
    echo ""
    echo "使用方法:"
    echo "  bash run_27class.sh [command] [options]"
    echo ""
    echo "命令:"
    echo "  test          测试环境配置"
    echo "  validate      验证数据集格式"
    echo "  train         开始训练"
    echo "  test-model    测试训练好的模型"
    echo "  help          显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --dataset PATH    数据集路径"
    echo "  --weight PATH     模型权重路径（用于test-model）"
    echo ""
    echo "示例:"
    echo "  bash run_27class.sh test --dataset /path/to/dataset"
    echo "  bash run_27class.sh validate --dataset /path/to/dataset"
    echo "  bash run_27class.sh train"
    echo "  bash run_27class.sh test-model --weight output/xxx/weight.pth"
}

# 测试环境
test_env() {
    print_info "测试环境配置..."
    python test_setup.py "$@"
}

# 验证数据集
validate_dataset() {
    if [ -z "$DATASET_PATH" ]; then
        print_error "请使用 --dataset 指定数据集路径"
        exit 1
    fi
    
    print_info "验证数据集: $DATASET_PATH"
    python -m utils.validate_custom27_dataset --dataset_path "$DATASET_PATH"
}

# 训练模型
train_model() {
    print_info "开始训练..."
    print_warn "请确保已在 engine/custom27_trainer.py 中配置了正确的数据集路径"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m engine.custom27_trainer
    else
        print_info "训练已取消"
    fi
}

# 测试模型
test_model() {
    if [ -z "$WEIGHT_PATH" ]; then
        print_error "请使用 --weight 指定模型权重路径"
        exit 1
    fi
    
    print_info "测试模型: $WEIGHT_PATH"
    print_warn "请确保已在 engine/custom27_test.py 中配置了正确的路径"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m engine.custom27_test
    else
        print_info "测试已取消"
    fi
}

# 解析参数
COMMAND=""
DATASET_PATH=""
WEIGHT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        test|validate|train|test-model|help)
            COMMAND="$1"
            shift
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --weight)
            WEIGHT_PATH="$2"
            shift 2
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行命令
case $COMMAND in
    test)
        if [ -n "$DATASET_PATH" ]; then
            test_env --dataset_path "$DATASET_PATH"
        else
            test_env
        fi
        ;;
    validate)
        validate_dataset
        ;;
    train)
        train_model
        ;;
    test-model)
        test_model
        ;;
    help|"")
        show_help
        ;;
    *)
        print_error "未知命令: $COMMAND"
        show_help
        exit 1
        ;;
esac
