"""
文件名: config.py
功能: CSV-Bench模块全局配置
作者: Paperbox
创建日期: 2025-01-29

配置说明:
1. 路径配置：定义所有数据、日志、结果的存储路径
2. 自动创建目录：首次运行时自动创建必要的目录结构
3. 日志配置：提供多种日志文件命名策略
4. 项目常量：定义随机种子、难度映射等全局常量

使用示例:
    from src.csv_bench.config import DATA_DIR, LOG_DIR, get_dated_log_file

    # 使用预定义路径
    data_path = DATA_DIR / "raw" / "questions.json"

    # 生成带日期的日志文件
    log_file = get_dated_log_file("data_loader")
"""

from pathlib import Path
from datetime import datetime
import os

# ==================== 路径配置 ====================

# 项目根目录（v3muti/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CSV-Bench模块根目录（src/csv_bench/）
MODULE_ROOT = Path(__file__).resolve().parent

# 数据目录
DATA_ROOT = PROJECT_ROOT / "data" / "csv_bench"
DATA_RAW_DIR = DATA_ROOT / "raw"  # 原始数据（JSON文件）
DATA_PROCESSED_DIR = DATA_ROOT / "processed"  # 处理后的数据
DATA_VIDEOS_DIR = DATA_ROOT / "videos"  # 视频文件

# 日志目录
LOG_ROOT = PROJECT_ROOT / "logs" / "csv_bench"
LOG_DATA_DIR = LOG_ROOT / "data"  # 数据处理日志
LOG_EVAL_DIR = LOG_ROOT / "evaluation"  # 评估日志
LOG_TRAIN_DIR = LOG_ROOT / "training"  # 训练日志

# 结果目录
RESULTS_ROOT = PROJECT_ROOT / "results" / "csv_bench"
RESULTS_BASELINE_DIR = RESULTS_ROOT / "baseline"  # Baseline结果
RESULTS_FIGURES_DIR = RESULTS_ROOT / "figures"  # 图表
RESULTS_REPORTS_DIR = RESULTS_ROOT / "reports"  # 报告
RESULTS_CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"  # 模型检查点

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models" / "csv_bench"
MODELS_PRETRAINED_DIR = MODELS_DIR / "pretrained"  # 预训练模型
MODELS_FINETUNED_DIR = MODELS_DIR / "finetuned"  # 微调后的模型

# 文档目录
DOCS_DIR = PROJECT_ROOT / "docs" / "csv_bench"


# ==================== 自动创建目录 ====================

def create_directories():
    """
    创建所有必要的目录
    首次导入config时自动执行
    """
    directories = [
        # 数据目录
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_VIDEOS_DIR,

        # 日志目录
        LOG_DATA_DIR,
        LOG_EVAL_DIR,
        LOG_TRAIN_DIR,

        # 结果目录
        RESULTS_BASELINE_DIR,
        RESULTS_FIGURES_DIR,
        RESULTS_REPORTS_DIR,
        RESULTS_CHECKPOINTS_DIR,

        # 模型目录
        MODELS_PRETRAINED_DIR,
        MODELS_FINETUNED_DIR,

        # 文档目录
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# 执行目录创建
create_directories()


# ==================== 日志配置函数 ====================

def get_default_log_file(module_name: str = "csv_bench") -> Path:
    """
    获取默认日志文件路径

    Args:
        module_name: 模块名称

    Returns:
        日志文件路径

    Example:
        logs/csv_bench/data/csv_bench.log
    """
    return LOG_DATA_DIR / f"{module_name}.log"


def get_dated_log_file(prefix: str = "csv_bench", log_type: str = "data") -> Path:
    """
    生成带日期的日志文件名

    Args:
        prefix: 日志文件前缀
        log_type: 日志类型 ("data" | "evaluation" | "training")

    Returns:
        日志文件路径

    Example:
        logs/csv_bench/data/csv_bench_20250129.log
    """
    date_str = datetime.now().strftime("%Y%m%d")

    # 根据类型选择目录
    log_dir_map = {
        "data": LOG_DATA_DIR,
        "evaluation": LOG_EVAL_DIR,
        "training": LOG_TRAIN_DIR
    }
    log_dir = log_dir_map.get(log_type, LOG_DATA_DIR)

    return log_dir / f"{prefix}_{date_str}.log"


def get_timestamped_log_file(prefix: str = "csv_bench", log_type: str = "data") -> Path:
    """
    生成带时间戳的日志文件名（用于多次实验）

    Args:
        prefix: 日志文件前缀
        log_type: 日志类型

    Returns:
        日志文件路径

    Example:
        logs/csv_bench/data/csv_bench_20250129_153045.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir_map = {
        "data": LOG_DATA_DIR,
        "evaluation": LOG_EVAL_DIR,
        "training": LOG_TRAIN_DIR
    }
    log_dir = log_dir_map.get(log_type, LOG_DATA_DIR)

    return log_dir / f"{prefix}_{timestamp}.log"


def get_experiment_log_file(experiment_name: str) -> Path:
    """
    为实验生成专用日志文件

    Args:
        experiment_name: 实验名称

    Returns:
        日志文件路径

    Example:
        logs/csv_bench/evaluation/pilot_study_gpt4v_20250129_153045.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_EVAL_DIR / f"{experiment_name}_{timestamp}.log"


# ==================== 难度映射配置 ====================

# 问题类型到难度等级的映射
TYPE_TO_DIFFICULTY = {
    "interaction": "L1",  # 感知层 - 识别交互动作
    "sequence": "L2",  # 语义层 - 理解时序关系
    "prediction": "L3",  # 推理层 - 预测下一步动作
    "feasibility": "L4"  # 评估层 - 判断可行性/安全性
}

# 难度等级描述
DIFFICULTY_DESCRIPTIONS = {
    "L1": "感知层 - 基础视觉识别",
    "L2": "语义层 - 场景理解与描述",
    "L3": "推理层 - 时序推理与预测",
    "L4": "评估层 - 安全评估与规范判断"
}

# ==================== 项目常量 ====================

# 随机种子（保证实验可复现）
RANDOM_SEED = 42

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# 支持的问题类型
QUESTION_TYPES = ["interaction", "sequence", "prediction", "feasibility"]

# 支持的难度等级
DIFFICULTY_LEVELS = ["L1", "L2", "L3", "L4"]

# 数据集划分默认比例
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15  # 1 - train - val

# ==================== 模型配置 ====================

# 评估的Baseline模型列表
BASELINE_MODELS = [
    "gpt4v",  # GPT-4V (OpenAI)
    "gemini",  # Gemini 1.5 Pro (Google)
    "claude",  # Claude 3.5 Sonnet (Anthropic)
    "llava_video",  # LLaVA-Video
    "video_llama",  # Video-LLaMA
    "qwen2_vl",  # Qwen2-VL
    "internvl2",  # InternVL2
    "yolo_rule",  # YOLOv10 + 规则基线
]

# 模型默认参数
MODEL_DEFAULT_PARAMS = {
    "temperature": 0.0,  # 温度（0表示贪心解码，确保可复现）
    "max_tokens": 100,  # 最大生成token数
    "top_p": 1.0,  # nucleus sampling参数
}

# ==================== 评估配置 ====================

# 评估指标
EVALUATION_METRICS = {
    "L1": ["top1_accuracy", "top5_accuracy", "mAP"],  # 感知层
    "L2": ["bleu4", "rouge_l", "cider"],  # 语义层
    "L3": ["accuracy", "gpt4o_score"],  # 推理层
    "L4": ["accuracy", "safety_f1", "gpt4o_safety_score"]  # 评估层
}

# GPT-4o辅助评分配置
GPT4O_EVAL_CONFIG = {
    "model": "gpt-4o",
    "max_tokens": 300,
    "temperature": 0.0,
    "dimensions": ["accuracy", "completeness", "safety"]  # 评分维度
}

# ==================== 环境变量配置 ====================

# OpenAI API Key（从环境变量读取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# Google API Key（从环境变量读取）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# Anthropic API Key（从环境变量读取）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)


# ==================== 配置验证 ====================

def validate_config():
    """
    验证配置是否正确
    检查关键路径和环境变量
    """
    # 检查项目根目录是否存在
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"项目根目录不存在: {PROJECT_ROOT}")

    # 检查必要的目录是否创建成功
    required_dirs = [DATA_ROOT, LOG_ROOT, RESULTS_ROOT]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"必要目录不存在: {directory}")

    # 警告：如果没有设置API Key
    if not OPENAI_API_KEY:
        print("警告: 未设置 OPENAI_API_KEY 环境变量，GPT-4V评估将无法运行")

    print("✓ 配置验证通过")


# ==================== 配置信息打印 ====================

def print_config_info():
    """
    打印当前配置信息（用于调试）
    """
    print("=" * 60)
    print("CSV-Bench 配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模块根目录: {MODULE_ROOT}")
    print()
    print("数据路径:")
    print(f"  - 原始数据: {DATA_RAW_DIR}")
    print(f"  - 处理数据: {DATA_PROCESSED_DIR}")
    print(f"  - 视频文件: {DATA_VIDEOS_DIR}")
    print()
    print("日志路径:")
    print(f"  - 数据日志: {LOG_DATA_DIR}")
    print(f"  - 评估日志: {LOG_EVAL_DIR}")
    print(f"  - 训练日志: {LOG_TRAIN_DIR}")
    print()
    print("结果路径:")
    print(f"  - Baseline结果: {RESULTS_BASELINE_DIR}")
    print(f"  - 图表: {RESULTS_FIGURES_DIR}")
    print(f"  - 报告: {RESULTS_REPORTS_DIR}")
    print()
    print("API配置:")
    print(f"  - OpenAI API Key: {'已设置' if OPENAI_API_KEY else '未设置'}")
    print(f"  - Google API Key: {'已设置' if GOOGLE_API_KEY else '未设置'}")
    print(f"  - Anthropic API Key: {'已设置' if ANTHROPIC_API_KEY else '未设置'}")
    print("=" * 60)


# ==================== 模块初始化 ====================

# 首次导入时验证配置
if __name__ != "__main__":
    try:
        validate_config()
    except Exception as e:
        print(f"配置验证失败: {e}")

# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试配置
    print_config_info()

    # 测试日志文件生成
    print("\n日志文件示例:")
    print(f"  - 默认日志: {get_default_log_file('data_loader')}")
    print(f"  - 按日期日志: {get_dated_log_file('data_loader')}")
    print(f"  - 按时间戳日志: {get_timestamped_log_file('pilot_study', 'evaluation')}")
    print(f"  - 实验日志: {get_experiment_log_file('baseline_gpt4v')}")