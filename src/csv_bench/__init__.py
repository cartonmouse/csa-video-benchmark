"""
文件名: __init__.py
功能: CSV-Bench模块初始化文件
作者: Paperbox
创建日期: 2025-01-29

模块说明:
CSV-Bench (Construction Site Video Benchmark) 是首个面向建筑施工现场的
第一人称视频理解基准测试系统。

主要功能:
1. 数据加载与处理
2. 多层次评估指标计算
3. Baseline模型评估
4. 结果分析与可视化

使用示例:
    # 方式1: 导入主要类
    from src.csv_bench import CSVBenchDataLoader

    loader = CSVBenchDataLoader()
    loader.load_from_json("data/questions.json")

    # 方式2: 导入配置
    from src.csv_bench import config
    print(config.DATA_RAW_DIR)

    # 方式3: 导入所有
    import src.csv_bench as csv_bench
    loader = csv_bench.CSVBenchDataLoader()
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "Paperbox"
__email__ = "your.email@bupt.edu.cn"  # 替换为你的邮箱

# 导入配置模块（让用户可以直接使用 csv_bench.config）
from . import config

# 导入核心类
from .data_loader import (
    CSVBenchDataLoader,
    VideoSample,
    Question,
    Segment
)

# 定义 __all__ (控制 from src.csv_bench import * 的行为)
__all__ = [
    # 版本信息
    "__version__",
    "__author__",

    # 配置模块
    "config",

    # 核心类
    "CSVBenchDataLoader",
    "VideoSample",
    "Question",
    "Segment",

    # 后续添加的类会在这里列出
    # "BenchmarkMetrics",  # 评估指标类（后续添加）
    # "GPT4VEvaluator",    # GPT-4V评估器（后续添加）
]


# 模块初始化时的欢迎信息（可选）
def _print_welcome():
    """打印模块欢迎信息（调试用）"""
    import sys
    if hasattr(sys, 'ps1'):  # 只在交互式环境打印
        print(f"CSV-Bench v{__version__} 已加载")
        print(f"项目路径: {config.PROJECT_ROOT}")

# 可选：自动打印欢迎信息
# _print_welcome()