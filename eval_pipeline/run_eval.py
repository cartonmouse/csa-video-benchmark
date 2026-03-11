"""
文件名: run_eval.py
功能介绍: CSV-Bench评估Pipeline主入口，串联数据加载、模型推理、指标计算全流程
输入:
    - 命令行参数（见下方参数说明）
    - config.py中的全局配置
输出:
    - 各模型推理结果: RESULT_DIR/{model_name}/results.jsonl
    - 各模型指标报告: RESULT_DIR/{model_name}/metrics.json 和 metrics.csv
    - 横向对比报告:   RESULT_DIR/model_comparison.csv
    - 运行日志:       LOG_DIR/run_eval.log

命令行参数:
    --models        指定要评估的模型名称，多个用逗号分隔，不填则评估所有启用模型
                    示例: --models GPT-4o,Qwen2-VL-7B
    --levels        只评估指定难度等级，示例: --levels L1,L2
    --categories    只评估指定类别，示例: --categories Action,Object
    --max_samples   每个任务最多评估样本数（调试用），示例: --max_samples 10
    --debug         调试模式，只跑10条样本快速验证pipeline是否正常
    --no_resume     关闭断点续跑，从头重新评估

使用示例:
    # 调试模式，快速验证pipeline
    python run_eval.py --debug

    # 只跑闭源模型，只评估L1和L2
    python run_eval.py --models GPT-4o,Gemini-1.5-Pro --levels L1,L2

    # 全量评估所有启用模型
    python run_eval.py

路径:
    - 日志文件: config.LOG_DIR/run_eval.log
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

from config import (
    EVAL_CONFIG, LOG_DIR,
    get_enabled_models, get_model_by_name, ModelConfig
)
from data_loader import load_from_json
from metrics import compare_models, compute_metrics
from model_runner import load_results, run_model_eval

# ============================================================
# 日志配置
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("run_eval")
logger.setLevel(logging.INFO)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "run_eval.log"), encoding="utf-8")
_ch = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV-Bench 评估Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="指定评估的模型名称，多个用逗号分隔\n示例: --models GPT-4o,Qwen2-VL-7B"
    )
    parser.add_argument(
        "--levels", type=str, default=None,
        help="只评估指定难度等级\n示例: --levels L1,L2"
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="只评估指定任务类别\n示例: --categories Action,Object"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="每个任务最多评估样本数（调试用）"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="调试模式，只跑10条样本"
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="关闭断点续跑，从头重新评估"
    )
    parser.add_argument(
        "--metrics_only", action="store_true",
        help="跳过推理，只对已有结果重新计算指标"
    )
    return parser.parse_args()


# ============================================================
# 模型列表解析
# ============================================================

def resolve_models(model_arg: Optional[str]) -> List[ModelConfig]:
    """
    根据命令行参数解析要评估的模型列表。
    不指定则返回所有enabled=True的模型。
    """
    if model_arg is None:
        models = get_enabled_models(
            include_closed=True,
            include_local=True,
            include_remote_open=False,
        )
        logger.info(f"未指定模型，使用所有启用模型，共 {len(models)} 个")
        return models

    model_names = [n.strip() for n in model_arg.split(",")]
    models = []
    for name in model_names:
        cfg = get_model_by_name(name)
        if cfg is None:
            logger.warning(f"未找到模型配置: {name}，已跳过")
        else:
            models.append(cfg)

    logger.info(f"指定评估模型: {[m.name for m in models]}")
    return models


# ============================================================
# 单模型评估流程
# ============================================================

def eval_one_model(model_cfg: ModelConfig,
                   samples: list,
                   metrics_only: bool = False) -> Dict:
    """
    对单个模型执行完整评估：推理 + 指标计算。
    返回该模型的指标字典。
    """
    logger.info(f"{'─'*50}")
    logger.info(f"开始评估模型: {model_cfg.name}  后端: {model_cfg.backend}")
    t_start = time.time()

    if metrics_only:
        logger.info(f"[{model_cfg.name}] metrics_only模式，跳过推理，直接读取已有结果")
        records = load_results(model_cfg.name)
    else:
        records = run_model_eval(samples, model_cfg)

    if not records:
        logger.warning(f"[{model_cfg.name}] 无推理结果，跳过指标计算")
        return {}

    metrics = compute_metrics(records, model_cfg.name)

    elapsed = time.time() - t_start
    logger.info(
        f"[{model_cfg.name}] 评估完成 | "
        f"总体准确率: {metrics.get('overall_acc', 0)}% | "
        f"耗时: {elapsed:.1f}s"
    )
    return metrics


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    args = parse_args()
    t_global_start = time.time()

    logger.info("=" * 60)
    logger.info("CSV-Bench 评估Pipeline 启动")
    logger.info(f"调试模式: {args.debug}")
    logger.info(f"断点续跑: {not args.no_resume}")
    logger.info(f"仅计算指标: {args.metrics_only}")
    logger.info("=" * 60)

    # 应用命令行参数到全局配置
    if args.no_resume:
        EVAL_CONFIG.resume = False
    if args.debug:
        EVAL_CONFIG.debug_mode = True
        EVAL_CONFIG.max_samples_per_task = EVAL_CONFIG.debug_samples
        logger.info(f"调试模式：只加载 {EVAL_CONFIG.debug_samples} 条样本")

    # 解析目标模型列表
    models = resolve_models(args.models)
    if not models:
        logger.error("没有可用的模型配置，退出")
        return

    # 解析过滤条件
    levels     = args.levels.split(",")     if args.levels     else None
    categories = args.categories.split(",") if args.categories else None
    max_samples = args.max_samples or EVAL_CONFIG.max_samples_per_task

    # 加载数据集
    logger.info("加载数据集...")
    samples = load_from_json(
        task_levels=levels,
        categories=categories,
        max_samples=max_samples,
    )

    if not samples and not args.metrics_only:
        logger.error("数据集为空，请检查QA_JSON_PATH和数据文件是否正确")
        return

    logger.info(f"数据集加载完成，共 {len(samples)} 条样本")
    logger.info(f"计划评估模型: {[m.name for m in models]}")

    # 预估总时间（粗略估算）
    _log_time_estimate(models, len(samples))

    # 逐个模型评估
    all_metrics: Dict[str, Dict] = {}

    for i, model_cfg in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] 当前模型: {model_cfg.name}")
        metrics = eval_one_model(model_cfg, samples, args.metrics_only)
        if metrics:
            all_metrics[model_cfg.name] = metrics

    # 多模型横向对比
    if len(all_metrics) > 1:
        logger.info("\n生成横向对比报告...")
        compare_models(all_metrics)
    elif len(all_metrics) == 1:
        logger.info("只有一个模型的结果，跳过横向对比")

    # 全局耗时统计
    total_elapsed = time.time() - t_global_start
    logger.info("=" * 60)
    logger.info(f"全部评估完成！共评估 {len(all_metrics)} 个模型")
    logger.info(f"总耗时: {_format_time(total_elapsed)}")
    logger.info(f"结果保存至: {os.path.abspath(os.environ.get('RESULT_DIR', 'results'))}")
    logger.info("=" * 60)


# ============================================================
# 工具函数
# ============================================================

def _format_time(seconds: float) -> str:
    """将秒数格式化为易读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def _log_time_estimate(models: List[ModelConfig], n_samples: int) -> None:
    """
    粗略预估总评估时间并输出日志。
    API模型按每条约3秒估算，本地模型按每条约10秒估算。
    """
    total_seconds = 0
    for m in models:
        per_sample = 10 if m.backend == "local" else 3
        workers    = 1  if m.backend == "local" else EVAL_CONFIG.api_workers
        est = n_samples * per_sample / workers
        total_seconds += est
        logger.info(
            f"预估 [{m.name}]: {_format_time(est)}"
            f"（{n_samples}条 × {per_sample}s / {workers}并发）"
        )
    logger.info(f"预估总时间: {_format_time(total_seconds)}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()