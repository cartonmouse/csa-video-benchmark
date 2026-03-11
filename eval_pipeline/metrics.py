"""
文件名: metrics.py
功能介绍: 计算评估指标，包括Accuracy、按任务级别/类别分类统计，并生成汇总报告
输入:
    - List[Dict]: model_runner.py输出的推理结果记录列表
    - model_name: 模型名称（用于报告标题）
输出:
    - Dict: 包含总体accuracy和各维度分类统计的指标字典
    - CSV报告: 保存至 RESULT_DIR/{model_name}/metrics.csv
    - 汇总JSON: 保存至 RESULT_DIR/{model_name}/metrics.json
需要改动的变量:
    - 无需改动，指标分类自动从结果数据中读取
路径:
    - 日志文件: config.LOG_DIR/metrics.log
    - 指标报告: config.RESULT_DIR/{model_name}/metrics.csv 和 metrics.json
"""

import csv
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from config import LOG_DIR, RESULT_DIR

# ============================================================
# 日志配置
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("metrics")
logger.setLevel(logging.INFO)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "metrics.log"), encoding="utf-8")
_ch = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)


# ============================================================
# 基础统计工具
# ============================================================

def _accuracy(correct: int, total: int) -> float:
    """计算准确率，total为0时返回0.0"""
    return round(correct / total * 100, 2) if total > 0 else 0.0


def _count_correct(records: List[Dict]) -> Tuple[int, int]:
    """返回 (正确数, 总数)"""
    total   = len(records)
    correct = sum(1 for r in records if r.get("is_correct", False))
    return correct, total


# ============================================================
# 分维度统计
# ============================================================

def compute_by_level(records: List[Dict]) -> Dict[str, Dict]:
    """
    按任务难度等级（L1/L2/L3/L4）统计准确率
    返回格式: {"L1": {"correct": 10, "total": 20, "accuracy": 50.0}, ...}
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[r.get("task_level", "unknown")].append(r)

    result = {}
    for level, recs in sorted(grouped.items()):
        correct, total = _count_correct(recs)
        result[level] = {
            "correct":  correct,
            "total":    total,
            "accuracy": _accuracy(correct, total),
        }
    return result


def compute_by_category(records: List[Dict]) -> Dict[str, Dict]:
    """
    按任务类别（Action/Object/Scene等）统计准确率
    返回格式: {"Action": {"correct": 8, "total": 15, "accuracy": 53.3}, ...}
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[r.get("category", "unknown")].append(r)

    result = {}
    for cat, recs in sorted(grouped.items()):
        correct, total = _count_correct(recs)
        result[cat] = {
            "correct":  correct,
            "total":    total,
            "accuracy": _accuracy(correct, total),
        }
    return result


def compute_by_task_type(records: List[Dict]) -> Dict[str, Dict]:
    """
    按具体任务类型（interaction/sequence/prediction/feasibility）统计准确率
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[r.get("task_type", "unknown")].append(r)

    result = {}
    for task_type, recs in sorted(grouped.items()):
        correct, total = _count_correct(recs)
        result[task_type] = {
            "correct":  correct,
            "total":    total,
            "accuracy": _accuracy(correct, total),
        }
    return result


# ============================================================
# 汇总指标计算
# ============================================================

def compute_metrics(records: List[Dict], model_name: str) -> Dict:
    """
    计算单个模型的完整指标，返回指标字典并保存报告。

    Args:
        records:    推理结果列表（来自model_runner）
        model_name: 模型名称
    Returns:
        包含总体及分维度指标的字典
    """
    t_start = time.time()
    logger.info(f"[{model_name}] 开始计算指标，共 {len(records)} 条记录")

    if not records:
        logger.warning(f"[{model_name}] 结果为空，无法计算指标")
        return {}

    # 过滤掉未能解析到答案的记录
    valid   = [r for r in records if r.get("parsed_answer", "")]
    skipped = len(records) - len(valid)
    if skipped > 0:
        logger.warning(f"[{model_name}] {skipped} 条记录答案解析失败，已排除")

    correct, total = _count_correct(valid)

    metrics = {
        "model_name":    model_name,
        "total":         total,
        "correct":       correct,
        "skipped":       skipped,
        "overall_acc":   _accuracy(correct, total),
        "by_level":      compute_by_level(valid),
        "by_category":   compute_by_category(valid),
        "by_task_type":  compute_by_task_type(valid),
    }

    _log_metrics_summary(metrics)
    _save_metrics_json(metrics, model_name)
    _save_metrics_csv(metrics, model_name)

    elapsed = time.time() - t_start
    logger.info(f"[{model_name}] 指标计算完成，耗时 {elapsed:.2f}s")

    return metrics


# ============================================================
# 日志输出
# ============================================================

def _log_metrics_summary(metrics: Dict) -> None:
    """在日志中打印指标汇总"""
    name = metrics["model_name"]
    logger.info(f"{'='*50}")
    logger.info(f"模型: {name}")
    logger.info(f"总体准确率: {metrics['overall_acc']}%  ({metrics['correct']}/{metrics['total']})")

    logger.info("--- 按难度等级 ---")
    for level, stat in metrics["by_level"].items():
        logger.info(f"  {level}: {stat['accuracy']}%  ({stat['correct']}/{stat['total']})")

    logger.info("--- 按任务类别 ---")
    for cat, stat in metrics["by_category"].items():
        logger.info(f"  {cat}: {stat['accuracy']}%  ({stat['correct']}/{stat['total']})")

    logger.info("--- 按任务类型 ---")
    for tp, stat in metrics["by_task_type"].items():
        logger.info(f"  {tp}: {stat['accuracy']}%  ({stat['correct']}/{stat['total']})")
    logger.info(f"{'='*50}")


# ============================================================
# 结果保存
# ============================================================

def _get_model_dir(model_name: str) -> str:
    model_dir = os.path.join(RESULT_DIR, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _save_metrics_json(metrics: Dict, model_name: str) -> None:
    """保存完整指标为JSON文件"""
    path = os.path.join(_get_model_dir(model_name), "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"[{model_name}] 指标JSON已保存: {path}")


def _save_metrics_csv(metrics: Dict, model_name: str) -> None:
    """
    保存指标为CSV文件，方便直接导入Excel或论文表格。
    格式：维度 | 子类 | 正确数 | 总数 | 准确率
    """
    path = os.path.join(_get_model_dir(model_name), "metrics.csv")
    rows = []

    # 总体
    rows.append(["overall", "all",
                 metrics["correct"], metrics["total"], metrics["overall_acc"]])

    # 按难度
    for level, stat in metrics["by_level"].items():
        rows.append(["level", level, stat["correct"], stat["total"], stat["accuracy"]])

    # 按类别
    for cat, stat in metrics["by_category"].items():
        rows.append(["category", cat, stat["correct"], stat["total"], stat["accuracy"]])

    # 按任务类型
    for tp, stat in metrics["by_task_type"].items():
        rows.append(["task_type", tp, stat["correct"], stat["total"], stat["accuracy"]])

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "sub_type", "correct", "total", "accuracy(%)"])
        writer.writerows(rows)

    logger.info(f"[{model_name}] 指标CSV已保存: {path}")


# ============================================================
# 多模型横向对比
# ============================================================

def compare_models(all_metrics: Dict[str, Dict]) -> None:
    """
    输入多个模型的指标字典，打印横向对比表并保存汇总CSV。

    Args:
        all_metrics: {model_name: metrics_dict}
    """
    if not all_metrics:
        logger.warning("没有可对比的模型指标")
        return

    logger.info("========== 模型横向对比 ==========")

    # 收集所有出现过的level和category
    all_levels = sorted({
        lv for m in all_metrics.values()
        for lv in m.get("by_level", {}).keys()
    })
    all_cats = sorted({
        cat for m in all_metrics.values()
        for cat in m.get("by_category", {}).keys()
    })

    # 打印对比表头
    header = f"{'模型':<25} {'总体':>8}"
    for lv in all_levels:
        header += f" {lv:>8}"
    for cat in all_cats:
        header += f" {cat:>12}"
    logger.info(header)
    logger.info("-" * len(header))

    rows = []
    for model_name, metrics in all_metrics.items():
        row = f"{model_name:<25} {metrics.get('overall_acc', 0):>7.1f}%"
        csv_row = [model_name, metrics.get("overall_acc", 0)]

        for lv in all_levels:
            acc = metrics.get("by_level", {}).get(lv, {}).get("accuracy", 0)
            row += f" {acc:>7.1f}%"
            csv_row.append(acc)

        for cat in all_cats:
            acc = metrics.get("by_category", {}).get(cat, {}).get("accuracy", 0)
            row += f" {acc:>11.1f}%"
            csv_row.append(acc)

        logger.info(row)
        rows.append(csv_row)

    logger.info("=" * len(header))

    # 保存对比CSV
    compare_path = os.path.join(RESULT_DIR, "model_comparison.csv")
    with open(compare_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "overall"] + all_levels + all_cats)
        writer.writerows(rows)

    logger.info(f"横向对比CSV已保存: {compare_path}")