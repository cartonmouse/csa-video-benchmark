"""
文件名: model_runner.py
功能介绍: 模型推理调度入口，负责遍历所有样本、调用model_backends执行推理、
         断点续跑、并发控制、结果保存和进度日志
输入:
    - List[QASample]: 标准化QA样本列表（来自data_loader.py）
    - ModelConfig: 模型配置（来自config.py）
输出:
    - results_{model_name}.jsonl: 每个模型的推理结果文件，保存至RESULT_DIR
      每行一条JSON记录，包含sample_id、模型回答、正确答案、是否答对等字段
需要改动的变量:
    - config.py中的 RESULT_DIR（结果保存路径）
    - config.py中的 EVAL_CONFIG.api_workers（API并发数，建议3-5）
路径:
    - 推理结果: config.RESULT_DIR/{model_name}/results.jsonl
    - 日志文件: config.LOG_DIR/model_runner.log
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from config import EVAL_CONFIG, LOG_DIR, RESULT_DIR, ModelConfig
from data_loader import QASample, extract_frames
from model_backends import run_inference

# ============================================================
# 日志配置
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

logger = logging.getLogger("model_runner")
logger.setLevel(logging.INFO)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "model_runner.log"), encoding="utf-8")
_ch = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)


# ============================================================
# 结果记录结构
# ============================================================

def _make_result_record(sample: QASample, raw_response: str,
                        parsed_answer: str, elapsed: float) -> Dict:
    """构建单条推理结果记录"""
    is_correct = parsed_answer.upper() == sample.answer.upper()
    return {
        "sample_id":     sample.sample_id,
        "video_name":    sample.video_name,
        "task_type":     sample.task_type,
        "task_level":    sample.task_level,
        "category":      sample.category,
        "question":      sample.question,
        "choices":       sample.choices,
        "ground_truth":  sample.answer,
        "raw_response":  raw_response,
        "parsed_answer": parsed_answer,
        "is_correct":    is_correct,
        "elapsed_s":     round(elapsed, 3),
    }


# ============================================================
# 答案解析
# ============================================================

def parse_answer(raw_response: str) -> str:
    """
    从模型原始回答中提取选项字母（A/B/C/D）。
    优先匹配开头的单个字母，其次在全文中查找。
    """
    if not raw_response:
        return ""

    text = raw_response.strip().upper()

    # 情况1：回答直接就是单个字母
    if text in ("A", "B", "C", "D"):
        return text

    # 情况2：回答以字母开头，如 "A." 或 "A："
    if text and text[0] in ("A", "B", "C", "D"):
        return text[0]

    # 情况3：在文本中搜索"答案是X"或"选X"
    import re
    match = re.search(r"答案[是为：:]\s*([A-D])", text)
    if match:
        return match.group(1)

    # 情况4：全文中第一个出现的ABCD字母
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)

    logger.warning(f"无法解析答案: {raw_response[:50]}")
    return ""


# ============================================================
# 断点续跑：加载已有结果
# ============================================================

def _get_result_path(model_name: str) -> str:
    """返回该模型结果文件的完整路径"""
    model_dir = os.path.join(RESULT_DIR, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, "results.jsonl")


def _load_finished_ids(result_path: str) -> set:
    """读取已完成的sample_id集合，用于断点续跑"""
    finished = set()
    if not os.path.exists(result_path):
        return finished
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                finished.add(record["sample_id"])
            except Exception:
                continue
    return finished


def _append_result(result_path: str, record: Dict) -> None:
    """追加写入一条结果记录（JSONL格式，断点安全）"""
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 单样本推理（含帧提取 + 推理 + 答案解析）
# ============================================================

def _run_single(sample: QASample, config: ModelConfig) -> Dict:
    """
    对单个样本执行完整推理流程：
    帧提取 → 推理 → 答案解析 → 返回结果记录
    """
    # 按需提取帧（API模式下每次提取，本地模式可在外部批量提取）
    if not sample.frames:
        extract_frames(sample, num_frames=config.max_frames)

    t0 = time.time()
    raw_response = run_inference(sample, config)
    elapsed = time.time() - t0

    parsed_answer = parse_answer(raw_response)
    return _make_result_record(sample, raw_response, parsed_answer, elapsed)


# ============================================================
# 主调度函数
# ============================================================

def run_model_eval(samples: List[QASample],
                   config: ModelConfig,
                   workers: Optional[int] = None) -> List[Dict]:
    """
    对给定模型运行完整评估，支持断点续跑和并发调用。

    Args:
        samples:  QASample列表
        config:   模型配置
        workers:  并发线程数，None则使用config中的默认值
                  本地模型请设为1（GPU推理不支持并发）
    Returns:
        所有推理结果记录的列表
    """
    result_path = _get_result_path(config.name)
    n_workers = workers if workers is not None else (
        1 if config.backend == "local" else EVAL_CONFIG.api_workers
    )

    # 断点续跑：过滤已完成样本
    finished_ids = set()
    if EVAL_CONFIG.resume:
        finished_ids = _load_finished_ids(result_path)
        if finished_ids:
            logger.info(f"[{config.name}] 断点续跑，已完成 {len(finished_ids)} 条，跳过")

    pending = [s for s in samples if s.sample_id not in finished_ids]
    total   = len(samples)
    done    = len(finished_ids)

    logger.info(
        f"[{config.name}] 开始评估 | 总样本: {total} | "
        f"待处理: {len(pending)} | 并发数: {n_workers}"
    )

    if not pending:
        logger.info(f"[{config.name}] 所有样本已完成，直接读取结果")
        return _load_all_results(result_path)

    results = []
    t_start = time.time()

    if n_workers == 1:
        # 串行推理（本地模型）
        for sample in pending:
            record = _run_single(sample, config)
            _append_result(result_path, record)
            results.append(record)
            done += 1
            _log_progress(config.name, done, total, t_start, record["is_correct"])
    else:
        # 并发推理（API模型）
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(_run_single, s, config): s
                for s in pending
            }
            for future in as_completed(future_map):
                try:
                    record = future.result()
                    _append_result(result_path, record)
                    results.append(record)
                    done += 1
                    _log_progress(config.name, done, total, t_start, record["is_correct"])
                except Exception as e:
                    sample = future_map[future]
                    logger.error(f"[{config.name}] 样本推理异常 [{sample.sample_id}]: {e}")

    elapsed_total = time.time() - t_start
    correct = sum(1 for r in results if r["is_correct"])
    logger.info(
        f"[{config.name}] 本轮完成 | "
        f"处理: {len(results)} 条 | "
        f"正确: {correct} | "
        f"本轮耗时: {elapsed_total:.1f}s"
    )

    return _load_all_results(result_path)


# ============================================================
# 进度日志
# ============================================================

def _log_progress(model_name: str, done: int, total: int,
                  t_start: float, is_correct: bool) -> None:
    """每处理一定数量样本输出一次进度和预估剩余时间"""
    log_interval = max(1, total // 20)   # 每完成5%输出一次
    if done % log_interval != 0 and done != total:
        return

    elapsed   = time.time() - t_start
    avg_time  = elapsed / max(done, 1)
    remaining = avg_time * (total - done)
    pct       = done / total * 100

    logger.info(
        f"[{model_name}] 进度: {done}/{total} ({pct:.1f}%) | "
        f"已用时: {elapsed:.0f}s | "
        f"预计剩余: {remaining:.0f}s"
    )


# ============================================================
# 读取已保存的全量结果
# ============================================================

def _load_all_results(result_path: str) -> List[Dict]:
    """从JSONL文件读取所有结果记录"""
    records = []
    if not os.path.exists(result_path):
        return records
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                continue
    return records


def load_results(model_name: str) -> List[Dict]:
    """对外接口：按模型名称读取已有推理结果"""
    path = _get_result_path(model_name)
    records = _load_all_results(path)
    logger.info(f"读取结果: {model_name}，共 {len(records)} 条，路径: {path}")
    return records