"""
文件名: data_loader.py
功能介绍: 加载QA标注JSON文件，提取视频帧，输出pipeline标准格式的数据样本
输入:
    - QA标注JSON文件（路径在config.py中的QA_JSON_PATH指定）
    - 视频文件目录（config.py中的DATA_ROOT）
输出:
    - List[QASample]：标准化的QA样本列表，供model_runner.py使用
需要改动的变量:
    - config.py中的 QA_JSON_PATH、DATA_ROOT、FIELD_MAP
    - 如果JSON格式变动，只需修改本文件的 _parse_sample() 函数
路径:
    - 日志文件输出至 config.LOG_DIR/data_loader.log
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import cv2

from config import (
    DATA_ROOT, QA_JSON_PATH, LOG_DIR,
    VIDEO_CONFIG, EVAL_CONFIG, FIELD_MAP
)

# ============================================================
# 日志配置
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_loader")
logger.setLevel(logging.INFO)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "data_loader.log"), encoding="utf-8")
_ch = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)


# ============================================================
# 标准数据结构
# ============================================================

@dataclass
class QASample:
    """Pipeline内部统一的QA样本格式"""
    sample_id: str                        # 唯一ID，格式：video_id + "_" + 题号
    video_name: str                       # 视频文件名
    video_path: str                       # 视频完整路径
    question: str                         # 问题文本
    choices: Dict[str, str]               # 选项，如 {"A": "是", "B": "否"}
    answer: str                           # 正确答案，如 "A"
    task_type: str                        # 任务类型，如 "interaction"/"sequence" 等
    task_level: str                       # 难度等级 L1/L2/L3/L4（可由task_type映射得到）
    category: str                         # 任务类别，如 "Action"
    reasoning: str = ""                   # GPT生成的推理说明（可选）
    duration: float = 0.0                 # 视频时长（秒）
    frames: List = field(default_factory=list)  # 提取的视频帧（numpy数组列表）


# ============================================================
# task_type → task_level 映射
# 如果你的任务类型定义有变化，只改这里
# ============================================================

TASK_TYPE_TO_LEVEL: Dict[str, str] = {
    "interaction":  "L1",
    "sequence":     "L2",
    "prediction":   "L3",
    "feasibility":  "L4",
}

# task_type → category 映射（当前JSON无category字段，暂时统一归为Action）
# 后续JSON包含category字段后，直接从JSON读取即可
TASK_TYPE_TO_CATEGORY: Dict[str, str] = {
    "interaction":  "Action",
    "sequence":     "Action",
    "prediction":   "Action",
    "feasibility":  "Cognition",
}


# ============================================================
# 核心解析函数
# ============================================================

def _parse_sample(video_name: str, q_dict: dict, q_index: int,
                  video_full_path: str, duration: float) -> Optional[QASample]:
    """
    将单条原始QA字典转换为QASample。
    JSON格式变动时，只需修改此函数内的字段读取逻辑。
    """
    try:
        # 使用FIELD_MAP做字段映射（兼容字段名变化）
        task_type = q_dict.get(FIELD_MAP.get("task_type", "type"), "unknown")
        question  = q_dict.get(FIELD_MAP.get("question", "question"), "")
        choices   = q_dict.get(FIELD_MAP.get("choices", "options"), {})
        answer    = q_dict.get(FIELD_MAP.get("answer", "correct_answer"), "")
        reasoning = q_dict.get("reasoning", "")

        if not question or not choices or not answer:
            logger.warning(f"样本字段缺失，跳过: {video_name} 第{q_index}题")
            return None

        sample_id  = f"{Path(video_name).stem}_{q_index:03d}"
        task_level = TASK_TYPE_TO_LEVEL.get(task_type, "L1")
        category   = TASK_TYPE_TO_CATEGORY.get(task_type, "Action")

        return QASample(
            sample_id=sample_id,
            video_name=video_name,
            video_path=video_full_path,
            question=question,
            choices=choices,
            answer=answer,
            task_type=task_type,
            task_level=task_level,
            category=category,
            reasoning=reasoning,
            duration=duration,
        )
    except Exception as e:
        logger.error(f"解析样本失败 {video_name} 第{q_index}题: {e}")
        return None


def _find_video_path(video_name: str) -> str:
    """
    在DATA_ROOT下递归查找视频文件，返回完整路径。
    找不到则返回空字符串。
    """
    root = Path(DATA_ROOT)
    matches = list(root.rglob(video_name))
    if matches:
        return str(matches[0])
    logger.warning(f"视频文件未找到: {video_name}（在 {DATA_ROOT} 下搜索）")
    return ""


# ============================================================
# JSON加载入口
# ============================================================

def load_from_json(json_path: str = QA_JSON_PATH,
                   task_levels: Optional[List[str]] = None,
                   categories: Optional[List[str]] = None,
                   max_samples: Optional[int] = None) -> List[QASample]:
    """
    加载QA标注JSON，返回QASample列表。

    Args:
        json_path:    JSON文件路径
        task_levels:  只加载指定难度等级，如 ["L1","L2"]，None表示全部
        categories:   只加载指定类别，None表示全部
        max_samples:  最多返回样本数（调试用），None表示全部
    Returns:
        List[QASample]
    """
    t_start = time.time()
    logger.info(f"开始加载数据集: {json_path}")

    if not os.path.exists(json_path):
        logger.error(f"JSON文件不存在: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = raw.get("results", [])
    logger.info(f"JSON共包含 {len(results)} 个视频的标注")

    samples: List[QASample] = []

    for video_entry in results:
        video_name = video_entry.get("video_name", "")
        duration   = video_entry.get("original_annotation", {}).get("duration", 0.0)
        questions  = video_entry.get("questions", [])
        video_path = _find_video_path(video_name)

        for idx, q_dict in enumerate(questions):
            sample = _parse_sample(video_name, q_dict, idx, video_path, duration)
            if sample is None:
                continue

            # 按难度过滤
            if task_levels and sample.task_level not in task_levels:
                continue
            # 按类别过滤
            if categories and sample.category not in categories:
                continue

            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                logger.info(f"已达到max_samples上限({max_samples})，停止加载")
                break
        else:
            continue
        break

    elapsed = time.time() - t_start
    logger.info(f"数据加载完成，共 {len(samples)} 条样本，耗时 {elapsed:.2f}s")
    _log_sample_stats(samples)

    return samples


def _log_sample_stats(samples: List[QASample]) -> None:
    """打印样本分布统计"""
    if not samples:
        return

    level_count: Dict[str, int] = {}
    type_count:  Dict[str, int] = {}

    for s in samples:
        level_count[s.task_level] = level_count.get(s.task_level, 0) + 1
        type_count[s.task_type]   = type_count.get(s.task_type, 0) + 1

    logger.info("--- 样本分布 ---")
    for lv, cnt in sorted(level_count.items()):
        logger.info(f"  {lv}: {cnt} 条")
    for tp, cnt in sorted(type_count.items()):
        logger.info(f"  {tp}: {cnt} 条")
    logger.info("----------------")


# ============================================================
# 视频帧提取
# ============================================================

def extract_frames(sample: QASample, num_frames: int = 8) -> List:
    """
    从视频中均匀采样num_frames帧，结果存入sample.frames。
    返回帧列表（numpy数组），提取失败返回空列表。
    """
    if not sample.video_path or not os.path.exists(sample.video_path):
        logger.warning(f"视频不存在，跳过帧提取: {sample.video_name}")
        return []

    cap = cv2.VideoCapture(sample.video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {sample.video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        logger.warning(f"视频帧数为0: {sample.video_name}")
        cap.release()
        return []

    # 均匀采样帧索引
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # resize到统一分辨率
        frame = cv2.resize(frame, VIDEO_CONFIG.frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    sample.frames = frames
    return frames


def batch_extract_frames(samples: List[QASample], num_frames: int = 8) -> None:
    """
    批量为所有样本提取视频帧，带进度日志和耗时预估。
    """
    total = len(samples)
    t_start = time.time()
    logger.info(f"开始批量提取视频帧，共 {total} 个样本，每个采样 {num_frames} 帧")

    for i, sample in enumerate(samples, 1):
        extract_frames(sample, num_frames)

        if i % 50 == 0 or i == total:
            elapsed   = time.time() - t_start
            avg_time  = elapsed / i
            remaining = avg_time * (total - i)
            logger.info(
                f"帧提取进度: {i}/{total} | "
                f"已用时: {elapsed:.1f}s | "
                f"预计剩余: {remaining:.1f}s"
            )

    logger.info(f"批量帧提取完成，总耗时 {time.time() - t_start:.2f}s")