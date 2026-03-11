"""
文件名: data_loader.py
功能: CSV-Bench数据集加载器
作者: Paperbox
创建日期: 2025-01-29

主要功能:
1. 读取STAR格式的JSON数据文件
2. 自动添加difficulty level (L1-L4)
3. 按任务类型/难度筛选数据
4. 支持训练集/验证集/测试集划分
5. 提供数据统计和验证功能
6. 完整的日志记录功能

使用示例:
    from data_loader import CSVBenchDataLoader

    loader = CSVBenchDataLoader()
    loader.load_from_json("data/generated_questions.json")

    # 获取所有L3难度的问题
    l3_questions = loader.filter_by_difficulty("L3")

    # 划分数据集
    train, val, test = loader.split_data(train_ratio=0.7, val_ratio=0.15)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random

# 导入配置
# 导入配置（兼容直接运行和模块导入）
try:
    # 尝试相对导入（作为模块导入时）
    from .config import (
        get_dated_log_file,
        TYPE_TO_DIFFICULTY,
        RANDOM_SEED
    )
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行时）
    from config import (
        get_dated_log_file,
        TYPE_TO_DIFFICULTY,
        RANDOM_SEED
    )


# ==================== 配置日志 ====================
def setup_logger(log_file: str = None) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_file: 日志文件路径（默认从config读取）

    Returns:
        配置好的logger对象
    """
    # 如果没有指定日志文件，使用配置中的默认路径
    if log_file is None:
        log_file = str(get_dated_log_file("data_loader"))
    """
    配置日志系统

    Args:
        log_file: 日志文件路径

    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger("CSVBenchDataLoader")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()

# ==================== 难度映射配置 ====================
# ==================== 难度映射配置 ====================
# 从config导入，不需要重复定义
# TYPE_TO_DIFFICULTY 已在上面从 .config 导入


# ==================== 数据类定义 ====================
@dataclass
class Segment:
    """视频片段标注信息"""
    start_time: float
    end_time: float
    description: str
    noun: str
    verb: str
    actor: str
    location: str
    result: str
    next_action: str
    procedure_type: str
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'Segment':
        """从字典创建Segment对象"""
        return cls(
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            description=data.get("description", ""),
            noun=data.get("noun", ""),
            verb=data.get("verb", ""),
            actor=data.get("actor", ""),
            location=data.get("location", ""),
            result=data.get("result", ""),
            next_action=data.get("next_action", ""),
            procedure_type=data.get("procedure_type", ""),
            tags=data.get("tags", [])
        )


@dataclass
class Question:
    """单个问题数据"""
    type: str  # interaction/sequence/prediction/feasibility
    difficulty: str  # L1/L2/L3/L4 (自动添加)
    question: str  # 问题文本
    options: Dict[str, str]  # 选项 {"A": "...", "B": "...", ...}
    correct_answer: str  # 正确答案 (A/B/C/D)
    reasoning: str  # 推理说明
    distractor_types: List[str]  # 干扰项类型

    @classmethod
    def from_dict(cls, data: dict) -> 'Question':
        """
        从字典创建Question对象，自动添加difficulty
        """
        q_type = data.get("type", "interaction")
        difficulty = TYPE_TO_DIFFICULTY.get(q_type, "L1")

        return cls(
            type=q_type,
            difficulty=difficulty,
            question=data.get("question", ""),
            options=data.get("options", {}),
            correct_answer=data.get("correct_answer", ""),
            reasoning=data.get("reasoning", ""),
            distractor_types=data.get("distractor_types", [])
        )

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "type": self.type,
            "difficulty": self.difficulty,
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "reasoning": self.reasoning,
            "distractor_types": self.distractor_types
        }


@dataclass
class VideoSample:
    """单个视频样本（包含多个问题）"""
    video_name: str
    video_path: Optional[str] = None  # 视频文件完整路径
    duration: float = 0.0
    questions: List[Question] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, video_dir: Optional[str] = None) -> 'VideoSample':
        """
        从字典创建VideoSample对象

        Args:
            data: 原始JSON数据
            video_dir: 视频文件所在目录（可选）
        """
        video_name = data.get("video_name", "")

        # 构造视频完整路径
        video_path = None
        if video_dir:
            video_path = str(Path(video_dir) / video_name)

        # 解析questions
        questions = [
            Question.from_dict(q)
            for q in data.get("questions", [])
        ]

        # 解析segments
        annotation = data.get("original_annotation", {})
        segments = [
            Segment.from_dict(seg)
            for seg in annotation.get("segments", [])
        ]

        return cls(
            video_name=video_name,
            video_path=video_path,
            duration=annotation.get("duration", 0.0),
            questions=questions,
            segments=segments
        )

    def get_questions_by_type(self, q_type: str) -> List[Question]:
        """获取指定类型的所有问题"""
        return [q for q in self.questions if q.type == q_type]

    def get_questions_by_difficulty(self, difficulty: str) -> List[Question]:
        """获取指定难度的所有问题"""
        return [q for q in self.questions if q.difficulty == difficulty]


# ==================== 主数据加载器 ====================
class CSVBenchDataLoader:
    """
    CSV-Bench数据集加载器主类

    功能:
    - 加载JSON格式的数据文件
    - 自动添加difficulty level
    - 数据筛选和统计
    - 训练/验证/测试集划分
    """

    def __init__(self, video_dir: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            video_dir: 视频文件所在目录（可选，用于验证视频路径）
        """
        self.video_dir = video_dir
        self.data: List[VideoSample] = []
        self.metadata: Dict = {}

        logger.info("数据加载器初始化完成")

    def load_from_json(self, json_path: str) -> None:
        """
        从JSON文件加载数据

        Args:
            json_path: JSON文件路径

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        start_time = datetime.now()
        logger.info(f"开始加载数据: {json_path}")

        # 检查文件是否存在
        if not Path(json_path).exists():
            error_msg = f"数据文件不存在: {json_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 读取JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"JSON格式错误: {e}"
            logger.error(error_msg)
            raise

        # 解析metadata
        self.metadata = raw_data.get("metadata", {})
        logger.info(f"数据集元信息: {self.metadata}")

        # 解析每个视频样本
        results = raw_data.get("results", [])
        self.data = []

        for i, sample_data in enumerate(results):
            try:
                sample = VideoSample.from_dict(sample_data, self.video_dir)
                self.data.append(sample)
                logger.debug(f"加载样本 {i + 1}/{len(results)}: {sample.video_name}")
            except Exception as e:
                logger.warning(f"加载样本失败 [{sample_data.get('video_name')}]: {e}")

        # 计算加载时间
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"数据加载完成! 共{len(self.data)}个视频, "
                    f"{sum(len(s.questions) for s in self.data)}个问题, "
                    f"耗时{elapsed:.2f}秒")

        # 验证难度映射
        self.validate_difficulty_mapping()

    def validate_difficulty_mapping(self) -> None:
        """
        验证难度映射是否合理
        输出每个难度级别的问题数量和类型分布
        """
        logger.info("=" * 50)
        logger.info("验证难度映射...")

        stats = {
            "L1": {"count": 0, "types": set()},
            "L2": {"count": 0, "types": set()},
            "L3": {"count": 0, "types": set()},
            "L4": {"count": 0, "types": set()}
        }

        for sample in self.data:
            for q in sample.questions:
                stats[q.difficulty]["count"] += 1
                stats[q.difficulty]["types"].add(q.type)

        logger.info("难度映射统计:")
        for level in ["L1", "L2", "L3", "L4"]:
            info = stats[level]
            logger.info(f"  {level}: {info['count']}个问题 - 类型{info['types']}")

        logger.info("=" * 50)

    def filter_by_type(self, q_type: str) -> List[Tuple[VideoSample, Question]]:
        """
        按问题类型筛选数据

        Args:
            q_type: 问题类型 (interaction/sequence/prediction/feasibility)

        Returns:
            List of (VideoSample, Question) 元组
        """
        results = []
        for sample in self.data:
            for q in sample.questions:
                if q.type == q_type:
                    results.append((sample, q))

        logger.info(f"筛选类型'{q_type}': 共{len(results)}个问题")
        return results

    def filter_by_difficulty(self, difficulty: str) -> List[Tuple[VideoSample, Question]]:
        """
        按难度筛选数据

        Args:
            difficulty: 难度等级 (L1/L2/L3/L4)

        Returns:
            List of (VideoSample, Question) 元组
        """
        results = []
        for sample in self.data:
            for q in sample.questions:
                if q.difficulty == difficulty:
                    results.append((sample, q))

        logger.info(f"筛选难度'{difficulty}': 共{len(results)}个问题")
        return results

    def split_data(
            self,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            seed: int = None,  # 改为None
            split_by: str = "video"
    ) -> Tuple[List, List, List]:
        """
        划分训练集/验证集/测试集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子（默认从config读取）
            split_by: 划分方式

        Returns:
            (train_data, val_data, test_data) 元组
        """
        # 使用配置中的随机种子
        if seed is None:
            seed = RANDOM_SEED

        random.seed(seed)
        # ... 后续代码不变
        """
        划分训练集/验证集/测试集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            split_by: 划分方式 ("video"按视频划分, "question"按问题划分)

        Returns:
            (train_data, val_data, test_data) 元组
        """
        start_time = datetime.now()
        logger.info(f"开始数据划分 (train={train_ratio}, val={val_ratio}, "
                    f"test={1 - train_ratio - val_ratio}, split_by={split_by})")

        random.seed(seed)

        if split_by == "video":
            # 按视频划分
            videos = self.data.copy()
            random.shuffle(videos)

            total = len(videos)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            train_data = videos[:train_end]
            val_data = videos[train_end:val_end]
            test_data = videos[val_end:]

        elif split_by == "question":
            # 按问题划分（展平所有问题）
            all_pairs = []
            for sample in self.data:
                for q in sample.questions:
                    all_pairs.append((sample, q))

            random.shuffle(all_pairs)

            total = len(all_pairs)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            train_data = all_pairs[:train_end]
            val_data = all_pairs[train_end:val_end]
            test_data = all_pairs[val_end:]

        else:
            raise ValueError(f"不支持的split_by参数: {split_by}")

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"数据划分完成! 训练集:{len(train_data)}, "
                    f"验证集:{len(val_data)}, 测试集:{len(test_data)}, "
                    f"耗时{elapsed:.2f}秒")

        return train_data, val_data, test_data

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            统计信息字典
        """
        total_videos = len(self.data)
        total_questions = sum(len(s.questions) for s in self.data)

        # 按类型统计
        type_stats = {}
        for sample in self.data:
            for q in sample.questions:
                type_stats[q.type] = type_stats.get(q.type, 0) + 1

        # 按难度统计
        difficulty_stats = {}
        for sample in self.data:
            for q in sample.questions:
                difficulty_stats[q.difficulty] = difficulty_stats.get(q.difficulty, 0) + 1

        # 视频时长统计
        durations = [s.duration for s in self.data]
        avg_duration = sum(durations) / len(durations) if durations else 0

        stats = {
            "total_videos": total_videos,
            "total_questions": total_questions,
            "avg_questions_per_video": total_questions / total_videos if total_videos > 0 else 0,
            "type_distribution": type_stats,
            "difficulty_distribution": difficulty_stats,
            "avg_video_duration": avg_duration,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0
        }

        logger.info("=" * 50)
        logger.info("数据集统计信息:")
        logger.info(f"  总视频数: {stats['total_videos']}")
        logger.info(f"  总问题数: {stats['total_questions']}")
        logger.info(f"  平均每视频问题数: {stats['avg_questions_per_video']:.2f}")
        logger.info(f"  类型分布: {stats['type_distribution']}")
        logger.info(f"  难度分布: {stats['difficulty_distribution']}")
        logger.info(f"  平均视频时长: {stats['avg_video_duration']:.2f}秒")
        logger.info("=" * 50)

        return stats

    def export_to_json(self, output_path: str, include_difficulty: bool = True) -> None:
        """
        导出数据为JSON格式（添加difficulty字段）

        Args:
            output_path: 输出文件路径
            include_difficulty: 是否包含difficulty字段
        """
        logger.info(f"导出数据到: {output_path}")

        export_data = {
            "metadata": self.metadata,
            "results": []
        }

        for sample in self.data:
            sample_dict = {
                "video_name": sample.video_name,
                "questions": [q.to_dict() for q in sample.questions],
                "original_annotation": {
                    "video_name": sample.video_name,
                    "duration": sample.duration,
                    "segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "description": seg.description,
                            "noun": seg.noun,
                            "verb": seg.verb,
                            "actor": seg.actor,
                            "location": seg.location,
                            "result": seg.result,
                            "next_action": seg.next_action,
                            "procedure_type": seg.procedure_type,
                            "tags": seg.tags
                        }
                        for seg in sample.segments
                    ]
                }
            }

            export_data["results"].append(sample_dict)

        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"导出完成! 文件: {output_path}")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 使用示例
    loader = CSVBenchDataLoader(video_dir="data/videos")

    # 加载数据
    loader.load_from_json(r"F:\huafeng_labeled\马开剑（钢筋工）_PU_20012513/generated_questions_20251113_155438.json")

    # 获取统计信息
    stats = loader.get_statistics()

    # 筛选L3难度的问题
    l3_questions = loader.filter_by_difficulty("L3")
    print(f"\nL3难度问题数量: {len(l3_questions)}")

    # 划分数据集（按视频）
    train, val, test = loader.split_data(
        train_ratio=0.5,
        val_ratio=0.25,
        split_by="video"
    )

    print(f"\n训练集: {len(train)} 个视频")
    print(f"验证集: {len(val)} 个视频")
    print(f"测试集: {len(test)} 个视频")

    # 导出带difficulty的JSON
    loader.export_to_json("F:\huafeng_labeled\马开剑（钢筋工）_PU_20012513\export/output_with_difficulty.json")