"""
文件名：data_loader.py
所在路径：src/seven_generate_question/data_loader.py
功能介绍：负责加载、解析、验证 labeler 导出的标注 JSON 文件。
         将原始 JSON 解析为 VideoAnnotation / VideoSegment 强类型对象，
         并提供按动词、名词、视频名查询的接口。
输入：
  PathConfig.INPUT_FILE — all_annotations_auto.json（labeler 导出）
输出：
  List[VideoAnnotation] — 供 template_engine / option_generator 使用
  data_statistics.json  — 可选，调用 export_statistics() 时导出

需要改动的变量：
  PathConfig.INPUT_FILE（在 config.py 中修改）
  GenerationConfig.REQUIRED_FIELDS — 决定哪些字段为必填，影响过滤结果
  GenerationConfig.MIN_SEGMENT_DURATION — 最短有效段时长（秒）
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import PathConfig, GenerationConfig, ValidationConfig

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 数据结构定义 ====================
@dataclass
class VideoSegment:
    """视频段数据结构"""
    start_time: float
    end_time: float
    verb: str
    noun: str
    actor: str
    location: str
    result: str
    next_action: str
    procedure_type: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """段持续时间（秒）"""
        return self.end_time - self.start_time

    @property
    def is_valid(self) -> bool:
        """判断段是否有效"""
        # 检查必需字段
        required_fields = GenerationConfig.REQUIRED_FIELDS
        for field_name in required_fields:
            if not getattr(self, field_name, None):
                return False

        # 检查时长
        if self.duration < GenerationConfig.MIN_SEGMENT_DURATION:
            return False

        return True

    @property
    def action(self) -> str:
        """完整动作描述（verb + noun）"""
        return f"{self.verb}{self.noun}"

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "verb": self.verb,
            "noun": self.noun,
            "actor": self.actor,
            "location": self.location,
            "result": self.result,
            "next_action": self.next_action,
            "procedure_type": self.procedure_type,
            "description": self.description,
            "tags": self.tags,
            "action": self.action,
        }


@dataclass
class VideoAnnotation:
    """视频标注数据结构"""
    video_path: str
    video_name: str
    duration: float
    segments: List[VideoSegment]
    annotated: bool
    status: str
    annotator: str = ""
    timestamp: str = ""

    @property
    def valid_segments(self) -> List[VideoSegment]:
        """获取所有有效的段"""
        return [seg for seg in self.segments if seg.is_valid]

    @property
    def segment_count(self) -> int:
        """段数量"""
        return len(self.segments)

    @property
    def valid_segment_count(self) -> int:
        """有效段数量"""
        return len(self.valid_segments)

    def get_segments_by_noun(self, noun: str) -> List[VideoSegment]:
        """根据物体获取段"""
        return [seg for seg in self.valid_segments if seg.noun == noun]

    def get_segments_by_verb(self, verb: str) -> List[VideoSegment]:
        """根据动作获取段"""
        return [seg for seg in self.valid_segments if seg.verb == verb]

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "duration": self.duration,
            "segment_count": self.segment_count,
            "valid_segment_count": self.valid_segment_count,
            "segments": [seg.to_dict() for seg in self.segments],
            "valid_segments": [seg.to_dict() for seg in self.valid_segments],
            "annotated": self.annotated,
            "status": self.status,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
        }


# ==================== 数据加载器 ====================
class AnnotationDataLoader:
    """标注数据加载器"""

    def __init__(self, json_path: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            json_path: JSON文件路径，默认使用配置中的路径
        """
        self.json_path = json_path or PathConfig.INPUT_FILE
        self.videos: List[VideoAnnotation] = []
        self.load_time: Optional[datetime] = None

        logger.info(f"初始化数据加载器，数据源: {self.json_path}")

    def load(self) -> Tuple[int, int]:
        """
        加载数据

        Returns:
            (总视频数, 有效段总数)
        """
        start_time = datetime.now()
        logger.info("开始加载标注数据...")

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 解析视频数据
            self.videos = self._parse_videos(data)
            self.load_time = datetime.now()

            # 统计信息
            total_videos = len(self.videos)
            total_segments = sum(v.segment_count for v in self.videos)
            valid_segments = sum(v.valid_segment_count for v in self.videos)

            elapsed = (self.load_time - start_time).total_seconds()

            logger.info(f"✅ 数据加载完成！耗时: {elapsed:.2f}秒")
            logger.info(f"📊 视频总数: {total_videos}")
            logger.info(f"📊 段总数: {total_segments}")
            logger.info(f"📊 有效段数: {valid_segments}")
            logger.info(f"📊 无效段数: {total_segments - valid_segments}")

            return total_videos, valid_segments

        except FileNotFoundError:
            logger.error(f"❌ 文件不存在: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 加载数据时发生错误: {e}")
            raise

    def _parse_videos(self, data: Dict) -> List[VideoAnnotation]:
        """
        解析视频数据

        Args:
            data: 原始JSON数据

        Returns:
            视频标注列表
        """
        videos = []
        annotations = data.get("annotations", [])

        for anno in annotations:
            try:
                video = self._parse_single_video(anno)
                videos.append(video)
            except Exception as e:
                logger.warning(f"⚠️ 解析视频失败: {anno.get('video_name', 'unknown')} - {e}")
                continue

        return videos

    def _parse_single_video(self, anno: Dict) -> VideoAnnotation:
        """
        解析单个视频

        Args:
            anno: 单个视频的标注数据

        Returns:
            VideoAnnotation对象
        """
        # 解析段
        segments = []
        for seg_data in anno.get("segments", []):
            segment = VideoSegment(
                start_time=seg_data.get("start_time", 0.0),
                end_time=seg_data.get("end_time", 0.0),
                verb=seg_data.get("verb", "").strip(),
                noun=seg_data.get("noun", "").strip(),
                actor=seg_data.get("actor", "").strip(),
                location=seg_data.get("location", "").strip(),
                result=seg_data.get("result", "").strip(),
                next_action=seg_data.get("next_action", "").strip(),
                procedure_type=seg_data.get("procedure_type", "").strip(),
                description=seg_data.get("description", "").strip(),
                tags=seg_data.get("tags", []),
            )
            segments.append(segment)

        # 创建视频对象
        video = VideoAnnotation(
            video_path=anno.get("video_path", ""),
            video_name=anno.get("video_name", ""),
            duration=anno.get("duration", 0.0),
            segments=segments,
            annotated=anno.get("annotated", False),
            status=anno.get("status", ""),
            annotator=anno.get("annotator", ""),
            timestamp=anno.get("timestamp", ""),
        )

        return video

    # ==================== 数据查询接口 ====================

    def get_all_valid_segments(self) -> List[Tuple[VideoAnnotation, VideoSegment]]:
        """
        获取所有有效段（包含所属视频信息）

        Returns:
            [(VideoAnnotation, VideoSegment), ...]
        """
        result = []
        for video in self.videos:
            for segment in video.valid_segments:
                result.append((video, segment))
        return result

    def get_segments_by_video_name(self, video_name: str) -> List[VideoSegment]:
        """根据视频名称获取段"""
        for video in self.videos:
            if video.video_name == video_name:
                return video.valid_segments
        return []

    def get_all_verbs(self) -> List[str]:
        """获取所有出现的动作"""
        verbs = set()
        for video in self.videos:
            for seg in video.valid_segments:
                if seg.verb:
                    verbs.add(seg.verb)
        return sorted(list(verbs))

    def get_all_nouns(self) -> List[str]:
        """获取所有出现的物体"""
        nouns = set()
        for video in self.videos:
            for seg in video.valid_segments:
                if seg.noun:
                    nouns.add(seg.noun)
        return sorted(list(nouns))

    def get_verb_noun_pairs(self) -> List[Tuple[str, str]]:
        """获取所有动作-物体组合"""
        pairs = set()
        for video in self.videos:
            for seg in video.valid_segments:
                if seg.verb and seg.noun:
                    pairs.add((seg.verb, seg.noun))
        return sorted(list(pairs))

    def get_statistics(self) -> Dict:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        all_segments = self.get_all_valid_segments()

        return {
            "total_videos": len(self.videos),
            "total_segments": sum(v.segment_count for v in self.videos),
            "valid_segments": len(all_segments),
            "unique_verbs": len(self.get_all_verbs()),
            "unique_nouns": len(self.get_all_nouns()),
            "unique_verb_noun_pairs": len(self.get_verb_noun_pairs()),
            "verbs": self.get_all_verbs(),
            "nouns": self.get_all_nouns(),
            "load_time": self.load_time.isoformat() if self.load_time else None,
        }

    def export_statistics(self, output_path: Optional[str] = None) -> str:
        """
        导出统计信息到JSON文件

        Args:
            output_path: 输出路径，默认为output/statistics.json

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = str(Path(PathConfig.OUTPUT_DIR) / "data_statistics.json")

        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 统计信息已导出: {output_path}")
        return output_path


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 确保目录存在
    PathConfig.ensure_dirs()

    # 测试数据加载
    loader = AnnotationDataLoader()

    try:
        total_videos, valid_segments = loader.load()

        print("\n" + "=" * 50)
        print("📊 数据统计")
        print("=" * 50)

        stats = loader.get_statistics()
        print(f"视频总数: {stats['total_videos']}")
        print(f"段总数: {stats['total_segments']}")
        print(f"有效段数: {stats['valid_segments']}")
        print(f"唯一动作数: {stats['unique_verbs']}")
        print(f"唯一物体数: {stats['unique_nouns']}")
        print(f"动作-物体组合数: {stats['unique_verb_noun_pairs']}")

        print(f"\n动作列表: {', '.join(stats['verbs'])}")
        print(f"物体列表: {', '.join(stats['nouns'])}")

        # 导出统计信息
        output_file = loader.export_statistics()
        print(f"\n✅ 统计信息已保存到: {output_file}")

    except Exception as e:
        print(f"❌ 错误: {e}")