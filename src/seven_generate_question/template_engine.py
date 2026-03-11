"""
文件名：template_engine.py
所在路径：src/seven_generate_question/template_engine.py
功能介绍：问题模板引擎。根据 VideoSegment 数据填充预设模板，
         生成问题文本和正确答案。支持4种问题类型：
         交互(interaction) / 序列(sequence) / 预测(prediction) / 可行性(feasibility)。
         不生成干扰项选项，选项填充由 option_generator 负责。
输入：
  VideoAnnotation — 视频标注对象（来自 data_loader）
  VideoSegment    — 视频段对象（来自 data_loader）
输出：
  Question        — 问题对象（含题干和正确答案，options 字段为空列表待填充）

需要改动的变量（均在 config.py 中）：
  QuestionTemplateConfig.INTERACTION_TEMPLATES  — 交互问题模板列表
  QuestionTemplateConfig.SEQUENCE_TEMPLATES     — 序列问题模板列表
  QuestionTemplateConfig.PREDICTION_TEMPLATES   — 预测问题模板列表
  QuestionTemplateConfig.FEASIBILITY_TEMPLATES  — 可行性问题模板列表
  GenerationConfig.QUESTION_TYPE_WEIGHTS        — 各类型问题的生成权重
"""

import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import QuestionTemplateConfig, GenerationConfig
from data_loader import VideoAnnotation, VideoSegment

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 问题数据结构 ====================
@dataclass
class Question:
    """问题数据结构"""
    question_id: str
    question_type: str  # interaction, sequence, prediction, feasibility
    question_text: str
    correct_answer: str
    options: List[str]  # 4个选项（包含正确答案，已随机排序）
    video_name: str
    segment_index: int
    segment_start_time: float
    segment_end_time: float
    template_id: int
    difficulty: str
    metadata: Dict

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "options": self.options,
            "video_name": self.video_name,
            "segment_index": self.segment_index,
            "segment_start_time": self.segment_start_time,
            "segment_end_time": self.segment_end_time,
            "template_id": self.template_id,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


# ==================== 模板引擎 ====================
class TemplateEngine:
    """问题模板引擎"""

    def __init__(self):
        """初始化模板引擎"""
        self.templates = {
            "interaction": QuestionTemplateConfig.INTERACTION_TEMPLATES,
            "sequence": QuestionTemplateConfig.SEQUENCE_TEMPLATES,
            "prediction": QuestionTemplateConfig.PREDICTION_TEMPLATES,
            "feasibility": QuestionTemplateConfig.FEASIBILITY_TEMPLATES,
        }

        # 问题类型权重
        self.type_weights = GenerationConfig.QUESTION_TYPE_WEIGHTS

        logger.info("模板引擎初始化完成")
        logger.info(f"  - 交互问题模板: {len(self.templates['interaction'])}个")
        logger.info(f"  - 序列问题模板: {len(self.templates['sequence'])}个")
        logger.info(f"  - 预测问题模板: {len(self.templates['prediction'])}个")
        logger.info(f"  - 可行性问题模板: {len(self.templates['feasibility'])}个")

    def generate_question(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int,
            question_type: Optional[str] = None
    ) -> Optional[Question]:
        """
        为视频段生成一道问题

        Args:
            video: 视频标注对象
            segment: 视频段对象
            segment_index: 段索引
            question_type: 指定问题类型，None则随机选择

        Returns:
            Question对象，生成失败返回None
        """
        # 选择问题类型
        if question_type is None:
            question_type = self._select_question_type(video, segment, segment_index)

        if question_type not in self.templates:
            logger.warning(f"未知的问题类型: {question_type}")
            return None

        # 根据类型生成问题
        try:
            if question_type == "interaction":
                return self._generate_interaction_question(video, segment, segment_index)
            elif question_type == "sequence":
                return self._generate_sequence_question(video, segment, segment_index)
            elif question_type == "prediction":
                return self._generate_prediction_question(video, segment, segment_index)
            elif question_type == "feasibility":
                return self._generate_feasibility_question(video, segment, segment_index)
            else:
                return None
        except Exception as e:
            logger.error(f"生成{question_type}问题失败: {e}")
            return None

    def _select_question_type(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int
    ) -> str:
        """
        智能选择问题类型

        Args:
            video: 视频对象
            segment: 当前段
            segment_index: 段索引

        Returns:
            问题类型字符串
        """
        valid_segments = video.valid_segments

        # 如果只有一个段，只能生成交互或预测问题
        if len(valid_segments) == 1:
            return random.choice(["interaction", "prediction"])

        # 如果是第一个段，不能生成序列问题（没有"之前"）
        if segment_index == 0:
            return random.choices(
                ["interaction", "prediction", "feasibility"],
                weights=[0.4, 0.4, 0.2],
                k=1
            )[0]

        # 如果是最后一个段，更倾向生成预测或可行性问题
        if segment_index == len(valid_segments) - 1:
            return random.choices(
                ["interaction", "sequence", "prediction", "feasibility"],
                weights=[0.2, 0.3, 0.3, 0.2],
                k=1
            )[0]

        # 中间段，按权重随机选择
        types = list(self.type_weights.keys())
        weights = list(self.type_weights.values())
        return random.choices(types, weights=weights, k=1)[0]

    # ==================== 交互问题生成 ====================
    def _generate_interaction_question(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int
    ) -> Question:
        """生成交互问题"""
        # 随机选择模板
        template_data = random.choice(self.templates["interaction"])
        template_text = template_data["template"]

        # 填充模板
        question_text = self._fill_template(
            template_text,
            actor=segment.actor,
            verb=segment.verb,
            noun=segment.noun,
            location=segment.location
        )

        # 正确答案（动作）
        correct_answer = segment.verb

        # 生成问题ID
        question_id = self._generate_question_id(video.video_name, segment_index, "INT")

        # 创建问题对象（选项稍后添加）
        question = Question(
            question_id=question_id,
            question_type="interaction",
            question_text=question_text,
            correct_answer=correct_answer,
            options=[],  # 稍后由option_generator填充
            video_name=video.video_name,
            segment_index=segment_index,
            segment_start_time=segment.start_time,
            segment_end_time=segment.end_time,
            template_id=self.templates["interaction"].index(template_data),
            difficulty=template_data["difficulty"],
            metadata={
                "verb": segment.verb,
                "noun": segment.noun,
                "actor": segment.actor,
                "location": segment.location,
                "procedure_type": segment.procedure_type,
            }
        )

        return question

    # ==================== 序列问题生成 ====================
    def _generate_sequence_question(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int
    ) -> Optional[Question]:
        """生成序列问题"""
        valid_segments = video.valid_segments

        # 找到当前段在valid_segments中的实际索引
        try:
            actual_index = valid_segments.index(segment)
        except ValueError:
            logger.debug("段不在valid_segments中")
            return None

        # 必须有前后段才能生成序列问题
        if len(valid_segments) == 1:
            logger.debug("只有一个段，无法生成序列问题")
            return None

        # 随机选择询问"之前"还是"之后"
        if actual_index == 0:
            # 第一个段只能问"之后"
            direction = "after"
            target_segment = valid_segments[actual_index + 1]
        elif actual_index == len(valid_segments) - 1:
            # 最后一个段只能问"之前"
            direction = "before"
            target_segment = valid_segments[actual_index - 1]
        else:
            # 中间段随机选择
            direction = random.choice(["before", "after"])
            if direction == "before":
                target_segment = valid_segments[actual_index - 1]
            else:
                target_segment = valid_segments[actual_index + 1]

        # 选择对应方向的模板
        templates = [t for t in self.templates["sequence"] if direction in t["template"]]
        if not templates:
            templates = self.templates["sequence"]

        template_data = random.choice(templates)
        template_text = template_data["template"]

        # 填充模板
        question_text = self._fill_template(
            template_text,
            actor=segment.actor,
            verb=segment.verb,
            noun=segment.noun,
            location=segment.location
        )

        # 正确答案（完整动作）
        correct_answer = target_segment.action

        # 生成问题ID
        question_id = self._generate_question_id(video.video_name, segment_index, "SEQ")

        question = Question(
            question_id=question_id,
            question_type="sequence",
            question_text=question_text,
            correct_answer=correct_answer,
            options=[],
            video_name=video.video_name,
            segment_index=segment_index,
            segment_start_time=segment.start_time,
            segment_end_time=segment.end_time,
            template_id=self.templates["sequence"].index(template_data),
            difficulty=template_data["difficulty"],
            metadata={
                "direction": direction,
                "current_action": segment.action,
                "target_action": target_segment.action,
                "actor": segment.actor,
            }
        )

        return question

    # ==================== 预测问题生成 ====================
    def _generate_prediction_question(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int
    ) -> Question:
        """生成预测问题"""
        # 随机选择模板
        template_data = random.choice(self.templates["prediction"])
        template_text = template_data["template"]

        # 填充模板
        question_text = self._fill_template(
            template_text,
            actor=segment.actor,
            verb=segment.verb,
            noun=segment.noun,
            location=segment.location
        )

        # 正确答案：基于next_action字段或当前动作
        # 如果有明确的next_action描述，尝试提取动词
        correct_answer = self._extract_next_action_verb(segment)

        # 生成问题ID
        question_id = self._generate_question_id(video.video_name, segment_index, "PRE")

        question = Question(
            question_id=question_id,
            question_type="prediction",
            question_text=question_text,
            correct_answer=correct_answer,
            options=[],
            video_name=video.video_name,
            segment_index=segment_index,
            segment_start_time=segment.start_time,
            segment_end_time=segment.end_time,
            template_id=self.templates["prediction"].index(template_data),
            difficulty=template_data["difficulty"],
            metadata={
                "current_action": segment.action,
                "next_action_description": segment.next_action,
                "noun": segment.noun,
                "actor": segment.actor,
            }
        )

        return question

    # ==================== 可行性问题生成 ====================
    def _generate_feasibility_question(
            self,
            video: VideoAnnotation,
            segment: VideoSegment,
            segment_index: int
    ) -> Question:
        """生成可行性问题"""
        # 随机选择模板
        template_data = random.choice(self.templates["feasibility"])
        template_text = template_data["template"]

        # 填充模板
        question_text = self._fill_template(
            template_text,
            actor=segment.actor,
            verb=segment.verb,
            noun=segment.noun,
            location=segment.location
        )

        # 正确答案：当前实际执行的动作
        correct_answer = segment.verb

        # 生成问题ID
        question_id = self._generate_question_id(video.video_name, segment_index, "FEA")

        question = Question(
            question_id=question_id,
            question_type="feasibility",
            question_text=question_text,
            correct_answer=correct_answer,
            options=[],
            video_name=video.video_name,
            segment_index=segment_index,
            segment_start_time=segment.start_time,
            segment_end_time=segment.end_time,
            template_id=self.templates["feasibility"].index(template_data),
            difficulty=template_data["difficulty"],
            metadata={
                "current_action": segment.action,
                "noun": segment.noun,
                "location": segment.location,
                "procedure_type": segment.procedure_type,
            }
        )

        return question

    # ==================== 辅助方法 ====================
    def _fill_template(self, template: str, **kwargs) -> str:
        """
        填充模板占位符

        Args:
            template: 模板字符串
            **kwargs: 占位符对应的值

        Returns:
            填充后的字符串
        """
        result = template

        # 替换占位符
        replacements = {
            "[ACTOR]": kwargs.get("actor", "工人"),
            "[VERB]": kwargs.get("verb", ""),
            "[NOUN]": kwargs.get("noun", ""),
            "[LOCATION]": kwargs.get("location", ""),
        }

        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _extract_next_action_verb(self, segment: VideoSegment) -> str:
        """
        从next_action描述中提取动词

        Args:
            segment: 视频段

        Returns:
            提取的动词
        """
        next_action = segment.next_action

        if not next_action:
            # 如果没有next_action，返回当前动作作为默认
            return segment.verb

        # 尝试提取动词（简单规则：取第一个动词）
        # 例如："将钢筋搬运到施工位置" -> "搬运"
        common_verbs = ["搬运", "拿起", "放下", "移动", "使用", "测量",
                        "整理", "准备", "安装", "传递", "检查"]

        for verb in common_verbs:
            if verb in next_action:
                return verb

        # 如果没找到，返回当前动作
        return segment.verb

    def _generate_question_id(self, video_name: str, segment_index: int, type_code: str) -> str:
        """
        生成问题ID

        Args:
            video_name: 视频名称
            segment_index: 段索引
            type_code: 类型代码（INT/SEQ/PRE/FEA）

        Returns:
            问题ID字符串
        """
        # 提取视频名称的关键部分
        video_id = video_name.split("_part_")[-1].replace(".mp4", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        return f"{type_code}_{video_id}_{segment_index:03d}_{timestamp}"


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import sys

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 导入数据加载器
    from data_loader import AnnotationDataLoader
    from config import PathConfig

    # 确保目录存在
    PathConfig.ensure_dirs()

    # 加载数据
    logger.info("加载标注数据...")
    loader = AnnotationDataLoader()
    loader.load()

    # 创建模板引擎
    engine = TemplateEngine()

    # 测试生成问题
    all_segments = loader.get_all_valid_segments()

    if not all_segments:
        logger.error("没有有效的视频段")
        sys.exit(1)

    logger.info(f"\n{'=' * 50}")
    logger.info("开始测试问题生成")
    logger.info(f"{'=' * 50}\n")

    # 为每个段生成一道问题
    generated_questions = []

    for idx, (video, segment) in enumerate(all_segments):
        logger.info(f"处理段 {idx + 1}/{len(all_segments)}: {segment.action}")

        question = engine.generate_question(video, segment, idx)

        if question:
            generated_questions.append(question)
            logger.info(f"  ✅ 生成{question.question_type}问题")
            logger.info(f"  问题: {question.question_text}")
            logger.info(f"  答案: {question.correct_answer}\n")
        else:
            logger.warning(f"  ❌ 生成失败\n")

    logger.info(f"{'=' * 50}")
    logger.info(f"生成完成: {len(generated_questions)}/{len(all_segments)} 道题")
    logger.info(f"{'=' * 50}")