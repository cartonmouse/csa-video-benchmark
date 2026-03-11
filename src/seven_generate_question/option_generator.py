"""
文件名：option_generator.py
所在路径：src/seven_generate_question/option_generator.py
功能介绍：干扰项生成器。接收 template_engine 生成的 Question 对象，
         用3种策略为其补充3个干扰项，最终输出含4个选项的完整问题。
         策略1-组合型：同视频其他段的动作
         策略2-随机型：其他视频中相同物体对应的动作
         策略3-高频型：数据集中出现频率最高的动作
输入：
  Question        — 来自 template_engine，options 字段为空
  VideoAnnotation — 视频标注对象（提供干扰项候选池）
  VideoSegment    — 当前段（确定物体类型，引导干扰项方向）
输出：
  List[str]       — 4个选项（含正确答案，已随机打乱顺序）

需要改动的变量（均在 config.py 中）：
  GenerationConfig.OPTIONS_PER_QUESTION — 每题选项数量，默认4
  VocabularyConfig.NOUNS                — 物体词汇表，影响后备干扰项质量
  VocabularyConfig.VERBS                — 动作词汇表，影响后备干扰项质量
"""

import random
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

from config import GenerationConfig, VocabularyConfig
from data_loader import VideoAnnotation, VideoSegment, AnnotationDataLoader
from template_engine import Question

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 选项生成器 ====================
class OptionGenerator:
    """选项生成器 - 实现3种干扰项策略"""

    def __init__(self, data_loader: AnnotationDataLoader):
        """
        初始化选项生成器

        Args:
            data_loader: 数据加载器对象
        """
        self.data_loader = data_loader

        # 预计算统计信息
        self._prepare_statistics()

        logger.info("选项生成器初始化完成")
        logger.info(f"  - 可用动作数: {len(self.all_verbs)}")
        logger.info(f"  - 可用物体数: {len(self.all_nouns)}")
        logger.info(f"  - 动作-物体组合数: {len(self.verb_noun_pairs)}")

    def _prepare_statistics(self):
        """预计算数据集的统计信息"""
        # 所有动作和物体
        self.all_verbs = self.data_loader.get_all_verbs()
        self.all_nouns = self.data_loader.get_all_nouns()
        self.verb_noun_pairs = self.data_loader.get_verb_noun_pairs()

        # 动作频率统计
        verb_counts = Counter()
        noun_counts = Counter()
        action_counts = Counter()  # 完整动作（verb+noun）

        for video in self.data_loader.videos:
            for seg in video.valid_segments:
                verb_counts[seg.verb] += 1
                noun_counts[seg.noun] += 1
                action_counts[seg.action] += 1

        self.verb_frequency = verb_counts
        self.noun_frequency = noun_counts
        self.action_frequency = action_counts

        # 物体对应的动作映射
        self.noun_to_verbs = self._build_noun_verb_mapping()

    def _build_noun_verb_mapping(self) -> Dict[str, List[str]]:
        """
        构建物体到动作的映射

        Returns:
            {物体: [动作列表]}
        """
        mapping = {}

        for video in self.data_loader.videos:
            for seg in video.valid_segments:
                if seg.noun not in mapping:
                    mapping[seg.noun] = []
                if seg.verb not in mapping[seg.noun]:
                    mapping[seg.noun].append(seg.verb)

        return mapping

    def generate_options(
            self,
            question: Question,
            video: VideoAnnotation,
            segment: VideoSegment
    ) -> List[str]:
        """
        为问题生成4个选项（1个正确答案 + 3个干扰项）

        Args:
            question: 问题对象
            video: 视频对象
            segment: 当前视频段

        Returns:
            随机排序的4个选项列表
        """
        correct_answer = question.correct_answer
        options = [correct_answer]

        # 根据问题类型确定答案类型
        answer_type = self._get_answer_type(question)

        # 生成3个干扰项
        distractors = self._generate_distractors(
            correct_answer=correct_answer,
            answer_type=answer_type,
            question=question,
            video=video,
            segment=segment
        )

        options.extend(distractors)

        # 验证选项唯一性
        if len(set(options)) < GenerationConfig.OPTIONS_PER_QUESTION:
            logger.warning(f"选项重复: {options}，尝试补充")
            options = self._ensure_unique_options(options, answer_type, segment.noun)

        # 随机打乱选项顺序
        random.shuffle(options)

        return options

    def _get_answer_type(self, question: Question) -> str:
        """
        确定答案类型

        Args:
            question: 问题对象

        Returns:
            答案类型：'verb' 或 'action'
        """
        if question.question_type == "sequence":
            return "action"  # 序列问题答案是完整动作
        else:
            return "verb"  # 其他问题答案是动作

    def _generate_distractors(
            self,
            correct_answer: str,
            answer_type: str,
            question: Question,
            video: VideoAnnotation,
            segment: VideoSegment
    ) -> List[str]:
        """
        生成3个干扰项

        策略分配：
        - 1个组合型干扰项（从同一视频的其他段）
        - 1个随机干扰项（从其他视频）
        - 1个高频干扰项（最常见的答案）

        Args:
            correct_answer: 正确答案
            answer_type: 答案类型（verb或action）
            question: 问题对象
            video: 视频对象
            segment: 当前段

        Returns:
            3个干扰项列表
        """
        distractors = []

        # 策略1: 组合型干扰项
        compositional = self._generate_compositional_distractor(
            correct_answer, answer_type, video, segment
        )
        if compositional and compositional != correct_answer:
            distractors.append(compositional)

        # 策略2: 随机干扰项
        random_distractor = self._generate_random_distractor(
            correct_answer, answer_type, video, segment
        )
        if random_distractor and random_distractor != correct_answer:
            distractors.append(random_distractor)

        # 策略3: 高频干扰项
        frequent = self._generate_frequent_distractor(
            correct_answer, answer_type, question.question_type
        )
        if frequent and frequent != correct_answer:
            distractors.append(frequent)

        # 如果不足3个，补充随机选项
        while len(distractors) < 3:
            fallback = self._generate_fallback_distractor(
                correct_answer, answer_type, segment.noun, distractors
            )
            if fallback and fallback not in distractors:
                distractors.append(fallback)
            else:
                break

        # 确保不包含正确答案
        distractors = [d for d in distractors if d != correct_answer]

        # 去重
        distractors = list(dict.fromkeys(distractors))

        return distractors[:3]

    # ==================== 策略1: 组合型干扰项 ====================
    def _generate_compositional_distractor(
            self,
            correct_answer: str,
            answer_type: str,
            video: VideoAnnotation,
            segment: VideoSegment
    ) -> Optional[str]:
        """
        组合型干扰项：从同一视频的其他段中选择

        Args:
            correct_answer: 正确答案
            answer_type: 答案类型
            video: 当前视频
            segment: 当前段

        Returns:
            干扰项字符串
        """
        valid_segments = video.valid_segments

        # 必须有其他段才能生成
        if len(valid_segments) <= 1:
            return None

        candidates = []

        if answer_type == "verb":
            # 找到操作相同物体的不同动作
            for seg in valid_segments:
                if seg != segment and seg.noun == segment.noun:
                    if seg.verb != correct_answer:
                        candidates.append(seg.verb)

        elif answer_type == "action":
            # 找到其他完整动作
            for seg in valid_segments:
                if seg != segment and seg.action != correct_answer:
                    candidates.append(seg.action)

        if candidates:
            return random.choice(candidates)

        return None

    # ==================== 策略2: 随机干扰项 ====================
    def _generate_random_distractor(
            self,
            correct_answer: str,
            answer_type: str,
            current_video: VideoAnnotation,
            segment: VideoSegment
    ) -> Optional[str]:
        """
        随机干扰项：从其他视频中随机选择合理的选项

        Args:
            correct_answer: 正确答案
            answer_type: 答案类型
            current_video: 当前视频
            segment: 当前段

        Returns:
            干扰项字符串
        """
        candidates = []

        # 从其他视频中收集候选项
        for video in self.data_loader.videos:
            if video == current_video:
                continue

            for seg in video.valid_segments:
                if answer_type == "verb":
                    # 找到操作相同物体的动作
                    if seg.noun == segment.noun and seg.verb != correct_answer:
                        candidates.append(seg.verb)
                elif answer_type == "action":
                    # 找到不同的完整动作
                    if seg.action != correct_answer:
                        candidates.append(seg.action)

        # 如果其他视频没有相同物体，从配置中的合理组合中选择
        if not candidates and answer_type == "verb":
            noun_config = VocabularyConfig.NOUNS.get(segment.noun, {})
            common_actions = noun_config.get("common_actions", [])
            candidates = [v for v in common_actions if v != correct_answer]

        if candidates:
            return random.choice(candidates)

        return None

    # ==================== 策略3: 高频干扰项 ====================
    def _generate_frequent_distractor(
            self,
            correct_answer: str,
            answer_type: str,
            question_type: str
    ) -> Optional[str]:
        """
        高频干扰项：选择数据集中最常见的答案

        Args:
            correct_answer: 正确答案
            answer_type: 答案类型
            question_type: 问题类型

        Returns:
            干扰项字符串
        """
        if answer_type == "verb":
            # 获取最高频的动作
            most_common = self.verb_frequency.most_common(5)
            for verb, count in most_common:
                if verb != correct_answer:
                    return verb

        elif answer_type == "action":
            # 获取最高频的完整动作
            most_common = self.action_frequency.most_common(5)
            for action, count in most_common:
                if action != correct_answer:
                    return action

        return None

    # ==================== 辅助方法 ====================
    def _generate_fallback_distractor(
            self,
            correct_answer: str,
            answer_type: str,
            noun: str,
            existing_distractors: List[str]
    ) -> Optional[str]:
        """
        后备干扰项：当其他策略失败时使用

        Args:
            correct_answer: 正确答案
            answer_type: 答案类型
            noun: 物体名称
            existing_distractors: 已有的干扰项

        Returns:
            干扰项字符串
        """
        if answer_type == "verb":
            # 从该物体的可能动作中选择
            possible_verbs = self.noun_to_verbs.get(noun, [])
            candidates = [v for v in possible_verbs
                          if v != correct_answer and v not in existing_distractors]

            if not candidates:
                # 从所有动作中随机选择
                candidates = [v for v in self.all_verbs
                              if v != correct_answer and v not in existing_distractors]

            if candidates:
                return random.choice(candidates)

        elif answer_type == "action":
            # 从所有动作组合中选择
            all_actions = [f"{v}{n}" for v, n in self.verb_noun_pairs]
            candidates = [a for a in all_actions
                          if a != correct_answer and a not in existing_distractors]

            if candidates:
                return random.choice(candidates)

        return None

    def _ensure_unique_options(
            self,
            options: List[str],
            answer_type: str,
            noun: str
    ) -> List[str]:
        """
        确保选项唯一性，补充重复的选项

        Args:
            options: 原始选项列表
            answer_type: 答案类型
            noun: 物体名称

        Returns:
            去重后的选项列表
        """
        # 去重保留第一次出现的
        unique_options = list(dict.fromkeys(options))
        correct_answer = options[0]  # 第一个是正确答案

        # 补充缺失的选项
        while len(unique_options) < GenerationConfig.OPTIONS_PER_QUESTION:
            fallback = self._generate_fallback_distractor(
                correct_answer, answer_type, noun, unique_options
            )
            if fallback and fallback not in unique_options:
                unique_options.append(fallback)
            else:
                # 实在找不到了，从配置中随机选
                if answer_type == "verb":
                    all_candidates = list(VocabularyConfig.VERBS.keys())
                else:
                    all_candidates = [f"{v}{n}" for v in VocabularyConfig.VERBS.keys()
                                      for n in VocabularyConfig.NOUNS.keys()]

                remaining = [c for c in all_candidates if c not in unique_options]
                if remaining:
                    unique_options.append(random.choice(remaining))
                else:
                    break

        return unique_options[:GenerationConfig.OPTIONS_PER_QUESTION]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import sys
    from datetime import datetime

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from config import PathConfig
    from template_engine import TemplateEngine

    # 确保目录存在
    PathConfig.ensure_dirs()

    # 加载数据
    logger.info("=" * 60)
    logger.info("开始测试选项生成器")
    logger.info("=" * 60)

    start_time = datetime.now()

    # 加载数据
    loader = AnnotationDataLoader()
    loader.load()

    # 创建模板引擎和选项生成器
    template_engine = TemplateEngine()
    option_generator = OptionGenerator(loader)

    # 测试生成问题和选项
    all_segments = loader.get_all_valid_segments()

    logger.info(f"\n{'=' * 60}")
    logger.info("开始生成问题和选项")
    logger.info(f"{'=' * 60}\n")

    success_count = 0

    for idx, (video, segment) in enumerate(all_segments):
        logger.info(f"[{idx + 1}/{len(all_segments)}] 处理段: {segment.action}")

        # 生成问题
        question = template_engine.generate_question(video, segment, idx)

        if not question:
            logger.warning("  ❌ 问题生成失败\n")
            continue

        # 生成选项
        options = option_generator.generate_options(question, video, segment)
        question.options = options

        # 显示结果
        logger.info(f"  问题类型: {question.question_type}")
        logger.info(f"  问题: {question.question_text}")
        logger.info(f"  正确答案: {question.correct_answer}")
        logger.info(f"  选项: {options}")

        # 验证
        if question.correct_answer in options and len(set(options)) == 4:
            logger.info("  ✅ 验证通过\n")
            success_count += 1
        else:
            logger.warning("  ⚠️ 验证失败（选项重复或缺少正确答案）\n")

    # 统计
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    logger.info(f"{'=' * 60}")
    logger.info(f"生成完成!")
    logger.info(f"  成功: {success_count}/{len(all_segments)}")
    logger.info(f"  耗时: {elapsed:.2f}秒")
    logger.info(f"{'=' * 60}")