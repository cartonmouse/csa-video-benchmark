"""
文件名：config.py
所在路径：src/seven_generate_question/config.py
功能介绍：问题生成系统的集中配置文件。包含路径配置、词汇表、
         问题模板、生成参数和验证参数，供其他模块 import 使用。
         本模块不调用 LLM，所有生成逻辑基于规则和模板。
输入：无（配置文件，直接修改变量即可）
输出：无（被 data_loader / template_engine / option_generator import）

需要改动的变量（均在 PathConfig 类中）：
  INPUT_FILE  — 标注数据 JSON 文件路径，默认读取同目录 data/ 下的文件
  OUTPUT_DIR  — 生成结果输出目录
  LOG_DIR     — 日志输出目录
"""

import os
from typing import Dict, List
from datetime import datetime


# ==================== 路径配置 ====================
class PathConfig:
    """路径配置"""
    # 项目根目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 数据目录
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    INPUT_FILE = os.path.join(DATA_DIR, "all_annotations_auto.json")

    # 输出目录
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    QUESTIONS_FILE = os.path.join(OUTPUT_DIR, "generated_questions.json")

    # 日志目录
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    LOG_FILE = os.path.join(LOG_DIR, f"question_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    @classmethod
    def ensure_dirs(cls):
        """确保所有必要的目录存在"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)


# ==================== 日志配置 ====================
class LogConfig:
    """日志配置"""
    LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==================== 词汇表配置 ====================
class VocabularyConfig:
    """建筑工地专用词汇表"""

    # 动作词汇（verb）- 基于你的数据扩展
    VERBS = {
        "拿起": {"category": "材料操作", "requires_object": True},
        "放下": {"category": "材料操作", "requires_object": True},
        "移动": {"category": "材料操作", "requires_object": True},
        "搬运": {"category": "材料操作", "requires_object": True},
        "抓握": {"category": "材料操作", "requires_object": True},
        "传递": {"category": "材料操作", "requires_object": True},
        "绑扎": {"category": "施工操作", "requires_object": True},
        "切割": {"category": "施工操作", "requires_object": True},
        "焊接": {"category": "施工操作", "requires_object": True},
        "测量": {"category": "准备工作", "requires_object": True},
        "检查": {"category": "准备工作", "requires_object": True},
        "整理": {"category": "清理整理", "requires_object": True},
        "穿戴": {"category": "安全防护", "requires_object": True},
        "使用": {"category": "工具操作", "requires_object": True},
    }

    # 物体词汇（noun）- 基于你的数据扩展
    NOUNS = {
        "钢筋": {"category": "建筑材料", "common_actions": ["拿起", "放下", "移动", "搬运", "绑扎"]},
        "卷尺": {"category": "测量工具", "common_actions": ["拿起", "放下", "使用", "测量"]},
        "鞋子": {"category": "个人物品", "common_actions": ["放下", "整理", "穿戴"]},
        "模板": {"category": "建筑材料", "common_actions": ["搭建", "拆除", "固定"]},
        "混凝土": {"category": "建筑材料", "common_actions": ["浇筑", "搅拌", "运送"]},
        "脚手架": {"category": "施工设备", "common_actions": ["搭建", "拆除", "检查", "固定"]},
        "安全帽": {"category": "安全防护", "common_actions": ["穿戴", "检查", "取下"]},
        "安全绳": {"category": "安全防护", "common_actions": ["系紧", "检查", "解开"]},
        "木板": {"category": "建筑材料", "common_actions": ["搬运", "切割", "测量"]},
        "工具箱": {"category": "工具设备", "common_actions": ["打开", "关闭", "搬运"]},
    }

    # 位置词汇（location）
    LOCATIONS = [
        "地面附近",
        "脚手架平台上",
        "工作台上",
        "施工现场",
        "材料堆放区",
        "楼层内",
        "室外场地",
    ]

    # 执行者（actor）
    ACTORS = [
        "主视角工人",
        "多人协作",
        "其他工人",
        "施工队",
    ]

    # 工序类型（procedure_type）
    PROCEDURE_TYPES = [
        "材料搬运",
        "工具准备",
        "施工操作",
        "清理整理",
        "安全防护",
        "质量检查",
        "测量定位",
    ]


# ==================== 问题模板配置 ====================
class QuestionTemplateConfig:
    """
    问题模板配置
    占位符说明：
    [ACTOR] - 执行者
    [VERB] - 动作
    [NOUN] - 物体
    [LOCATION] - 位置
    [TIME] - 时间表达
    """

    # 交互问题模板（Interaction Question）
    # 目的：测试对人-物交互的理解
    INTERACTION_TEMPLATES = [
        {
            "template": "[ACTOR]对[NOUN]做了什么？",
            "answer_type": "verb",
            "difficulty": "easy"
        },
        {
            "template": "在[LOCATION]，[ACTOR]对[NOUN]进行了什么操作？",
            "answer_type": "verb",
            "difficulty": "medium"
        },
        {
            "template": "[ACTOR]如何处理[NOUN]的？",
            "answer_type": "verb",
            "difficulty": "easy"
        },
        {
            "template": "视频中[ACTOR]对[NOUN]进行了哪种操作？",
            "answer_type": "verb",
            "difficulty": "easy"
        },
    ]

    # 序列问题模板（Sequence Question）
    # 目的：测试对动作时序的理解
    SEQUENCE_TEMPLATES = [
        {
            "template": "[ACTOR]在[VERB][NOUN]之前做了什么？",
            "answer_type": "action",  # 完整动作（verb + noun）
            "difficulty": "medium"
        },
        {
            "template": "[ACTOR]在[VERB][NOUN]之后做了什么？",
            "answer_type": "action",
            "difficulty": "medium"
        },
        {
            "template": "在[VERB][NOUN]之前，[ACTOR]进行了什么操作？",
            "answer_type": "action",
            "difficulty": "medium"
        },
        {
            "template": "[VERB][NOUN]之后，[ACTOR]的下一步动作是什么？",
            "answer_type": "action",
            "difficulty": "hard"
        },
    ]

    # 预测问题模板（Prediction Question）
    # 目的：测试对动作目的和后续的预测
    PREDICTION_TEMPLATES = [
        {
            "template": "[ACTOR]接下来会对[NOUN]做什么？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
        {
            "template": "根据当前情况，[ACTOR]下一步会如何处理[NOUN]？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
        {
            "template": "[ACTOR]拿起[NOUN]后，最可能进行什么操作？",
            "answer_type": "verb",
            "difficulty": "medium"
        },
        {
            "template": "在[LOCATION]，[ACTOR]对[NOUN]的下一步操作是什么？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
    ]

    # 可行性问题模板（Feasibility Question）
    # 目的：测试对工序逻辑和安全规范的理解
    FEASIBILITY_TEMPLATES = [
        {
            "template": "在当前情况下，[ACTOR]能够对[NOUN]做什么？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
        {
            "template": "在[LOCATION]，以下哪个操作对[NOUN]是可行的？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
        {
            "template": "根据工序要求，[ACTOR]此时应该对[NOUN]进行什么操作？",
            "answer_type": "verb",
            "difficulty": "hard"
        },
        {
            "template": "当前条件下，对[NOUN]最合适的操作是什么？",
            "answer_type": "verb",
            "difficulty": "medium"
        },
    ]


# ==================== 生成配置 ====================
class GenerationConfig:
    """问题生成配置"""

    # 每个视频段生成的问题数量
    QUESTIONS_PER_SEGMENT = 1

    # 每道题的选项数量
    OPTIONS_PER_QUESTION = 4  # 1个正确答案 + 3个干扰项

    # 问题类型权重（用于随机选择问题类型）
    QUESTION_TYPE_WEIGHTS = {
        "interaction": 0.3,  # 30%
        "sequence": 0.25,  # 25%
        "prediction": 0.25,  # 25%
        "feasibility": 0.2,  # 20%
    }

    # 过滤条件：跳过字段不完整的段
    REQUIRED_FIELDS = ["verb", "noun", "actor"]  # 必须有这些字段才能生成问题

    # 最小段持续时间（秒）
    MIN_SEGMENT_DURATION = 0.5  # 少于0.5秒的段可能是标注错误


# ==================== 验证配置 ====================
class ValidationConfig:
    """质量验证配置"""

    # 语法检查
    GRAMMAR_CHECK_ENABLED = False  # 暂时关闭，后续可启用

    # 答案验证
    VALIDATE_ANSWER_EXISTS = True  # 确保正确答案存在于数据中

    # 选项验证
    VALIDATE_OPTIONS_UNIQUE = True  # 确保选项不重复
    VALIDATE_OPTIONS_REASONABLE = True  # 确保选项语义合理


# ==================== 导出配置 ====================
def get_all_configs() -> Dict:
    """获取所有配置的字典形式"""
    return {
        "paths": PathConfig.__dict__,
        "log": LogConfig.__dict__,
        "vocabulary": VocabularyConfig.__dict__,
        "templates": QuestionTemplateConfig.__dict__,
        "generation": GenerationConfig.__dict__,
        "validation": ValidationConfig.__dict__,
    }


if __name__ == "__main__":
    # 测试配置
    PathConfig.ensure_dirs()
    print("✅ 配置文件加载成功")
    print(f"📁 日志文件: {PathConfig.LOG_FILE}")
    print(f"📁 输出文件: {PathConfig.QUESTIONS_FILE}")
    print(f"📊 动作词汇数: {len(VocabularyConfig.VERBS)}")
    print(f"📊 物体词汇数: {len(VocabularyConfig.NOUNS)}")
    print(f"📊 交互问题模板数: {len(QuestionTemplateConfig.INTERACTION_TEMPLATES)}")