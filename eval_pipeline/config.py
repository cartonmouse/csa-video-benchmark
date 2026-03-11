"""
文件名: config.py
功能介绍: CSV-Bench 评估Pipeline的全局配置文件，集中管理所有路径、模型参数、API密钥和评估设置
输入: 无（直接import使用）
输出: 各模块通过 from config import cfg 获取配置项
需要改动的变量:
    - API_KEYS: 填入你的各平台API Key
    - DATA_ROOT: 你的视频数据集根目录
    - QA_JSON_PATH: 你的QA标注文件路径
    - RESULT_DIR: 评估结果保存目录
    - LOCAL_MODEL_DIR: 本地模型权重存放目录
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional



# ============================================================
# 路径配置（根据实际情况修改）
# ============================================================

# 数据集根目录（视频文件存放位置）
DATA_ROOT = "/data/csv_bench/videos"

# QA标注JSON文件路径
QA_JSON_PATH = "/data/csv_bench/annotations/qa_pairs.json"

# 评估结果输出目录
RESULT_DIR = "/data/csv_bench/results"

# 本地开源模型权重目录
LOCAL_MODEL_DIR = "/data/models"

# 日志文件目录
LOG_DIR = "/data/csv_bench/logs"


# ============================================================
# API Key 配置（填入你的真实Key，勿上传至git）
# ============================================================

API_KEYS: Dict[str, str] = {
    "openai":  os.environ.get("OPENAI_API_KEY", "sk-YOUR_OPENAI_KEY"),
    "gemini":  os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY"),
    "together": os.environ.get("TOGETHER_API_KEY", "YOUR_TOGETHER_KEY"),  # 开源模型API后端
}


# ============================================================
# 模型配置
# ============================================================

@dataclass
class ModelConfig:
    name: str               # 模型显示名称
    model_id: str           # 模型ID（API名称或本地路径下的文件夹名）
    backend: str            # 推理后端: "openai_api" | "gemini_api" | "anthropic_api" | "local" | "together_api"
    max_frames: int = 8     # 视频最多采样帧数
    max_tokens: int = 512   # 最大输出token数
    temperature: float = 0.0  # 温度（0表示确定性输出）
    enabled: bool = True    # 是否参与本次评估


# 闭源商业模型
CLOSED_SOURCE_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="GPT-4o",
        model_id="gpt-4o",
        backend="openai_api",
        max_frames=8,
    ),
    ModelConfig(
        name="GPT-4V",
        model_id="gpt-4-vision-preview",
        backend="openai_api",
        max_frames=8,
        enabled=False,  # 默认关闭，需要时手动开启
    ),
    ModelConfig(
        name="Gemini-1.5-Pro",
        model_id="gemini-1.5-pro",
        backend="gemini_api",
        max_frames=16,
    ),
    ModelConfig(
        name="Claude-3.5-Sonnet",
        model_id="claude-3-5-sonnet-20241022",
        backend="anthropic_api",
        max_frames=8,
    ),
]

# 开源视频大模型（本地部署）
LOCAL_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="LLaVA-Video-7B",
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        backend="local",
        max_frames=8,
    ),
    ModelConfig(
        name="Qwen2-VL-7B",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        backend="local",
        max_frames=8,
    ),
    ModelConfig(
        name="InternVL2-8B",
        model_id="OpenGVLab/InternVL2-8B",
        backend="local",
        max_frames=8,
    ),
    ModelConfig(
        name="VideoChat2-7B",
        model_id="OpenGVLab/VideoChat2-HD",
        backend="local",
        max_frames=16,
    ),
    ModelConfig(
        name="Video-LLaMA-7B",
        model_id="DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned",
        backend="local",
        max_frames=8,
    ),
]

# 开源模型通过API调用（Together.ai等）
REMOTE_OPEN_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="LLaVA-Video-7B-API",
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        backend="together_api",
        max_frames=8,
        enabled=False,
    ),
]


# ============================================================
# 评估配置
# ============================================================

@dataclass
class EvalConfig:
    # 任务类型（对应QA JSON中的 task_level 字段）
    task_levels: List[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])

    # 任务类别（对应QA JSON中的 category 字段）
    categories: List[str] = field(default_factory=lambda: [
        "Action", "Object", "Position_Movement",
        "Scene", "Count", "Attribute", "Pose", "Cognition"
    ])

    # 每个任务最多评估的样本数（调试时设小，正式跑设None表示全量）
    max_samples_per_task: Optional[int] = None

    # 调试模式：只跑少量样本快速验证pipeline是否正常
    debug_mode: bool = False
    debug_samples: int = 10

    # 推理并发数（API调用时使用，本地推理设为1）
    api_workers: int = 4

    # 断点续跑：已有结果的样本跳过
    resume: bool = True

    # 答案提取方式："choice"（多选题提取ABCD）| "open"（开放题）
    answer_format: str = "choice"


EVAL_CONFIG = EvalConfig()


# ============================================================
# 视频处理配置
# ============================================================

@dataclass
class VideoConfig:
    # 帧采样策略: "uniform"（均匀采样）| "keyframe"（关键帧）
    sample_strategy: str = "uniform"

    # 帧分辨率（resize到此大小再送入模型）
    frame_size: tuple = (336, 336)

    # 视频最大时长（秒），超过则截断
    max_duration: int = 60

    # 临时帧缓存目录
    frame_cache_dir: str = "/tmp/csv_bench_frames"


VIDEO_CONFIG = VideoConfig()


# ============================================================
# 快捷访问：获取所有启用的模型列表
# ============================================================

def get_enabled_models(include_closed: bool = True,
                       include_local: bool = True,
                       include_remote_open: bool = False) -> List[ModelConfig]:
    """返回所有enabled=True的模型配置列表"""
    models = []
    if include_closed:
        models += [m for m in CLOSED_SOURCE_MODELS if m.enabled]
    if include_local:
        models += [m for m in LOCAL_MODELS if m.enabled]
    if include_remote_open:
        models += [m for m in REMOTE_OPEN_MODELS if m.enabled]
    return models


def get_model_by_name(name: str) -> Optional[ModelConfig]:
    """按名称查找模型配置"""
    all_models = CLOSED_SOURCE_MODELS + LOCAL_MODELS + REMOTE_OPEN_MODELS
    for m in all_models:
        if m.name == name:
            return m
    return None

# ============================================================
# 字段映射（JSON字段名变动时只改右边的值）
# ============================================================
FIELD_MAP: Dict[str, str] = {
    "task_type":  "type",           # JSON中任务类型的字段名
    "question":   "question",
    "choices":    "options",        # JSON中选项的字段名
    "answer":     "correct_answer", # JSON中正确答案的字段名
}