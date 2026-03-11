"""
文件名: model_backends.py
功能介绍: 各模型后端的具体推理实现，包括OpenAI、Gemini、Anthropic API以及本地模型调用
输入:
    - QASample: 单条QA样本（含视频帧、问题、选项）
    - ModelConfig: 模型配置
输出:
    - str: 模型的原始回答文本
需要改动的变量:
    - 本地模型加载逻辑需根据实际部署的模型调整（见 _load_local_model）
路径:
    - 日志输出至 config.LOG_DIR/model_backends.log
"""

import base64
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import io

from config import API_KEYS, LOG_DIR, ModelConfig
from data_loader import QASample

# ============================================================
# 日志配置
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("model_backends")
logger.setLevel(logging.INFO)

_fh = logging.FileHandler(os.path.join(LOG_DIR, "model_backends.log"), encoding="utf-8")
_ch = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)


# ============================================================
# Prompt构建
# ============================================================

def build_prompt(sample: QASample) -> str:
    """
    将QASample的问题和选项拼成统一的prompt文本。
    所有模型共用同一个prompt格式，保证评估公平性。
    """
    choices_text = "\n".join(
        f"{k}. {v}" for k, v in sample.choices.items()
    )
    prompt = (
        f"请观看视频并回答以下单选题，只需回答选项字母（A/B/C/D），不要解释。\n\n"
        f"问题：{sample.question}\n\n"
        f"{choices_text}\n\n"
        f"答案："
    )
    return prompt


def frames_to_base64(frames: list) -> list:
    """
    将numpy帧列表转换为base64编码的JPEG字符串列表，供API调用使用。
    """
    b64_frames = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        b64_frames.append(b64)
    return b64_frames


# ============================================================
# OpenAI后端（GPT-4o / GPT-4V）
# ============================================================

def run_openai(sample: QASample, config: ModelConfig) -> str:
    """调用OpenAI API进行推理"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("未安装openai库，请运行: pip install openai")
        return ""

    client = OpenAI(api_key=API_KEYS["openai"])
    prompt = build_prompt(sample)

    # 构建消息内容：文字 + 视频帧图片
    content = []
    if sample.frames:
        b64_frames = frames_to_base64(sample.frames[:config.max_frames])
        for b64 in b64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    content.append({"type": "text", "text": prompt})

    try:
        response = client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API调用失败 [{sample.sample_id}]: {e}")
        return ""


# ============================================================
# Gemini后端
# ============================================================

def run_gemini(sample: QASample, config: ModelConfig) -> str:
    """调用Gemini API进行推理"""
    try:
        import google.generativeai as genai
    except ImportError:
        logger.error("未安装google-generativeai库，请运行: pip install google-generativeai")
        return ""

    genai.configure(api_key=API_KEYS["gemini"])
    model = genai.GenerativeModel(config.model_id)
    prompt = build_prompt(sample)

    # 构建多模态输入
    parts = []
    if sample.frames:
        for frame in sample.frames[:config.max_frames]:
            img = Image.fromarray(frame.astype(np.uint8))
            parts.append(img)
    parts.append(prompt)

    try:
        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API调用失败 [{sample.sample_id}]: {e}")
        return ""


# ============================================================
# Anthropic后端（Claude）
# ============================================================

def run_anthropic(sample: QASample, config: ModelConfig) -> str:
    """调用Anthropic API进行推理"""
    try:
        import anthropic
    except ImportError:
        logger.error("未安装anthropic库，请运行: pip install anthropic")
        return ""

    client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
    prompt = build_prompt(sample)

    content = []
    if sample.frames:
        b64_frames = frames_to_base64(sample.frames[:config.max_frames])
        for b64 in b64_frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                }
            })
    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error(f"Anthropic API调用失败 [{sample.sample_id}]: {e}")
        return ""


# ============================================================
# Together.ai后端（开源模型远程API）
# ============================================================

def run_together(sample: QASample, config: ModelConfig) -> str:
    """调用Together.ai API进行推理"""
    try:
        from together import Together
    except ImportError:
        logger.error("未安装together库，请运行: pip install together")
        return ""

    client = Together(api_key=API_KEYS["together"])
    prompt = build_prompt(sample)

    content = []
    if sample.frames:
        b64_frames = frames_to_base64(sample.frames[:config.max_frames])
        for b64 in b64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    content.append({"type": "text", "text": prompt})

    try:
        response = client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Together API调用失败 [{sample.sample_id}]: {e}")
        return ""


# ============================================================
# 本地模型后端
# ============================================================

# 全局模型缓存，避免重复加载
_local_model_cache: dict = {}


def _load_local_model(config: ModelConfig) -> Tuple[object, object]:
    """
    加载本地模型和processor，结果缓存到内存。
    目前支持HuggingFace格式的模型，具体加载方式根据模型类型分发。
    """
    if config.model_id in _local_model_cache:
        return _local_model_cache[config.model_id]

    logger.info(f"正在加载本地模型: {config.name}，首次加载可能需要几分钟...")
    t0 = time.time()

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        model_path = os.path.join(
            __import__("config").LOCAL_MODEL_DIR, config.model_id
        )
        # 如果本地路径不存在，尝试从HuggingFace下载
        load_path = model_path if os.path.exists(model_path) else config.model_id

        processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            device_map="auto",          # 自动分配到可用GPU
            trust_remote_code=True,
        )
        model.eval()

        _local_model_cache[config.model_id] = (model, processor)
        logger.info(f"模型加载完成: {config.name}，耗时 {time.time() - t0:.1f}s")
        return model, processor

    except Exception as e:
        logger.error(f"本地模型加载失败 [{config.name}]: {e}")
        return None, None


def run_local(sample: QASample, config: ModelConfig) -> str:
    """
    调用本地部署的开源模型进行推理。
    注意：不同模型的输入格式有差异，如遇到具体模型报错，
    需要在此函数中针对该模型做适配。
    """
    model, processor = _load_local_model(config)
    if model is None:
        return ""

    prompt = build_prompt(sample)

    try:
        import torch
        from PIL import Image as PILImage

        images = [PILImage.fromarray(f.astype(np.uint8))
                  for f in sample.frames[:config.max_frames]]

        inputs = processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                do_sample=False,
            )

        output_text = processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output_text.strip()

    except Exception as e:
        logger.error(f"本地模型推理失败 [{config.name}] [{sample.sample_id}]: {e}")
        return ""


# ============================================================
# 统一推理入口
# ============================================================

BACKEND_MAP = {
    "openai_api":    run_openai,
    "gemini_api":    run_gemini,
    "anthropic_api": run_anthropic,
    "together_api":  run_together,
    "local":         run_local,
}


def run_inference(sample: QASample, config: ModelConfig) -> str:
    """
    根据模型配置选择对应后端执行推理，返回模型原始回答文本。
    """
    backend_fn = BACKEND_MAP.get(config.backend)
    if backend_fn is None:
        logger.error(f"未知后端类型: {config.backend}")
        return ""
    return backend_fn(sample, config)