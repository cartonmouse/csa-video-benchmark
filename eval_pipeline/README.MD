# CSV-Bench 评估Pipeline 使用说明手册

## 目录

1. [环境要求](#1-环境要求)
2. [依赖安装](#2-依赖安装)
3. [配置文件修改](#3-配置文件修改)
4. [数据准备](#4-数据准备)
5. [运行方式](#5-运行方式)
6. [结果文件说明](#6-结果文件说明)
7. [常见报错处理](#7-常见报错处理)

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9 及以上 |
| GPU | 本地模型需要至少单张24GB显存（双卡更佳） |
| CUDA | 11.8 及以上（本地模型必须） |
| 操作系统 | Linux（推荐）/ Windows |

> 闭源API模型（GPT/Gemini/Claude）不需要GPU，只需要网络和API Key。

---

## 2. 依赖安装

**基础依赖（必装）：**
```bash
pip install opencv-python pillow numpy
```

**闭源API模型依赖（按需安装）：**
```bash
# OpenAI（GPT-4o / GPT-4V）
pip install openai

# Google Gemini
pip install google-generativeai

# Anthropic Claude
pip install anthropic

# Together.ai（开源模型远程API）
pip install together
```

**本地开源模型依赖（本地部署时安装）：**
```bash
pip install torch torchvision transformers accelerate
```

---

## 3. 配置文件修改

所有配置集中在 `config.py`，运行前必须修改以下几项：

### 3.1 路径配置

打开 `config.py`，找到路径配置区块，修改为你的实际路径：

```python
DATA_ROOT     = "/你的路径/videos"        # 视频文件所在目录
QA_JSON_PATH  = "/你的路径/qa_pairs.json" # QA标注JSON文件路径
RESULT_DIR    = "/你的路径/results"       # 评估结果保存目录
LOCAL_MODEL_DIR = "/你的路径/models"      # 本地模型权重目录
LOG_DIR       = "/你的路径/logs"          # 日志文件目录
```

### 3.2 API Key配置

**方式一（推荐）：使用环境变量**
```bash
export OPENAI_API_KEY="sk-你的key"
export GEMINI_API_KEY="你的key"
export ANTHROPIC_API_KEY="你的key"
```

**方式二：直接填入config.py**
```python
API_KEYS = {
    "openai":    "sk-你的key",
    "gemini":    "你的key",
    "anthropic": "你的key",
}
```
> 注意：方式二不要将config.py上传至Git，避免Key泄露。

### 3.3 模型开关

不想评估某个模型时，将对应模型的 `enabled` 改为 `False`：
```python
ModelConfig(
    name="GPT-4V",
    ...
    enabled=False,  # 改为False即可跳过
),
```

---

## 4. 数据准备

### 4.1 视频文件

将所有视频文件放入 `DATA_ROOT` 目录下，支持子目录嵌套：
```
videos/
├── Action/
│   ├── site_001.mp4
│   └── site_002.mp4
└── Object/
    └── site_003.mp4
```

### 4.2 QA标注JSON

JSON文件需符合标准格式（详见 `DATA_FORMAT.md`），关键字段如下：

```json
{
  "results": [
    {
      "video_name": "site_001.mp4",
      "duration": 15.6,
      "questions": [
        {
          "task_type": "interaction",
          "task_level": "L1",
          "category": "Action",
          "question": "工人执行了什么操作？",
          "choices": {"A": "拿起卷尺", "B": "放下卷尺", "C": "测量", "D": "整理"},
          "answer": "A"
        }
      ]
    }
  ]
}
```

> 字段名必须与上面一致，否则pipeline无法正确读取。

---

## 5. 运行方式

所有命令在 `eval_pipeline/` 目录下执行。

### 5.1 调试模式（首次运行必做）

只跑10条样本，快速验证pipeline是否正常：
```bash
python run_eval.py --debug
```

验证内容：路径是否正确、API Key是否有效、视频帧是否能正常提取。

### 5.2 单模型评估

只评估指定模型，模型名称来自 `config.py` 中的 `name` 字段：
```bash
python run_eval.py --models GPT-4o
python run_eval.py --models GPT-4o,Gemini-1.5-Pro
```

### 5.3 按难度/类别筛选评估

```bash
# 只评估L1和L2难度
python run_eval.py --models GPT-4o --levels L1,L2

# 只评估Action类别
python run_eval.py --models GPT-4o --categories Action
```

### 5.4 全量评估（正式跑）

评估所有 `enabled=True` 的模型：
```bash
python run_eval.py
```

### 5.5 断点续跑

程序中断后直接重新运行，会自动跳过已完成的样本：
```bash
python run_eval.py  # 直接重跑，自动续跑
```

如需从头重新评估（覆盖已有结果）：
```bash
python run_eval.py --no_resume
```

### 5.6 只重新计算指标（不重跑推理）

已有推理结果，只想重新统计指标时：
```bash
python run_eval.py --metrics_only
```

---

## 6. 结果文件说明

评估完成后，`RESULT_DIR` 目录结构如下：

```
results/
├── GPT-4o/
│   ├── results.jsonl     # 每条样本的推理结果（断点续跑依赖此文件）
│   ├── metrics.json      # 完整指标数据
│   └── metrics.csv       # 指标表格（可直接用于论文）
├── Gemini-1.5-Pro/
│   ├── results.jsonl
│   ├── metrics.json
│   └── metrics.csv
└── model_comparison.csv  # 所有模型横向对比表
```

### results.jsonl 字段说明

每行一条JSON记录，包含：

| 字段 | 说明 |
|------|------|
| `sample_id` | 样本唯一ID |
| `video_name` | 视频文件名 |
| `task_level` | 难度等级 |
| `category` | 任务类别 |
| `ground_truth` | 正确答案 |
| `raw_response` | 模型原始回答 |
| `parsed_answer` | 解析后的答案（A/B/C/D） |
| `is_correct` | 是否答对（true/false） |
| `elapsed_s` | 本条推理耗时（秒） |

### model_comparison.csv 示例

| model | overall | L1 | L2 | L3 | L4 | Action | Cognition |
|-------|---------|----|----|----|----|--------|-----------|
| GPT-4o | 72.5% | 85.0% | 75.0% | 65.0% | 55.0% | 78.0% | 55.0% |
| Gemini-1.5-Pro | 68.3% | 80.0% | 70.0% | 60.0% | 50.0% | 73.0% | 50.0% |

---

## 7. 常见报错处理

**报错：`QA_JSON_PATH` 文件不存在**
```
ERROR - JSON文件不存在: /your/path/qa_pairs.json
```
→ 检查 `config.py` 中 `QA_JSON_PATH` 路径是否正确。

---

**报错：视频文件未找到**
```
WARNING - 视频文件未找到: site_001.mp4
```
→ 检查 `DATA_ROOT` 路径，以及视频文件名是否与JSON中的 `video_name` 完全一致（包括大小写）。

---

**报错：API Key无效**
```
ERROR - OpenAI API调用失败: AuthenticationError
```
→ 检查 `config.py` 或环境变量中的API Key是否正确填写，注意不要有多余空格。

---

**报错：本地模型显存不足**
```
ERROR - CUDA out of memory
```
→ 在 `config.py` 中减小 `max_frames`（如从8改为4），或关闭其他占用显存的进程。

---

**报错：答案解析失败数量过多**
```
WARNING - 50条记录答案解析失败，已排除
```
→ 查看 `results.jsonl` 中的 `raw_response` 字段，检查模型是否按要求输出了A/B/C/D。
可以在 `model_backends.py` 的 `build_prompt()` 中加强prompt的指令约束。