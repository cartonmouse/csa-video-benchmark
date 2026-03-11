# CSV-Bench 数据集标准字段说明

## 概述

本文档定义了CSV-Bench评估Pipeline的标准JSON输入格式。
数据生成脚本的输出字段名需与本文档一致，以避免字段映射转换。

---

## 文件结构

```json
{
  "metadata": { ... },
  "results": [ ... ]
}
```

---

## metadata 字段

| 字段名 | 类型 | 必须 | 说明 |
|--------|------|------|------|
| `generation_time` | string | 否 | 生成时间，ISO格式 |
| `total_videos` | int | 否 | 视频总数 |
| `total_questions` | int | 否 | 问题总数 |
| `model` | string | 否 | 生成QA使用的模型名称 |

---

## results 字段（每条视频记录）

| 字段名 | 类型 | 必须 | 说明 |
|--------|------|------|------|
| `video_name` | string | ✅ | 视频文件名，含扩展名 |
| `duration` | float | 建议 | 视频时长（秒） |
| `questions` | list | ✅ | 该视频对应的QA列表 |

---

## questions 字段（每条QA记录）

这是pipeline最核心的部分，字段名必须严格对齐。

| 字段名 | 类型 | 必须 | 取值范围 | 说明 |
|--------|------|------|----------|------|
| `task_type` | string | ✅ | 见下方任务类型表 | 任务类型 |
| `task_level` | string | ✅ | `L1` / `L2` / `L3` / `L4` | 难度等级 |
| `category` | string | ✅ | 见下方类别表 | 任务类别 |
| `question` | string | ✅ | 任意问题文本 | 问题内容 |
| `choices` | dict | ✅ | `{"A":..,"B":..,"C":..,"D":..}` | 四个选项 |
| `answer` | string | ✅ | `A` / `B` / `C` / `D` | 正确答案 |
| `reasoning` | string | 否 | 任意文本 | GPT推理说明，供人工核查 |

---

## task_type 取值表

| task_type | 对应 task_level | 对应 category | 说明 |
|-----------|----------------|---------------|------|
| `interaction` | `L1` | `Action` | 工人与物体的交互动作 |
| `sequence` | `L2` | `Action` | 动作先后顺序判断 |
| `prediction` | `L3` | `Action` | 预测下一步动作 |
| `feasibility` | `L4` | `Cognition` | 安全规范可行性判断 |

> 后续扩展新任务类型时，在此表中新增一行，并同步更新 `data_loader.py` 中的
> `TASK_TYPE_TO_LEVEL` 和 `TASK_TYPE_TO_CATEGORY` 两个映射字典。

---

## category 取值表

| category | 说明 |
|----------|------|
| `Action` | 动作类任务 |
| `Object` | 物体检测类任务 |
| `Position_Movement` | 位置与移动类任务 |
| `Scene` | 场景类任务 |
| `Count` | 计数类任务 |
| `Attribute` | 属性类任务 |
| `Pose` | 姿态类任务 |
| `Cognition` | 认知推理类任务 |

---

## 标准格式完整示例

```json
{
  "metadata": {
    "generation_time": "2025-11-13T15:55:24",
    "total_videos": 2,
    "total_questions": 8,
    "model": "gpt-4o"
  },
  "results": [
    {
      "video_name": "site_001.mp4",
      "duration": 15.6,
      "questions": [
        {
          "task_type": "interaction",
          "task_level": "L1",
          "category": "Action",
          "question": "在视频中，工人对卷尺执行了什么操作？",
          "choices": {
            "A": "拿起卷尺",
            "B": "放下卷尺",
            "C": "测量长度",
            "D": "整理卷尺"
          },
          "answer": "A",
          "reasoning": "根据标注数据，工人在视频中有两次拿起卷尺的动作。"
        },
        {
          "task_type": "feasibility",
          "task_level": "L4",
          "category": "Cognition",
          "question": "当卷尺不在工人手中时，以下哪个操作是不可行的？",
          "choices": {
            "A": "放下鞋子",
            "B": "移动钢筋",
            "C": "观察现场",
            "D": "进行精确测量"
          },
          "answer": "D",
          "reasoning": "没有卷尺时，工人无法进行精确的测量操作。"
        }
      ]
    }
  ]
}
```

---

## 与当前JSON的差异对照

当前数据生成脚本输出的字段名与标准格式有以下差异，修改时对照此表：

| 位置 | 当前字段名 | 标准字段名 | 改动方式 |
|------|-----------|-----------|----------|
| questions级别 | `type` | `task_type` | 重命名 |
| questions级别 | `options` | `choices` | 重命名 |
| questions级别 | `correct_answer` | `answer` | 重命名 |
| questions级别 | 无此字段 | `task_level` | 新增，根据task_type映射生成 |
| questions级别 | 无此字段 | `category` | 新增，根据task_type映射生成 |
| results级别 | `original_annotation.duration` | `duration` | 提升到results层级 |

> `task_level` 和 `category` 可以在数据生成脚本中根据 `task_type` 自动填入，
> 对照上方的任务类型表即可。

---

## 修改数据生成脚本的建议步骤

1. 将输出字段 `type` 改为 `task_type`
2. 将输出字段 `options` 改为 `choices`
3. 将输出字段 `correct_answer` 改为 `answer`
4. 在每条QA记录中新增 `task_level` 和 `category` 字段（根据task_type映射自动填入）
5. 将 `original_annotation.duration` 提升为 `duration`，放在results层级
6. `original_annotation` 字段可以保留（pipeline会自动忽略）也可以删除

完成以上修改后，`data_loader.py` 中的 `FIELD_MAP` 和转换逻辑可以全部删除。