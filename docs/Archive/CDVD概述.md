# 建筑工地视频数据集标注流程概述

## 标注流程总览

### 四个核心阶段

**阶段一：数据预处理**（全自动化）
- 视频质量筛选（时长、光线、稳定性评估）
- 关键帧提取（帧间隔视内容而定，场景变化检测）
- 元数据生成（时长、分辨率、质量评分等）

#### 
为每个视频生成结构化元数据：
```json
{
  "video_id": "construction_001",
  "duration": 120.5,
  "resolution": [1920, 1080],
  "fps": 30,
  "keyframes": [1.0, 2.5, 4.0, ...],
  "quality_score": 0.85,
  "scene_type": "outdoor_construction"
}
```

**阶段二：自动预标注**
- AI模型批量处理生成初始标注
- 动作识别：识别搬运、焊接、测量等工地动作
- 物体检测：检测安全帽、工具、设备、材料等

#### 预训练模型
| 任务类型 | 模型选择 |
|---------|---------|
| 动作识别 |  VideoMAE | 
| 物体检测 | YOLOv8-large | 
| 质量评估 | BRISQUE + OpenCV | 

**预标注输出格式（VideoMAE）（动作识别）**：
```json
{
  "video_id": "construction_001",
  "frame_rate": 30,
  "duration": 120.5,
  "actions": [
    {
      "action_id": "act_001",
      "class": "welding",
      "class_id": 2,
      "confidence": 0.89,
      "start_time": 15.2,
      "end_time": 28.7,
      "start_frame": 456,
      "end_frame": 861,
      "safety_level": "safe",
      "bbox": [320, 180, 640, 480],
      "attributes": {
        "tool_used": "welding_torch",
        "ppe_compliance": true,
        "work_quality": "standard"
      }
    },
    {
      "action_id": "act_002", 
      "class": "unsafe_operation",
      "class_id": 11,
      "confidence": 0.76,
      "start_time": 45.1,
      "end_time": 47.3,
      "start_frame": 1353,
      "end_frame": 1419,
      "safety_level": "violation",
      "violation_type": "no_hard_hat",
      "severity": "high"
    }
  ]
}
```

**预标注输出格式（姿态估计）（OpenPose）**：
```json
{
  "video_id": "construction_001",
  "frame_detections": [
    {
      "frame_id": 456,
      "timestamp": 15.2,
      "objects": [
        {
          "object_id": "obj_001",
          "class": "hard_hat",
          "class_id": 1,
          "confidence": 0.94,
          "bbox": [150, 80, 220, 160],
          "attributes": {
            "color": "yellow",
            "condition": "good",
            "worn_properly": true
          }
        },
        {
          "object_id": "obj_002",
          "class": "welding_equipment", 
          "class_id": 15,
          "confidence": 0.87,
          "bbox": [300, 200, 450, 350],
          "attributes": {
            "type": "arc_welder",
            "status": "active",
            "safety_certified": true
          }
        },
        {
          "object_id": "obj_003",
          "class": "scaffolding",
          "class_id": 25,
          "confidence": 0.91,
          "bbox": [0, 300, 800, 600],
          "attributes": {
            "height_level": "level_2",
            "stability": "stable",
            "safety_rails": true
          }
        }
      ]
    }
  ]
}
```



**阶段三：人工校正** 
- 标注员通过Web界面修正AI预标注结果
- 修正错误分类、补充遗漏物体、调整边界框
- 添加安全违规标注和复合动作细分



### 建筑工地专用标注体系

**动作类别**（12类）：
- 基础作业：搬运、焊接、切割、测量、钻孔、清理
- 安全相关：佩戴防护设备、违规操作、检查、休息

**物体类别**（20+类）：
- 安全装备：安全帽、安全背心、防护眼镜、工作手套
- 工具设备：电钻、锤子、扳手、焊接设备、切割机
- 材料环境：钢筋、混凝土、脚手架、起重机、警示标识
