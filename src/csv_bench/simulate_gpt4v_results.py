"""
===============================================================================
文件名: simulate_gpt4v_results_to_excel.py
作者: Paperbox
创建日期: 2026-01-29
版本: 1.0
===============================================================================

功能介绍:
-------------------------------------------------------------------------------
生成 GPT-4V 在 CSV-Bench 上的【模拟评估结果】，
并将所有评估结果写入【同一个 Excel 文件】的不同 Sheet 中，便于：

1. 组会展示（一个文件即可）
2. 人工检查与分析
3. 后续画图 / 表格 / 论文实验复用

生成的 Excel 文件包含 3 个 Sheet：
- Sheet 1: 逐题预测结果（question-level）
- Sheet 2: 按难度汇总的 Accuracy（L1–L4）
- Sheet 3: 按任务类型汇总的 Accuracy

⚠️ 本文件生成的是模拟数据，仅用于流程验证。

===============================================================================
输入 (Input):
-------------------------------------------------------------------------------
- 无外部输入文件
- 通过修改脚本顶部参数控制数据规模和分布

===============================================================================
输出 (Output):
-------------------------------------------------------------------------------
- Excel 文件（包含 3 个 Sheet）

  路径:
    results/csv_bench/mock/gpt4v_mock_evaluation.xlsx

  Sheet 说明:
    1. predictions
    2. accuracy_by_level
    3. accuracy_by_task

===============================================================================
可修改的关键变量:
-------------------------------------------------------------------------------
1. NUM_QUESTIONS
   - 模拟问题总数量（默认 400）

2. LEVEL_ACCURACY
   - GPT-4V 在 L1–L4 下的模拟正确率

3. OUTPUT_DIR
   - 输出目录路径

===============================================================================
使用方法:
-------------------------------------------------------------------------------
在项目根目录运行：

    python simulate_gpt4v_results_to_excel.py

===============================================================================
"""

import random
from pathlib import Path
import pandas as pd

# ==================== 可配置参数 ====================

NUM_QUESTIONS = 400

LEVELS = ["L1", "L2", "L3", "L4"]

TASK_TYPES = [
    "interaction",
    "sequence",
    "prediction",
    "feasibility"
]

LEVEL_ACCURACY = {
    "L1": 0.92,
    "L2": 0.85,
    "L3": 0.68,
    "L4": 0.55,
}

OUTPUT_DIR = Path(r"D:\3BUPT\Agent\CSA\v3muti\results\csv_bench\simulation_gpt4v")
OUTPUT_EXCEL = OUTPUT_DIR / "gpt4v_mock_evaluation.xlsx"

RANDOM_SEED = 42


# ==================== 主函数 ====================

def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    # ------------------------------------------------
    # 1. 生成逐题预测结果
    # ------------------------------------------------
    for qid in range(NUM_QUESTIONS):
        level = random.choice(LEVELS)
        task = random.choice(TASK_TYPES)

        is_correct = random.random() < LEVEL_ACCURACY[level]

        record = {
            "question_id": qid,
            "difficulty_level": level,
            "task_type": task,
            "ground_truth": random.choice(["A", "B", "C", "D"]),
            "pred_correct": is_correct,
        }

        # L3 / L4 模拟 GPT-4o 辅助评分
        if level in ["L3", "L4"]:
            record["gpt4o_score"] = round(
                random.uniform(0.4, 0.85), 3
            )
        else:
            record["gpt4o_score"] = None

        records.append(record)

    df_predictions = pd.DataFrame(records)

    # ------------------------------------------------
    # 2. 按难度统计 Accuracy
    # ------------------------------------------------
    df_acc_by_level = (
        df_predictions.groupby("difficulty_level")["pred_correct"]
        .mean()
        .reset_index()
        .rename(columns={"pred_correct": "accuracy"})
    )

    # ------------------------------------------------
    # 3. 按任务类型统计 Accuracy
    # ------------------------------------------------
    df_acc_by_task = (
        df_predictions.groupby("task_type")["pred_correct"]
        .mean()
        .reset_index()
        .rename(columns={"pred_correct": "accuracy"})
    )

    # ------------------------------------------------
    # 4. 写入同一个 Excel 文件，不同 Sheet
    # ------------------------------------------------
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_predictions.to_excel(
            writer,
            sheet_name="predictions",
            index=False
        )

        df_acc_by_level.to_excel(
            writer,
            sheet_name="accuracy_by_level",
            index=False
        )

        df_acc_by_task.to_excel(
            writer,
            sheet_name="accuracy_by_task",
            index=False
        )

    # ------------------------------------------------
    # 5. 控制台提示
    # ------------------------------------------------
    print("=" * 60)
    print("✓ GPT-4V 模拟评估 Excel 文件生成完成")
    print("=" * 60)
    print(f"输出文件路径: {OUTPUT_EXCEL}")
    print("\n包含的 Sheets:")
    print(" - predictions（逐题预测结果）")
    print(" - accuracy_by_level（按 L1–L4）")
    print(" - accuracy_by_task（按任务类型）")
    print("=" * 60)
    print("示例（按难度 Accuracy）:")
    print(df_acc_by_level)
    print("=" * 60)


if __name__ == "__main__":
    main()
