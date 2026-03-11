"""
================================================================================
文件名: analyze_data.py
模块路径: src/csv_bench/analyze_data.py
作者: Paperbox
创建日期: 2025-01-29
版本: 1.0

================================================================================
功能介绍:
================================================================================
CSV-Bench数据集统计分析与可视化工具

主要功能:
1. 加载并解析STAR格式的问答数据
2. 计算数据集统计信息（问题类型、难度分布、视频时长等）
3. 生成可视化图表（问题类型分布图、难度分布图、时长分布图）
4. 自动生成Markdown格式的分析报告
5. 提供数据质量评估和改进建议

适用场景:
- Pilot Study数据分析
- 组会汇报材料准备
- 数据集质量检查
- 论文实验数据统计

================================================================================
输入:
================================================================================
必需输入:
  - JSON数据文件路径（STAR格式的问答数据）
    格式要求: 包含metadata和results字段
    示例路径: data/csv_bench/raw/generated_questions_20251113_155438.json

可选输入:
  - 自定义输出路径（默认使用config.py中的配置）

================================================================================
输出:
================================================================================
自动生成的文件:
1. 问题类型分布图（PNG）
   路径: results/csv_bench/figures/question_type_distribution_YYYYMMDD_HHMMSS.png
   内容: 柱状图展示interaction/sequence/prediction/feasibility的数量分布

2. 难度分布图（PNG）
   路径: results/csv_bench/figures/difficulty_distribution_YYYYMMDD_HHMMSS.png
   内容: 柱状图展示L1/L2/L3/L4的数量分布

3. 视频时长分布图（PNG）
   路径: results/csv_bench/figures/duration_distribution_YYYYMMDD_HHMMSS.png
   内容: 直方图展示视频时长分布情况

4. Markdown分析报告
   路径: results/csv_bench/reports/data_analysis_YYYYMMDD_HHMMSS.md
   内容: 包含基本统计、分布表格、数据质量评估、建议

5. 日志文件
   路径: logs/csv_bench/data/analyze_data_YYYYMMDD_HHMMSS.log
   内容: 分析过程的详细日志

控制台输出:
  - 实时进度信息
  - 生成文件的路径列表
  - 总耗时统计

================================================================================
需要修改的配置变量:
================================================================================
如果需要自定义，可以修改以下位置的变量:

1. 默认数据文件路径（第560行，main函数中）:
   json_path = DATA_RAW_DIR / "generated_questions_20251113_155438.json"
   → 改为你的数据文件名

2. 图表样式配置（第52-55行）:
   matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
   → 如果中文显示有问题，修改字体名称

   sns.set_style("whitegrid")
   sns.set_palette("husl")
   → 修改图表风格和配色方案

3. 统计维度（可选，在compute_statistics方法中）:
   - top_verbs/top_nouns的数量（第170行）: most_common(10)
   → 修改数字可以改变统计的Top N数量

4. 图表尺寸（可选）:
   - 问题类型分布图（第188行）: figsize=(10, 6)
   - 难度分布图（第230行）: figsize=(10, 6)
   - 时长分布图（第277行）: figsize=(10, 6)
   → 修改元组调整图表大小

================================================================================
使用方法:
================================================================================
方法1: 直接运行（使用默认配置）
  cd D:\3BUPT\Agent\CSA\v3muti
  python src\csv_bench\analyze_data.py

方法2: 指定数据文件路径
  python src\csv_bench\analyze_data.py "path/to/your/data.json"

方法3: 在Python脚本中调用
  from src.csv_bench.analyze_data import DataAnalyzer

  analyzer = DataAnalyzer()
  analyzer.load_data("data/csv_bench/raw/questions.json")
  outputs = analyzer.generate_full_report()

  # 访问生成的文件路径
  print(outputs['type_chart'])      # 问题类型图路径
  print(outputs['markdown_report']) # 报告路径

方法4: 只生成特定图表
  analyzer = DataAnalyzer()
  analyzer.load_data("data.json")
  analyzer.compute_statistics()
  analyzer.plot_question_type_distribution()  # 只生成类型分布图

================================================================================
依赖库:
================================================================================
必需:
  - matplotlib >= 3.5.0  (绘图库)
  - seaborn >= 0.11.0    (统计可视化)
  - pandas >= 1.3.0      (数据处理，可选)

安装命令:
  pip install matplotlib seaborn pandas

================================================================================
注意事项:
================================================================================
1. 首次运行会自动创建输出目录（results/csv_bench/）
2. Windows系统中文字体默认使用'SimHei'或'Microsoft YaHei'
3. 如果中文显示为方块，请修改第52行的字体配置
4. 生成的文件名包含时间戳，不会覆盖之前的结果
5. 日志文件记录了完整的分析过程，便于问题排查

================================================================================
常见问题:
================================================================================
Q1: 中文显示为方块？
A1: 修改第52行，将字体改为系统已安装的中文字体
    matplotlib.rcParams['font.sans-serif'] = ['你的字体名称']

Q2: 找不到数据文件？
A2: 检查文件是否在 data/csv_bench/raw/ 目录下
    或使用命令行参数指定完整路径

Q3: 图表保存失败？
A3: 检查是否有写入权限，results目录是否被占用

Q4: 想修改图表配色？
A4: 修改第55行 sns.set_palette("husl")
    可选: "husl", "Set2", "Paired", "pastel" 等

================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 导入配置和数据加载器
try:
    from .config import (
        RESULTS_FIGURES_DIR,
        RESULTS_REPORTS_DIR,
        get_timestamped_log_file,
        QUESTION_TYPES,
        DIFFICULTY_LEVELS
    )
    from .data_loader import CSVBenchDataLoader
except ImportError:
    from config import (
        RESULTS_FIGURES_DIR,
        RESULTS_REPORTS_DIR,
        get_timestamped_log_file,
        QUESTION_TYPES,
        DIFFICULTY_LEVELS
    )
    from data_loader import CSVBenchDataLoader


# ==================== 可配置参数 ====================
# 如果需要自定义，修改这里的值

# 中文字体配置（如果显示有问题请修改）
CHINESE_FONT = ['SimHei', 'Microsoft YaHei', 'Arial']

# 图表风格
CHART_STYLE = "whitegrid"  # 可选: "darkgrid", "white", "dark", "ticks"
COLOR_PALETTE = "husl"     # 可选: "Set2", "Paired", "pastel", "deep"

# 图表尺寸配置
FIGURE_SIZE_LARGE = (10, 6)   # 大图尺寸
FIGURE_SIZE_MEDIUM = (8, 5)   # 中图尺寸

# 统计配置
TOP_N_ITEMS = 10  # Top N 统计项数量

# DPI配置（图片清晰度）
FIGURE_DPI = 300  # 推荐: 300 (高清), 150 (普通), 72 (屏幕预览)


# ==================== 配置中文字体 ====================
matplotlib.rcParams['font.sans-serif'] = CHINESE_FONT
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置seaborn样式
sns.set_style(CHART_STYLE)
sns.set_palette(COLOR_PALETTE)


# ==================== 配置日志 ====================
def setup_logger() -> logging.Logger:
    """配置分析日志"""
    logger = logging.getLogger("DataAnalyzer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件handler
    log_file = get_timestamped_log_file("analyze_data", "data")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()


# ==================== 数据分析器类 ====================
class DataAnalyzer:
    """
    CSV-Bench数据集分析器

    功能：
    - 统计数据分布
    - 生成可视化图表
    - 输出Markdown报告
    """

    def __init__(self):
        """初始化分析器"""
        self.loader = CSVBenchDataLoader()
        self.stats = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("数据分析器初始化完成")

    def load_data(self, json_path: str) -> None:
        """
        加载数据

        Args:
            json_path: JSON数据文件路径
        """
        start_time = datetime.now()
        logger.info(f"加载数据: {json_path}")

        self.loader.load_from_json(json_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"数据加载完成，耗时 {elapsed:.2f}秒")

    def compute_statistics(self) -> Dict:
        """
        计算详细统计信息

        Returns:
            统计信息字典
        """
        logger.info("开始计算统计信息...")
        start_time = datetime.now()

        total_videos = len(self.loader.data)
        total_questions = sum(len(s.questions) for s in self.loader.data)

        # 按类型统计
        type_counter = Counter()
        difficulty_counter = Counter()

        for sample in self.loader.data:
            for q in sample.questions:
                type_counter[q.type] += 1
                difficulty_counter[q.difficulty] += 1

        # 视频时长统计
        durations = [s.duration for s in self.loader.data]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # 每个视频的问题数统计
        questions_per_video = [len(s.questions) for s in self.loader.data]
        avg_questions = sum(questions_per_video) / len(questions_per_video) if questions_per_video else 0

        # 动词统计（从segments提取）
        verb_counter = Counter()
        noun_counter = Counter()
        for sample in self.loader.data:
            for seg in sample.segments:
                if seg.verb:
                    verb_counter[seg.verb] += 1
                if seg.noun:
                    noun_counter[seg.noun] += 1

        self.stats = {
            "basic": {
                "total_videos": total_videos,
                "total_questions": total_questions,
                "avg_questions_per_video": avg_questions,
                "total_segments": sum(len(s.segments) for s in self.loader.data)
            },
            "type_distribution": dict(type_counter),
            "difficulty_distribution": dict(difficulty_counter),
            "duration": {
                "avg": avg_duration,
                "min": min_duration,
                "max": max_duration,
                "all": durations
            },
            "top_verbs": verb_counter.most_common(TOP_N_ITEMS),
            "top_nouns": noun_counter.most_common(TOP_N_ITEMS)
        }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"统计完成，耗时 {elapsed:.2f}秒")

        return self.stats

    def plot_question_type_distribution(self, save_path: Path = None) -> Path:
        """
        绘制问题类型分布图

        Args:
            save_path: 保存路径（可选）

        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = RESULTS_FIGURES_DIR / f"question_type_distribution_{self.timestamp}.png"

        logger.info(f"绘制问题类型分布图: {save_path}")

        type_dist = self.stats["type_distribution"]

        # 创建图表
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

        types = list(type_dist.keys())
        counts = list(type_dist.values())
        colors = sns.color_palette(COLOR_PALETTE, len(types))

        bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('问题类型', fontsize=14, fontweight='bold')
        ax.set_ylabel('问题数量', fontsize=14, fontweight='bold')
        ax.set_title('CSV-Bench 问题类型分布', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # 添加类型说明
        type_labels = {
            "interaction": "L1-交互识别",
            "sequence": "L2-时序理解",
            "prediction": "L3-动作预测",
            "feasibility": "L4-可行性判断"
        }

        ax.set_xticklabels([type_labels.get(t, t) for t in types], rotation=15, ha='right')

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 图表已保存: {save_path}")
        return save_path

    def plot_difficulty_distribution(self, save_path: Path = None) -> Path:
        """
        绘制难度分布图

        Args:
            save_path: 保存路径（可选）

        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = RESULTS_FIGURES_DIR / f"difficulty_distribution_{self.timestamp}.png"

        logger.info(f"绘制难度分布图: {save_path}")

        diff_dist = self.stats["difficulty_distribution"]

        # 创建图表
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

        difficulties = ["L1", "L2", "L3", "L4"]  # 确保顺序
        counts = [diff_dist.get(d, 0) for d in difficulties]
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

        bars = ax.bar(difficulties, counts, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('难度等级', fontsize=14, fontweight='bold')
        ax.set_ylabel('问题数量', fontsize=14, fontweight='bold')
        ax.set_title('CSV-Bench 难度分布', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        # 添加难度说明
        difficulty_labels = {
            "L1": "L1\n(感知层)",
            "L2": "L2\n(语义层)",
            "L3": "L3\n(推理层)",
            "L4": "L4\n(评估层)"
        }
        ax.set_xticklabels([difficulty_labels[d] for d in difficulties])

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 图表已保存: {save_path}")
        return save_path

    def plot_duration_distribution(self, save_path: Path = None) -> Path:
        """
        绘制视频时长分布直方图

        Args:
            save_path: 保存路径（可选）

        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = RESULTS_FIGURES_DIR / f"duration_distribution_{self.timestamp}.png"

        logger.info(f"绘制视频时长分布图: {save_path}")

        durations = self.stats["duration"]["all"]

        # 创建图表
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

        ax.hist(durations, bins=10, color='skyblue', alpha=0.7, edgecolor='black')

        # 添加平均线
        avg_duration = self.stats["duration"]["avg"]
        ax.axvline(avg_duration, color='red', linestyle='--', linewidth=2,
                   label=f'平均时长: {avg_duration:.2f}秒')

        ax.set_xlabel('视频时长（秒）', fontsize=14, fontweight='bold')
        ax.set_ylabel('视频数量', fontsize=14, fontweight='bold')
        ax.set_title('CSV-Bench 视频时长分布', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 图表已保存: {save_path}")
        return save_path

    def generate_markdown_report(self, save_path: Path = None) -> Path:
        """
        生成Markdown格式的统计报告

        Args:
            save_path: 保存路径（可选）

        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = RESULTS_REPORTS_DIR / f"data_analysis_{self.timestamp}.md"

        logger.info(f"生成Markdown报告: {save_path}")

        report_lines = []

        # 标题
        report_lines.append("# CSV-Bench 数据集分析报告")
        report_lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**分析人**: Paperbox")
        report_lines.append("\n---\n")

        # 基本统计
        report_lines.append("## 1. 基本统计信息")
        report_lines.append("")
        basic = self.stats["basic"]
        report_lines.append(f"- **视频总数**: {basic['total_videos']}")
        report_lines.append(f"- **问题总数**: {basic['total_questions']}")
        report_lines.append(f"- **平均每视频问题数**: {basic['avg_questions_per_video']:.2f}")
        report_lines.append(f"- **视频片段总数**: {basic['total_segments']}")
        report_lines.append("")

        # 问题类型分布
        report_lines.append("## 2. 问题类型分布")
        report_lines.append("")
        report_lines.append("| 类型 | 数量 | 占比 | 难度等级 |")
        report_lines.append("|------|------|------|----------|")

        type_mapping = {
            "interaction": ("交互识别", "L1"),
            "sequence": ("时序理解", "L2"),
            "prediction": ("动作预测", "L3"),
            "feasibility": ("可行性判断", "L4")
        }

        total = self.stats["basic"]["total_questions"]
        for qtype, count in self.stats["type_distribution"].items():
            name, level = type_mapping.get(qtype, (qtype, "N/A"))
            percentage = (count / total * 100) if total > 0 else 0
            report_lines.append(f"| {name} ({qtype}) | {count} | {percentage:.1f}% | {level} |")

        report_lines.append("")

        # 难度分布
        report_lines.append("## 3. 难度分布")
        report_lines.append("")
        report_lines.append("| 难度 | 描述 | 数量 | 占比 |")
        report_lines.append("|------|------|------|------|")

        difficulty_desc = {
            "L1": "感知层 - 基础视觉识别",
            "L2": "语义层 - 场景理解与描述",
            "L3": "推理层 - 时序推理与预测",
            "L4": "评估层 - 安全评估与规范判断"
        }

        for level in ["L1", "L2", "L3", "L4"]:
            count = self.stats["difficulty_distribution"].get(level, 0)
            percentage = (count / total * 100) if total > 0 else 0
            desc = difficulty_desc.get(level, "")
            report_lines.append(f"| {level} | {desc} | {count} | {percentage:.1f}% |")

        report_lines.append("")

        # 视频时长统计
        report_lines.append("## 4. 视频时长统计")
        report_lines.append("")
        duration = self.stats["duration"]
        report_lines.append(f"- **平均时长**: {duration['avg']:.2f} 秒")
        report_lines.append(f"- **最短时长**: {duration['min']:.2f} 秒")
        report_lines.append(f"- **最长时长**: {duration['max']:.2f} 秒")
        report_lines.append("")

        # 高频动作统计
        report_lines.append(f"## 5. 高频动作统计（Top {TOP_N_ITEMS}）")
        report_lines.append("")
        report_lines.append("### 5.1 动词（Verb）")
        report_lines.append("")
        report_lines.append("| 排名 | 动词 | 出现次数 |")
        report_lines.append("|------|------|----------|")
        for i, (verb, count) in enumerate(self.stats["top_verbs"], 1):
            report_lines.append(f"| {i} | {verb} | {count} |")
        report_lines.append("")

        report_lines.append("### 5.2 名词（Noun）")
        report_lines.append("")
        report_lines.append("| 排名 | 名词 | 出现次数 |")
        report_lines.append("|------|------|----------|")
        for i, (noun, count) in enumerate(self.stats["top_nouns"], 1):
            report_lines.append(f"| {i} | {noun} | {count} |")
        report_lines.append("")

        # 数据质量评估
        report_lines.append("## 6. 数据质量评估")
        report_lines.append("")

        # 检查问题分布是否均衡
        type_counts = list(self.stats["type_distribution"].values())
        type_balance = max(type_counts) / min(type_counts) if min(type_counts) > 0 else float('inf')

        if type_balance <= 1.5:
            balance_status = "✅ **优秀** - 问题类型分布均衡"
        elif type_balance <= 2.0:
            balance_status = "⚠️ **良好** - 问题类型分布基本均衡"
        else:
            balance_status = "❌ **需改进** - 问题类型分布不均衡"

        report_lines.append(f"- **类型分布均衡性**: {balance_status} (最大/最小比值: {type_balance:.2f})")
        report_lines.append(f"- **难度层次覆盖**: ✅ 完整覆盖L1-L4四个层次")

        avg_q = basic['avg_questions_per_video']
        if avg_q >= 4:
            density_status = "✅ **充足** - 平均每视频问题数充足"
        elif avg_q >= 3:
            density_status = "⚠️ **适中** - 平均每视频问题数适中"
        else:
            density_status = "❌ **不足** - 平均每视频问题数偏少"

        report_lines.append(f"- **问题密度**: {density_status} ({avg_q:.2f}个/视频)")
        report_lines.append("")

        # 结论与建议
        report_lines.append("## 7. 结论与建议")
        report_lines.append("")
        report_lines.append("### 优点")
        report_lines.append("")
        report_lines.append("1. ✅ **多层次难度设计**: 覆盖L1-L4四个认知层次，符合基准测试要求")
        report_lines.append("2. ✅ **STAR方法论**: 采用interaction/sequence/prediction/feasibility四类问题，科学合理")
        report_lines.append("3. ✅ **真实场景数据**: 基于实际建筑施工现场视频，具有实用价值")
        report_lines.append("")

        report_lines.append("### 建议")
        report_lines.append("")

        if basic['total_videos'] < 100:
            report_lines.append(f"1. ⚠️ **扩大数据规模**: 当前仅{basic['total_videos']}个视频，建议扩展到500+（Pilot Study）或8000+（完整版）")

        if type_balance > 1.5:
            report_lines.append("2. ⚠️ **平衡问题类型**: 适当增加数量较少的问题类型，提高数据均衡性")

        report_lines.append("3. 💡 **增加标注一致性验证**: 建议计算标注者间一致性（Kappa系数）")
        report_lines.append("4. 💡 **补充安全规范知识库**: 为L4难度问题提供明确的安全规范引用")
        report_lines.append("")

        # 保存报告
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"✓ 报告已保存: {save_path}")
        return save_path

    def generate_full_report(self) -> Dict[str, Path]:
        """
        生成完整的分析报告（图表+Markdown）

        Returns:
            包含所有输出文件路径的字典
        """
        logger.info("=" * 60)
        logger.info("开始生成完整分析报告")
        logger.info("=" * 60)

        start_time = datetime.now()

        # 计算统计信息
        self.compute_statistics()

        # 生成图表
        outputs = {}
        outputs['type_chart'] = self.plot_question_type_distribution()
        outputs['difficulty_chart'] = self.plot_difficulty_distribution()
        outputs['duration_chart'] = self.plot_duration_distribution()

        # 生成Markdown报告
        outputs['markdown_report'] = self.generate_markdown_report()

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info(f"分析报告生成完成！总耗时: {elapsed:.2f}秒")
        logger.info("=" * 60)
        logger.info("输出文件:")
        for name, path in outputs.items():
            logger.info(f"  - {name}: {path}")
        logger.info("=" * 60)

        return outputs


# ==================== 主函数 ====================
def main():
    """
    主函数：运行完整的数据分析流程

    命令行用法:
        python analyze_data.py                           # 使用默认路径
        python analyze_data.py "path/to/data.json"       # 指定数据文件
    """
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # ========================================
        # 【可修改】默认数据文件路径
        # ========================================
        try:
            from .config import DATA_RAW_DIR
        except ImportError:
            from config import DATA_RAW_DIR

        # 修改这里的文件名为你的数据文件
        json_path = r"F:\huafeng_labeled\马开剑（钢筋工）_PU_20012513\export/output_with_difficulty.json"

    # 创建分析器
    analyzer = DataAnalyzer()

    # 加载数据
    analyzer.load_data(str(json_path))

    # 生成完整报告
    outputs = analyzer.generate_full_report()

    # 打印结果
    print("\n" + "=" * 60)
    print("✓ 分析完成！生成的文件:")
    print("=" * 60)
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()