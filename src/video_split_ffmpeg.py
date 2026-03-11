"""
文件名：video_splitter.py
功能介绍：使用 FFmpeg 将视频批量切割为固定时长的片段（默认15秒，去除音频）。
         支持两层目录结构：输入根目录下按工人文件夹组织，
         输出同样保留工人文件夹 → 视频名文件夹的层级结构。
输入：
  - 输入文件夹（含多个工人子文件夹，每个工人文件夹下有若干视频）
  - 切割时长（秒）
输出：
  - 输出文件夹/工人名/视频名/视频名_part_000.mp4 ...
  - 日志文件 → 输出文件夹/logs/video_split_时间戳.log
依赖：FFmpeg 需已安装并加入系统 PATH
路径配置：优先读取同目录下 config.json，也可通过命令行参数覆盖

需要改动的变量（config.json 中）：
  input_folder    — 输入视频根目录
  output_folder   — 输出根目录
  segment_duration — 切割时长（秒），默认 15
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分割工具
按指定时长切割视频文件
"""
# ==================== 需要改动的变量 ====================
INPUT_FOLDER     = r"F:\raw_videos_huafeng\刘宗锋（木工）_PU_31806153"       # 输入视频根目录
OUTPUT_FOLDER    = r"G:\huafeng_seg"   # 输出根目录
SEGMENT_DURATION = 15                  # 切割时长（秒）
# ======================================================
import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def setup_logging(output_folder):
    """配置日志系统"""
    log_folder = Path(output_folder) / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / f"video_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def check_ffmpeg():
    """检查FFmpeg是否已安装"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def scan_video_files(input_folder, video_extensions):
    """扫描两层文件夹结构中的视频文件
    返回: [(视频文件路径, 工人文件夹名称), ...]
    """
    video_files = []
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")

    # 遍历所有子文件夹（工人文件夹）
    for worker_folder in input_path.iterdir():
        if not worker_folder.is_dir():
            continue

        worker_name = worker_folder.name

        # 在工人文件夹中查找视频文件
        for ext in video_extensions:
            for video_file in worker_folder.glob(f"*.{ext}"):
                video_files.append((video_file, worker_name))
            for video_file in worker_folder.glob(f"*.{ext.upper()}"):
                video_files.append((video_file, worker_name))

    return sorted(video_files, key=lambda x: (x[1], x[0].name))


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logging.warning(f"无法获取视频时长 {video_path.name}: {e}")
    return None


def split_video(video_path, worker_name, output_folder, segment_duration, logger):
    """切割单个视频文件"""
    video_name = video_path.stem
    video_ext = video_path.suffix

    # 创建输出子文件夹：输出根目录/工人名/视频名
    output_subfolder = Path(output_folder) / worker_name / video_name
    output_subfolder.mkdir(parents=True, exist_ok=True)

    # 输出文件命名模板
    output_pattern = output_subfolder / f"{video_name}_part_%03d{video_ext}"

    logger.info(f"开始处理: {video_path.name}")

    # 获取视频时长
    duration = get_video_duration(video_path)
    if duration:
        expected_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
        logger.info(f"视频时长: {duration:.2f}秒, 预计切割为 {expected_segments} 段")

    # FFmpeg切割命令
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-c:v', 'copy',  # 视频流直接复制
        '-an',  # 去除所有音频流
        '-segment_time', str(segment_duration),
        '-f', 'segment',
        '-reset_timestamps', '1',
        str(output_pattern)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )

        if result.returncode == 0:
            # 统计生成的片段数量
            segments = list(output_subfolder.glob(f"{video_name}_part_*{video_ext}"))
            logger.info(f"✓ 成功切割 {video_path.name} -> {len(segments)} 个片段")
            return True
        else:
            logger.error(f"✗ 切割失败 {video_path.name}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ 切割超时 {video_path.name}")
        return False
    except Exception as e:
        logger.error(f"✗ 切割出错 {video_path.name}: {e}")
        return False


def parse_arguments(default_config):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='视频分割工具 - 按指定时长切割视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python video_splitter.py
  python video_splitter.py --input "D:\\videos" --output "D:\\output"
  python video_splitter.py --duration 30
        """
    )

    parser.add_argument(
        '-i', '--input',
        default=default_config.get('input_folder', ''),
        help='输入视频文件夹路径'
    )

    parser.add_argument(
        '-o', '--output',
        default=default_config.get('output_folder', ''),
        help='输出文件夹路径'
    )

    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=default_config.get('segment_duration', 15),
        help='切割时长（秒），默认15秒'
    )

    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("视频分割工具 v1.0")
    print("=" * 60)

    config = {
        "input_folder": INPUT_FOLDER,
        "output_folder": OUTPUT_FOLDER,
        "segment_duration": SEGMENT_DURATION
    }
    args = parse_arguments(config)

    # 检查必要参数
    if not args.input or not args.output:
        print("\n错误: 必须指定输入和输出文件夹路径")
        print("方式1: 在 config.json 中配置")
        print("方式2: 使用命令行参数 --input 和 --output")
        sys.exit(1)

    # 设置日志
    logger = setup_logging(args.output)

    logger.info("=" * 60)
    logger.info("视频分割任务开始")
    logger.info(f"输入文件夹: {args.input}")
    logger.info(f"输出文件夹: {args.output}")
    logger.info(f"切割时长: {args.duration}秒")
    logger.info("=" * 60)

    # 检查FFmpeg
    if not check_ffmpeg():
        logger.error("错误: 未检测到FFmpeg，请先安装FFmpeg")
        logger.error("下载地址: https://ffmpeg.org/download.html")
        sys.exit(1)

    logger.info("✓ FFmpeg环境检查通过")

    # 扫描视频文件
    video_extensions = config.get('video_extensions', ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'])

    try:
        video_files = scan_video_files(args.input, video_extensions)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not video_files:
        logger.warning(f"在 {args.input} 中未找到视频文件")
        sys.exit(0)

    logger.info(f"找到 {len(video_files)} 个视频文件")

    # 批量处理
    success_count = 0
    fail_count = 0

    for idx, (video_file, worker_name) in enumerate(video_files, 1):
        logger.info(f"\n[{idx}/{len(video_files)}] 处理中...")
        logger.info(f"工人: {worker_name}")

        if split_video(video_file, worker_name, args.output, args.duration, logger):
            success_count += 1
        else:
            fail_count += 1

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("任务完成")
    logger.info(f"成功: {success_count} 个")
    logger.info(f"失败: {fail_count} 个")
    logger.info(f"总计: {len(video_files)} 个")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()