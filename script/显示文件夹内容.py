"""
文件名：显示文件夹内容.py
功能介绍：递归打印指定文件夹下的所有文件和子文件夹，以树形结构展示，用于快速了解目录内容。
输入：TARGET_PATH — 需要查看的文件夹路径
输出：控制台打印目录树（无文件写入）
路径配置：修改下方 TARGET_PATH 即可

需要改动的变量：
"""

# ==================== 需要改动的变量 ====================
TARGET_PATH = r"D:\3BUPT\Agent\CSA\v3muti\data\epic-kitchens-100-annotations"
# ======================================================

import os


def show_folder_contents(path, depth=0):
    """展示指定路径下的所有内容，包括子文件夹"""
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return

    indent = "  " * depth
    if depth == 0:
        print(f"📁 {path} 的内容:")
        print("-" * 50)

    items = os.listdir(path)
    for item in sorted(items):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            # 递归显示子文件夹内容
            show_folder_contents(item_path, depth + 1)
        else:
            print(f"{indent}📄 {item}")


# 使用
show_folder_contents(TARGET_PATH)