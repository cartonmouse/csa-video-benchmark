"""
文件名：批量下载视频.py
功能介绍：从华凤平台批量下载违纪视频和整改视频。
         读取Excel清单，通过Selenium自动操作浏览器获取真实下载链接，
         再用requests下载视频文件，支持断点续传（跳过已存在文件）。
输入：Excel文件（含"隐患编号"、"违纪视频"、"整改视频"三列，视频列为embed链接）
输出：
  - 违纪视频 → DOWNLOAD_DIR/违纪视频/
  - 整改视频 → DOWNLOAD_DIR/整改视频/
  - 日志文件 → DOWNLOAD_DIR/download_log_时间戳.txt
依赖：pip install pandas requests selenium webdriver-manager openpyxl
路径配置：修改下方三个变量即可

需要改动的变量：
"""

# ==================== 需要改动的变量 ====================
EXCEL_FILE    = r"D:\3BUPT\Agent\CSA\v3muti\docs\下载清单\去冗余隐患记录数据可编辑.xlsx"
DOWNLOAD_DIR  = r"F:\v3muti\data\AntiVedio"   # 视频保存目录，None则保存到Excel同目录
DEBUG_MODE    = False                          # True时浏览器窗口可见，便于调试
# ======================================================

import pandas as pd
import requests
import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urlparse
import re
from datetime import datetime


class VideoDownloader:
    def __init__(self, excel_file_path, download_base_dir=None):
        self.excel_file = excel_file_path

        # 如果指定了下载目录，使用指定目录，否则使用Excel文件所在目录
        if download_base_dir:
            self.base_dir = os.path.abspath(download_base_dir)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(excel_file_path))

        self.violation_dir = os.path.join(self.base_dir, "违纪视频")
        self.rectification_dir = os.path.join(self.base_dir, "整改视频")
        self.log_file = os.path.join(self.base_dir, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        # 创建目录
        os.makedirs(self.violation_dir, exist_ok=True)
        os.makedirs(self.rectification_dir, exist_ok=True)

        # 配置日志
        self.setup_logging()

        # 初始化浏览器选项 (设置为False可以看到浏览器窗口进行调试)
        self.setup_browser_options(headless=True)

        # 统计信息
        self.total_videos = 0
        self.downloaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0

    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_browser_options(self, headless=True):
        """设置浏览器选项"""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')  # 无界面模式
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        # 增加页面加载策略
        self.chrome_options.add_argument('--page-load-strategy=eager')

    def read_excel_data(self):
        """读取Excel数据"""
        try:
            df = pd.read_excel(self.excel_file)
            self.logger.info(f"成功读取Excel文件，共{len(df)}条记录")
            return df
        except Exception as e:
            self.logger.error(f"读取Excel文件失败: {e}")
            return None

    def convert_embed_to_download_url(self, embed_url):
        """将embed链接转换为下载页面链接"""
        if not embed_url or pd.isna(embed_url):
            return None
        return embed_url.replace('/videos/embed/', '/w/')

    def get_download_link(self, page_url, max_wait_time=30):
        """使用Selenium获取真实下载链接"""
        driver = None
        try:
            # 自动下载并设置ChromeDriver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)

            self.logger.info(f"正在访问页面: {page_url}")
            driver.get(page_url)

            # 增加等待时间
            wait = WebDriverWait(driver, max_wait_time)

            # 等待页面完全加载
            self.logger.info("等待页面加载...")
            time.sleep(3)

            # 查找并点击下载按钮
            self.logger.info("查找下载按钮...")
            download_button = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button[aria-label='Open the modal to download this video']"))
            )
            self.logger.info("找到下载按钮，准备点击")
            download_button.click()

            # 等待模态框出现
            self.logger.info("等待下载模态框出现...")
            time.sleep(2)

            # 查找type选择框并选择"video-files"
            self.logger.info("查找类型选择框...")
            select_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='type']"))
            )

            # 选择"视频文件"选项
            self.logger.info("选择'视频文件'选项...")
            select = Select(select_element)
            select.select_by_value("video-files")

            # 等待页面更新显示下载链接
            self.logger.info("等待下载链接出现...")
            time.sleep(3)

            # 查找包含下载链接的input元素
            self.logger.info("查找下载链接...")
            download_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input.form-control[readonly]"))
            )

            download_link = download_input.get_attribute('value')

            if download_link and download_link.startswith('https://'):
                self.logger.info(f"成功获取下载链接: {download_link[:80]}...")
                return download_link
            else:
                self.logger.error(f"获取的链接格式不正确: {download_link}")
                return None

        except TimeoutException:
            self.logger.error(f"获取下载链接超时: {page_url}")
            return None
        except NoSuchElementException as e:
            self.logger.error(f"页面元素未找到: {page_url}, 错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"获取下载链接时发生错误: {page_url}, 错误: {e}")
            return None
        finally:
            if driver:
                driver.quit()

    def download_video(self, download_url, file_path, max_retries=3):
        """下载视频文件"""
        if not download_url:
            return False

        for attempt in range(max_retries):
            try:
                self.logger.info(f"开始下载: {os.path.basename(file_path)} (尝试 {attempt + 1}/{max_retries})")

                response = requests.get(download_url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            downloaded_size += len(chunk)

                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\r下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')

                print()  # 换行
                self.logger.info(f"下载成功: {os.path.basename(file_path)}")
                return True

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # 等待5秒后重试

        self.logger.error(f"下载失败，已达最大重试次数: {os.path.basename(file_path)}")
        return False

    def is_file_downloaded(self, file_path):
        """检查文件是否已下载（用于断点续传）"""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def process_video(self, embed_url, hazard_id, video_type):
        """处理单个视频的下载"""
        if not embed_url or pd.isna(embed_url):
            self.logger.info(f"跳过空链接: {hazard_id} - {video_type}")
            self.skipped_count += 1
            return

        # 确定文件名和保存目录
        if video_type == "违纪":
            file_name = f"{hazard_id}WJ.mp4"
            save_dir = self.violation_dir
        else:
            file_name = f"{hazard_id}ZG.mp4"
            save_dir = self.rectification_dir

        file_path = os.path.join(save_dir, file_name)

        # 检查文件是否已存在
        if self.is_file_downloaded(file_path):
            self.logger.info(f"文件已存在，跳过: {file_name}")
            self.skipped_count += 1
            return

        # 转换URL
        page_url = self.convert_embed_to_download_url(embed_url)
        if not page_url:
            self.logger.error(f"URL转换失败: {embed_url}")
            self.failed_count += 1
            return

        self.logger.info(f"处理视频: {hazard_id} - {video_type}")
        self.logger.info(f"页面URL: {page_url}")

        # 获取下载链接
        download_link = self.get_download_link(page_url)
        if not download_link:
            self.logger.error(f"获取下载链接失败: {hazard_id} - {video_type}")
            self.failed_count += 1
            return

        self.logger.info(f"获取到下载链接: {download_link}")

        # 下载文件
        if self.download_video(download_link, file_path):
            self.downloaded_count += 1
            self.logger.info(f"✅ 下载成功: {file_name}")
        else:
            self.failed_count += 1
            self.logger.error(f"❌ 下载失败: {file_name}")

    def run(self):
        """运行主程序"""
        self.logger.info("=== 开始视频批量下载 ===")
        self.logger.info(f"Excel文件: {self.excel_file}")
        self.logger.info(f"违纪视频保存目录: {self.violation_dir}")
        self.logger.info(f"整改视频保存目录: {self.rectification_dir}")

        # 读取Excel数据
        df = self.read_excel_data()
        if df is None:
            return

        # 计算总视频数量
        violation_count = df['违纪视频'].notna().sum()
        rectification_count = df['整改视频'].notna().sum()
        self.total_videos = violation_count + rectification_count

        self.logger.info(
            f"待下载视频总数: {self.total_videos} (违纪视频: {violation_count}, 整改视频: {rectification_count})")

        start_time = time.time()

        # 处理每一行数据
        for index, row in df.iterrows():
            hazard_id = row['隐患编号']
            violation_url = row['违纪视频']
            rectification_url = row['整改视频']

            self.logger.info(f"\n处理第 {index + 1}/{len(df)} 条记录: {hazard_id}")

            # 处理违纪视频
            if pd.notna(violation_url):
                self.process_video(violation_url, hazard_id, "违纪")

            # 处理整改视频
            if pd.notna(rectification_url):
                self.process_video(rectification_url, hazard_id, "整改")

            # 显示总体进度
            completed = self.downloaded_count + self.failed_count + self.skipped_count
            progress = (completed / self.total_videos) * 100 if self.total_videos > 0 else 0
            self.logger.info(f"总进度: {progress:.1f}% ({completed}/{self.total_videos})")

        # 显示最终统计
        end_time = time.time()
        duration = end_time - start_time

        self.logger.info("\n=== 下载完成 ===")
        self.logger.info(f"总耗时: {duration:.1f} 秒")
        self.logger.info(f"下载成功: {self.downloaded_count}")
        self.logger.info(f"下载失败: {self.failed_count}")
        self.logger.info(f"跳过文件: {self.skipped_count}")
        self.logger.info(f"日志文件: {self.log_file}")


def main():
    """主函数"""
    excel_file = EXCEL_FILE
    download_dir = DOWNLOAD_DIR
    debug_mode = DEBUG_MODE

    if not os.path.exists(excel_file):
        print(f"错误: Excel文件不存在: {excel_file}")
        print("请确保Excel文件在脚本同一目录下，或修改文件路径")
        return

    # 如果指定了下载目录，检查目录是否存在，不存在则创建
    if download_dir:
        if not os.path.exists(download_dir):
            try:
                os.makedirs(download_dir, exist_ok=True)
                print(f"创建下载目录: {download_dir}")
            except Exception as e:
                print(f"创建下载目录失败: {e}")
                return
        print(f"视频将下载到: {download_dir}")
    else:
        print(f"视频将下载到Excel文件同一目录: {os.path.dirname(os.path.abspath(excel_file))}")

    downloader = VideoDownloader(excel_file, download_dir)

    # 如果是调试模式，修改浏览器选项
    if debug_mode:
        print("🐛 调试模式已启用 - 浏览器窗口将可见")
        downloader.setup_browser_options(headless=False)

    downloader.run()


if __name__ == "__main__":
    main()