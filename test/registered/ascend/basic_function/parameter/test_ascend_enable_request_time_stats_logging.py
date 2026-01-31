import unittest
import requests
import time
import importlib.util
from pathlib import Path
import os
import glob

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
# 直接导入ascend模块（无try，满足你的要求）
from sglang.test import ascend

# 注册CI，保持原有配置
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

def get_ascend_test_dir():
    """获取 sglang.test.ascend 包的绝对目录路径（无try，直接导入模块）"""
    # 直接使用已导入的ascend模块，获取其__file__属性
    ascend_file_path = Path(ascend.__file__)
    # 处理.pyc编译缓存文件，切换为.py文件路径（保证目录一致性）
    if ascend_file_path.suffix == ".pyc":
        ascend_file_path = ascend_file_path.with_suffix(".py")
    # 返回模块文件的父目录（即ascend包的目录）
    return ascend_file_path.parent

# 基础目录
ASCEND_TEST_DIR = get_ascend_test_dir()
# 请求日志输出目录（通过 --log-requests-target 指定该路径）
REQUEST_LOG_DIR = ASCEND_TEST_DIR / "request_logs"
# 日志文件匹配模式（匹配.log格式，与实际生成的日志文件一致）
REQUEST_LOG_PATTERN = str(REQUEST_LOG_DIR / "*.log")

# 打印路径信息（用于验证目录是否精准匹配，方便排查）
print(f"ascend 测试目录：{ASCEND_TEST_DIR.absolute()}")
print(f"日志存储目录：{REQUEST_LOG_DIR.absolute()}")
print(f"日志匹配规则：{REQUEST_LOG_PATTERN}")

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase：Verify --enable-request-time-stats-logging via --log-requests-target (directory path)

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging / --log-requests-target
    [Core Check] Request log file contains "Req Time Stats"
    """

    @classmethod
    def setUpClass(cls):
        # 步骤1：清理旧的请求日志（保证测试环境干净）
        cls.clear_old_request_logs()

        # 步骤2：创建请求日志目录（若不存在）
        REQUEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # 步骤3：启动服务，指定 --log-requests-target 为目录路径（核心参数，单个目录目标）
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging",
            # 1. 开启请求日志总开关（必须添加，解决日志为空的核心）
            "--log-requests",
            # 2. 开启请求时间统计（原有配置，保留）
            "--enable-request-time-stats-logging",
            # 3. 可选：指定日志详细级别（使用默认 2 也可，显式配置更清晰）
            "--log-requests-level", "2",
            "--log-requests-target", str(REQUEST_LOG_DIR)  # 单个目录目标，确保日志落地
        ]

        # 启动模型服务
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args
        )

        # 等待服务初始化 + 校验服务是否启动成功（无try，直接轮询验证）
        max_wait = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start_time = time.time()
        service_ready = False
        while time.time() - start_time < max_wait:
            resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info", timeout=5)
            if resp.status_code == 200:
                service_ready = True
                break
            time.sleep(1)
        
        # 服务未启动成功则终止测试
        if not service_ready:
            raise RuntimeError(f"服务启动超时，超过 {max_wait} 秒，无法继续测试")

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程及其子进程
        kill_process_tree(cls.process.pid)

    @staticmethod
    def clear_old_request_logs():
        """清理旧的请求日志文件（无try，直接删除匹配文件）"""
        old_log_files = glob.glob(REQUEST_LOG_PATTERN)
        for log_file in old_log_files:
            os.remove(log_file)

    def get_latest_request_log_content(self, timeout=30, interval=2):
        """获取最新生成的请求日志文件内容（增加轮询，确保获取非空内容）"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            log_files = glob.glob(REQUEST_LOG_PATTERN)
            if not log_files:
                time.sleep(interval)
                continue
            
            # 按修改时间排序，获取最新日志文件
            latest_log_file = max(log_files, key=os.path.getmtime)
            
            # 读取日志内容（忽略编码错误，确保正常读取）
            with open(latest_log_file, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read().strip()
                if log_content:
                    return log_content
            
            time.sleep(interval)
        
        # 超时返回空字符串
        return ""

    def test_enable_request_time_stats_logging(self):
        """Core Test: Verify parameter takes effect via request log file"""
        # Step 1: 发送推理请求，触发 Req Time Stats 日志生成
        requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )

        # Step 2: 等待日志异步写入（延长轮询，无需额外sleep）
        log_content = self.get_latest_request_log_content()

        # Step 3: 校验1 - 服务端配置中 enable-request-time-stats-logging 已开启
        server_info_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(server_info_resp.status_code, 200, "Server info API request failed")
        self.assertTrue(
            server_info_resp.json().get("enable_request_time_stats_logging", False),
            "--enable-request-time-stats-logging not enabled in server config"
        )

        # Step 4: 核心校验2 - 最新日志文件中包含 Req Time Stats
        self.assertNotEqual(log_content, "", "No request log content found (log file may not be generated)")
        self.assertIn(
            "Req Time Stats", log_content,
            "Missing 'Req Time Stats' in request log file -- parameter may not take effect"
        )


if __name__ == "__main__":
    unittest.main()
