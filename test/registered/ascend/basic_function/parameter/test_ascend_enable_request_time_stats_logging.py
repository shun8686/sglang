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

# 注册CI，保持原有配置
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

def get_ascend_test_dir():
    """获取 sglang.test.ascend 包的绝对目录路径"""
    spec = importlib.util.find_spec("sglang.test.ascend")
    if spec and spec.origin:
        return Path(spec.origin).parent
    from sglang.test import ascend
    return Path(ascend.__file__).parent

# 基础目录
ASCEND_TEST_DIR = get_ascend_test_dir()
# 请求日志输出目录（通过 --log-requests-target 指定该路径）
REQUEST_LOG_DIR = ASCEND_TEST_DIR / "request_logs"
# 日志文件匹配模式（sglang 会在该目录下生成 .log 或 .txt 日志文件）
REQUEST_LOG_PATTERN = str(REQUEST_LOG_DIR / "*.log")

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

        # 步骤3：启动服务，指定 --log-requests-target 为目录路径（核心修改）
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging",
            "--log-requests-target", str(REQUEST_LOG_DIR)  # 传递目录路径，让日志写入该目录
        ]

        # 启动模型服务（无需传递 stdout/stderr，避开封装限制）
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args
        )

        # 等待服务初始化（确保日志功能就绪）
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程
        kill_process_tree(cls.process.pid)

    @staticmethod
    def clear_old_request_logs():
        """清理旧的请求日志文件"""
        # 匹配所有日志文件
        old_log_files = glob.glob(REQUEST_LOG_PATTERN)
        for log_file in old_log_files:
            try:
                os.remove(log_file)
            except Exception as e:
                print(f"Failed to delete old log file {log_file}: {e}")

    def get_latest_request_log_content(self):
        """获取最新生成的请求日志文件内容"""
        # 匹配所有日志文件
        log_files = glob.glob(REQUEST_LOG_PATTERN)
        if not log_files:
            return ""
        
        # 找到最新生成的日志文件（按修改时间排序）
        latest_log_file = max(log_files, key=os.path.getmtime)
        
        # 读取日志内容
        with open(latest_log_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

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

        # Step 2: 等待日志写入文件（sglang 异步写入，需给足够延迟）
        time.sleep(3)

        # Step 3: 校验1 - 服务端配置中 enable-request-time-stats-logging 已开启
        server_info_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(server_info_resp.status_code, 200, "Server info API request failed")
        self.assertTrue(
            server_info_resp.json().get("enable_request_time_stats_logging", False),
            "--enable-request-time-stats-logging not enabled in server config"
        )

        # Step 4: 核心校验2 - 最新日志文件中包含 Req Time Stats
        log_content = self.get_latest_request_log_content()
        self.assertNotEqual(log_content, "", "No request log content found (log file may not be generated)")
        self.assertIn(
            "Req Time Stats", log_content,
            "Missing 'Req Time Stats' in request log file -- parameter may not take effect"
        )

        # 可选：校验关键耗时字段
        self.assertIn(
            "forward_duration", log_content,
            "Missing 'forward_duration' in request log file"
        )

if __name__ == "__main__":
    unittest.main()
