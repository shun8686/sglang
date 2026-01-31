import unittest
import requests
import time
import importlib.util
from io import StringIO

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
        return importlib.util.find_spec("sglang.test.ascend").origin.parent
    from sglang.test import ascend
    return ascend.__file__.parent

ASCEND_TEST_DIR = get_ascend_test_dir()

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase：Verify --enable-request-time-stats-logging writes core time stats to console (via --log-requests-target stdout)

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging / --log-requests-target
    [Core Check] Console request log contains "Req Time Stats"
    """

    # 类级变量：内存缓存控制台请求日志
    server_request_console_log = StringIO()

    @classmethod
    def setUpClass(cls):
        # 核心配置：添加 --log-requests-target stdout，指定请求日志输出到控制台
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging",
            "--log-requests-target", "stdout"  # 关键参数：请求日志输出到控制台
        ]

        # 启动模型服务，捕获控制台输出（stdout/stderr）到内存缓存
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            stdout=cls.server_request_console_log,
            stderr=cls.server_request_console_log,
            universal_newlines=True
        )

        # 等待服务初始化（确保请求日志功能就绪）
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程
        kill_process_tree(cls.process.pid)
        # 关闭内存缓存，释放资源
        cls.server_request_console_log.close()

    def get_captured_console_log(self):
        """读取捕获的控制台请求日志内容"""
        # 移动文件指针到缓存开头，避免读取遗漏
        self.server_request_console_log.seek(0)
        return self.server_request_console_log.read()

    def test_enable_request_time_stats_logging(self):
        """Core Test: Verify parameter takes effect via console request log"""
        # Step 1: 发送推理请求，触发 Req Time Stats 日志生成
        requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )

        # Step 2: 等待请求日志异步输出到控制台（短暂延迟，确保缓存完整）
        time.sleep(2)

        # Step 3: 校验1 - 服务端配置中 enable-request-time-stats-logging 已开启
        server_info_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(server_info_resp.status_code, 200, "Server info API request failed")
        self.assertTrue(
            server_info_resp.json().get("enable_request_time_stats_logging", False),
            "--enable-request-time-stats-logging not enabled in server config"
        )

        # Step 4: 核心校验2 - 捕获的控制台日志中包含 Req Time Stats（请求日志核心标识）
        console_log = self.get_captured_console_log()
        self.assertIn(
            "Req Time Stats", console_log,
            "Missing 'Req Time Stats' in console request log -- parameter may not take effect"
        )

        # 可选：校验关键耗时字段（按需保留，极简不冗余）
        self.assertIn(
            "forward_duration", console_log,
            "Missing 'forward_duration' in console request log"
        )

if __name__ == "__main__":
    unittest.main()
