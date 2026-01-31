import unittest
import requests
import time
from pathlib import Path
import os
import importlib.util

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 获取 sglang.test.ascend 包的目录，基于该目录创建日志路径
def get_ascend_test_dir():
    """获取 sglang.test.ascend 包的绝对目录路径"""
    spec = importlib.util.find_spec("sglang.test.ascend")
    if spec and spec.origin:
        return Path(spec.origin).parent
    from sglang.test import ascend
    return Path(ascend.__file__).parent

# 日志路径：sglang/test/ascend/logs/server.log
ASCEND_TEST_DIR = get_ascend_test_dir()
SGLANG_SERVER_LOG_PATH = ASCEND_TEST_DIR / "logs" / "server.log"


class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase：Verify --enable-request-time-stats-logging writes core time stats to server log

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    [Core Check] Server log contains "Req Time Stats"
    """

    @classmethod
    def setUpClass(cls):
        # 核心新增：启动服务前，先清理原有日志文件（保证测试环境干净）
        cls.clear_server_log()
        
        # 启动服务，携带目标参数
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging"
        ]

        # 启动模型服务
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        # 等待服务初始化和日志文件创建
        time.sleep(2)
        # 确保日志目录存在
        SGLANG_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程
        kill_process_tree(cls.process.pid)
    
    # 核心新增：定义清理日志文件的静态方法
    @staticmethod
    def clear_server_log():
        """清理服务端日志文件（删除原有文件，或清空文件内容）"""
        # 方案1：直接删除日志文件（下次写入会自动创建，更简洁）
        if SGLANG_SERVER_LOG_PATH.exists():
            os.remove(SGLANG_SERVER_LOG_PATH)
        
        # 方案2：清空文件内容（保留文件本身，按需选择）
        # if SGLANG_SERVER_LOG_PATH.exists():
        #     with open(SGLANG_SERVER_LOG_PATH, "w", encoding="utf-8") as f:
        #         f.truncate(0)

    def read_server_log(self):
        """读取完整服务端日志（简化版，直接读取全部内容）"""
        if not SGLANG_SERVER_LOG_PATH.exists():
            return ""
        with open(SGLANG_SERVER_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_request_time_stats_logging(self):
        """Core Test: Simplified verification for the parameter's core effect"""
        # Step 1: 发送1条推理请求，触发日志生成
        requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )

        # Step 2: 等待日志异步落盘（短暂延迟，确保日志写入）
        time.sleep(2)

        # Step 3: 核心校验1 - 服务端配置中参数已开启（简单校验）
        server_info_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(server_info_resp.status_code, 200, "Server info API request failed")
        self.assertTrue(
            server_info_resp.json().get("enable_request_time_stats_logging", False),
            "Parameter not enabled in server config"
        )

        # Step 4: 核心校验2 - 日志中包含 "Req Time Stats"（参数生效的核心标识，极简）
        log_content = self.read_server_log()
        self.assertIn(
            "Req Time Stats", log_content,
            "Missing core time stats log 'Req Time Stats' -- parameter may not take effect"
        )

        # 可选：额外校验1个关键耗时字段（按需保留，不冗余）
        self.assertIn(
            "forward_duration", log_content,
            "Missing key time stats field 'forward_duration'"
        )


if __name__ == "__main__":
    unittest.main()
