import unittest
import requests
import time
from pathlib import Path
import os
import importlib.util
from datetime import datetime

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

# 日志路径：sglang/test/ascend/logs/server.log（本次会让服务端真的写入这个路径）
ASCEND_TEST_DIR = get_ascend_test_dir()
SGLANG_SERVER_LOG_PATH = ASCEND_TEST_DIR / "logs" / "server.log"
# 转储日志目录：sglang/test/ascend/logs/backup/
SGLANG_LOG_BACKUP_DIR = ASCEND_TEST_DIR / "logs" / "backup"

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase：Verify --enable-request-time-stats-logging writes core time stats to server log

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    [Core Check] Server log contains "Req Time Stats"
    """

    @classmethod
    def setUpClass(cls):
        # 清理原有日志文件（保证测试环境干净）
        cls.clear_server_log()
        
        # 核心修复：添加日志相关启动参数，告诉服务端日志写入路径和输出格式
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--enable-request-time-stats-logging",
            # 新增：指定日志输出到文件（而非控制台）
            "--log-file", str(SGLANG_SERVER_LOG_PATH),
            # 新增：可选 - 指定日志级别（INFO 级别足够捕获 Req Time Stats）
            "--log-level", "INFO"
        ]

        # 启动模型服务（此时服务端会读取 --log-file 参数，将日志写入我们指定的路径）
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        # 等待服务初始化和日志文件创建（服务端会自动创建日志文件）
        time.sleep(3)  # 稍延长一点，确保服务端完成日志文件初始化
        # 确保转储目录存在（日志目录由服务端创建，转储目录手动创建）
        SGLANG_LOG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程
        kill_process_tree(cls.process.pid)
        # 测试结束后，自动转储本次日志
        cls.dump_server_log()

    @staticmethod
    def clear_server_log():
        """清理服务端日志文件（删除原有文件，或清空文件内容）"""
        if SGLANG_SERVER_LOG_PATH.exists():
            os.remove(SGLANG_SERVER_LOG_PATH)

    @staticmethod
    def dump_server_log():
        """转储（备份）本次测试的日志文件，带时间戳避免重名"""
        # 1. 若当前日志文件不存在，直接返回（无内容可转储）
        if not SGLANG_SERVER_LOG_PATH.exists():
            print("No server log file to dump, skip backup.")
            return
        
        # 2. 生成带时间戳的转储文件名（格式：server_YYYYMMDD_HHMMSS.log）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_log_path = SGLANG_LOG_BACKUP_DIR / f"server_{timestamp}.log"
        
        # 3. 核心：复制当前日志文件到转储目录（实现转储/备份）
        try:
            with open(SGLANG_SERVER_LOG_PATH, "rb") as src_file, open(backup_log_path, "wb") as dst_file:
                dst_file.write(src_file.read())
            print(f"Log dumped successfully: {backup_log_path}")
        except Exception as e:
            print(f"Failed to dump log: {e}")

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

        # Step 2: 等待日志异步落盘（服务端写入文件有延迟，确保写入完成）
        time.sleep(3)

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
