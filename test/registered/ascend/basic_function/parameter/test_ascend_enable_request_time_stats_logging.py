import unittest
import requests
import os
import sys
import time
from datetime import datetime

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# 定义日志转储文件路径
LOG_DUMP_FILE = f"server_request_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 保存原始stdout和stderr（用于后续恢复）
        cls.original_stdout = sys.stdout
        cls.original_stderr = sys.stderr

        # 2. 打开日志文件，用于重定向输出（a+模式：可读可写，追加创建）
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8", buffering=1)  # buffering=1：行缓冲，实时写入

        # 3. 关键：临时重定向全局stdout和stderr到日志文件
        sys.stdout = cls.log_file
        sys.stderr = cls.log_file

        # 4. 启动服务器（完全不修改popen_launch_server，直接调用原有逻辑）
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-request-time-stats-logging",
            ]
            if is_npu()
            else ["--enable-request-time-stats-logging"]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 5. 立即恢复原始stdout和stderr，不影响测试用例后续输出
        sys.stdout = cls.original_stdout
        sys.stderr = cls.original_stderr

        # 6. 给服务器预留启动时间，确保日志文件正常写入内容
        time.sleep(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH // 2)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树
        kill_process_tree(cls.process.pid)

        # 2. 关闭日志文件句柄，确保所有缓冲内容写入文件
        cls.log_file.close()

        # 3. 提示日志文件路径
        print(f"服务器日志已完整转储到：{os.path.abspath(LOG_DUMP_FILE)}")

    def read_log_file(self):
        """读取日志文件完整内容，返回字符串"""
        # 以只读模式打开，避免覆盖已有内容
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_request_time_stats_logging(self):
        # 1. 发送请求，触发服务端生成 Req Time Stats 日志
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        # 断言请求发送成功（先确保请求正常，再判断日志）
        self.assertEqual(response.status_code, 200, "请求生成接口失败")

        # 2. 等待日志写入（解决服务端输出缓冲延迟问题）
        time.sleep(3)

        # 3. 读取完整日志文件内容
        server_logs = self.read_log_file()

        # 4. 断言日志中包含 Req Time Stats 关键字
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            server_logs,
            f"未在服务端日志中找到关键字：{target_keyword}\n日志文件路径：{os.path.abspath(LOG_DUMP_FILE)}\n日志内容预览（最后1000字符）：\n{server_logs[-1000:] if len(server_logs) > 1000 else server_logs}",
        )

if __name__ == "__main__":
    unittest.main()
