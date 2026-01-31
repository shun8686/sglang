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

# 自定义合理的服务器启动等待时间
CUSTOM_SERVER_WAIT_TIME = 15  # 推荐值：10-20秒

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

        # 【关键修改1】：删除此处的立即IO恢复操作，保持重定向状态
        # 注释掉原有恢复代码
        # sys.stdout = cls.original_stdout
        # sys.stderr = cls.original_stderr

        # 5. 自定义等待时间，等待服务器完全启动
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")
        # 注意：此处print的内容会被写入日志文件（因为IO还未恢复），属于正常现象
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树
        kill_process_tree(cls.process.pid)

        # 2. 关闭日志文件句柄，确保所有缓冲内容写入文件
        cls.log_file.close()

        # 3. 提示日志文件路径（此时IO已恢复，会输出到控制台）
        print(f"服务器日志已完整转储到：{os.path.abspath(LOG_DUMP_FILE)}")

    def read_log_file(self):
        """读取日志文件完整内容，返回字符串"""
        # 以只读模式打开，避免覆盖已有内容
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_request_time_stats_logging(self):
        # 1. 发送请求，触发服务端生成 Req Time Stats 日志（此时IO仍处于重定向状态）
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

        # 2. 等待日志写入（确保Req Time Stats被完整写入文件）
        time.sleep(3)

        # 【关键修改2】：此时请求已处理完成，日志已生成，恢复原始IO
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # 3. 断言请求发送成功（此时IO已恢复，断言失败信息会输出到控制台）
        self.assertEqual(response.status_code, 200, "请求生成接口失败")

        # 4. 读取完整日志文件内容
        server_logs = self.read_log_file()

        # 5. 断言日志中包含 Req Time Stats 关键字
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            server_logs,
            f"未在服务端日志中找到关键字：{target_keyword}\n日志文件路径：{os.path.abspath(LOG_DUMP_FILE)}\n日志内容预览（最后1000字符）：\n{server_logs[-1000:] if len(server_logs) > 1000 else server_logs}",
        )

if __name__ == "__main__":
    unittest.main()
