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
CUSTOM_SERVER_WAIT_TIME = 20  # 小幅上调，确保服务器完全加载并输出日志

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 保存操作系统层面的原始stdout/stderr文件句柄（关键：不是sys.stdout，而是fileno）
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. 打开日志文件（操作系统层面的文件句柄，用于重定向）
        # os.O_WRONLY | os.O_CREAT | os.O_APPEND：只写、不存在则创建、追加写入
        cls.log_fd = os.open(
            LOG_DUMP_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644  # 文件权限：可读可写
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")  # 用于后续读取

        # 3. 关键：操作系统层面重定向stdout和stderr到日志文件句柄
        # 替换当前进程的stdout（fileno=1）和stderr（fileno=2）
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. 启动服务器（此时子进程会继承操作系统层面的文件句柄，输出写入日志）
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

        # 5. 等待服务器完全启动（加载模型、初始化服务）
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")  # 此输出会写入日志
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树
        kill_process_tree(cls.process.pid)

        # 2. 恢复操作系统层面的stdout/stderr（关键：还原文件句柄）
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. 关闭文件句柄和文件对象
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. 提示日志文件路径（此时已恢复，输出到控制台）
        print(f"\n服务器日志已完整转储到：{os.path.abspath(LOG_DUMP_FILE)}")

    def read_log_file(self):
        """读取日志文件完整内容，返回字符串"""
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        # 以只读模式打开，避免覆盖，忽略特殊编码错误
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_request_time_stats_logging(self):
        # 1. 发送请求，触发服务端生成 Req Time Stats 日志（此时输出仍写入日志文件）
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

        # 2. 延长日志写入等待时间（确保服务端有足够时间输出Req Time Stats）
        # 服务器处理请求并输出统计日志可能有轻微延迟，上调到5秒
        time.sleep(5)

        # 3. 先恢复IO（方便后续断言信息输出到控制台）
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())

        # 4. 断言请求发送成功
        self.assertEqual(response.status_code, 200, "请求生成接口失败")

        # 5. 读取完整日志文件内容
        server_logs = self.read_log_file()

        # 6. 断言日志中包含 Req Time Stats 关键字
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            server_logs,
            f"未在服务端日志中找到关键字：{target_keyword}\n日志文件路径：{os.path.abspath(LOG_DUMP_FILE)}\n日志内容预览（最后2000字符）：\n{server_logs[-2000:] if len(server_logs) > 2000 else server_logs}",
        )

if __name__ == "__main__":
    unittest.main()
