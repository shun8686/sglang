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
CUSTOM_SERVER_WAIT_TIME = 20  # 确保服务器完全加载并输出日志

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # 1. 保存操作系统层面的原始stdout/stderr文件句柄
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. 打开日志文件（操作系统层面的文件句柄，用于重定向）
        cls.log_fd = os.open(
            LOG_DUMP_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644  # 文件权限：可读可写
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")  # 用于后续读取

        # 3. 操作系统层面重定向stdout和stderr到日志文件句柄
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. 启动服务器
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

        # 5. 等待服务器完全启动
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树
        kill_process_tree(cls.process.pid)

        # 2. 恢复操作系统层面的stdout/stderr（方便后续打印日志和提示）
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. 关闭所有文件句柄和文件对象（释放文件占用）
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. 新增：打印完整日志内容到控制台
        cls.print_full_log()

        # 5. 新增：删除日志文件（清理冗余文件）
        cls.delete_log_file()

    @classmethod
    def print_full_log(cls):
        """打印完整的日志文件内容到控制台"""
        if not os.path.exists(LOG_DUMP_FILE):
            print("\n【日志提示】日志文件不存在，无内容可打印")
            return
        
        print("\n" + "="*80)
        print("完整服务器日志内容：")
        print("="*80)
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
            # 打印完整日志（若日志过大，可考虑只打印最后5000字符，避免控制台刷屏）
            print(full_log if len(full_log) <= 5000 else f"【日志过长，仅展示最后5000字符】\n{full_log[-5000:]}")
        print("="*80)
        print("日志打印完毕")

    @classmethod
    def delete_log_file(cls):
        """删除已生成的日志文件，清理冗余"""
        try:
            if os.path.exists(LOG_DUMP_FILE):
                os.remove(LOG_DUMP_FILE)
                print(f"\n日志文件已删除：{os.path.abspath(LOG_DUMP_FILE)}")
            else:
                print("\n【删除提示】日志文件不存在，无需删除")
        except Exception as e:
            # 此处恢复少量异常处理，避免删除失败导致测试报错（仅提示，不终止）
            print(f"\n【删除警告】日志文件删除失败：{e}")

    def read_log_file(self):
        """读取日志文件完整内容，返回字符串"""
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

        # 2. 延长日志写入等待时间，确保Req Time Stats完整写入
        time.sleep(5)

        # 3. 恢复IO，方便后续断言信息输出到控制台
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
