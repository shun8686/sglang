import unittest
import requests
import os
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

# 定义日志转储文件路径（重定向直接写入该文件）
LOG_DUMP_FILE = f"server_request_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class TestEnableRequestTimeStatsLogging(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # 基础参数（原有逻辑不变）
        base_other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-request-time-stats-logging",
            ]
            if is_npu()
            else ["--enable-request-time-stats-logging"]
        )

        # 关键修改1：添加命令行重定向，将所有输出写入日志文件（Linux/Mac 语法）
        # 说明：> 表示将标准输出(stdout)写入文件（覆盖原有内容）
        # 说明：2>&1 表示将标准错误(stderr)重定向到标准输出，最终一起写入文件
        # Windows 环境需改为：> {LOG_DUMP_FILE} 2> {LOG_DUMP_FILE}
        redirect_args = [f"> {LOG_DUMP_FILE} 2>&1"]

        # 合并基础参数和重定向参数（注意：部分启动脚本需支持命令行拼接，此处直接传入即可）
        cls.other_args = base_other_args + redirect_args

        # 启动服务器（无需修改popen_launch_server，重定向已包含在命令参数中）
        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

        # 给服务器预留启动时间，确保日志文件正常创建
        time.sleep(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH // 2)

    @classmethod
    def tearDownClass(cls):
        # 终止服务器进程树
        kill_process_tree(cls.process.pid)
        # 日志文件已通过重定向自动生成，无需额外处理，直接提示路径即可
        print(f"服务器日志已通过命令行重定向转储到：{os.path.abspath(LOG_DUMP_FILE)}")

    def read_log_file(self):
        """读取日志文件内容，返回完整字符串"""
        # 确保日志文件存在（避免服务器启动失败导致文件缺失）
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        # 以utf-8编码读取文件，兼容中文日志（若有）
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

        # 2. 等待日志写入（服务端输出日志可能有延迟，避免读取不完整）
        time.sleep(3)

        # 3. 读取完整日志文件内容
        server_logs = self.read_log_file()

        # 4. 关键：判断日志中是否包含 Req Time Stats
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            server_logs,
            f"未在服务端日志中找到关键字：{target_keyword}\n日志文件路径：{os.path.abspath(LOG_DUMP_FILE)}\n日志内容预览（最后1000字符）：\n{server_logs[-1000:] if len(server_logs) > 1000 else server_logs}",
        )

if __name__ == "__main__":
    unittest.main()
