import unittest
import requests
import time
from datetime import datetime

# 仅保留必要模块导入，移除所有文件句柄相关依赖
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

# 极简配置项，仅保留核心参数
TARGET_TOKEN_COUNT = 2500
CHUNK_SIZE = 1024

def build_long_input_text_for_token():
    """构造足够token数的超长输入，满足mixed chunk触发前置条件"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (TARGET_TOKEN_COUNT // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

class TestEnableMixedChunk(CustomTestCase):
    """极简测试用例：仅保留核心功能，移除所有文件句柄操作"""

    @classmethod
    def setUpClass(cls):
        """启动服务器：仅配置核心参数，移除所有句柄相关代码"""
        # 服务器启动核心参数（enable-mixed-chunk + chunked-prefill-size）
        other_args = [
            "--enable-mixed-chunk",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--chunked-prefill-size", str(CHUNK_SIZE)
        ]

        # 启动服务器，无额外句柄操作，无阻塞优化（保留原生逻辑）
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 简短等待服务器就绪，无额外日志写入
        time.sleep(20)

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程，移除所有句柄相关清理代码"""
        kill_process_tree(cls.process.pid)

    def test_enable_mixed_chunk_core(self):
        """极简核心测试：仅发送请求，无额外复杂验证"""
        # 1. 构造超长输入
        long_input_text = build_long_input_text_for_token()

        # 2. 发送/generate请求（触发mixed chunk逻辑）
        generate_response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=70
        )

        # 3. 仅保留最基础的响应状态验证，无其他复杂断言
        self.assertEqual(generate_response.status_code, 200, "核心请求失败")

        # 4. 验证server_info参数配置（仅核心参数，无额外输出）
        server_info_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(server_info_response.status_code, 200)
        self.assertEqual(server_info_response.json().get("enable_mixed_chunk"), True)

if __name__ == "__main__":
    unittest.main()
