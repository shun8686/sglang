import unittest
import requests
import time
import threading
from datetime import datetime

# 必要模块导入
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

# 配置项
TARGET_TOKEN_COUNT = 2500  # 超长输入token数（>2048）
CHUNK_SIZE = 1024  # 分块预填充大小
REQUEST_COUNT = 3  # 同时发送3个请求，制造排队（≥2即可）

# 全局变量：记录请求结果
request_results = []

def build_long_input_text_for_token():
    """构造足够token数的超长输入"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (TARGET_TOKEN_COUNT // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id):
    """单个请求发送函数（供线程调用）"""
    global request_results
    try:
        # 1. 构造超长输入
        long_input_text = build_long_input_text_for_token()
        
        # 2. 发送/generate请求
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=120  # 延长超时，适配多请求排队场景
        )
        
        # 3. 记录请求结果
        request_results.append({
            "task_id": task_id,
            "status_code": response.status_code,
            "has_paris": "Paris" in response.text
        })
        
        print(f"【任务 {task_id}】请求完成，状态码：{response.status_code}")
    except Exception as e:
        request_results.append({
            "task_id": task_id,
            "status_code": -1,
            "error": str(e)
        })
        print(f"【任务 {task_id}】请求失败，报错：{e}")

class TestEnableMixedChunk(CustomTestCase):
    """多请求测试用例：制造排队场景，触发mixed chunk生效"""

    @classmethod
    def setUpClass(cls):
        """启动服务器：保留chunked + mixed核心配置"""
        other_args = [
            "--enable-mixed-chunk",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--chunked-prefill-size", str(CHUNK_SIZE)
        ]

        # 启动服务器
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 等待服务器完全就绪（多请求场景，需更长等待）
        time.sleep(30)

    @classmethod
    def tearDownClass(cls):
        """终止服务器进程"""
        kill_process_tree(cls.process.pid)

    def test_mixed_chunk_with_multi_requests(self):
        """核心测试：同时发送多个请求，制造排队"""
        global request_results
        request_results = []  # 重置结果列表

        # 1. 创建多个线程（对应多个请求）
        threads = []
        for task_id in range(REQUEST_COUNT):
            t = threading.Thread(target=send_generate_request, args=(task_id,))
            threads.append(t)

        # 2. 同时启动所有线程（关键：制造并发请求，触发排队）
        print(f"=== 开始启动 {REQUEST_COUNT} 个请求线程，制造排队场景 ===")
        for t in threads:
            t.start()

        # 3. 等待所有线程执行完成
        for t in threads:
            t.join()

        # 4. 极简验证：所有请求是否正常返回（重点看日志，而非断言）
        print(f"\n=== 所有请求执行完毕，结果汇总 ===")
        for result in request_results:
            if result["status_code"] == 200:
                self.assertEqual(result["status_code"], 200, f"任务 {result['task_id']} 请求失败")
                self.assertTrue(result["has_paris"], f"任务 {result['task_id']} 未返回Paris")
                print(f"任务 {result['task_id']}：✅ 成功（返回200，包含Paris）")
            else:
                self.fail(f"任务 {result['task_id']} 执行失败，状态码：{result['status_code']}")

if __name__ == "__main__":
    unittest.main()
