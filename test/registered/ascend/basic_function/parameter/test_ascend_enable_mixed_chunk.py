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

# 全局变量：记录请求结果和时间戳
request_results = []

def build_long_input_text_for_token():
    """构造足够token数的超长输入"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (TARGET_TOKEN_COUNT // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id):
    """单个请求发送函数（供线程调用），记录单个请求耗时"""
    global request_results
    try:
        # 1. 记录单个请求的开始时间（发送前）
        single_start_time = time.time()
        
        # 2. 构造超长输入
        long_input_text = build_long_input_text_for_token()
        
        # 3. 发送/generate请求
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
        
        # 4. 记录单个请求的结束时间（完成后）
        single_end_time = time.time()
        
        # 5. 计算单个请求耗时（秒，保留4位小数，更精准）
        single_elapsed_time = round(single_end_time - single_start_time, 4)
        
        # 6. 记录请求结果和耗时
        request_results.append({
            "task_id": task_id,
            "status_code": response.status_code,
            "has_paris": "Paris" in response.text,
            "single_start_time": single_start_time,
            "single_end_time": single_end_time,
            "single_elapsed_time": single_elapsed_time
        })
        
        print(f"【任务 {task_id}】请求完成，状态码：{response.status_code}，单个耗时：{single_elapsed_time} 秒")
    except Exception as e:
        # 异常场景也记录耗时（便于排查问题）
        single_end_time = time.time()
        single_elapsed_time = round(single_end_time - time.time(), 4)
        request_results.append({
            "task_id": task_id,
            "status_code": -1,
            "error": str(e),
            "single_elapsed_time": single_elapsed_time
        })
        print(f"【任务 {task_id}】请求失败，报错：{e}，耗时：{single_elapsed_time} 秒")

class TestEnableMixedChunk(CustomTestCase):
    """多请求测试用例：添加端到端耗时统计"""

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
        """核心测试：同时发送多个请求，统计端到端耗时"""
        global request_results
        request_results = []  # 重置结果列表

        # 1. 记录「端到端整体开始时间」（所有线程启动前）
        overall_start_time = time.time()
        print(f"=== 整体任务开始，时间戳：{datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== 开始启动 {REQUEST_COUNT} 个请求线程，制造排队场景 ===")

        # 2. 创建多个线程（对应多个请求）
        threads = []
        for task_id in range(REQUEST_COUNT):
            t = threading.Thread(target=send_generate_request, args=(task_id,))
            threads.append(t)

        # 3. 同时启动所有线程（关键：制造并发请求，触发排队）
        for t in threads:
            t.start()

        # 4. 等待所有线程执行完成
        for t in threads:
            t.join()

        # 5. 记录「端到端整体结束时间」（所有线程完成后）
        overall_end_time = time.time()
        # 6. 计算端到端整体耗时（秒，保留4位小数）
        overall_elapsed_time = round(overall_end_time - overall_start_time, 4)

        # 7. 格式化输出耗时统计结果（核心：端到端+单个请求）
        print(f"\n=== 所有请求执行完毕，整体任务结束 ===")
        print(f"=== 端到端整体耗时：{overall_elapsed_time} 秒 ===")
        print(f"=== 单个请求耗时详情 ===")
        for result in request_results:
            if result["status_code"] == 200:
                self.assertEqual(result["status_code"], 200, f"任务 {result['task_id']} 请求失败")
                self.assertTrue(result["has_paris"], f"任务 {result['task_id']} 未返回Paris")
                print(f"任务 {result['task_id']}：✅ 成功 | 耗时：{result['single_elapsed_time']} 秒")
            else:
                self.fail(f"任务 {result['task_id']} 执行失败，状态码：{result['status_code']} | 耗时：{result['single_elapsed_time']} 秒")

        # 8. 额外统计：平均单个请求耗时（仅成功请求）
        success_requests = [r for r in request_results if r["status_code"] == 200]
        if success_requests:
            avg_single_time = round(sum([r["single_elapsed_time"] for r in success_requests]) / len(success_requests), 4)
            print(f"\n=== 成功请求平均耗时：{avg_single_time} 秒 ===")
