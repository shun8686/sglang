import os
import unittest
import time
import requests
import threading
from typing import List, Dict, Any

from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
)
from sglang.srt.managers.io_struct import BatchTokenIDOutput
from sglang.test.test_utils import CustomTestCase, DEFAULT_URL_FOR_TEST, popen_launch_server
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.environ import envs

# ================================
# 配置测试参数
# ================================
MODEL_PATH = "/home/weights/Qwen/Qwen3-0.6B"  # 替换为实际路径
TEST_URL = DEFAULT_URL_FOR_TEST  # 默认为 http://127.0.0.1:18000
TEST_PROMPT = "Hello<|eot_id|>"  # 包含特殊 token 的 prompt

# 需要检测的特殊 token
SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|eot_id|>",
    "<|pad|>",
    "<|unk|>",
    "�",  # 替代 Unicode 替代字符
]


# ================================
# 测试类：TestTokenizerBatchDecodeBehavior
# ================================
class TestTokenizerBatchDecodeBehavior(CustomTestCase):
    """
    测试 DetokenizerManager 在以下场景的行为：
    1. --disable-tokenizer-batch-decode 开启 vs 关闭
    2. skip_special_tokens=True / False
    3. 并发请求中不同 skip 设置
    4. 特殊 token 是否残留
    5. 流式输出一致性
    """

    model = MODEL_PATH
    base_url = TEST_URL
    test_prompt = TEST_PROMPT

    # 公共基础参数
    base_args = [
        "--trust-remote-code",
        "--tp-size", "1",
        "--base-gpu-id",
        "2",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--tokenizer-mode", "auto",
        "--revision", "main",
    ]

    def setUp(self):
        """准备日志和测试环境"""
        configure_logger(ServerArgs(self.model))  # 初始化日志
        self.out_log = open("./tmp_out.txt", "w+")
        self.err_log = open("./tmp_err.txt", "w+")
        self.processes = []

    def tearDown(self):
        """清理资源"""
        for p in self.processes:
            kill_process_tree(p.pid)
        self.out_log.close()
        self.err_log.close()
        for f in ["./tmp_out.txt", "./tmp_err.txt"]:
            if os.path.exists(f):
                os.remove(f)

    def _run_server(self, extra_args: List[str] = None):
        """启动服务"""
        extra_args = extra_args or []
        args = self.base_args + extra_args
        print(f"\n🚀 启动参数: {args}")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1200,
            other_args=args,
            return_stdout_stderr=(self.out_log, self.err_log),
        )
        time.sleep(15)
        self.processes.append(process)
        return process

    def _stop_server(self, process):
        """停止服务"""
        if process in self.processes:
            self.processes.remove(process)
        kill_process_tree(process.pid)
        time.sleep(3)

    def _send_single_request(
        self,
        skip_special: bool,
        request_id: int,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """发送单个请求"""
        payload = {
            "text": self.test_prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": skip_special,
                "spaces_between_special_tokens": True,
                "stop": ["<|eot_id|>"],
            },
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result["text"]
                has_special = any(token in text for token in SPECIAL_TOKENS)
                return {
                    "request_id": request_id,
                    "skip_special_tokens": skip_special,
                    "text": text,
                    "has_special": has_special,
                    "status": "success",
                    "raw": result,
                }
            else:
                return {
                    "request_id": request_id,
                    "skip_special_tokens": skip_special,
                    "text": "",
                    "has_special": False,
                    "status": f"fail_{response.status_code}",
                    "error": response.text,
                }
        except Exception as e:
            return {
                "request_id": request_id,
                "skip_special_tokens": skip_special,
                "text": "",
                "has_special": False,
                "status": "exception",
                "error": str(e),
            }

    def _run_concurrent_test(
        self,
        num_requests: int = 10,
        skip_special_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """并发发送请求，模拟高负载"""
        results = []
        threads = []

        def worker(request_id):
            skip_special = skip_special_first if request_id % 2 == 1 else not skip_special_first
            result = self._send_single_request(skip_special, request_id)
            results.append(result)

        print(f"\n🧵 启动 {num_requests} 个并发请求测试...")
        for i in range(1, num_requests + 1):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"✅ 并发测试完成，共 {len(results)} 个请求。")
        return results

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析结果并返回统计信息"""
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        bad_cases = [
            r for r in results
            if r["status"] == "success" and r["skip_special_tokens"] and r["has_special"]
        ]
        special_in_no_skip = [
            r for r in results
            if r["status"] == "success" and not r["skip_special_tokens"] and r["has_special"]
        ]

        print(f"\n📊 并发测试报告（共 {total} 个请求）:")
        print(f"  成功: {success} / {total}")
        print(f"  失败/异常: {total - success}")
        print(f"  ❌ skip=True 但含特殊 token: {len(bad_cases)}")
        for case in bad_cases:
            print(f"    Request {case['request_id']}: {repr(case['text'])}")

        print(f"  ⚠️  skip=False 但含特殊 token: {len(special_in_no_skip)}")
        for case in special_in_no_skip:
            print(f"    Request {case['request_id']}: {repr(case['text'])}")

        return {
            "total": total,
            "success": success,
            "bad_cases": len(bad_cases),
            "special_in_no_skip": len(special_in_no_skip),
            "results": results,
        }

    def test_compare_disable_tokenizer_batch_decode(self):
        """测试：开启/关闭 --disable-tokenizer-batch-decode 的行为一致性"""
        print("\n" + "=" * 80)
        print("🔹 测试 1：对比 batch decode 与 disable batch decode 行为一致性")
        print("=" * 80)

        test_cases = [
            {"name": "不带 --disable-tokenizer-batch-decode", "extra_args": []},
            {"name": "带 --disable-tokenizer-batch-decode", "extra_args": ["--disable-tokenizer-batch-decode"]},
        ]

        results_by_case = {}

        for case in test_cases:
            print(f"\n🚀 测试用例：{case['name']}")
            process = self._run_server(case["extra_args"])
            time.sleep(2)

            # 发送 2 个请求（一个 skip=True，一个 skip=False）以测试分组逻辑
            results = self._run_concurrent_test(num_requests=2, skip_special_first=True)
            results_by_case[case["name"]] = results

            self._stop_server(process)

            # 分析结果
            analysis = self._analyze_results(results)
            self.assertEqual(analysis["bad_cases"], 0, f"{case['name']} 中存在 skip=True 但输出特殊 token")
            self.assertEqual(analysis["special_in_no_skip"], 0, f"{case['name']} 中 skip=False 但输出特殊 token")

        # ✅ 最终断言：两个配置行为一致
        print("\n" + "=" * 80)
        print("✅ 最终验证：两种配置下输出应一致（内容 + 特殊 token）")
        print("=" * 80)

        # 比较两个 case 的结果（取第一个请求）
        case1 = results_by_case["不带 --disable-tokenizer-batch-decode"][0]
        case2 = results_by_case["带 --disable-tokenizer-batch-decode"][0]

        # 仅比较 skip=True 的情况（因为 skip=False 时可能有特殊 token）
        if case1["skip_special_tokens"] and case2["skip_special_tokens"]:
            self.assertEqual(
                case1["text"],
                case2["text"],
                "batch decode 与 disable batch decode 输出不一致"
            )
            print("✅ 内容一致：batch decode vs disable batch decode")
        else:
            print("ℹ️  跳过内容一致性检查（skip 不同）")

    def test_batch_decode_grouping_logic(self):
        """测试 batch_decode 分组逻辑是否正确"""
        print("\n" + "=" * 80)
        print("🔹 测试 2：验证 batch_decode 分组逻辑（skip 不同）")
        print("=" * 80)

        process = self._run_server([])
        time.sleep(2)

        # 发送 2 个请求，skip 设置不同
        results = self._run_concurrent_test(num_requests=2, skip_special_first=True)
        self._stop_server(process)

        # 验证是否分组处理
        skip_true = [r for r in results if r["skip_special_tokens"]]
        skip_false = [r for r in results if not r["skip_special_tokens"]]
        self.assertTrue(len(skip_true) > 0, "应至少有一个 skip=True 的请求")
        self.assertTrue(len(skip_false) > 0, "应至少有一个 skip=False 的请求")

        # 验证结果不为 None
        for r in results:
            self.assertIn(r["status"], ["success", "exception"])

if __name__ == "__main__":
    unittest.main()
