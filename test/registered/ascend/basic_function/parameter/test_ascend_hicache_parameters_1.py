import unittest
import requests
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

HICACHE_CONFIGS = [
    ("lru", "direct", "layer_first", "lru_direct_layer_first"),
    ("lfu", "kernel", "page_first", "lfu_kernel_page_first"),
    ("lru", "kernel_ascend", "page_first_direct", "lru_kernel_ascend_page_first_direct"),
    ("lru", "direct", "page_first_kv_split", "lru_direct_page_first_kv_split"),
]

BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "2",
    "--base-gpu-id", "4",
    "--enable-hierarchical-cache",
]

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class BaseQwenHiCacheTest(CustomTestCase):
    accuracy = 0.86
    model_name = QWEN3_32B_WEIGHTS_PATH

    @classmethod
    def launch_server_with_hicache(cls, eviction_policy, io_backend, mem_layout):
        other_args = BASE_OTHER_ARGS.copy()
        other_args.extend([
            "--radix-eviction-policy", eviction_policy,
            "--hicache-io-backend", io_backend,
            "--hicache-mem-layout", mem_layout,
        ])

        return popen_launch_server(
            cls.model_name,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def run_gsm8k_accuracy_test(self, scenario):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f"[{scenario}] 精度恶化！实际: {metrics['accuracy']}, 阈值: {self.accuracy}",
        )

for eviction_policy, io_backend, mem_layout, scenario in HICACHE_CONFIGS:
    class TestQwenHiCache(BaseQwenHiCacheTest):
        @classmethod
        def setUpClass(cls):
            cls.base_url = DEFAULT_URL_FOR_TEST
            cls.process = cls.launch_server_with_hicache(eviction_policy, io_backend, mem_layout)

        @classmethod
        def tearDownClass(cls):
            kill_process_tree(cls.process.pid)

        def test_gsm8k_hicache_accuracy(self):
            self.run_gsm8k_accuracy_test(scenario)

    TestQwenHiCache.__name__ = f"TestQwen32BHiCache_{scenario}"
    globals()[TestQwenHiCache.__name__] = TestQwenHiCache

if __name__ == "__main__":
    unittest.main()
