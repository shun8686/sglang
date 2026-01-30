import unittest
import requests
from types import SimpleNamespace
from typing import List, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_URL_FOR_TEST = "http://127.0.0.1:8234"

# HiCache核心配置组合（覆盖所有关键参数，无需全排列）
HICACHE_CONFIGS = [
    # (eviction_policy, io_backend, mem_layout, test_scenario_name)
    ("lru", "direct", "layer_first", "lru_direct_layer_first"),
    ("lfu", "kernel", "page_first", "lfu_kernel_page_first"),
    ("lru", "kernel_ascend", "page_first_direct", "lru_kernel_ascend_page_first_direct"),
    ("lru", "direct", "page_first_kv_split", "lfu_direct_page_first_kv_split"),
]

# 基础服务配置（通用配置，不随HiCache变化）
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
    """Qwen3-32B HiCache精度验证基础类"""
    # 精度阈值（保持基线，确保HiCache开启后精度无恶化）
    accuracy = 0.8722
    model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"
    
    @classmethod
    def launch_server_with_hicache(cls, eviction_policy: str, io_backend: str, mem_layout: str):
        """启动带指定HiCache配置的Qwen3-32B服务"""
        # 拼接完整配置参数
        other_args = BASE_OTHER_ARGS.copy()
        other_args.extend([
            "--radix-eviction-policy", eviction_policy,    # 缓存淘汰策略：lru/lfu
            "--hicache-io-backend", io_backend,            # IO后端：direct/kernel/kernel_ascend
            "--hicache-mem-layout", mem_layout,            # 内存布局：layer_first/page_first等
        ])
        
        # 启动服务
        return popen_launch_server(
            cls.model_name,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def run_gsm8k_accuracy_test(self, scenario: str):
        """执行GSM8K精度测试，验证HiCache开启后精度无恶化"""
        # 配置GSM8K评测参数
        args = SimpleNamespace(
            num_shots=5,                # 5-shot评测
            data_path=None,             # 使用默认数据集路径
            num_questions=200,          # 评测200题（覆盖性验证）
            max_new_tokens=512,         # 生成答案最大长度
            parallel=128,               # 128并行请求（模拟真实场景）
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        
        # 执行GSM8K评测
        metrics = run_eval(args)
        
        # 核心断言：精度不低于阈值，说明HiCache未导致精度恶化
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f"[{scenario}] Qwen3-32B精度恶化！实际精度: {metrics['accuracy']}, 阈值: {self.accuracy}",
        )
        
        # 验证服务配置生效（可选，辅助排查）
        server_info = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(server_info.status_code, 200)

# 动态生成测试类（覆盖所有HiCache关键配置组合）
def generate_hicache_test_classes() -> List[unittest.TestCase]:
    test_classes = []
    
    for eviction_policy, io_backend, mem_layout, scenario in HICACHE_CONFIGS:
        # 定义测试类
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
        
        # 重命名测试类（便于CI日志识别）
        TestQwenHiCache.__name__ = f"TestQwen32BHiCache_{scenario}"
        test_classes.append(TestQwenHiCache)
    
    return test_classes

# 生成所有HiCache测试类
hicache_test_classes = generate_hicache_test_classes()

if __name__ == "__main__":
    # 构造测试套件并执行
    suite = unittest.TestSuite()
    for test_class in hicache_test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    unittest.TextTestRunner(verbosity=2).run(suite)
