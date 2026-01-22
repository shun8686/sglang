import unittest
import requests  # 补充缺失的requests模块导入
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_URL_FOR_TEST="http://127.0.0.1:8234"
class TestQwenPPTieWeightsAccuracy(CustomTestCase):
    # 保持精度阈值不变，确保测试通过即说明精度无恶化
    accuracy = 0.88
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"
        other_args = [
            "--chunked-prefill-size",
            "256",  
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8", 
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--enable-hierarchical-cache",
            "--radix-eviction-policy",
            "lru",
            "--hicache-io-backend",
            "direct",
            "--hicache-mem-layout",
            "page_first_kv_split",    
        ]
        cls.process = popen_launch_server(
            cls.model_name,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
        
        # 断言精度不低于阈值，验证精度无恶化
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model_name} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
        
        # 调用服务器信息接口，输出相关信息
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{server_info=}")
       
if __name__ == "__main__":
    unittest.main()
