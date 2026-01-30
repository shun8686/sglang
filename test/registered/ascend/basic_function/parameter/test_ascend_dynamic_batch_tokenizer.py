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

DEFAULT_URL_FOR_TEST = "http://127.0.0.1:8234"

BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "4",
    "--base-gpu-id", "4",
    "--enable-dynamic-batch-tokenizer",
    "--dynamic-batch-tokenizer-batch-size", "4",
    "--dynamic-batch-tokenizer-batch-timeout", "0", 
    "--log-level", "debug"
]
MODEL_NAME = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"

def launch_server_with_tokenizer_timeout(model_name, base_url, tokenizer_timeout, other_args_base):
    other_args = other_args_base.copy()
    if "--dynamic-batch-tokenizer-batch-timeout" in other_args:
        idx = other_args.index("--dynamic-batch-tokenizer-batch-timeout") + 1
        other_args[idx] = str(tokenizer_timeout)
    
    process = popen_launch_server(
        model_name,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 
        other_args=other_args,
    )
    return process

class BaseQwenTest(CustomTestCase):
    accuracy = 0.38

    def _run_gsm8k_test(self, scenario):
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
            f'accuracy {metrics["accuracy"]} < {self.accuracy}',
        )
        
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{scenario}: {server_info=}")

class TestQwenPPTieWeightsAccuracyTokenizerTimeout0(BaseQwenTest):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=0, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_0(self):
        self._run_gsm8k_test("tokenizer_timeout=0")

class TestQwenPPTieWeightsAccuracyTokenizerTimeout1(BaseQwenTest):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=1, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_1(self):
        self._run_gsm8k_test("tokenizer_timeout=1")

if __name__ == "__main__":
    unittest.main()
