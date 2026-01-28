import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_npu_ci
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNoChunkedPrefill(CustomTestCase):
    """Testcase：Verify service availability and request processing accuracy of Llama-3.1-8B-Instruct model when chunked prefill is disabled

    [Test Category] Parameter
    [Test Target] --chunked-prefill-size
    """
    
    @classmethod
    def setUpClass(cls):
        # Start server: disable chunked prefill (-1) and cache, adapt to NPU environment
        cls.model = "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        # Stop server to release resources
        kill_process_tree(cls.process.pid)
    
    def test_no_chunked_prefill_without_radix_cache(cls):  
        # Configure MMLU test parameters and evaluation returns accuracy ≥ 0.65
        args = SimpleNamespace(
            base_url=cls.base_url,
            model=cls.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        cls.assertGreaterEqual(metrics["score"], 0.65)


if __name__ == "__main__":
    unittest.main()
