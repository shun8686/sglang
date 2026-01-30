import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import Llama_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


# 封装通用父类，提取公共逻辑和可配置参数
class BaseTestRadixCacheChunkedPrefill(CustomTestCase):
    """Testcase: Verify that the model can successfully process inference requests and achieve an accuracy of ≥ 0.65 when the overlap scheduler is disabled,
    covering all combination scenarios of radix cache (enabled/disabled) and chunked prefill (enabled/disabled).

    [Test Category] Parameter
    [Test Target] --disable-radix-cache;--disable-overlap
    """
    _disable_radix_cache = None
    _chunked_prefill_size = None

    @classmethod
    def setUpClass(cls):
        cls.model = Llama_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                cls._disable_radix_cache,  
                "--chunked-prefill-size",
                cls._chunked_prefill_size,
                "--disable-overlap",
                "True",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestRadixCacheFalseChunk128(BaseTestRadixCacheChunkedPrefill):
    # Test class for disable_radix_cache=False, chunked_prefill_size=128
    _disable_radix_cache = "False"
    _chunked_prefill_size = "128"


class TestRadixCacheFalseChunkMinus1(BaseTestRadixCacheChunkedPrefill):
    # Test class for disable_radix_cache=False, chunked_prefill_size=-1
    _disable_radix_cache = "False"
    _chunked_prefill_size = "-1"


class TestRadixCacheTrueChunk128(BaseTestRadixCacheChunkedPrefill):
    # Test class for disable_radix_cache=True, chunked_prefill_size=128
    _disable_radix_cache = "True"
    _chunked_prefill_size = "128"


class TestRadixCacheTrueChunkMinus1(BaseTestRadixCacheChunkedPrefill):
    # Test class for disable_radix_cache=True, chunked_prefill_size=-1
    _disable_radix_cache = "True"
    _chunked_prefill_size = "-1"


if __name__ == "__main__":
    unittest.main()
