import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
)

# RadixAttention server integration tests
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

MODEL_PATH = Llama_3_2_1B_WEIGHTS_PATH

class TestRadixCacheFCFS(CustomTestCase):
    """
    Testcase：Verify the correctness and performance of the Radix Attention mechanism in the SGLang inference service

    [Test Category] Parameter
    [Test Target] --schedule-policy fcfs/lpm, --disable-overlap-schedule
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "fcfs",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestRadixCacheLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


class TestRadixCacheNonOverlapLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


if __name__ == "__main__":
    envs.SGLANG_TEST_RETRACT.set(True)
    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.set(1)
    unittest.main()
