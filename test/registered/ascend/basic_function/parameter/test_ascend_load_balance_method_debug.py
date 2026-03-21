import random
import unittest
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    popen_with_error_check,
)

register_npu_ci(est_time=500, suite="nightly-16-npu-a3", nightly=True)


class TestDPAttentionRoundBinLoadBalance(CustomTestCase):
    """
    Testcase：Verify that the inference is successful when --load-balance-method is set to round_robin, auto,
    follow_bootstrap_room, total_requests, total_tokens

    [Test Category] Parameter
    [Test Target] --load-balance-method
    """

    mode = "round_robin"

    @classmethod
    def setUpClass(cls):
        # cls.model_path = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.model_path =  "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
        # cls.model_path = "/home/weights/DeepSeek-R1-0528-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--load-balance-method",
            cls.mode,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            "0.75",
        ]

        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model_path,
            eval_name="mgsm_en",
            num_examples=10,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)

    def test_server_info(self):
        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.mode, response.text)


class _TestDPAttentionAutoLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "auto"


class _TestDPAttentionFollowBootstrapRoomLoadBalance(
    TestDPAttentionRoundBinLoadBalance
):
    mode = "follow_bootstrap_room"

    @classmethod
    def setUpClass(cls):
        cls.model_path = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
        cls.worker_url = "http://127.0.0.1:22222"
        other_args = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--load-balance-method",
            cls.mode,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            "0.75",
        ]

        cls.process = popen_launch_server(
            cls.model_path,
            cls.worker_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        cls.wait_server_ready(cls.worker_url + "/health")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)

        router_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            cls.worker_url,
            "--host",
            cls.url.hostname,
            "--port",
            str(cls.url.port),
            "--model-path",
            cls.model,
        ]
        cls.router_process = popen_with_error_check(router_command)
        cls.wait_server_ready(cls.base_url + "/health")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.router_process.pid)
        kill_process_tree(cls.worker_process.pid)

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(5)

    @classmethod
    def setUpClass(cls):
        # cls.model_path = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.model_path = "/home/weights/DeepSeek-R1-0528-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--load-balance-method",
            cls.mode,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            "0.75",
        ]

        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class _TestDPAttentionTotalRequestsLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "total_requests"


class _TestDPAttentionTotalTokensLoadBalance(TestDPAttentionRoundBinLoadBalance):
    mode = "total_tokens"


if __name__ == "__main__":
    # To reduce the CI execution time.
    if is_in_ci():
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        RUN_FLAG = [
            TestDPAttentionRoundBinLoadBalance,
            _TestDPAttentionAutoLoadBalance,
            _TestDPAttentionFollowBootstrapRoomLoadBalance,
            _TestDPAttentionTotalRequestsLoadBalance,
            _TestDPAttentionTotalTokensLoadBalance,
        ]
        suite.addTests(loader.loadTestsFromTestCase(random.choice(RUN_FLAG)))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # unittest.main()
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromTestCase(_TestDPAttentionFollowBootstrapRoomLoadBalance))
        runner = unittest.TextTestRunner()
        runner.run(suite)
