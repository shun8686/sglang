import os
import random
import tempfile
import unittest
import time
from time import sleep
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
from sglang.bench_serving import get_tokenizer

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import PDDisaggregationServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    popen_with_error_check, popen_launch_pd_server,
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
        sleep(600)
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
    PDDisaggregationServerBase
):
    mode = "follow_bootstrap_room"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-piecewise-cuda-graph",
            "--load-balance-method",
            cls.mode,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--load-balance-method",
            cls.mode,
            "--base-gpu-id",
            "8",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )


    def test_mgsm_en(self):
        sleep(600)
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=10,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


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
