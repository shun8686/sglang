import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.performance.test_ascend_performance_utils import run_bench_serving
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH, run_command
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server, CustomTestCase, DEFAULT_URL_FOR_TEST,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class DisaggregationHiCacheBase(CustomTestCase):
    """Base class for disaggregation with HiCache tests"""
    model = QWEN3_32B_WEIGHTS_PATH
    decode_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--base-gpu-id",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-decode-enable-fake-auto",
        "--load-balance-method",
        "follow_bootstrap_room",
    ]
    keyword = "gen throughput (token/s): "
    out_log = "./out_log.txt"
    err_log = "./err_log.txt"
    out_file = open(out_log, "w+", encoding="utf-8")
    err_file = open(err_log, "w+", encoding="utf-8")

    @classmethod
    def setUpClass(cls):
        cls.decode_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.process_decode = popen_launch_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.decode_args,
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process_decode.pid)

        if hasattr(cls, 'out_file') and cls.out_file:
            cls.out_file.close()
        if hasattr(cls, 'err_file') and cls.err_file:
            cls.err_file.close()

    def test_disaggregation_decode_enable_fake_auto(self):
        response = requests.get(f"{self.decode_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        text1 = "The capital of France is"

        response = requests.post(
            f"{self.decode_url}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn(self.keyword, content)
        res = run_command(f"cat {self.err_log} | grep '{self.keyword}'")
        gen_throughput = res.split(self.keyword)[1].split(',')[0]
        self.assertGreater(float(gen_throughput), 0)

        metrics = run_bench_serving(
            host=self.url.hostname,
            port=int(self.url.port),
            dataset_name="random",
            num_prompts=16,
            input_len=3500,
            output_len=1,
        )
        print(metrics)
        self.assertGreater(float(metrics["mean_ttft"]), 0)


if __name__ == "__main__":
    unittest.main()
