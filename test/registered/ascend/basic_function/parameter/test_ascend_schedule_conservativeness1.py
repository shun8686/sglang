import unittest

import requests

from types import SimpleNamespace

import sglang.bench_serving
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestScheduleConservativeness(CustomTestCase):
    """Testcase: Test the schedule policy, and use the GSM8K dataset ensure an inference accuracy of at least 0.86.

    [Test Category] Parameter
    [Test Target] --schedule-conservativeness
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--schedule-conservativeness",
            2.0,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--mem-fraction-static",
            "0.5"
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        cls.res = sglang.bench_serving.run_benchmark(
            backend=sglang,
            host="127.0.0.1",
            port="8080",
            num_prompts=128,
            model=QWEN3_32B_WEIGHTS_PATH,
            dataset_path=None,
            dataset_name="random",
            random_input_len=128,
            random_output_len=3512,
            max_concurrency=64,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_schedule_conservativeness(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            parallel=512,
            max_new_tokens=2500,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(metrics["accuracy"], 0.86)


if __name__ == "__main__":
    unittest.main()
