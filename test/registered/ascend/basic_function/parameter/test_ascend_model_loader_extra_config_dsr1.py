import json
import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


@unittest.skip("test skip")
class TestModelLoaderExtraConfig(CustomTestCase):
    """Testcase: Configure the --model-loader-extra-configparameter to ensure no degradation in accuracy,
    and verify that the startup log contains "Multi-thread".
    Without configuring this parameter, the startup log should contain "Loading safetensors".
    After configuring the parameter, the model loading time should be reduced.

    [Test Category] Parameter
    [Test Target] --model-loader-extra-config {"enable_multithread_load": True, "num_threads": 2}
    """

    models = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
    accuracy = 0.95
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        32768,
        "--tp-size",
        16,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--cuda-graph-max-bs",
        16,
        "--enable-torch-compile",
        "--disable-cuda-graph",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 2}),
    ]
    out_log_file = open("./model_loader_extra_config_out_log.txt", "w+", encoding="utf-8")
    err_log_file = open("./model_loader_extra_config_err_log.txt", "w+", encoding="utf-8")
    log_info = "Multi-thread"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.extra_envs = {
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "STREAMS_PER_DEVICE": "32",
        }
        os.environ.update(cls.extra_envs)

        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_loader_extra_config(self):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn(self.log_info, content)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.models} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


class TestNOModelLoaderExtraConfig(CustomTestCase):
    models = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
    accuracy = 0.95
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        32768,
        "--tp-size",
        16,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--cuda-graph-max-bs",
        16,
        "--enable-torch-compile",
        "--disable-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.extra_envs = {
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "STREAMS_PER_DEVICE": "32",
        }
        os.environ.update(cls.extra_envs)

        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
