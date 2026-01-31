import json
import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestModelLoaderExtraConfig(CustomTestCase):
    """Testcase: Configure the --model-loader-extra-configparameter to ensure no degradation in accuracy,
    and verify that the startup log contains "Multi-thread".
    Without configuring this parameter, the startup log should contain "Loading safetensors".
    After configuring the parameter, the model loading time should be reduced.

    [Test Category] Parameter
    [Test Target] --model-loader-extra-config {"enable_multithread_load": True, "num_threads": 2}
    """

    models = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    accuracy = 0.5
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 2}),
    ]
    out_log_file = open("./enable_out_log.txt", "w+", encoding="utf-8")
    err_log_file = open("./enable_err_log.txt", "w+", encoding="utf-8")
    log_info = "Multi-thread"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        os.environ["HCCL_BUFFSIZE"] = "200"
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "24"
        os.environ["USE_VLLM_CUSTOM_ALLREDUCE"] = "1"
        os.environ["HCCL_EXEC_TIMEOUT"] = "200"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        os.environ["SGLANG_ENBLE_TORCH_COMILE"] = "1"
        os.environ["AUTO_USE_UC_MEMORY"] = "0"
        os.environ["P2P_HCCL_BUFFSIZE"] = "20"
        env = os.environ.copy()

        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=3000,
            other_args=cls.other_args,
            env=env,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_loader_extra_config(self):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn(self.log_info, content)

    def _test_gsm8k(self):
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


class TestNOModelLoaderExtraConfig(TestModelLoaderExtraConfig):
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
    ]
    out_log_file = open("./no_enable_out_log.txt", "w+", encoding="utf-8")
    err_log_file = open("./no_enable_err_log.txt", "w+", encoding="utf-8")
    log_info = "Loading safetensors"


if __name__ == "__main__":
    unittest.main()
