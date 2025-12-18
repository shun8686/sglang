import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="stage-b-test-2-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

class TestGrammarBackend(CustomTestCase):
    accuracy = 0
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--grammar-backend",
            "xgrammar",
        ]
        cls.process = popen_launch_server(
            cls.model_name,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
            f'Accuracy of {self.model_name} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
