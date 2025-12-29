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


class TestQwenPPTieWeightsAccuracy(CustomTestCase):
    accuracy = 0.38
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # cls.model_name = "Qwen/Qwen3-0.6B"
        cls.model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-0___6B"
        other_args = [
            "--chunked-prefill-size",
            256,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
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
