import unittest
import ssl
from types import SimpleNamespace

from sglang.test.run_eval import run_eval # type: ignore
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k # type: ignore
from sglang.test.test_utils import CustomTestCase # type: ignore

ssl._create_default_https_context = ssl._create_unverified_context

QWEN3_CODER_480B_A35B_W8A8_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
DEFAULT_URL_FOR_TEST = "http://127.0.0.1:6688"

class TestDeepEpQwen(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_480B_A35B_W8A8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmlu(self):
        expect_score = 0.56

        print("Starting mmlu test...")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.9

        print("Starting gsm8k test...")
        _, host, port = self.base_url.split(":")
        host = host[2:]
        print(f"{host=}, {port=}")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{host}",
            port=int(port),
        )
        metrics = run_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )

if __name__ == "__main__":
    unittest.main()
