import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    write_github_step_summary,
)


class TestDeepseekV32CPInSeqSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--quantization",
            "modelslim",
            "--tp",
            "16",
            "--enable-dp-attention",
            "--dp",
            "1",
            "--attn-cp-size",
            "4",
            "--enable-nsa-prefill-context-parallel",
            "--nsa-prefill-cp-mode",
            "in-seq-split",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.7",
            # "--cuda-graph-max-bs",
            # "32",
            # "--max-running-requests",
            # "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
            "--attention-backend",
            "ascend",
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self,):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=500,
            parallel=32,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        write_github_step_summary(
            f"### test_a_gsm8k (deepseek-v32-cp-in-seq-split)\n"
            f'{metrics["accuracy"]=:.3f}\n'
        )
        self.assertGreater(metrics["accuracy"], 0.935)


if __name__ == "__main__":
    unittest.main()

