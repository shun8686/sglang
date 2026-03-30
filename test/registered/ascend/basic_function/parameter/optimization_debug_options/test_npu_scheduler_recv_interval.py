import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH


class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen3-0.6B model (with tie_word_embeddings) between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = QWEN3_0_6B_WEIGHTS_PATH  # qwen3 < 8B all have tie_word_embeddings = True

    def run_gsm8k_test(self, interval):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--scheduler-recv-interval",
                interval,
                "--attention-backend",
                "ascend",
                "--disable-radix-cache",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(interval=1)
        print("============================baseline=====================================")
        print(baseline)
        pp_metrics = self.run_gsm8k_test(interval=50000)
        print("============================100===========================================")
        print(pp_metrics)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")




        self.assertGreaterEqual(baseline["accuracy"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )
        self.assertGreaterEqual(pp_metrics["latency"], baseline["latency"])

if __name__ == "__main__":
    unittest.main()