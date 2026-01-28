import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNoChunkedPrefill(CustomTestCase):
    """Testcase：Verify service availability and request processing accuracy of Llama-3.1-8B-Instruct model when chunked prefill is disabled

    [Test Category] Parameter
    [Test Target] --chunked-prefill-size
    """

    def test_no_chunked_prefill_without_radix_cache(self):
        model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        )
        other_args = (
            [
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        
        res = run_bench_serving(
            model=model,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=other_args,
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
