import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

OTHER_ARGS = [
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


class TestNPUDeepSeek_V3_2_8P_AIME2025(TestAscendAccuracyTestCaseBase):

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    other_args = OTHER_ARGS
    accuracy = 93.1
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen"
    output_len = 65536
    max_concurrency = 64
    generation_kwargs = "dict(temperature=1.0)"

    def test_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
