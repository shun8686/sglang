import unittest

from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestMistralModels(TestVLMModels):
    """
    Accuracy of the Mistral-Small-3.1-24B-Instruct-2503 model was tested using the mmmu dataset.
    """

    model = "/root/.cache/modelscope/hub/models/mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
