import unittest

from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestPhi4Multimodal(TestVLMModels):
    """Testcase:Accuracy of the microsoft/Phi-4-multimodal-instruct model was tested using the mmmu dataset.

    [Test Category] Model
    [Test Target] microsoft/Phi-4-multimodal-instruct
    """

    model = "/root/.cache/modelscope/hub/models/microsoft/Phi-4-multimodal-instruct"
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
