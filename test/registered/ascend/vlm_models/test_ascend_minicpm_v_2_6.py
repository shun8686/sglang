import unittest

from sglang.test.ascend.test_ascend_utils import MINICPM_V_2_6_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestMiniCPMModelsV(TestVLMModels):
    """Testcase:Test the accuracy of the openbmb/MiniCPM-V-2_6 model using the mmmu dataset.

    [Test Category] Model
    [Test Target] openbmb/MiniCPM-V-2_6
    """

    model = MINICPM_V_2_6_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
