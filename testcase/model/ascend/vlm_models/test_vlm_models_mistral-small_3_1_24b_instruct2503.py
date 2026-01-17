import unittest

from vlm_utils import TestVLMModels


class TestMistralModels(TestVLMModels):
    model="/root/.cache/modelscope/hub/models/mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
