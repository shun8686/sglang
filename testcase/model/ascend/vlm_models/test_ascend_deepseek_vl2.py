import argparse
import random
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip
from mmmu_vlm_mixin import DEFAULT_MEM_FRACTION_STATIC, MMMUVLMMixin
from sglang.test.test_utils import CustomTestCase, is_in_ci

_is_hip = is_hip()
# VLM models for testing
if _is_hip:
    MODELS = [SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4)]
else:
    MODELS = [
        #SimpleNamespace(model="google/gemma-3-4b-it", mmmu_accuracy=0.38),
        #SimpleNamespace(model="Qwen/Qwen2.5-VL-3B-Instruct", mmmu_accuracy=0.4),
        #SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4),
        SimpleNamespace(model="/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/deepseek-ai/deepseek-vl2", mmmu_accuracy=0.2)
    ]


class TestVLMModels(MMMUVLMMixin, CustomTestCase):
    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            self._run_vlm_mmmu_test(model, "./logs")


if __name__ == "__main__":
    # Define and parse arguments here, before unittest.main
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Static memory fraction for the model",
        default=0.95,
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
    )

    # Parse args intended for unittest
    args = parser.parse_args()

    # Store the parsed args object on the class
    TestVLMModels.parsed_args = args
    TestVLMModels.other_arga = [
            "--tp-size",
            4,
            "--mem-fraction-static",
            0.95,
            ]


    # Pass args to unittest
    unittest.main(argv=[sys.argv[0]])
