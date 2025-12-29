import unittest

from test_ascend_llm_models import TestMistral


class TestMiMo(TestMistral):
    #model = "/root/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-7B-RL"
    model = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/allenai/OLMoE-1B-7B-0924"
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
