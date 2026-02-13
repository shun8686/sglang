import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.vlm_utils import (
    OmniOpenAITestMixin, ImageOpenAITestMixin, VideoOpenAITestMixin,
)


class TestQwen3OmniServer(VideoOpenAITestMixin):
    model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
    extra_args = [
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        0.9,
        "--disable-cuda-graph",
        "--disable-fast-image-processor",
        "--grammar-backend",
        "none",
        "--tp-size",
        4,
    ]


if __name__ == "__main__":
    unittest.main()
