import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.vlm_utils import (
    OmniOpenAITestMixin, ImageOpenAITestMixin, VideoOpenAITestMixin,
)


class TestQwen3OmniServer(VideoOpenAITestMixin):
    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.7,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
        "--disable-fast-image-processor",
    ]


if __name__ == "__main__":
    unittest.main()
