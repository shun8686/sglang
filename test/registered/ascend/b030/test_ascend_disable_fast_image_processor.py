import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.vlm_utils import (
    OmniOpenAITestMixin, ImageOpenAITestMixin, VideoOpenAITestMixin,
)


class TestQwen3OmniServer(VideoOpenAITestMixin):
    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        # "--mem-fraction-static",
        # 0.9,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-fast-image-processor",
    ]


if __name__ == "__main__":
    unittest.main()
