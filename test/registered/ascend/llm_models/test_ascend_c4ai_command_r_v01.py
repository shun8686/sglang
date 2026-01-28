import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestC4AI(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/baichuan-inc/Baichuan2-13B-Chat"
    accuracy = 0.05
    chat_template_path = "/__w/sglang/sglang/test/nightly/ascend/llm_models/tool_chat_template_c4ai_command_r_v01.jinja"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chat-template",
        chat_template_path,
        "--tp-size",
        "2",
        "--dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
