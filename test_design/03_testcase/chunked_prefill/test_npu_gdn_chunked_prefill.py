import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

QWEN3_5_9B_WEIGHTS_PATH = os.path.join(
    "/root/.cache/modelscope/hub/models", "Qwen/Qwen3.5-9B"
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


NUM_QUESTIONS = 20
ACCURACY = 0.8


class TestGDNChunkedPrefillEnabled(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify Qwen3.5-9B (GDN hybrid linear attention) GSM8K accuracy >= 0.8
    with chunked prefill enabled (size=128) on NPU.

    This covers the GDN + chunked prefill state-passing path in AscendGDNAttnBackend,
    where SSM recurrent state (last_recurrent_state) is serialized between prefill
    chunks. The chunked prefill path is exercised across the GDN linear attention
    layers (Gated DeltaNet blocks), verifying that multi-chunk state transfer
    produces correct generation output.

    Regression coverage for: PR #25839 and any future changes to
    NPU chunked prefill + hybrid linear attention interaction.

    [Test Category] Memory & Scheduling
    [Test Target] --chunked-prefill-size (GDN hybrid linear attention + Ascend NPU)
    """

    model = QWEN3_5_9B_WEIGHTS_PATH
    accuracy = ACCURACY
    num_questions = NUM_QUESTIONS
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--cuda-graph-bs",
        "4",
        "16",
        "--chunked-prefill-size",
        "128",
        "--max-running-requests",
        "16",
        "--tp-size",
        "2",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


class TestGDNChunkedPrefillDisabled(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify Qwen3.5-9B (GDN hybrid linear attention) GSM8K accuracy >= 0.8
    with chunked prefill disabled on NPU — serves as the baseline reference for the
    enabled variant to detect chunked prefill induced regressions.

    [Test Category] Memory & Scheduling
    [Test Target] --chunked-prefill-size -1 (GDN hybrid linear attention + Ascend NPU, baseline)
    """

    model = QWEN3_5_9B_WEIGHTS_PATH
    accuracy = ACCURACY
    num_questions = NUM_QUESTIONS
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--cuda-graph-bs",
        "4",
        "16",
        "--chunked-prefill-size",
        "-1",
        "--max-running-requests",
        "16",
        "--tp-size",
        "2",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


class TestGDNChunkedPrefillEnabledWithSpec(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify Qwen3.5-9B (GDN hybrid linear attention) GSM8K accuracy >= 0.8
    with chunked prefill (size=128) and speculative decoding (NEXTN) enabled on NPU.

    This covers the GDN + chunked prefill + speculative decoding state-passing path
    in AscendGDNAttnBackend, where SSM recurrent state (last_recurrent_state) is
    serialized between prefill chunks while speculative draft tokens are being
    verified. The PR #25839 bug (incorrect transpose of last_recurrent_state under
    spec_algorithm) specifically manifests in this combination of flags.

    [Test Category] Memory & Scheduling
    [Test Target] --chunked-prefill-size + --speculative-algorithm (GDN + Ascend NPU)
    """

    model = QWEN3_5_9B_WEIGHTS_PATH
    accuracy = ACCURACY
    num_questions = NUM_QUESTIONS
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--cuda-graph-bs",
        "4",
        "16",
        "--chunked-prefill-size",
        "128",
        "--max-running-requests",
        "16",
        "--tp-size",
        "2",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--dtype",
        "bfloat16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
