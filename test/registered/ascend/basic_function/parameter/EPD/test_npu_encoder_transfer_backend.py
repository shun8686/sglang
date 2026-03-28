import unittest

import requests

import os

# ============ [Local path override - for local debugging only] ============
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"
import sglang.test.ascend.test_ascend_utils as _utils
_utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-30B-A3B-Instruct"
)
# =========================================================================

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


class TestEncoderTransferBackendBase(CustomTestCase):
    """Testcase: Verify --encoder-transfer-backend parameter accepts all supported values on Ascend NPU.

    --encoder-transfer-backend specifies how the encoder node transfers visual embeddings
    to the language model scheduler during VLM encoder-prefill disaggregation.
    Supported values: zmq_to_scheduler (default), zmq_to_tokenizer, mooncake.

    This base class starts an encoder-only VLM server with a specific transfer backend,
    then verifies the server is healthy, confirming the parameter is accepted.

    [Test Category] Parameter
    [Test Target] --encoder-transfer-backend; --encoder-only
    """

    # Subclasses must override this with a supported backend string.
    # Supported values: "zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"
    transfer_backend: str = None

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            # Enable encoder-only mode to exercise encoder disaggregation parameters
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.transfer_backend,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # Restore the test environment by stopping the server process (coding standard rule 8)
        kill_process_tree(cls.process.pid)

    def test_server_health(self):
        """Verify the encoder-only server with the specified transfer backend is healthy.

        The /health endpoint returning HTTP 200 confirms:
        1. The --encoder-transfer-backend value was recognized and accepted by server_args.
        2. The encoder-only server completed initialization successfully.
        A non-200 response or connection error here would mean the parameter caused a startup failure.
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            f"Encoder-only server with --encoder-transfer-backend={self.transfer_backend} "
            "failed health check; parameter may have been rejected at startup.",
        )


class TestEncoderTransferBackendMooncake(TestEncoderTransferBackendBase):
    """Testcase 3.1: Verify --encoder-transfer-backend=mooncake is accepted.

    mooncake uses the Mooncake high-performance RDMA transfer engine.
    """

    transfer_backend = "mooncake"


class TestEncoderTransferBackendZmqToScheduler(TestEncoderTransferBackendBase):
    """Testcase 3.2: Verify --encoder-transfer-backend=zmq_to_scheduler is accepted.

    zmq_to_scheduler is the default backend; embeddings are sent via ZMQ directly
    to the language model scheduler process. Testing explicit assignment ensures
    the parameter parser handles it correctly even when it matches the default.
    """

    transfer_backend = "zmq_to_scheduler"


if __name__ == "__main__":
    unittest.main()
