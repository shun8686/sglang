from time import sleep
import requests
import unittest
import subprocess
import time
from urllib.parse import urlparse
from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_with_error_check,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendGrpcModePDMixed(CustomTestCase):
    """
    Testcase：Verify that gRPC requests are correctly received and process when gRPC mode is enabled.

    [Test Category] Parameter
    [Test Target] --grpc-mode
    """

    @classmethod
    def setUpClass(cls):
        # cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
        cls.grpc_base_url = f"grpc://127.0.0.1:30111"
        cls.grpc_url = urlparse(cls.grpc_base_url)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)

        worker_command = [
            "python3",
            "-m", "sglang.launch_server",
            "--model-path", cls.model,
            "--grpc-mode",
            "--host", cls.grpc_url.hostname, "--port", str(cls.grpc_url.port),
        ]
        cls.worker_process = subprocess.Popen(worker_command, stdout=None, stderr=None)
        sleep(100)

        router_command = [
            "python3",
            "-m", "sglang_router.launch_router",
            "--worker-urls", cls.grpc_base_url,
            "--host", cls.url.hostname, "--port", str(cls.url.port),
            "--model-path", cls.model,
        ]
        cls.router_process = popen_with_error_check(router_command)
        cls.wait_server_ready(cls.base_url + "/health")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.router_process.pid)
        kill_process_tree(cls.worker_process.pid)

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    def test_grpc_mode(self):
        response = requests.post(
            f"http://127.0.0.1:21000/generate",
            json={
                "text": "The capital of France is",
                "model": self.model,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        print("============base==============")
        print(f"{response.status_code=}")
        print(f"{response.text=}")

# grpc
# P
export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24667"
python -m sglang.launch_server \
  --model /home/weights/Qwen3-0.6B \
  --trust-remote-code --attention-backend ascend --device npu \
  --disable-cuda-graph \
  --mem-fraction-static 0.4 --tp-size 1 --dp-size 2 --base-gpu-id 12 \
  --grpc-mode --disaggregation-transfer-backend ascend --disaggregation-mode prefill \
  --port 20000 --enable-metrics --metrics-port 10000
# D
export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24667"
python -m sglang.launch_server \
  --model /home/weights/Qwen3-0.6B \
  --trust-remote-code --attention-backend ascend --device npu \
  --disable-cuda-graph --prefill-round-robin-balance \
  --mem-fraction-static 0.4 --tp-size 1 --dp-size 2 --base-gpu-id 14 \
  --grpc-mode --disaggregation-transfer-backend ascend --disaggregation-mode decode \
  --port 20001 --enable-metrics --metrics-port 10001
# Router
export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill-policy cache_aware --dp-minimum-tokens-scheduler --worker-load-check-interval 1 \
  --prefill grpc://127.0.0.1:20000 \
  --decode grpc://127.0.0.1:20001 \
  --model-path /home/weights/Qwen3-0.6B \
  --tokenizer-path /home/weights/Qwen3-0.6B \
  --host 0.0.0.0 --port 4567


if __name__ == "__main__":
    unittest.main()
