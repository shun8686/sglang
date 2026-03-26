import os
import unittest
import tempfile
import time
import requests
from sglang.bench_serving import get_tokenizer
import random
from typing import Dict

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):

    @classmethod
    def setUpClass(cls):
        """Test class initialization: Launch Prefill/Decode disaggregated services and load balancer, then wait for services to be ready"""
        super(DisaggregationHiCacheBase, cls).setUpClass()
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Launch the Prefill service with configuration for Ascend NPU
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "ascend",
                "disaggregation-bootstrap-port",
                8998,
                "--disaggregation-decode-enable-offload-kvcache",
                "--tp-size",
                "2",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667"
        }

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        pass

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """Send a generate request and return response"""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache"""
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        requests.post(self.prefill_url + "/flush_cache")

class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    @classmethod
    def start_decode(cls):
        # Launch the Decode service with specified configuration for Ascend NPU (disaggregated architecture)
        ascend_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend_devices
        base_gpu_id = ascend_devices.split(",")[2] if len(ascend_devices.split(",")) >= 3 else "2"
        decode_args = (
            [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                base_gpu_id,
                "--disaggregation-transfer-backend",
                "ascend",
                "--num-reserved-decode-tokens",
                128,
                "--disaggregation-decode-polling-interval",
                2,
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667"
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    # def test_PD_disaggregation(self):
    #     response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
    #     self.assertEqual(response.status_code, 200)
    #
    #     response = requests.post(
    #         f"{DEFAULT_URL_FOR_TEST}/generate",
    #         json={
    #             "text": "The capital of France is",
    #             "sampling_params": {"temperature": 0, "max_new_tokens": 32},
    #         },
    #     )
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("Paris", response.text)


    def test_prefill_cache_hit(self):
        """Test that prefill cache works with repeated queries"""

        repeated_prompt = self.gen_prompt(800)

        # First request - should miss cache
        self.send_request(repeated_prompt, max_tokens=100)
        # Flush cache
        # Second request - should hit cache (faster)
        response2 = self.send_request(repeated_prompt, max_tokens=100)
        # Assert cached tokens cnt
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
