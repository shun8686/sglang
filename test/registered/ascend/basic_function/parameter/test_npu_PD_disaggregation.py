import logging
import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    popen_with_error_check,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Testcase: Test with offload enabled, cached_tokens continue to grow.

    [Test Category] Parameter
    [Test Target] --disaggregation-decode-enable-offload-kvcache
    """

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = QWEN3_32B_WEIGHTS_PATH

        cls.tokenizer = get_tokenizer(cls.model)
        cls.bootstrap_port = "8996"
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_router()
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def start_prefill(cls):
        # Prefill with HiCache enabled
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "2",
            "--base-gpu-id",
            4,
            "--enable-hierarchical-cache",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
        ]
        env = {
            **os.environ,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
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

    @classmethod
    def launch_router(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health")

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        # Send a generate request and return response
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
        # Helper method to trigger offloading and flush cache
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        requests.post(self.prefill_url + "/flush_cache")


class TestDisaggregationDecodeWithHiCache(DisaggregationHiCacheBase):
    """Decode startup parameters, enable offload-kvcache"""
    ascend_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    base_gpu_id = ascend_devices.split(",")[2] if len(ascend_devices.split(",")) >= 3 else "2"

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            2,
            "--mem-fraction-static",
            "0.9",
            "--base-gpu-id",
            2,
            # cls.base_gpu_id,
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--num-reserved-decode-tokens",
            128,
            "--disaggregation-decode-polling-interval",
            2,
        ]

        env = {
            **os.environ,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667"
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_multi_turn_conversation_cache(self):
        logging.warning("====================Testing request=======================")
        time.sleep(10)
        initial_prompt = self.gen_prompt(800)
        response1 = self.send_request(initial_prompt, max_tokens=200, temperature=0.1)
        current_context = initial_prompt + response1["text"]

        previous_cached_tokens = 0

        for turn in range(2, 5):
            print(f"\nTurn {turn}: Continuing from previous context")

            response = self.send_request(
                current_context, max_tokens=200, temperature=0.1
            )
            cached_tokens = response["meta_info"]["cached_tokens"]

            print(f"Turn {turn} cached tokens: {cached_tokens}")
            print(f"Improvement: {cached_tokens - previous_cached_tokens} tokens")

            # Assert cache improvement
            self.assertGreater(
                cached_tokens,
                previous_cached_tokens,
                f"Turn {turn} should have more cached tokens than turn {turn - 1}",
            )

            # Update context and cached tokens for next iteration
            current_context += response["text"]
            previous_cached_tokens = cached_tokens

            # Flush prefill cache
            self.trigger_offloading_and_flush()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
