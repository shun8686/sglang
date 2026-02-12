import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests
from types import SimpleNamespace
from  sglang.test.few_shot_gsm8k import run_eval

from sglang.bench_serving import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Base class for disaggregation with HiCache tests"""
    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Prefill with HiCache enabled
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--disaggregation-decode-enable-offload-kvcache",
            "--enable-hierarchical-cache",
            "--hicache-storage-backend",
            "file",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL":"tcp://127.0.0.1:26666",
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



class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled only on Prefill side"""

    @classmethod
    def start_decode(cls):
        # Decode without HiCache offload
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--attention-backend",
            "ascend",
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-storage-backend",
            "file",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL":"tcp://127.0.0.1:26666",
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
                num_shots=5,
                data_path="/tmp/test.jsonl",
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=self.lb_port,
            )
        metrics = run_eval(args)
        print(f"*************metrics={metrics['accuracy']}")


if __name__ == "__main__":
    unittest.main()
