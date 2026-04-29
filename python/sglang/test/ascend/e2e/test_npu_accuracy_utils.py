"""
E2E test utilities for NPU accuracy testing with PD separation.
"""

import os
import subprocess
import threading
import time
from abc import ABC
from typing import Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.simple_eval_gpqa import run_eval as run_gpqa_eval
from sglang.test.simple_eval_aime25 import run_eval as run_aime25_eval
from sglang.test.test_utils import CustomTestCase, is_in_ci
from sglang.utils import wait_for_http_ready


QWEN3_5_397B_W4A8_MODEL_PATH = os.path.join(
    "/root/.cache/modelscope/hub/models/", "Qwen/Qwen3.5-397B-w4a8"
)

GLM5_1_W4A8_MODEL_PATH = os.path.join(
    "/root/.cache/modelscope/hub/models/", "GLM-5.1-w4a8"
)

GPQA_DATASET = "gpqa"
AIME2025_DATASET = "aime2025"


class TestAscendAccuracyPdSepTestCaseBase(CustomTestCase, ABC):
    model_config: Dict = {}
    dataset: str = ""
    max_concurrency: int = 64
    num_prompts: int = 100
    max_tokens: int = 128

    @classmethod
    def setUpClass(cls):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        
        cls.prefill_host = "127.0.0.1"
        cls.decode_host = "127.0.0.1"
        cls.router_host = "127.0.0.1"
        cls.prefill_port = 30000
        cls.decode_port = 30001
        cls.router_port = 30002
        cls.bootstrap_port = 30003
        
        cls.prefill_url = f"http://{cls.prefill_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.decode_host}:{cls.decode_port}"
        cls.router_url = f"http://{cls.router_host}:{cls.router_port}"
        
        cls.process_prefill = None
        cls.process_decode = None
        cls.process_router = None

    @classmethod
    def launch_prefill(cls):
        env = os.environ.copy()
        if cls.model_config.get("prefill_envs"):
            for key, value in cls.model_config["prefill_envs"].items():
                env[key] = value
        
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", cls.model_config["model_path"],
            "--host", cls.prefill_host,
            "--port", str(cls.prefill_port),
        ] + cls.model_config["prefill_args"]
        
        print(f"Launching prefill server: {' '.join(cmd)}")
        cls.process_prefill = subprocess.Popen(cmd, env=env)
        
        wait_for_http_ready(cls.prefill_url + "/health", timeout=600, process=cls.process_prefill)
        print(f"Prefill server ready at {cls.prefill_url}")

    @classmethod
    def launch_decode(cls):
        env = os.environ.copy()
        if cls.model_config.get("decode_envs"):
            for key, value in cls.model_config["decode_envs"].items():
                env[key] = value
        
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", cls.model_config["model_path"],
            "--host", cls.decode_host,
            "--port", str(cls.decode_port),
            "--disaggregation-bootstrap-port", str(cls.bootstrap_port),
        ] + cls.model_config["decode_args"]
        
        print(f"Launching decode server: {' '.join(cmd)}")
        cls.process_decode = subprocess.Popen(cmd, env=env)
        
        wait_for_http_ready(cls.decode_url + "/health", timeout=600, process=cls.process_decode)
        print(f"Decode server ready at {cls.decode_url}")

    @classmethod
    def launch_router(cls):
        env = os.environ.copy()
        if cls.model_config.get("router_envs"):
            for key, value in cls.model_config["router_envs"].items():
                env[key] = value
        
        cmd = [
            "python3", "-m", "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill", cls.prefill_url,
            "--decode", cls.decode_url,
            "--host", cls.router_host,
            "--port", str(cls.router_port),
        ] + cls.model_config.get("router_args", [])
        
        print(f"Launching router: {' '.join(cmd)}")
        cls.process_router = subprocess.Popen(cmd, env=env)
        
        wait_for_http_ready(cls.router_url + "/health", timeout=120, process=cls.process_router)
        print(f"Router ready at {cls.router_url}")

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_router, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid, wait_timeout=60)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")
        time.sleep(5)

    def run_accuracy(self):
        if self.dataset == GPQA_DATASET:
            result = self._run_gpqa()
            print(f"\nGPQA Accuracy: {result['accuracy']:.4f}")
            if is_in_ci():
                self.assertGreater(result['accuracy'], 0.3)
        elif self.dataset == AIME2025_DATASET:
            result = self._run_aime2025()
            print(f"\nAIME2025 Accuracy: {result['accuracy']:.4f}")
            if is_in_ci():
                self.assertGreater(result['accuracy'], 0.5)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _run_gpqa(self):
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            num_shots=0,
            data_path=None,
            num_questions=self.num_prompts,
            max_new_tokens=self.max_tokens,
            parallel=self.max_concurrency,
            host=self.router_host,
            port=self.router_port,
        )
        
        return run_gpqa_eval(args)

    def _run_aime2025(self):
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            num_shots=0,
            data_path=None,
            num_questions=self.num_prompts,
            max_new_tokens=self.max_tokens,
            parallel=self.max_concurrency,
            host=self.router_host,
            port=self.router_port,
        )
        
        return run_aime25_eval(args)