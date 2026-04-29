"""
E2E test utilities for NPU performance testing with PD separation.
"""

import os
import subprocess
import threading
import time
from abc import ABC
from typing import Dict, List, Optional

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, is_in_ci
from sglang.utils import wait_for_http_ready


QWEN3_5_397B_W4A8_MODEL_PATH = os.path.join(
    "/root/.cache/modelscope/hub/models/", "Qwen/Qwen3.5-397B-w4a8"
)

GLM5_1_W4A8_MODEL_PATH = os.path.join(
    "/root/.cache/modelscope/hub/models/", "GLM-5.1-w4a8"
)


class TestAscendPerformancePdSepTestCaseBase(CustomTestCase, ABC):
    model_config: Dict = {}
    max_concurrency: int = 64
    num_prompts: int = 500
    tpot_threshold_ms: float = 50.0

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

    def run_performance(self):
        import asyncio
        
        args = type('Args', (), {
            'backend': 'sglang',
            'base_url': self.router_url,
            'host': None,
            'port': None,
            'dataset_name': 'random',
            'dataset_path': '',
            'model': None,
            'tokenizer': None,
            'num_prompts': self.num_prompts,
            'sharegpt_output_len': None,
            'sharegpt_context_len': None,
            'random_input_len': self.random_input_len,
            'random_output_len': self.random_output_len,
            'random_range_ratio': 0.0,
            'request_rate': float('inf'),
            'multi': None,
            'output_file': None,
            'disable_tqdm': False,
            'disable_stream': False,
            'return_logprob': False,
            'return_routed_experts': False,
            'seed': 0,
            'disable_ignore_eos': False,
            'extra_request_body': None,
            'apply_chat_template': False,
            'profile': None,
            'lora_name': None,
            'lora_request_distribution': 'uniform',
            'lora_zipf_alpha': 1.5,
            'prompt_suffix': '',
            'device': 'npu',
            'pd_separated': True,
            'gsp_num_groups': 4,
            'gsp_prompts_per_group': 4,
            'gsp_system_prompt_len': 128,
            'gsp_question_len': 32,
            'gsp_output_len': 32,
            'gsp_num_turns': 1,
            'header': None,
            'max_concurrency': self.max_concurrency,
        })()
        
        result = run_benchmark(args)
        
        print(f"\nPerformance Results:")
        print(f"  Completed: {result['completed']}")
        print(f"  Output throughput: {result['output_throughput']:.2f} token/s")
        print(f"  Total throughput: {result['total_throughput']:.2f} token/s")
        print(f"  Median TTFT: {result['median_ttft_ms']:.2f} ms")
        print(f"  Median ITL: {result['median_itl_ms']:.2f} ms")
        print(f"  Median TPOT: {result['median_tpot_ms']:.2f} ms")
        
        self.assertEqual(result['completed'], self.num_prompts)
        
        if is_in_ci():
            self.assertLess(result['median_tpot_ms'], self.tpot_threshold_ms)