import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


class BaseModelLoaderTest(CustomTestCase):
    """Base class for model loader tests"""

    models = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    accuracy = 0.5
    other_args = None
    out_log_file = None
    err_log_file = None
    log_info = None

    # Define log file names as class variables
    MULTITHREAD_OUT_LOG = "./multi_thread_out_log.txt"
    MULTITHREAD_ERR_LOG = "./multi_thread_err_log.txt"
    CHECKPOINT_OUT_LOG = "./checkpoint_out_log.txt"
    CHECKPOINT_ERR_LOG = "./checkpoint_err_log.txt"

    # Cache for service instances
    _service_instances = {}

    @classmethod
    def get_server_key(cls, use_multithread):
        """Generate a unique key for server configuration"""
        return f"{cls.models}_{use_multithread}"

    @classmethod
    def start_server(cls, use_multithread=True):
        """Start server with given configuration"""
        server_key = cls.get_server_key(use_multithread)

        # If already started, return cached instance
        if server_key in cls._service_instances:
            return cls._service_instances[server_key]

        # Build arguments
        base_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.9",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            "16",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
        ]

        if use_multithread:
            other_args = base_args + [
                "--model-loader-extra-config",
                json.dumps({"enable_multithread_load": True, "num_threads": 2}),
            ]
            out_file = open(cls.MULTITHREAD_OUT_LOG, "w+", encoding="utf-8")
            err_file = open(cls.MULTITHREAD_ERR_LOG, "w+", encoding="utf-8")
        else:
            other_args = base_args
            out_file = open(cls.CHECKPOINT_OUT_LOG, "w+", encoding="utf-8")
            err_file = open(cls.CHECKPOINT_ERR_LOG, "w+", encoding="utf-8")

        # Set environment variables
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        os.environ["HCCL_BUFFSIZE"] = "200"
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "24"
        os.environ["USE_VLLM_CUSTOM_ALLREDUCE"] = "1"
        os.environ["HCCL_EXEC_TIMEOUT"] = "200"
        os.environ["STREAMS_PER_DEVICE"] = "32"
        os.environ["SGLANG_ENBLE_TORCH_COMILE"] = "1"
        os.environ["AUTO_USE_UC_MEMORY"] = "0"
        os.environ["P2P_HCCL_BUFFSIZE"] = "20"
        env = os.environ.copy()

        # Start service once for warm-up to prevent cache affecting model load time
        process = popen_launch_server(
            cls.models,
            DEFAULT_URL_FOR_TEST,
            timeout=3000,
            other_args=other_args,
            env=env,
        )
        kill_process_tree(process.pid)

        # Start the actual service for testing
        process = popen_launch_server(
            cls.models,
            DEFAULT_URL_FOR_TEST,
            timeout=3000,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(out_file, err_file),
        )

        # Cache service instance
        cls._service_instances[server_key] = {
            'process': process,
            'out_file': out_file,
            'err_file': err_file,
            'use_multithread': use_multithread
        }

        return cls._service_instances[server_key]

    @classmethod
    def tearDownClass(cls):
        """Clean up all servers"""
        for server_key, instance in cls._service_instances.items():
            kill_process_tree(instance['process'].pid)
            instance['out_file'].close()
            instance['err_file'].close()
        cls._service_instances.clear()


class TestModelLoaderExtraConfig(BaseModelLoaderTest):
    """Testcase: Configure the --model-loader-extra-config parameter to ensure no degradation in accuracy,
    and verify that the startup log contains "Multi-thread".
    Without configuring this parameter, the startup log should contain "Loading safetensors".
    After configuring the parameter, the model loading time should be reduced.

    [Test Category] Parameter
    [Test Target] --model-loader-extra-config {"enable_multithread_load": True, "num_threads": 2}
    """

    @classmethod
    def setUpClass(cls):
        # Start server with multithread configuration
        cls.server_instance = cls.start_server(use_multithread=True)
        cls.process = cls.server_instance['process']
        cls.out_log_file = cls.server_instance['out_file']
        cls.err_log_file = cls.server_instance['err_file']
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.log_info = "Multi-thread"

    def test_model_loader_extra_config(self):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn(self.log_info, content)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.models} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


class TestNOModelLoaderExtraConfig(BaseModelLoaderTest):
    """Test without model loader extra config"""

    @classmethod
    def setUpClass(cls):
        # Start server without multithread configuration
        cls.server_instance = cls.start_server(use_multithread=False)
        cls.process = cls.server_instance['process']
        cls.out_log_file = cls.server_instance['out_file']
        cls.err_log_file = cls.server_instance['err_file']
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.log_info = "Loading safetensors"

    def test_model_loader_extra_config(self):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn(self.log_info, content)

    def test_model_loading_time_reduced(self):
        # Helper function to extract loading time
        def get_loading_seconds(filename, pattern):
            cmd = f"grep '{pattern}' ./{filename} | tail -1"
            line = run_command(cmd)
            if not line:
                return 0
            line = line.strip()
            print(f"{pattern}: {line}")
            if not line:
                return 0
            mm_ss = line.split('[')[1].split('<')[0]
            m, s = map(int, mm_ss.split(':'))
            return m * 60 + s

        # Get loading times
        multi_thread_seconds = get_loading_seconds(
            self.MULTITHREAD_ERR_LOG,
            "Multi-thread loading shards"
        )
        checkpoint_seconds = get_loading_seconds(
            self.CHECKPOINT_ERR_LOG,
            "Loading safetensors checkpoint shards"
        )

        # Print information
        print(f"Multi-thread: {multi_thread_seconds}s, Loading safetensors: {checkpoint_seconds}s.")

        # Assert
        self.assertGreater(checkpoint_seconds, multi_thread_seconds)


if __name__ == "__main__":
    unittest.main()
