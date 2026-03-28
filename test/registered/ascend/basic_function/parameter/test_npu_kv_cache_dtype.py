import os
import threading
import unittest
import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=150, suite="nightly-1-npu-a3", nightly=True)


class OutputCapturer:
    """Capture all console print information

    Class Description:
        Capture console output using low-level file descriptor redirection.
        Used to obtain print information from child processes, NPU processes,
        and underlying C/C++ modules that are not logged in sglang logs for test assertion.
        All captured output will be displayed normally in the console in real-time.
    """

    def __init__(self):
        """Initialize all member variables of the capturer"""
        self.old_stdout = None
        self.old_stderr = None
        self.pipe_out = None
        self.pipe_in = None
        self.pipe_err_out = None
        self.pipe_err_in = None
        self.captured_stdout = []
        self.captured_stderr = []
        self.stop_thread = False
        self.thread = None

    def start(self):
        """Start console output capture"""
        # Duplicate and save original stdout/stderr file descriptors
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # Create anonymous pipes for output redirection
        self.pipe_out, self.pipe_in = os.pipe()
        self.pipe_err_out, self.pipe_err_in = os.pipe()

        # Redirect system stdout/stderr to the write end of pipes
        os.dup2(self.pipe_in, 1)
        os.dup2(self.pipe_err_in, 2)

        # Close unused pipe write ends
        os.close(self.pipe_in)
        os.close(self.pipe_err_in)

        # Start daemon thread to read output in real time
        self.stop_thread = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """The background process reads and prints pipeline data records in a loop."""
        while not self.stop_thread:
            try:
                data = os.read(self.pipe_out, 4096)
                if data:
                    self.captured_stdout.append(data)
                    os.write(self.old_stdout, data)  # 实时打印
            except:
                self.stop()

            try:
                err_data = os.read(self.pipe_err_out, 4096)
                if err_data:
                    self.captured_stderr.append(err_data)
                    os.write(self.old_stderr, err_data)
            except:
                self.stop()

            time.sleep(0.001)

    def get_output(self):
        """Get all captured stdout as UTF-8 string

        Return: Decoded stdout string (ignore decoding errors)
        """
        return b''.join(self.captured_stdout).decode('utf-8', errors='ignore')

    def get_error(self):
        """Get all captured stderr as UTF-8 string

        Return: Decoded stderr string (ignore decoding errors)
        """
        return b''.join(self.captured_stderr).decode('utf-8', errors='ignore')

    def stop(self):
        """Stop capture and restore system environment"""
        self.stop_thread = True
        if self.thread:
            self.thread.join(timeout=0.5)

        # 恢复原始输出
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)

        # 关闭所有文件描述符
        for fd in [self.pipe_out, self.pipe_err_out, self.old_stdout, self.old_stderr]:
            try:
                os.close(fd)
            except:
                pass


class TestNPUKVCacheDtype(CustomTestCase):
    """Testcase：Verify set --kv_cache_dtype is auto, bf16 or bfloat16, request inference successful.

    [Test Category] Parameter
    [Test Target] --kv_cache_dtype
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    kv_cache_dtype = "auto"
    using_kv_cache_dtype = "torch.bfloat16"

    @classmethod
    def setUpClass(cls):
        cls.capturer = OutputCapturer()
        cls.capturer.start()

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--dtype",
            "auto",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--kv-cache-dtype",
            cls.kv_cache_dtype,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.capturer.stop()

    def test_dtype_options(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        response = requests.get(
            f"{self.base_url}/server_info",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'"kv_cache_dtype":"{self.kv_cache_dtype}"', response.text)

        output = self.__class__.capturer.get_output() + self.__class__.capturer.get_error()
        self.assertIn(f"Using KV cache dtype: {self.using_kv_cache_dtype}", output)


class TestNPUKVCacheDtypeBf16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bf16"


class TestNPUKVCacheDtypeBfloat16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bfloat16"


if __name__ == "__main__":
    unittest.main()
