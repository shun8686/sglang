import os
import sys
import threading
import unittest
import logging
from io import StringIO
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
    """底层文件描述符捕获，支持子进程/NPU打印，同时显示+捕获"""
    def __init__(self):
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
        """开始捕获"""
        # 保存原始 stdout/stderr
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # 创建管道
        self.pipe_out, self.pipe_in = os.pipe()
        self.pipe_err_out, self.pipe_err_in = os.pipe()

        # 重定向
        os.dup2(self.pipe_in, 1)
        os.dup2(self.pipe_err_in, 2)

        # 关闭无用端
        os.close(self.pipe_in)
        os.close(self.pipe_err_in)

        # 启动后台线程实时读取输出（关键：不阻塞、实时打印+捕获）
        self.stop_thread = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """后台循环读取输出 → 既存起来，又打印到终端"""
        while not self.stop_thread:
            # 读 stdout
            try:
                data = os.read(self.pipe_out, 4096)
                if data:
                    self.captured_stdout.append(data)
                    os.write(self.old_stdout, data)  # 实时打印
            except:
                break

            # 读 stderr
            try:
                err_data = os.read(self.pipe_err_out, 4096)
                if err_data:
                    self.captured_stderr.append(err_data)
                    os.write(self.old_stderr, err_data)
            except:
                break

            time.sleep(0.001)

    def get_output(self):
        """获取所有捕获的打印"""
        return b''.join(self.captured_stdout).decode('utf-8', errors='ignore')

    def get_error(self):
        """获取所有捕获的错误打印"""
        return b''.join(self.captured_stderr).decode('utf-8', errors='ignore')

    def stop(self):
        """停止捕获 + 恢复终端 + 清理资源"""
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

        # cls.old_stdout = os.dup(1)
        # cls.old_stderr = os.dup(2)
        # cls.pipe_out, cls.pipe_in = os.pipe()
        # cls.pipe_err_out, cls.pipe_err_in = os.pipe()
        # os.dup2(cls.pipe_in, 1)
        # os.dup2(cls.pipe_err_in, 2)

        cls.capturer = OutputCapturer()
        cls.capturer.start()



        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        # sys.stdout = cls.original_stdout
        # sys.stderr = cls.original_stderr
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

        # os.dup2(self.old_stdout, 1)
        # os.dup2(self.old_stderr, 2)
        # os.close(self.pipe_in)
        # os.close(self.pipe_err_in)



        # output = os.read(self.pipe_out, 1024 * 1024).decode("utf-8")
        # error = os.read(self.pipe_err_out, 1024 * 1024).decode("utf-8")
        # os.close(self.pipe_out)
        # os.close(self.pipe_err_out)
        # logger = logging.getLogger()
        # logger.info(output)
        # logger.info(error)
        # self.assertIn(f"Using KV cache dtype: {self.using_kv_cache_dtype}", error)
        output = self.__class__.capturer.get_output() +self.__class__.capturer.get_error()
        with open("output.log", "w", encoding="utf-8") as f:
            f.write(output)
        # print("=========================================================================")
        # print(output)
        # print("=========================================================================")
        # self.assertIn(f"Using KV cache dtype: {self.using_kv_cache_dtype}", self.tee_stdout.getvalue() + self.tee_stderr.getvalue())


#
# class TestNPUKVCacheDtypeBf16(TestNPUKVCacheDtype):
#     kv_cache_dtype = "bf16"
#
#
# class TestNPUKVCacheDtypeBfloat16(TestNPUKVCacheDtype):
#     kv_cache_dtype = "bfloat16"




if __name__ == "__main__":
    unittest.main()
