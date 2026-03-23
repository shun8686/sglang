import unittest
import os
import shutil
import json
from shutil import copy2

import requests

import tempfile

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestNpuTokenizer:#(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        #cls.model_path = tempfile.mkdtemp(prefix="model_path")
        cls.tokenizer_path = tempfile.mkdtemp(prefix="tokenizer_path")
        cls.file_names = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        for file_name in cls.file_names:
            if not os.path.exists(cls.tokenizer_path + "/" + file_name):
                copy2(cls.model + "/" + file_name, cls.tokenizer_path)
        cls.tokenizer_worker_num = 4
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--model-path",
            cls.model,
            "--tokenizer-path",
            cls.tokenizer_path,
            "--tokenizer-worker-num",
            cls.tokenizer_worker_num,
            "--tokenizer-mode",
            "auto",
            #"--skip-tokenizer-init",
            "--load-format",
            "safetensors",
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.tokenizer_path):
            shutil.rmtree(cls.tokenizer_path)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")


    def test_model_tokenizer_sending_request(self):
        long_prompt = "Explain the concept of machine learning in detail. " * 100
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text", response.text)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("Start multi-tokenizer worker process", content)
        self.assertIn("Registering detokenizer", content)
        self.out_log_file.close()
        self.err_log_file.close()


class TestNpuModelTokenizer(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tokenizer-mode",
            "slow",
            "--load-format",
            "auto",
            "--model-loader-extra-config",
            json.dumps({"enable_multithread_load": True, "num_threads": 2}),
            "context-length",
            "1000",
            "--is-embedding",
        ]
        # cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        # cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            # return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        # if os.path.exists(cls.tokenizer_path):
        #     shutil.rmtree(cls.tokenizer_path)
        # os.remove("./cache_out_log.txt")
        # os.remove("./cache_err_log.txt")

    def test_model_tokenizer_request(self):
        text1 = "The capital of France is"
        for i in range(3):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text1,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)



        # self.err_log_file.seek(0)
        # content = self.err_log_file.read()
        # self.assertIn("Start multi-tokenizer worker process", content)
        # self.assertIn("Registering detokenizer", content)
        # self.out_log_file.close()
        # self.err_log_file.close()

    def test_model_tokenizer_context_length(self):
        long_text = "hello " * 1200  # Will tokenize to more than context length
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                },
            },
        )
        print(f"{response.status_code = }")
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()


