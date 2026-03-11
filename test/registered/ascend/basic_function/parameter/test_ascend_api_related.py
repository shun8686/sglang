import logging
import unittest

import requests
from types import SimpleNamespace
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    run_command,
)
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestApiRelatedApiKey(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "test-api-key-12345"
        cls.custom_model_name = "Llama3.2"
        cls.weight_version = "v1.0.0"
        cls.hf_chat_template_name = "tool_use"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--api-key",
            cls.api_key,
            "--served-model-name",
            cls.custom_model_name,
            "--weight-version",
            cls.weight_version,
            "--hf-chat-template-name",
            cls.hf_chat_template_name
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process)

    def test_served_model_weight_version(self):
        response = requests.get(f"{self.base_url}/v1/models")
        result = response.json()

        self.assertIn("data", result)
        logging.warning(f"*******{result}")
        self.assertEqual(result["data"][0]["id"], self.custom_model_name)
        self.assertEqual(result["data"][0]["weight_version"], self.weight_version)
        logging.warning(f"Request with api-key auth succeeded: {result['text'][:50]}")
        logging.warning(f"Weight version works: {result['text'][:50]}")

    def test_template_name(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Explain neural networks:",
                "sampling_params": {
                    "max_new_tokens": 64,
                },
            },
            headers = headers,
        )
        result = response.json()

        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        logging.warning(f"Request with succeeded: {result['text'][:50]}")

class TestApiRelatedStoragePath(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_path = "/tmp/storage_path"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--hicache-storage-backend",
            "file",
            "--file-storage-path",
            cls.storage_path,
            "--tool-server",
            "demo",
            "--tool-call-parser",
            "llama3",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_storage_path(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=21000,
        )
        run_eval(args)
        hicache_file = run_command(f"ls {self.storage_path}")
        self.assertNotEqual(hicache_file, None)
        hicache_file_size = run_command(f"du -s {self.storage_path} | cut -f1")
        self.assertGreater(int(hicache_file_size), 0)
        run_command(f"rm -rf {self.storage_path}")

class TestApiRelatedChatTemplate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_path = "/tmp/storage_path"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--sampling-defaults",
            "openai",
            "--chat-template",
            "llama3",
            "--tool-call-parser",
            "pythonic",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_chat_template(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "llama-3.2",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                ]
            },
        )
        result = response.json()

        self.assertIn("choices", result)
        self.assertGreater(len(result["choices"]), 0)
        logging.warning(f"Builtin chat template works: {result['choices'][0]['messages'][0]['content'][:50]}...")

if __name__ == "__main__":
    unittest.main()




