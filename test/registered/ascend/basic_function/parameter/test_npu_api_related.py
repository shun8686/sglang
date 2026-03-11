import logging
import unittest

import requests
from types import SimpleNamespace
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestApiRelatedGHFChat(CustomTestCase):
    """The API test with combined parameters returned the correct values for model_name and weight_version, indicating successful inference.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --served-model-name; --weight-version; --hf-chat-template-name
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
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
        kill_process_tree(cls.process.pid)

    def test_served_model_weight_version(self):
        """Verify the weight version identifier and the served-model-name covered model name."""
        response = requests.get(f"{self.base_url}/v1/models")
        result = response.json()

        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], self.custom_model_name)

        response1 = requests.get(f"{self.base_url}/model_info")
        self.assertEqual(response1.json()["weight_version"], self.weight_version)

    def test_chat_template_name(self):
        """Send inference request"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Explain neural networks:",
                "sampling_params": {
                    "max_new_tokens": 64,
                },
            },
        )
        result = response.json()

        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        logging.warning(f"Request with succeeded: {result['text'][:50]}")

class TestApiRelatedToolCallParser(CustomTestCase):
    """Test combined parameter tool-server and tool-call-parser, indicating successful inference.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --tool-server; --tool-call-parser
    """

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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tool_call_parser(self):
        """Send batch request"""
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

class TestApiRelatedSamplingDefaults(CustomTestCase):
    """Test --chat-template, indicating successful inference.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --chat-template
    """

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
            "--sampling-defaults",
            "openai",
            "--chat-template",
            "llama-4",
            "--tool-call-parser",
            "pythonic",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
        logging.warning(f"Builtin chat template works: {result['choices'][0]['message']['content'][:50]}...")

class TestApiRelatedCacheReport(CustomTestCase):
    """Test verify set --enable-cache-report, sent openai request prompt_tokens_details will return cached_tokens.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --chat-template
    """

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
            "--chat-template",
            "llama-2",
            "--enable-cache-report"
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_cache_report(self):
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/completions",
                json={
                    "prompt": "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                              "just return me a string with of 5000 characters,just return me a string with of 5000 characters, ",
                    "max_tokens": 260,

                },
            )
            self.assertEqual(response.status_code, 200)
            if i == 1:
                cached_tokens = response.json()["usage"]['prompt_tokens_details']['cached_tokens']
                self.assertEqual(256, cached_tokens)


if __name__ == "__main__":
    unittest.main()




