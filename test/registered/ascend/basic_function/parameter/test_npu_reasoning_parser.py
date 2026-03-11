import logging
import unittest
import openai
import requests
from types import SimpleNamespace
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_1_3_B_BASE_PATH
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
    """Test for reasoning content API with DeepSeek R1 reasoning parser.

    [Test Category] Functional
    [Test Target] Api related on NPU
    --reasoning-parser
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_CODER_1_3_B_BASE_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--reasoning-parser",
            "deepseek-r1",
            "--completion-template",
            "deepseek_coder"
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )

        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    #def test_chat_template_name(self):
        """Send inference request"""
        # response = requests.post(
        #     f"{self.base_url}/v1/chat/completions",
        #     json={
        #         "model": "deepseek_coder",
        #         "messages": [
        #             {"role": "user", "content": "Hello, how are you?"},
        #         ]
        #     },
        # )
        # result = response.json()
        #
        # self.assertIn("choices", result)
        # self.assertGreater(len(result["choices"]), 0)
        # logging.warning(f"Builtin chat template works: {result['choices'][0]['message']['content'][:50]}...")


    def run_fim_completion(self, number_of_completion):
        prompt = "function sum(a: number, b: number): number{\n"
        client = openai.Client(
            api_key="sk-123456",
            base_url=self.base_url
        )

        response = client.completions.create(
            model = self.model,
            prompt = prompt,
            max_tokens = 512,
            stream = False,
            n=number_of_completion,
        )
        assert len(response.choices) == number_of_completion
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        logging.warning(response.choices[0].message.content)

    def test_fim_completion(self):
        for number_of_completion in [1, 3]:
            self.run_fim_completion(number_of_completion)

if __name__ == "__main__":
    unittest.main()
