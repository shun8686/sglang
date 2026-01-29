import json

import requests
import unittest
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestChatCompletionsInterface(CustomTestCase):
    """Testcase:The test is to verify whether the functions of each parameter of the v1/chat/completions interface are normal.

    [Test Category] Interface
    [Test Target] v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--reasoning-parser",
            "qwen3",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
            "--enable-return-hidden-states",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_model_and_messages(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], self.model)
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], "default")
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_max_completion_tokens(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 1,
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_stream(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        print(f"response.text:{response.text}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        has_reasoning = False
        has_content = False

        print("\n=== Stream With Reasoning ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertTrue(
            has_reasoning,
            "The reasoning content is not included in the stream response",
        )
        self.assertTrue(
            has_content, "The stream response does not contain normal content"
        )

    def test_temperature(self):
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 0,
            },
        )
        print(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 0,
            },
        )
        print(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertEqual(content1, content2)

        response3 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 2,
            },
        )
        print(f"response3.json:{response3.json()}")
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")
        content3 = response3.json()["choices"][0]["message"]["content"]

        response4 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "temperature": 2,
            },
        )
        print(f"response4.json:{response4.json()}")
        self.assertEqual(response4.status_code, 200, f"Failed with: {response4.text}")
        content4 = response4.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content3, content4)

    def test_return_hidden_states(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True,
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertIn("hidden_states", response.json()["choices"][0])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertNotIn("hidden_states", response.json()["choices"][0])

    def test_top_k(self):
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "top_k": 20,
            },
        )
        print(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "帮我写一首五言绝句"}],
                "top_k": 20,
            },
        )
        print(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content1, content2)

    def test_stop_token_ids(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stop_token_ids": [1, 13],
            },
        )
        print(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()['choices'][0]['matched_stop'], 13)

    def test_rid(self):
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "rid": "sssss",
            },
        )
        print(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        self.assertEqual(response1.json()['id'], 'sssss')


if __name__ == "__main__":
    unittest.main()
