import requests
import unittest
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEnableThinking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
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
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=cls.other_args,
        )
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_with_reasoning(self):
        # Test non-streaming with "enable_thinking": True, reasoning_content should not be empty
        client = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
                **self.additional_chat_kwargs,
            },
        )
        print(f"client:{client}")
        print(f"client.status_code:{client.status_code}")
        print(f"client.json:{client.json}")
        print(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("message", data["choices"][0])
        self.assertIn("reasoning_content", data["choices"][0]["message"])
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])


if __name__ == "__main__":
    unittest.main()
