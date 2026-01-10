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
            "--base-gpu-id",
            "6",
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

    # def test_model_and_messages(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
    #     data = client.json()
    #     self.assertEqual(data["model"], self.model)
    #     self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    # def test_max_completion_tokens(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "max_completion_tokens": 1,
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
    #     self.assertEqual(client.json()["choices"][0]["finish_reason"], "length")

    # def test_stream(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "stream": True,
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
    
    def test_temperature(self):
        client1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Can you write a poem?"}],
                "temperature": 0.3,
            },
        )
        print(f"client1.json:{client1.json()}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")
        
        client2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Can you write a poem?"}],
                "temperature": 1.0,
            },
        )
        print(f"client2.json:{client2.json()}")
        self.assertEqual(client2.status_code, 200, f"Failed with: {client2.text}")
    
    # def test_return_hidden_states(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "return_hidden_states": True,
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
    #
    # def test_top_k(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "top_k": 1,
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
    #
    # def test_stop_token_ids(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "stop_token_ids": [1,2],
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

    # def test_rid(self):
    #     client = requests.post(
    #         f"{self.base_url}/v1/chat/completions",
    #         json={
    #             "model": self.model,
    #             "messages": [{"role": "user", "content": "Hello"}],
    #             "rid": 1,
    #         },
    #     )
    #     print(f"client.json:{client.json()}")
    #     self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")


if __name__ == "__main__":
    unittest.main()
