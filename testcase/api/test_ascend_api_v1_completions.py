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
        cls.base_url = "http://127.0.0.1:30080"
        #cls.base_url = DEFAULT_URL_FOR_TEST
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
            "2",
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

    def _test_model_parameters_model(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": 'who are you?'
            },
        )
        print(f"client:{client}")
        print(f"client.status_code:{client.status_code}")
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()
        self.assertEqual(data["model"], self.model)

    def test_model_parameters_prompt(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?'
            },
        )
        print(f"client:{client}")
        print(f"client.status_code:{client.status_code}")
        print(f"client.json:{client.json()}")
        print(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])


    def test_model_parameters_max_tokens(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                'max_tokens': 1
            },
        )
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        print(f"client.json_choices:{client.json()['choices'][0]['finish_reason']}")
        self.assertEqual(client.json()['choices'][0]['finish_reason'], 'length')

    def test_model_parameters_stream(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stream": True
            },
        )
        #print(f"client.json:{client.json()}")
        # print(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

    def test_model_parameters_temperature(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 0
            },
        )
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # self.assertEqual(data["model"], self.model)
        # self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_model_parameters_hidden_states(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "return_hidden_status": True
            },
        )
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # self.assertEqual(data["model"], self.model)
        # self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_model_parameters_top_k(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "top_k": 1
            },
        )
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # self.assertEqual(data["model"], self.model)

    def test_model_parameters_stop_token_ids(self):
        list_ids = [1, 13]
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stop_token_ids": list_ids,
                "max_tokens": 1024
            },
        )
        print(f"client:{client}")
        print(f"client.status_code:{client.status_code}")
        print(f"client.json:{client.json()}")
        # print(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        self.assertEqual(client.json()['choices'][0]['matched_stop'], 13)

    def test_model_parameters_rid(self):
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "rid": "10086"
            },
        )
        print(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        self.assertEqual(client.json()['id'], '10086')



if __name__ == "__main__":
    unittest.main()
