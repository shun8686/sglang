import requests
import unittest
import json
import logging
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestEnableThinking(CustomTestCase):
    """Testcase: Test the basic functionality of the 'v1/completions' interface parameters.

    [Test Category] Interface
    [Test Target] v1/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
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

    def test_model_parameters_model(self):
        # Test model parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": 'who are you?'
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()
        # Assertion model configuration successful
        self.assertEqual(data["model"], self.model)

    def test_model_parameters_prompt(self):
        # Test prompt parameter
        # The input is in str format
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?'
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        # The input is in list[int] format
        list_int = [1, 2, 3, 4]
        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_int
            },
        )
        logging.warning(f"client1.json:{client1.json()}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")

        # The input is in list[str] format
        list_str = ["who is you", "hello world", "ABChello"]
        client2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_str
            },
        )
        logging.warning(f"client2.json:{client2.json()}")
        self.assertEqual(client2.status_code, 200, f"Failed with: {client2.text}")

        # The input is in list[list[int]] format
        list_list_int = [[14990], [1350, 445, 14990, 1879, 899], [14623, 525, 498, 30]]
        client3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_list_int
            },
        )
        logging.warning(f"client3.json:{client3.json()}")
        self.assertEqual(client3.status_code, 200, f"Failed with: {client3.text}")

    def test_model_parameters_max_tokens(self):
        # Test max_tokens parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                'max_tokens': 1
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        logging.warning(f"client.json_choices:{client.json()['choices'][0]['finish_reason']}")
        # Assertion output includes length
        self.assertEqual(client.json()['choices'][0]['finish_reason'], 'length')

    def test_model_parameters_stream(self):
        # Test stream parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stream": True
            },
        )
        logging.warning(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        # Decompose the response and determine if the output format is stream
        has_text = False
        logging.warning("\n=== Stream With Reasoning ===")
        for line in client.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:

                        if "text" in data["choices"][0]:
                            has_text = True
        self.assertTrue(
            has_text,
            "The text is a stream response",
        )

    def test_model_parameters_temperature(self):
        # Test temperature parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 1
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 1
            },
        )
        logging.warning(f"client1.json:{client.json()}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client.text}")
        # Asser that the configuration temperature is the same and the output response is the same
        self.assertNotEqual(client.json()['choices'][0]['text'], client1.json()['choices'][0]['text'])

    def test_model_parameters_hidden_states(self):
        # Test hidden_states parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "return_hidden_states": True
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # Assert the output response contains hidden_states
        self.assertIn("hidden_states", client.json()['choices'][0])

    def test_model_parameters_top_k(self):
        # Test top_k parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "top_k": 5
            },
        )
        logging.warning(f"client.json:{client.json()}")
        logging.warning(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "top_k": 5
            },
        )
        logging.warning(f"client1.json:{client1.json()}")
        logging.warning(f"client1.text:{client1.text}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")
        # Asser that the configuration top_k is the same and the output response is the same
        self.assertNotEqual(client.json()['choices'][0]['text'], client1.json()['choices'][0]['text'])

    def test_model_parameters_stop_token_ids(self):
        # Test stop_token_ids parameter
        list_ids = [1, 13]
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stop_token_ids": list_ids,
                "max_tokens": 1024
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # Asser can terminate the token according to the configured stop_token_ids
        self.assertEqual(client.json()['choices'][0]['matched_stop'], 13)

    def test_model_parameters_rid(self):
        # Test rid parameter
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "rid": "10086"
            },
        )
        logging.warning(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        # Assert the output response to include the configured rid
        self.assertEqual(client.json()['id'], '10086')


if __name__ == "__main__":
    unittest.main()
