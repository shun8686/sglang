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
    """Testcase: The test is to verify whether the functions of each parameter of the v1/completions interface are normal.

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
        # Test model parameter; configured model returns correct name, unconfigured defaults to "default", reasoning works
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": 'who are you?'
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()
        self.assertEqual(data["model"], self.model)

    def test_model_parameters_prompt(self):
        # Test prompt parameter, The input has str, list[int], list[str], and list[list[int]], reasoning works
        # The input is in str format
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?'
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        # The input is in list[int] format
        list_int = [1, 2, 3, 4]
        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_int
            },
        )
        logging.info(f"client1.json:{client1.json()}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")

        # The input is in list[str] format
        list_str = ["who is you", "hello world", "ABChello"]
        client2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_str
            },
        )
        logging.info(f"client2.json:{client2.json()}")
        self.assertEqual(client2.status_code, 200, f"Failed with: {client2.text}")

        # The input is in list[list[int]] format
        list_list_int = [[14990], [1350, 445, 14990, 1879, 899], [14623, 525, 498, 30]]
        client3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": list_list_int
            },
        )
        logging.info(f"client3.json:{client3.json()}")
        self.assertEqual(client3.status_code, 200, f"Failed with: {client3.text}")

    def test_model_parameters_max_tokens(self):
        # Test max_completion_tokens parameter; setting to 1 token forces immediate truncation, verify finish_reason is "length"
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                'max_tokens': 1
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        logging.info(f"client.json_choices:{client.json()['choices'][0]['finish_reason']}")
        # Assertion output includes length
        self.assertEqual(client.json()['choices'][0]['finish_reason'], 'length')

    def test_model_parameters_stream(self):
        # Test stream parameter; verify streaming response contains both reasoning_content and normal content chunks
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stream": True
            },
        )
        logging.info(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        # Decompose the response and determine if the output format is stream
        has_text = False
        logging.info("\n=== Stream With Reasoning ===")
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
        # Test temperature parameter; temperature=0 yields identical outputs across requests, temperature=2 yields varied outputs
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 0
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 0
            },
        )
        logging.info(f"client1.json:{client1.json()}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")
        # Asser that the configuration temperature is the same and the output response is the same
        self.assertEqual(client.json()['choices'][0]['text'], client1.json()['choices'][0]['text'])

        client2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 2
            },
        )
        logging.info(f"client.json:{client2.json()}")
        self.assertEqual(client2.status_code, 200, f"Failed with: {client2.text}")

        client3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "temperature": 2
            },
        )
        logging.info(f"client3.json:{client3.json()}")
        self.assertEqual(client3.status_code, 200, f"Failed with: {client3.text}")
        self.assertNotEqual(client2.json()['choices'][0]['text'], client3.json()['choices'][0]['text'])

    def test_model_parameters_hidden_states(self):
        # Test return_hidden_states parameter; verify hidden_states field appears when enabled
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "return_hidden_states": True
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        self.assertIn("hidden_states", client.json()['choices'][0])

    def test_model_parameters_top_k(self):
        # Test top_k parameter; with k=20, outputs vary between identical requests due to token sampling
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "top_k": 20
            },
        )
        logging.info(f"client.json:{client.json()}")
        logging.info(f"client.text:{client.text}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")

        client1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "top_k": 20
            },
        )
        logging.info(f"client1.json:{client1.json()}")
        logging.info(f"client1.text:{client1.text}")
        self.assertEqual(client1.status_code, 200, f"Failed with: {client1.text}")
        self.assertNotEqual(client.json()['choices'][0]['text'], client1.json()['choices'][0]['text'])

    def test_model_parameters_stop_token_ids(self):
        # Test stop_token_ids parameter; verify response stops at specified token ID (13 is a period) and matched_stop field is correct
        list_ids = [13]
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "stop_token_ids": list_ids,
                "max_tokens": 1024
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        self.assertEqual(client.json()['choices'][0]['matched_stop'], 13)

    def test_model_parameters_rid(self):
        # Test rid parameter; verify response ID matches the requested rid value '10086'
        client = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": 'who are you?',
                "rid": "10086"
            },
        )
        logging.info(f"client.json:{client.json()}")
        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        self.assertEqual(client.json()['id'], '10086')


if __name__ == "__main__":
    unittest.main()
