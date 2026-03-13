import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestLoraBasicFunction(CustomTestCase):
    """Testcase：Verify the use different lora, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --enable-lora, --lora-path,
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_use_different_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        text_no_lora = response.text

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        text_lora_a = response.text

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        text_lora_b = response.text

        self.assertNotEqual(
            text_no_lora,
            text_lora_a,
            f"same response.text"
        )

        self.assertNotEqual(
            text_no_lora,
            text_lora_b,
            f"same response.text"
        )

        self.assertNotEqual(
            text_lora_a,
            text_lora_b,
            f"same response.text"
        )

        # compare the consistency between streaming and non-streaming
        response_stream = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
                "stream": True,
            },
            stream=True,
        )
        stream_text = ""
        for chunk in response_stream.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                stream_text += data.get("text", "")
        self.assertIn(text_lora_a, stream_text)

    def test_batch_with_different_loras(self):
        # test different loras in batch requests can work properly
        prompts = [
            "What is AI",
            "Explain neural network",
            "What is deep learning",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                },
                "lora_path": ["lora_a", "lora_b"],
            },
        )
        results = response.json()

        self.assertEqual(len(results), len(prompts))

        for i, result in enumerate(results):
            self.assertEqual("text", result)
            self.assertGreater(len(result["text"]), 0)

    def test_lora_with_sampling_parameters(self):
        # test loras with temperature
        response_texts = []
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0.8,
                        "max_new_tokens": 32,
                    },
                    "lora_path": "lora_a",
                },
            )
            self.assertEqual(response.status_code, 200)
            response_text = response.json()["text"]
            response_texts.append(response_text)
        first_text = response_texts[0]
        for idx, text in enumerate(response_texts[1:], start=2):
            self.assertNotEqual(text, first_text, f"same response_text")

    def test_lora_with_json_schema(self):
        # test lora and json schema can work properly
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],

        })
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "Generate person information",
                "sampling_params": {
                    "temperature": 0.3,
                    "max_new_tokens": 128,
                    "json_schema": json_schema,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)
        parsed_json = json.loads(result["text"])
        self.assertIn("name", parsed_json)
        self.assertIn("age", parsed_json)
        self.assertIn("city", parsed_json)

    def test_lora_kv_cache(self):
        input_ids_first = [1] * 200
        input_ids_second = input_ids_first + [2] * 70

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids_first,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",

            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["cached_tokens"], 0)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids_first,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["cached_tokens"], 0)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids_second,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["cached_tokens"], 128)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids_second,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["meta_info"]["cached_tokens"], 128)

    def test_lora_session(self):
        # test the correct collaboration of lora  with session management functionality
        session_id_first = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()

        session_id_second = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        self.assertNotEqual(
            session_id_first,
            session_id_second,
            f"session_id"
        )

        # First conversation round - establish context
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物是一只猫，叫咪咪",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,

                },
                "session_params": {
                    "id": session_id_first,
                },
                "lora_path": "lora_a",

            },
        )
        self.assertEqual(response1.status_code, 200)
        rid = response1.json()["meta_info"]["id"]
        # Second conversation round - verify context
        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物叫什么名字？",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "session_params": {
                    "id": session_id_first,
                    "rid": rid,
                },
                "lora_path": "lora_a",

            },
        )
        self.assertEqual(response2.status_code, 200)
        response_text_2 = response2.text
        self.assertIn("咪咪", response_text_2,
                      f"Session should remember pet name '咪咪', but got: {response_text_2}")

        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物叫什么名字？",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "session_params": {
                    "id": session_id_second,
                },
                "lora_path": "lora_a",

            },
        )
        self.assertEqual(response3.status_code, 200)
        response_text_3 = response3.text

        # Verify new session doesn't remember previous context
        self.assertNotIn("咪咪", response_text_3,
                         f"New session should not remember old context, but got: {response_text_3}")


class TestLoraMemoryEvictionFifo(CustomTestCase):
    """Testcase：Verify the eviction policy works properly, inference request succeeded.
    [Test Category] Parameter
    [Test Target] --lora-eviction-policy
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH
    lora_eviction_policy = "fifo"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--lora-eviction-policy",
            cls.lora_eviction_policy,
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_eviction_policy"], self.lora_eviction_policy)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_eviction_policy"], self.lora_eviction_policy)


class TestLoraMemoryEvictionLru(CustomTestCase):
    lora_eviction_policy = "lru"


class TestLoraMaxLoraRank(CustomTestCase):
    """Testcase：Verify set the --max-load-rank parameter, can load lora corresponding to the number of ranks, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --max-load-rank
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-lora-rank",
            "64",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["max_lora_rank"], 64)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["max_lora_rank"], 64)


if __name__ == "__main__":
    unittest.main()
