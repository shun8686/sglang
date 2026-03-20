import json
import re
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestJSONModeMixin:
    """Mixin class containing JSON mode test methods"""

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "json mode") produces valid JSON, even without a system prompt that mentions JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        print(f"Response ({len(text)} characters): {text}")

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        # Verify it's actually an object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")
        self._verify_whitespace_constraint(text)

    def test_json_mode_with_streaming(self):
        """Test that streaming with json_object response (also known as "json mode") format works correctly, even without a system prompt that mentions JSON."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        print(
            f"Concatenated Response ({len(full_response)} characters): {full_response}"
        )

        # Verify the combined response is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)
        self._verify_whitespace_constraint(full_response)

    def _verify_whitespace_constraint(self, json_str):
        """Verify that the --constrained-json-disable-any-whitespace parameter only takes effect (removes all whitespace from JSON output) when the grammar backend is xgrammar/llguidance; otherwise, the parameter has no effect."""
        # Detect whitespace characters (spaces, newlines, tabs) in the JSON string
        has_whitespace = bool(re.search(r'[\n\t ]', json_str))

        if self.backend in ["xgrammar", "llguidance"]:
            # Expect no whitespace characters
            self.assertFalse(
                has_whitespace,
                f"[{self.backend}] Whitespace characters still exist after enabling --constrained-json-disable-any-whitespace! JSON: {json_str}"
            )
        else:
            # Expect whitespace characters to remain (parameter has no effect)
            self.assertTrue(
                has_whitespace,
                f"[{self.backend}] The --constrained-json-disable-any-whitespace parameter should not take effect, but no whitespace characters are present in JSON! JSON: {json_str}"
            )


class ServerWithGrammarBackend(CustomTestCase):
    """Base class for tests requiring a grammar backend server"""

    backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
            "--constrained-json-disable-any-whitespace",
            "--attention-backend",
            "ascend",
        ]


        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestJSONModeXGrammar(ServerWithGrammarBackend, TestJSONModeMixin):
    """Testcase: Verify that when the grammar backend is xgrammar and the --constrained-json-disable-any-whitespace parameter is enabled, the JSON output contains no whitespace characters

    [Test Category] Parameter
    [Test Target] --constrained-json-disable-any-whitespace
    """
    backend = "xgrammar"


class TestJSONModeOutlines(ServerWithGrammarBackend, TestJSONModeMixin):
    """Testcase: Verify that when the grammar backend is outlines and the --constrained-json-disable-any-whitespace parameter is enabled, the JSON output contains no whitespace characters

    [Test Category] Parameter
    [Test Target] --constrained-json-disable-any-whitespace
    """
    backend = "outlines"


class TestJSONModeLLGuidance(ServerWithGrammarBackend, TestJSONModeMixin):
    """Testcase: Verify that when the grammar backend is llguidance and the --constrained-json-disable-any-whitespace parameter is enabled, the parameter has no effect and the JSON output still contains whitespace characters

    [Test Category] Parameter
    [Test Target] --constrained-json-disable-any-whitespace
    """
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
