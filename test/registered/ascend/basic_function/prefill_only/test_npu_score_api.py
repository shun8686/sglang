import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH


class TestCausalLMScoringHTTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(self.base_url + "/v1/score", json=payload)

    def test_response_envelope(self):
        resp = self._post(
            {
                "query": "The capital of France is",
                "items": ["Paris", "Berlin"],
                "label_token_ids": [1, 2],
                "apply_softmax": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("scores", body)
        self.assertIn("model", body)
        self.assertEqual(body["object"], "scoring")

    def test_apply_softmax_false_by_default(self):
        resp = self._post(
            {
                "query": "The capital of France is",
                "items": ["Paris"],
                "label_token_ids": [1, 2],
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        scores = resp.json()["scores"]
        self.assertEqual(len(scores), 1)
        self.assertNotAlmostEqual(sum(scores[0]), 1.0, places=3)

    def test_apply_softmax_true_normalizes(self):
        resp = self._post(
            {
                "query": "The capital of France is",
                "items": ["Paris", "Berlin", "Rome"],
                "label_token_ids": [1, 2, 3],
                "apply_softmax": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        for row in resp.json()["scores"]:
            self.assertAlmostEqual(sum(row), 1.0, places=6)
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    def test_schema_rejection(self):
        bad_payloads = [
            {
                "query": "Q",
                "items": ["X"],
                "label_token_ids": "bad",
                "model": self.model,
            },
            {
                "query": "Q",
                "items": 42,
                "label_token_ids": [1, 2],
                "model": self.model,
            },
        ]
        for payload in bad_payloads:
            with self.subTest(payload=list(payload.keys())):
                self.assertGreaterEqual(self._post(payload).status_code, 400)


class TestCausalLMMISScoringHTTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--enable-mis",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _score(self, query, items, label_token_ids):
        resp = requests.post(
            self.base_url + "/v1/score",
            json={
                "query": query,
                "items": items,
                "label_token_ids": label_token_ids,
                "apply_softmax": True,
                "model": self.model,
            },
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def test_one_probability_vector_per_item(self):
        items = ["Sacramento", "San Jose", "San Francisco"]
        label_token_ids = [9454, 2753]
        scores = self._score("Is each the capital?", items, label_token_ids)["scores"]
        self.assertEqual(len(scores), len(items))
        for i, row in enumerate(scores):
            self.assertEqual(len(row), len(label_token_ids))
            self.assertAlmostEqual(sum(row), 1.0, places=6, msg=f"Item {i}")
            for v in row:
                self.assertGreaterEqual(v, 0.0)

    def test_empty_items_returns_empty_scores(self):
        result = self._score("Test query", [], [1, 2])
        self.assertEqual(len(result["scores"]), 0)

    def test_varying_item_counts(self):
        label_token_ids = [1, 2, 3, 4, 5]
        for items in (
            ["Single item"],
            ["Item 1", "Item 2"],
            ["A", "B", "C", "D"],
            ["X", "Y", "Z", "W", "V", "U"],
        ):
            with self.subTest(n=len(items)):
                scores = self._score("Rate each:", items, label_token_ids)["scores"]
                self.assertEqual(len(scores), len(items))
                for row in scores:
                    self.assertEqual(len(row), len(label_token_ids))
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_deterministic(self):
        items, label_token_ids = ["Option A", "Option B", "Option C"], [1, 2, 3]
        s1 = self._score("Choose:", items, label_token_ids)["scores"]
        s2 = self._score("Choose:", items, label_token_ids)["scores"]
        self.assertEqual(len(s1), len(s2))
        for r1, r2 in zip(s1, s2):
            for v1, v2 in zip(r1, r2):
                self.assertAlmostEqual(v1, v2, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=3)
