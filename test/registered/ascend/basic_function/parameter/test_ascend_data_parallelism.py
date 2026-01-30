import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import Llama_3_1_8B_Instruct_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDataParallelism(CustomTestCase):
    """Testcase：With data parallelism (DP=2)  enabled, verify model accuracy is greater than 0.65 and related API availability of Llama-3.1-8B-Instruct 

    [Test Category] Parameter
    [Test Target] --dp
    """

    @classmethod
    def setUpClass(cls):
        cls.model = Llama_3_1_8B_Instruct_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--dp",
                2,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.8,
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """Test MMLU accuracy with DP=2 enabled (core functionality validation)
        
        Key Validations:
        1. MMLU score ≥ 0.65 (accuracy baseline)
        2. Ensure DP=2 does not degrade model accuracy
        """
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        # Critical assertion: accuracy baseline validation
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_update_weight(self):
        #Test update_weights_from_disk API availability with DP=2
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": self.model},
        )

        # check if the response is 200
        assert response.status_code == 200

        # pause a few seconds then send again
        time.sleep(1)

        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": self.model},
        )

        # check if the response is 200
        assert response.status_code == 200

    def test_get_memory_pool_size(self):
        # use `get_server_info` instead since `get_memory_pool_size` is merged into `get_server_info`
        response = requests.get(self.base_url + "/get_server_info")
        assert response.status_code == 200

        time.sleep(1)

        response = requests.get(self.base_url + "/get_server_info")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
