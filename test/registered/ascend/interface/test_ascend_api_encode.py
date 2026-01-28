import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendApi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                2,
                "--is-embedding",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        
    def test_api_encode_01(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "2",
                "text": "what is the capital of France",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200,
                    "top_p": 1
                },
                
            },
        )
        print(response.json().keys())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "2")
        #self.assertEqual(response.json()['sampling_params']['temperature'], 0)
    def test_api_encode_02(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "3",
                "input_ids": [101, 7592, 2088, 102],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                },
                
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "3")
        #self.assertEqual(response.json()['sampling_params']['temperature'], 0)
        
    def test_api_encode_03(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "4",
                "text": "show me the words",
                "image_data": "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                },
                
            },
        )
        print(response.json().keys)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "4")
        #self.assertEqual(response.json()['sampling_params']['temperature'], 0)


if __name__ == "__main__":

    unittest.main()
