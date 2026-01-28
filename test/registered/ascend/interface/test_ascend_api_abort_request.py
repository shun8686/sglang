import json
import threading
import requests
import unittest
import time
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
DEFAULT_URL_FOR_TEST="http://127.0.0.1:2345"
responses = []
def send_requests(url, **kwargs):
    response = requests.post('http://127.0.0.1:2345' + url, json=kwargs)
    responses.append(response)

class TestAscendApi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
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

    def test_api_abort_request(self):
        thread1 = threading.Thread(target=send_requests, args=('/generate',), kwargs={'rid': '10086', 'text': 'who are you?', 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1024}})
        thread2 = threading.Thread(target=send_requests, args=('/abort_request',), kwargs={'rid': "10086"})
        thread1.start()
        time.sleep(0.5)
        thread2.start()
        thread1.join()
        thread2.join()
        print(responses[1].text)


if __name__ == "__main__":

    unittest.main()
