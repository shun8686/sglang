import os
from abc import ABC
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class GSM8KAscendMixin(ABC):
    model = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"
    accuracy = 0.5
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=self.gsm8k_num_shots,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://172.22.3.71",
            port=8001,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
