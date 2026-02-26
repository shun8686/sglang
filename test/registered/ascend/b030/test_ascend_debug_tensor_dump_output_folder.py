import unittest

import requests
import glob
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH, run_command
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server, CustomTestCase,
)


class TestEnableReturnRoutedExperts(CustomTestCase):
    model = QWEN3_32B_WEIGHTS_PATH
    tp_size = 4
    pp_size = 2
    dp_size = 1
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        tp_size,
        "--pp-size",
        pp_size,
        "--dp-size",
        dp_size,
        "--debug-tensor-dump-output-folder",
        "./",
        "--skip-server-warmup",
        "--base-gpu-id",
        "8",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        run_command("rm -rf ./TP*_PP*")

    def test_debug_tensor_dump_output_folder(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        text1 = "The capital of France is"

        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
            },
        )
        self.assertEqual(response1.status_code, 200)
        print(response1.text)
        res1 = run_command("ls -d TP*_PP*_Rank*_pid* | wc -l")
        self.assertEqual(int(res1), self.tp_size * self.pp_size)


        file_pattern = "./TP0_PP0_Rank0_pid*"
        matching_files = glob.glob(file_pattern)
        model_layers_list = []
        if matching_files:
            tensor_file_path = matching_files[0]
            tensor_data = torch.load(tensor_file_path + "/Pass00000.pt", map_location="cpu")
            for idx, key in enumerate(tensor_data.keys(), 1):
                print(f"{idx}. {key}")
                if "model.layers." in key:
                    model_layers_list.append(key.split(".")[2])
        model_layers_list = sorted(set(int(x) for x in model_layers_list))
        print(model_layers_list)
        self.assertEqual(model_layers_list[-1], 31)

        file_pattern = "./TP0_PP1_Rank4_pid*"
        matching_files = glob.glob(file_pattern)
        model_layers_list = []
        if matching_files:
            tensor_file_path = matching_files[0]
            tensor_data = torch.load(tensor_file_path + "/Pass00000.pt", map_location="cpu")
            for idx, key in enumerate(tensor_data.keys(), 1):
                print(f"{idx}. {key}")
                if "model.layers." in key:
                    model_layers_list.append(key.split(".")[2])
        model_layers_list = sorted(set(int(x) for x in model_layers_list))
        print(model_layers_list)
        self.assertEqual(model_layers_list[-1], 63)


if __name__ == "__main__":
    unittest.main()
