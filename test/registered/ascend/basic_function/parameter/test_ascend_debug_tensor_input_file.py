import os
import unittest

import numpy
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestDebugTensorInputFile(CustomTestCase):
    """Testcase：Verify set --debug-tensor-dump-input-file parameter, the -debug-tensor-dump-input-file is taking effect.

       [Test Category] Parameter
       [Test Target] --debug-tensor-dump-input-file
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    def test_tensor_input_file(self):
        vector = numpy.array([1001, 1002, 1003, 1004, 1005, 1006, 1007])
        numpy.save("./input_tensor.npy", vector)
        other_args = (
            [
                "--debug-tensor-dump-input-file",
                "./input_tensor.npy",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]

        )
        out_log_file = open("./tensor_input_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./tensor_input_err_log.txt", "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        except Exception as e:
            print("process is killed")
        err_log_file.seek(0)
        content = err_log_file.read()
        self.assertTrue(len(content) > 0)
        out_log_file.close()
        err_log_file.close()
        os.remove("./tensor_input_out_log.txt")
        os.remove("./tensor_input_err_log.txt")
        os.remove("./input_tensor.npy")


if __name__ == "__main__":
    unittest.main()
