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

    def setUp(self):
        self.input_npy_path = "./input_tensor.npy"
        self.out_log_path = "./tensor_input_out_log.txt"
        self.err_log_path = "./tensor_input_err_log.txt"
        self.server_process = None
        vector = numpy.array([1001, 1002, 1003, 1004, 1005, 1006, 1007])
        numpy.save(self.input_npy_path, vector)
        self.other_args = [
            "--debug-tensor-dump-input-file",
            self.input_npy_path,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

    def tearDown(self):
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
            print("Server process has been terminated.")
        for file_path in [self.out_log_path, self.err_log_path, self.input_npy_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted test file: {file_path}")
                except Exception as e:
                    print(f"Warning: Failed to delete {file_path}, error: {e}")

    def test_tensor_input_file(self):
        with open(self.out_log_path, "w+", encoding="utf-8") as out_log_file, \
            open(self.err_log_path, "w+", encoding="utf-8") as err_log_file:
            try:
                self.server_process = popen_launch_server(
                    self.model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=self.other_args,
                    return_stdout_stderr=(out_log_file, err_log_file),
                )
            except Exception as e:
                print(f"Server launch failed, error: {e}")
                print("process is killed")

            err_log_file.seek(0)
            content = err_log_file.read()
            self.assertTrue(len(content) > 0, "Error log is empty, parameter may not take effect!")


if __name__ == "__main__":
    unittest.main()
