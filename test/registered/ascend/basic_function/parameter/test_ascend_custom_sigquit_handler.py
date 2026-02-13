
import unittest

import requests


from sglang import Engine
from sglang.srt.tracing.trace import *
from custom_handler import custom_sigquit_handler
from sglang.test.test_utils import (
    CustomTestCase,
)


@dataclass
class Req:
    rid: int
    trace_context: Optional[Dict[str, Any]] = None


class TestTrace(CustomTestCase):
    # --custom-

    def test_trace_engine_enable(self):

        prompt = "Today is a sunny day and I like"
        model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B-Instruct"

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            custom_sigquit_handler=custom_sigquit_handler,
        )

        try:
            engine.generate(prompt, sampling_params)

            # sleep for a few seconds to wait for opentelemetry collector to asynchronously export data to file.
            time.sleep(100)

            # check trace file
        finally:
            engine.shutdown()



if __name__ == "__main__":
    unittest.main()
