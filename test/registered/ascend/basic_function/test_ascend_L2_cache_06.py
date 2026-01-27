import unittest

from sglang.srt.utils import is_npu
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
    run_mmlu_test,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

class TestNoChunkedPrefill(CustomTestCase):
    """Test DeepSeek R1 model
    Enable L2 cache increases hit rate by up to 50% and improves TTFT by 20%.
    --enable-hierarchical-cache: enable L2 cache
    """
    def test_no_chunked_prefill_without_radix_cache(self):
        TTFTS=[]
        model = (
            "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/vllm-ascend/DeepSeek-R1-W8A8"
            if is_npu()
            else "Qwen/Qwen3-32B"
        )
        common_args = [
            [
                "--trust-remote-code",
                "--tp-size",
                16,
                "--mem-fraction-static",
                0.8,
                "--max-running-requests",
                16,
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "512",
                "--disable-cuda-graph",
                "--quantization",
                "modelslim",
                "--attention-backend",
                "ascend",

            ],
            [
                "--trust-remote-code",
                "--tp-size",
                16,
                "--mem-fraction-static",
                0.8,
                "--max-running-requests",
                16,
                "--chunked-prefill-size",
                "512",
                "--disable-cuda-graph",
                "--quantization",
                "modelslim",
                "--attention-backend",
                "ascend",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                5,
                "--hicache-write-policy",
                "write_back",
            ]
        ]
        for common_arg in common_args:
            other_args=common_arg + (
                    ["--attention-backend",
                     "ascend",
                    ]
                    if is_npu()
                    else []
                    )
            res = run_bench_serving(
                model=model,
                dataset_name="generated-shared-prefix",
                num_prompts=128,
                random_input_len=3584,
                random_output_len=1,
                request_rate=float("inf"),
                max_concurrency=16,
                gsp_num_groups=1,
                gsp_prompts_per_group=128,
                gsp_system_prompt_len=1792,
                gsp_question_len=1792, 
                gsp_output_len=1,
                other_server_args=other_args,
                )
            print("---------------------------------------------",res)
            TTFT=res["mean_ttft_ms"]
            TTFTS.append(TTFT)

        print(f"***********{TTFTS[1]=}")
        print(f"***********{TTFTS[0]=}")
        assert float(TTFTS[1]) <= 0.8*float(TTFTS[0])


if __name__ == "__main__":
    unittest.main()
