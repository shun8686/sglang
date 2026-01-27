import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="embeddings are not all close",
)

TEXTS = "two Subway Series sandwiches with meats, cheese, lettuce, tomatoes, and onions on a black background, accompanied by the Subway Series logo, highlighting a new sandwich series."
IMAGES = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"

MODELS = [
    ("/root/.cache/modelscope/hub/models/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", 1e-3),
]
TORCH_DTYPES = [torch.float16]


class TestQmeQwenModels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_embeddings(self, model, prefill_tolerance, torch_dtype):

        prompts_no_image = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{TEXTS}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
        prompts_with_image = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
        with HFRunner(
            model,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            hf_text_embeddings = hf_runner.forward(prompts=[prompts_no_image])
            hf_image_embeddings = hf_runner.forward(
                prompts=[prompts_with_image], image_data=[IMAGES]
            )
        with SRTRunner(
            model,
            tp_size=1,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as srt_runner:
            srt_text_embeddings = srt_runner.forward(prompts=prompts_no_image)
            srt_image_embeddings = srt_runner.forward(
                prompts=prompts_with_image, image_data=IMAGES
            )

        similarity = get_similarities(
            hf_text_embeddings.embed_logits[0], srt_text_embeddings.embed_logits[0]
        )
        print("texts similarity diff", abs(similarity - 1))
        assert torch.all(
            abs(similarity - 1) < prefill_tolerance
        ), "embeddings are not all close"
        similarity = get_similarities(
            hf_image_embeddings.embed_logits[0], srt_image_embeddings.embed_logits[0]
        )
        print("images similarity diff", abs(similarity - 1))
        assert torch.all(
            abs(similarity - 1) < prefill_tolerance
        ), "embeddings are not all close"

    def test_accuracy(self):
        for model, prefill_tolerance in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_embeddings(model, prefill_tolerance, torch_dtype)


if __name__ == "__main__":
    unittest.main()
