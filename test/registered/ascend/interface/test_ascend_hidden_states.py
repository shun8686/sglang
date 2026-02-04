import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import  CustomTestCase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestHiddenState(CustomTestCase):
    """Testcase: Verify that sglang successfully return hidden states when generating text.

    [Test Category] Interface
    [Test Target] generate
    """

    @classmethod
    def setUpClass(cls):
        prompts = ["Today is", "Today is a sunny day and I like"]
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.input_ids = cls.tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        cls.engine = sgl.Engine(
            model_path=cls.model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            enable_return_hidden_states=True,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        cls.outputs = cls.engine.generate(
            input_ids=cls.input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )

    @classmethod
    def tearDownClass(self):
        self.engine.shutdown()

    def test_return_hidden_states(self):
        for output in self.outputs:
            self.assertEqual(len(output["meta_info"]["hidden_states"]), 8)
            for i in range(len(output["meta_info"]["hidden_states"])):
                assert isinstance(output["meta_info"]["hidden_states"][i], list)
                output["meta_info"]["hidden_states"][i] = torch.tensor(
                    output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                )
        # Checks that splicing of the batch was done correctly
        self.assertGreater(
            self.outputs[1]["meta_info"]["hidden_states"][0].shape[0],
            self.outputs[0]["meta_info"]["hidden_states"][0].shape[0],
        )

        device_map = "npu" if True else "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=device_map
        )

        for input_id, output in zip(self.input_ids, self.outputs):
            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["output_ids"][:-1]], device=model.device
                    ),
                    output_hidden_states=True,
                )
            print("=== HF Hiddens ===")
            print(hf_out["hidden_states"][-1][0])
            sg_hidden_states = torch.cat(
                [
                    i.unsqueeze(0) if len(i.shape) == 1 else i
                    for i in output["meta_info"]["hidden_states"]
                ]
            ).to("npu")
            print("=== SRT Hiddens ===")
            print(sg_hidden_states)

            print(
                f"Max diff: {torch.max(torch.abs(hf_out['hidden_states'][-1][0] - sg_hidden_states))}"
            )

            atol = 0.8
            self.assertTrue(
                torch.allclose(
                    hf_out["hidden_states"][-1][0],
                    sg_hidden_states,
                    atol=atol,
                    rtol=0,
                )
            )

# class TestChangeHiddenState(CustomTestCase):
#     def test_repeatedly_changes_hidden_states(self):
#         prompts = ["Today is", "Today is a sunny day and I like"]
#         model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         input_ids = tokenizer(prompts).input_ids
#
#         sampling_params = {
#             "temperature": 0,
#             "max_new_tokens": 8,
#         }
#
#         self.engine1 = sgl.Engine(
#             model_path=model_path,
#             random_seed=42,
#             skip_tokenizer_init=True,
#             enable_return_hidden_states=True,
#             attention_backend="ascend",
#             disable_cuda_graph=True,
#         )
#         outputs_completion_first_round = self.engine1.generate(
#             input_ids=input_ids,
#             sampling_params=sampling_params,
#             return_hidden_states=True,
#         )
#         outputs_hidden_state = self.engine1.generate(
#             input_ids=input_ids,
#             sampling_params=sampling_params,
#             return_hidden_states=False,
#         )
#
#         outputs_completion_last_round = self.engine1.generate(
#             input_ids=input_ids,
#             sampling_params=sampling_params,
#             return_hidden_states=True,
#         )
#
#         self.engine1.shutdown()
#
#         for (
#             output_completion_first_round,
#             output_hidden_state,
#             output_completion_last_round,
#         ) in zip(
#             outputs_completion_first_round,
#             outputs_hidden_state,
#             outputs_completion_last_round,
#         ):
#             self.assertEqual(
#                 len(output_completion_first_round["meta_info"]["hidden_states"]), 8
#             )
#             self.assertNotIn("hidden_states", output_hidden_state["meta_info"])
#             self.assertEqual(
#                 len(output_completion_last_round["meta_info"]["hidden_states"]), 8
#             )


if __name__ == "__main__":
    unittest.main()
