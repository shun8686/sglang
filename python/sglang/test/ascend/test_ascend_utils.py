"""Common utilities for testing and benchmarking on NPU"""

import os

# Model weights storage directory
MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"

# LLM model weights path
MiniCPM_O_2_6_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "openbmb/MiniCPM-o-2_6")
Llama_3_1_8B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "AI-ModelScope/Llama-3.1-8B-Instruct")
