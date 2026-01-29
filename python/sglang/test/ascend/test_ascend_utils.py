"""Common utilities for testing and benchmarking on NPU"""

import os

# Model weights storage directory
MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"

# LLM model weights path
MiniCPM_O_2_6_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "openbmb/MiniCPM-o-2_6")
Llama_3_1_8B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "AI-ModelScope/Llama-3.1-8B-Instruct")

Llama_3_2_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B")
Llama_3_2_1B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct")
Llama_3_2_1B_Instruct_Tool_Calling_Lora_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "codelion/Llama-3.2-1B-Instruct-tool-calling-lora")
Llama_3_2_11B_Vision_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-11B-Vision-Instruct")
Meta_Llama_3_1_8B_Instruct = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Meta-Llama-3.1-8B-Instruct")

DeepSeek_R1_0528_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-R1-0528-W8A8")
DeepSeek_V2_Lite_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V2-Lite-W8A8")

Qwen2_5_7B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-7B-Instruct")


AFM_4_5B_BASE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "arcee-ai/AFM-4.5B-Base")
BAICHUAN2_13B_CHAT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "baichuan-inc/Baichuan2-13B-Chat")
C4AI_COMMAND_R_V01_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "CohereForAI/c4ai-command-r-v01")
CHATGLM2_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/chatglm2-6b")
DBRX_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "AI-ModelScope/dbrx-instruct")
DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "DeepSeek-V3.2-Exp-W8A8")
ERNIE_4_5_21B_A3B_PT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "baidu/ERNIE-4.5-21B-A3B-PT")
EXAONE_3_5_7_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
GEMMA_3_4B_IT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-3-4b-it")
GLM_4_9B_CHAT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/glm-4-9b-chat")
GRANITE_3_0_3B_A800M_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.0-3b-a800m-instruct")
GRANITE_3_1_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.1-8b-instruct")
GROK_2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "huihui-ai/grok-2")
INTERNLM2_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Shanghai_AI_Laboratory/internlm2-7b")
KIMI_K2_THINKING_WEIGHTS_PATH  = os.path.join(MODEL_WEIGHTS_DIR, "Kimi/Kimi-K2-Thinking")
LING_LITE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "inclusionAI/Ling-lite")
LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "meta-llama/Llama-4-Scout-17B-16E-Instruct")
LLAMA_2_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-2-7B")
MIMO_7B_RL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-7B-RL")
MINICPM3_4B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "OpenBMB/MiniCPM3-4B")
MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "mistralai/Mistral-7B-Instruct-v0.2")
OLMOE_1B_7B_0924_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "allenai/OLMoE-1B-7B-0924")
PERSIMMON_8B_CHAT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Howeee/persimmon-8b-chat")
PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "microsoft/Phi-4-multimodal-instruct")
QWEN3_0_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-0.6B")
Qwen3_30B_A3B_Instruct_2507_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-Instruct-2507")
QWEN3_32B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B")
QWEN3_235B_A22B_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/Qwen3-235B-A22B-W8A8")
QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot")
QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-Next-80B-A3B-Instruct")
QWQ_32B_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/QWQ-32B-W8A8")
SMOLLM_1_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "HuggingFaceTB/SmolLM-1.7B")
STABLELM_2_1_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "stabilityai/stablelm-2-1_6b")
XVERSE_MOE_A36B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "xverse/XVERSE-MoE-A36B")

# VLM model weights path
# Embedding model weights path
# Rerank model weights path
# Reward model weights path
