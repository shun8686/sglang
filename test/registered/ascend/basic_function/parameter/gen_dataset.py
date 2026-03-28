import json


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8")

input_len = 3500
batch_size = 3000

dataset = []

