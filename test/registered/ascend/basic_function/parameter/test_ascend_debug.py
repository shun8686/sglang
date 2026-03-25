import json
from transformers import AutoTokenizer

# 以BERT为例（其他模型只需替换模型名称）
tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8")

batch_size = 3000
input_len = 3500

dataset = []
dataset_path = "/data/l30081563/test.jsonl"
with open(dataset_path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data['question'])


# repeat input_len
dataset_2k = []
for sentence in dataset:
    words = tokenizer.tokenize(sentence)
    #print(len(words))
    len_num = len(words) // input_len
    if len_num == 0:
        multiplier = (input_len // len(words)) + 1
        repeated_len = words * multiplier
        words = repeated_len[:input_len]
        decoded_text = tokenizer.convert_tokens_to_string(words)
        #print(len(words))
        dataset_2k.append(decoded_text)
    #else:
    #    words = words[:input_len]
    #    merged_sentence = " ".join(words)  # 合并
    #    print(len(words))
    #    dataset_2k.append(merged_sentence)


# repeat to batch_size
batch_num = len(dataset_2k) // batch_size
if batch_num == 0:
    multiplier = (batch_size // len(dataset_2k)) + 1
    repeated_batch = dataset_2k * multiplier
    dataset_2k = repeated_batch[:batch_size]
else:
    dataset_2k = dataset_2k[:batch_size]

print(len(dataset_2k))

#json_str = json.dumps(dataset_2k, ensure_ascii=False, indent=4)
final_result = []
for seq in dataset_2k:
    conversations = {"conversations": [{"from": "human", "value": seq}, {"from": "gpt", "value": ""}]}
    final_result.append(conversations)

with open(f'GSM8K-in{input_len}-bs{batch_size}_qwen3-30b.jsonl', 'w', encoding='utf-8') as f:
    json.dump(final_result, f, indent=4, ensure_ascii=False)
