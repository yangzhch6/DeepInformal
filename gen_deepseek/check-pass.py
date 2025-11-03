import json
from datasets import load_dataset

def count_reflections(response):
    reflection_words = ["i think", "in my opinion", "i believe", "it seems to me", "from my perspective", "however", "on the other hand", "but", "although", "yet", "nevertheless", "wait", "let me see", "let me think", "considering that", "taking into account", "after all", "as i see it", "to be honest", "frankly", "candidly", "honestly", "maybe", "perhaps", "it appears", "i wonder", "i question", "i doubt", "i realize", "i notice", "i observe", "it occurs to me", "might", "could", "seems", "seemingly", "apparently", "likely", "possibly"]
    count = 0
    for word in reflection_words:
        count += response.lower().count(word)
    return count

def count_thinking_length(response):
    start_tag = "<synthetic_think>"
    end_tag = "</synthetic_think>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return 0
    thinking_content = response[start_index + len(start_tag):end_index]
    return len(thinking_content)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Jiahao004/DeepTheorem
# yangzhch6/Putnam-Informal-1995-2024
original_data = load_dataset("yangzhch6/Putnam-Informal-1995-2024", split="train").to_pandas()
question_key = "question" # "informal_theorem_qa"
ground_truth_key = "proof"

# 读取/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/data下的每一个{id}.json文件
# iteration 读取
pass_count = 0
all_count = 0
reflection_avg = []
thinking_length_avg = []

synthetic_thinking_data = []
for index, row in original_data.iterrows():
    if index >= 99999999:
        break
    print(f"Checking ID: {index}")
    # 判断是否存在对应的json文件
    file_path = f"/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/putnam-prompt3-deepseekv3reasoner/{index}.json"
    try:
        data = load_json(file_path)
        # print(f"Loaded data for ID {index}.")
        assert row[question_key].strip() == data["question"].strip()
        for k in range(len(data["thinking_cots"])):
            if data["thinking_cots"][k]["tag"] == "pass":
                pass_count += 1
                reflection_count = count_reflections(data["thinking_cots"][k]["response"])
                thinking_length = count_thinking_length(data["thinking_cots"][k]["response"])
                reflection_avg.append(reflection_count)
                thinking_length_avg.append(thinking_length)

                synthetic_thinking_data.append({
                    "id": index,
                    "question": data["question"],
                    "ground_truth_proof": data["ground_truth"],
                    "synthetic_thinking_response": data["thinking_cots"][k]["response"].replace("<synthetic_think>", "<think>").replace("</synthetic_think>", "</think>").strip(),
                    "reflection_count": reflection_count,
                    "thinking_length": thinking_length
                })
            all_count += 1
    except FileNotFoundError:
        print(f"File not found for ID {index}: {file_path}")
        continue
    
    # print("-----------------------------------------")

print(f"Pass rate: {pass_count}/{all_count} = {pass_count/all_count:.2%}")
if pass_count > 0:
    print(f"Average reflection count in passing responses: {sum(reflection_avg)/len(reflection_avg):.2f}")
    print(f"Average thinking length in passing responses: {sum(thinking_length_avg)/len(thinking_length_avg):.2f}")

print(f"Total synthetic thinking data collected: {len(synthetic_thinking_data)}")

# convert synthetic_thinking_data to parquet, submit to huggingface dataset repo "yangzhch6/DeepInformal"
huggignface_token = os.environ.get('yangzhch6_HF_TOKEN')
from datasets import Dataset
synthetic_thinking_dataset = Dataset.from_list(synthetic_thinking_data)
synthetic_thinking_dataset.push_to_hub("yangzhch6/DeepInformal-Putnam-1995-2024", token=huggignface_token)