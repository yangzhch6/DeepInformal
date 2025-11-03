import json
from datasets import load_dataset

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

original_data = load_dataset("Jiahao004/DeepTheorem", split="train").to_pandas()

# 读取/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/data下的每一个{id}.json文件
# iteration 读取
for index, row in original_data.iterrows():
    print(f"Checking ID: {index}")
    # 判断是否存在对应的json文件
    file_path = f"/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/data/{index}.json"
    try:
        data = load_json(file_path)
        print(f"Loaded data for ID {index}.")
        assert row["informal_theorem_qa"].strip() == data["question"].strip()
    except FileNotFoundError:
        print(f"File not found for ID {index}: {file_path}")
        continue
    
    print("-----------------------------------------")