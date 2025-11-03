# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

SYS_PROMPT_1 = """You are a math AI assistant. You need to solve the given math problem. For calculation problem, show your work clearly and put your final answer within \\boxed{}. For proof problem, provide a rigorous logical derivation. Ensure your solution is clearly stated. (You can only use natural language, not formal language.)"""

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/train/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Elliott/Openr1-Math-46k-8192"

    dataset = datasets.load_dataset(data_source, split="train")
    dataset = dataset.to_list()[:1024]
    # print(dataset[0])

    processed_data = []
    for line in dataset:
        line["prompt"][0]["content"] = SYS_PROMPT_1
        processed_data.append(line)

    # convert to dataset and save to parquet
    train_dataset = datasets.Dataset.from_list(processed_data)
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(args.local_dir, "Openr1-Math-46k-Prompt1-debug.parquet"))
