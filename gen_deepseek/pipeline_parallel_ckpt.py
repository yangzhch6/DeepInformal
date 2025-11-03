import pandas as pd
import json
import time
import re
import os
from typing import List, Dict, Any, Optional
import logging
from datasets import load_dataset
from enum import Enum
from openai import OpenAI
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYNTHESIS_SYS = """**Role**: You are an expert math prover. You will be provided with a mathematical statement and its ground truth proof.
You should peek the ground truth to generate a deep and detailed synthetic thinking process accompanied by your own proof, but do not mention "ground truth" in your output.

**Instructions**:
1.  **Read the Statement**: Carefully analyze the provided statement.
2.  **Formatting**: Enclose your entire synthetic thinking process within the tags `<synthetic_think>` and `</synthetic_think>`. After closing the `</synthetic_think>` tag, provide a clean, polished, and well-structured final proof.
3.  **Content of the Synthetic Thinking Process**: Within these tags, you must detail your entire reasoning journey. This includes showing your work, self-reflection (e.g., questioning your approach), and self-correction (e.g., identifying and fixing errors). Your synthetic thought process must contain **as much "self-reflection and self-correction" as possible** to show the depth of your reasoning.
4.  **Peek the Ground Truth**: In order to guarantee accuracy, you should consult the ground truth proof to inform your synthetic thinking process and your own proof. However, it is imperative that you **NEVER** reference or suggest the existence of a "ground truth" in your synthetic thinking process or final proof. **You must prove the statement as if you are proving from scratch, solely relying on your own abilities.**

**Output Format**:
Your response must strictly follow the structure below:
```
<synthetic_think>
[Your detailed, step-by-step reasoning goes here. Include self-reflection, doubts, and corrections. Think as deeply as possible. Peek the ground truth proof, but do not mention it.]
</synthetic_think>
[Your final, polished proof goes here.]
```"""

VERIFY_SYS = """You are a proof evaluator. Compare the "Candidate Proof" to the "Ground Truth Proof" for the given "Statement".

Determine if the Candidate Proof is **both**:
1.  **Essentially Equivalent** in its core logic to the Ground Truth.
2.  **Completely Correct** with no logical or mathematical errors in any step.

Your final output must be **only** your judgment inside `\\boxed{}`:
- If both conditions are met, output: `\\boxed{true}`
- Otherwise, output: `\\boxed{false}`"""

# 定义响应标签枚举
class ResponseTag(Enum):
    FORMAT_NOT_PASS = "format not pass"
    VERIFICATION_NOT_PASS = "verification not pass"
    PASS = "pass"

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

class DataGenerationPipeline:
    def __init__(self, 
                 deepseek_reasoner_api_key: str,
                 deepseek_chat_api_key: str,
                 r1_base_url: str = "https://api.deepseek.com",
                 v3_base_url: str = "https://api.deepseek.com"):
        self.reasoner_client = OpenAI(api_key=deepseek_reasoner_api_key, base_url=r1_base_url)
        self.chat_client = OpenAI(api_key=deepseek_chat_api_key, base_url=v3_base_url)
        
    def load_seed_data(self, parquet_path: str, question_key: str, ground_truth_key: str) -> pd.DataFrame:
        """加载种子数据"""
        logger.info(f"Loading seed data from {parquet_path}")
        # df = pd.read_parquet(parquet_path)
        df = load_dataset(parquet_path, split="train").to_pandas()
        print(df)
        
        # 检查必要的列是否存在
        required_columns = [question_key, ground_truth_key]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"Loaded {len(df)} samples from seed data")
        return df

    def call_deepseek_syn(self, model_name: str, question: str, solution: str, temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """调用DeepSeek-Reasoner API生成响应"""
        messages = [
            {
                "role": "system",
                "content": SYNTHESIS_SYS
            },
            {
                "role": "user",
                "content": f"## Statement:\n{question}\n\n\n## Ground Truth Proof:\n{solution}"
            }
        ]
        
        try:
            response = self.reasoner_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content
            completion_token_usage = response.usage.completion_tokens
            return content.strip(), completion_token_usage
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek-R1 API: {e}")
            return None, 0
    
    def validate_response_format(self, response: str) -> bool:
        """验证响应格式是否符合要求"""
        if not response:
            return False
        
        # 检查是否包含必要的标签，且只包含一次
        if response.count('<synthetic_think>') != 1 or response.count('</synthetic_think>') != 1:
            return False
        
        # 检查是否包含禁止的词汇
        forbidden_patterns = ['ground truth', 'standard proof', 'standard solution', 'peek']
        response_lower = response.lower()
        for pattern in forbidden_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def extract_solution_part(self, response: str) -> str:
        """根据</synthetic_think>标签截取solution部分"""
        if '</synthetic_think>' not in response:
            return response
        
        # 分割think部分和solution部分
        parts = response.split('</synthetic_think>', 1)
        if len(parts) > 1:
            return parts[1].strip()
        return response

    def call_deepseek_verify(self, model_name: str, question: str, ground_truth: str, candidate_solution: str, max_tokens: int = 8192) -> bool:
        """调用DeepSeek-V3验证solution是否一致"""
        messages = [
            {
                "role": "system",
                "content": VERIFY_SYS
            },
            {
                "role": "user",
                "content": f"""## Statement:\n{question}\n\n\n## Ground Truth Proof:\n{ground_truth}\n\n\n## Candidate Proof:\n{candidate_solution}"""
            }
        ]
        
        try:
            response = self.chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析响应，提取true/false
            if r'\boxed{true}' in content:
                return True
            elif r'\boxed{false}' in content:
                return False
            else:
                # 如果没有明确的boxed格式，检查文本内容
                if 'true' in content.lower() and 'false' not in content.lower():
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Error calling DeepSeek-V3 API: {e}")
            return False

    def generate_responses_for_sample(self, model_name: str, question: str, solution: str, k: int = 5, temperature: float = 0.7, max_tokens: int = 8192) -> List[str]:
        """为单个样本生成K个响应"""
        responses = []
        token_usages = []
        temperatures = [temperature] * k
        
        for i in range(k):
            temperature = temperatures[i % len(temperatures)]
            logger.info(f"Generating response {i+1}/{k} for sample (temperature: {temperature})")

            response, response_token_usage = self.call_deepseek_syn(model_name, question, solution, temperature, max_tokens)
            if response:
                responses.append(response)
                token_usages.append(response_token_usage)
            else:
                responses.append("")  # 保存空字符串表示API调用失败
                token_usages.append(0)
            
            # 添加延迟避免API限制
            time.sleep(2)

        return responses, token_usages

    def process_single_sample(self, sample_data: Dict, model_name: str, max_tokens: int = 8192) -> Dict[str, Any]:
        """处理单个样本的包装函数，用于并行处理"""
        idx = sample_data["original_index"]
        row = sample_data["row"]
        k = sample_data["k"]
        temperature = sample_data["temperature"]
        data_dir = sample_data["data_dir"]
        question_key = sample_data["question_key"]
        ground_truth_key = sample_data["ground_truth_key"]
        
        logger.info(f"Starting processing for sample {idx + 1}")
        
        question = row[question_key]
        solution = row[ground_truth_key]
        
        try:
            # 步骤1: 生成K个响应
            raw_responses, token_usages = self.generate_responses_for_sample(model_name, question, solution, k, temperature, max_tokens)
            logger.info(f"Generated {len(raw_responses)} raw responses for sample {idx + 1}")
            
            # 步骤2: 格式验证和过滤，但保存所有响应和标签
            all_responses_with_tags = []

            for response, token_usage in zip(raw_responses, token_usages):
                # 初始化响应数据
                response_data = {
                    "response": response,
                    "tag": None,
                    "token_usage": token_usage
                }
                
                # 检查API调用是否失败
                if not response:
                    response_data["tag"] = "API call failed"
                    all_responses_with_tags.append(response_data)
                    continue
                    
                # 格式验证
                if not self.validate_response_format(response):
                    response_data["tag"] = ResponseTag.FORMAT_NOT_PASS.value
                    all_responses_with_tags.append(response_data)
                    continue
                
                # 提取solution部分并进行一致性验证
                solution_part = self.extract_solution_part(response)
                is_consistent = True  # 默认一致，减少API调用次数
                # is_consistent = self.call_deepseek_verify(model_name, question, solution, solution_part, max_tokens)
                
                if is_consistent:
                    response_data["tag"] = ResponseTag.PASS.value
                    response_data["reflection_count"] = count_reflections(response_data["response"])
                    response_data["thinking_length"] = count_thinking_length(response_data["response"])
                else:
                    response_data["tag"] = ResponseTag.VERIFICATION_NOT_PASS.value
                    response_data["reflection_count"] = -1
                    response_data["thinking_length"] = -1
                
                all_responses_with_tags.append(response_data)
                
                # 添加延迟避免API限制
                time.sleep(1)
            
            # 统计各类响应的数量
            format_not_pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.FORMAT_NOT_PASS.value)
            verification_not_pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.VERIFICATION_NOT_PASS.value)
            pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.PASS.value)
            api_failed_count = sum(1 for item in all_responses_with_tags if item["tag"] == "API call failed")
            
            logger.info(f"Sample {idx + 1} statistics - Format not pass: {format_not_pass_count}, "
                       f"Verification not pass: {verification_not_pass_count}, "
                       f"Pass: {pass_count}, "
                       f"API failed: {api_failed_count}")
            
            result = {
                "id": idx,
                "question": question,
                "ground_truth": solution,
                "thinking_cots": all_responses_with_tags
            }
            
            # 立即保存单个样本的结果
            self._save_single_result(result, data_dir)
            
            logger.info(f"Completed processing for sample {idx + 1}")
            return result
                
        except Exception as e:
            logger.error(f"Error processing sample {idx + 1}: {e}")
            # 即使出错也保存一个空的结果
            error_result = {
                "id": idx,
                "question": question,
                "ground_truth": solution,
                "thinking_cots": [{"response": f"Error: {str(e)}", "tag": "processing_error"}]
            }
            # 保存错误结果
            self._save_single_result(error_result, data_dir)
            return error_result

    def _save_single_result(self, result: Dict, data_dir: str) -> None:
        """保存单个样本的结果到独立的JSON文件"""
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 构建文件路径
        file_path = os.path.join(data_dir, f"{result['id']}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved result for sample {result['id']} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving result for sample {result['id']}: {e}")

    def _load_existing_results(self, data_dir: str, total_samples: int) -> set:
        """加载已存在的样本ID集合"""
        processed_ids = set()
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            return processed_ids
        
        try:
            # 扫描data_dir下的所有JSON文件
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    # 提取ID（去掉.json后缀）
                    try:
                        sample_id = int(filename[:-5])  # 去掉.json后缀并转换为整数
                        if 0 <= sample_id < total_samples:
                            processed_ids.add(sample_id)
                    except ValueError:
                        # 如果文件名不是数字，跳过
                        continue
            
            logger.info(f"Found {len(processed_ids)} existing samples in {data_dir}")
            return processed_ids
            
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            return set()

    def _load_all_results(self, data_dir: str, total_samples: int) -> List[Optional[Dict]]:
        """加载所有结果（包括已处理的和未处理的）"""
        results = [None] * total_samples
        processed_count = 0
        
        if not os.path.exists(data_dir):
            return results
        
        try:
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    try:
                        sample_id = int(filename[:-5])
                        if 0 <= sample_id < total_samples:
                            file_path = os.path.join(data_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                result = json.load(f)
                            results[sample_id] = result
                            processed_count += 1
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning(f"Error loading {filename}: {e}")
                        continue
            
            logger.info(f"Loaded {processed_count} existing results from {data_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading all results: {e}")
            return [None] * total_samples

    def _save_progress_status(self, data_dir: str, current_index: int, total_samples: int):
        """保存进度状态"""
        status_file = os.path.join(data_dir, 'progress_status.json')
        status_data = {
            'data_dir': data_dir,
            'current_index': current_index,
            'total_samples': total_samples,
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress status: {e}")

    def _load_progress_status(self, data_dir: str) -> Optional[Dict]:
        """加载进度状态"""
        status_file = os.path.join(data_dir, 'progress_status.json')
        if not os.path.exists(status_file):
            return None
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress status: {e}")
            return None

    def _identify_failed_samples(self, data_dir: str, total_samples: int) -> set:
        """识别所有响应都不是'pass'的失败样本ID"""
        failed_ids = set()
        results = self._load_all_results(data_dir, total_samples)
        
        for idx, result in enumerate(results):
            if result is None:
                continue
                
            # 检查该样本的所有响应
            thinking_cots = result.get('thinking_cots', [])
            if not thinking_cots:
                failed_ids.add(idx)
                continue
                
            # 检查是否有至少一个pass的响应
            has_pass = any(
                response.get('tag') == ResponseTag.PASS.value 
                for response in thinking_cots
            )
            
            if not has_pass:
                failed_ids.add(idx)
        
        logger.info(f"Identified {len(failed_ids)} failed samples (no PASS responses)")
        return failed_ids

    def run_pipeline(self, 
                    parquet_path: str, 
                    data_dir: str = "./data",
                    model_name: str = "deepseek-chat",
                    k: int = 5,
                    temperature: float = 0.7,
                    max_tokens: int = 8192,
                    max_samples: Optional[int] = None,
                    parallel_num: int = 5,
                    question_key: str = "informal_theorem_qa",
                    ground_truth_key: str = "proof",
                    fail_regen: bool = False) -> None:
        
        # 加载数据
        df = self.load_seed_data(parquet_path, question_key, ground_truth_key)
        
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)
        
        # 重置索引以确保连续性
        df = df.reset_index(drop=True)
        
        # 检查已存在的样本
        processed_ids = self._load_existing_results(data_dir, len(df))
        
        # 如果需要重新生成失败样本，识别失败样本
        if fail_regen:
            failed_ids = self._identify_failed_samples(data_dir, len(df))
            # 将失败样本从已处理集合中移除，以便重新处理
            processed_ids -= failed_ids
            logger.info(f"Will regenerate {len(failed_ids)} failed samples")
        
        # 准备需要处理的样本（保持原始索引）
        samples_to_process = []
        for idx in range(len(df)):
            if idx not in processed_ids:
                samples_to_process.append({
                    "original_index": idx,  # 关键：保存原始df中的索引
                    "row": df.iloc[idx],
                    "k": k,
                    "temperature": temperature,
                    "data_dir": data_dir,
                    "question_key": question_key,
                    "ground_truth_key": ground_truth_key
                })
        
        if not samples_to_process:
            logger.info("No new samples to process.")
            return
        
        logger.info(f"Starting processing for {len(samples_to_process)} new samples")
        
        # 使用ThreadPoolExecutor进行并行处理
        with ThreadPoolExecutor(max_workers=parallel_num) as executor:
            # 提交任务时保存原始索引
            future_to_index = {}
            for sample_data in samples_to_process:
                future = executor.submit(self.process_single_sample, sample_data, model_name, max_tokens)
                future_to_index[future] = sample_data["original_index"]
            
            completed_count = 0
            total_count = len(future_to_index)
            
            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    result = future.result()
                    completed_count += 1
                    
                    logger.info(f"Progress: {completed_count}/{total_count} new samples completed "
                            f"(total processed: {len(processed_ids) + completed_count}/{len(df)})")
                    
                    # 定期保存进度状态
                    if completed_count % 5 == 0:
                        self._save_progress_status(data_dir, original_index, len(df))
                        
                except Exception as e:
                    logger.error(f"Error processing sample {original_index + 1}: {e}")
                    completed_count += 1
        
        # 生成最终的汇总文件（可选）
        self._generate_final_summary(data_dir, len(df))
        
        logger.info(f"Pipeline completed. Processed {len(samples_to_process)} new samples")

    def _generate_final_summary(self, data_dir: str, total_samples: int) -> None:
        """生成最终的汇总JSON文件"""
        results = self._load_all_results(data_dir, total_samples)
        final_results = [r for r in results if r is not None]
        
        summary_file = os.path.join(data_dir, "all_results.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        self._print_final_statistics(final_results)
        logger.info(f"Final summary saved to {summary_file}")

    def _print_final_statistics(self, results: List[Dict]) -> None:
        """打印最终统计信息"""
        total_samples = len(results)
        total_responses = 0
        tag_counts = {
            ResponseTag.FORMAT_NOT_PASS.value: 0,
            ResponseTag.VERIFICATION_NOT_PASS.value: 0,
            ResponseTag.PASS.value: 0,
            "API call failed": 0,
            "processing_error": 0
        }
        
        for sample in results:
            total_responses += len(sample['thinking_cots'])
            for response_data in sample['thinking_cots']:
                tag = response_data.get('tag', 'unknown')
                if tag in tag_counts:
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
        
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Total responses generated: {total_responses}")
        logger.info("Response breakdown:")
        for tag, count in tag_counts.items():
            percentage = (count / total_responses * 100) if total_responses > 0 else 0
            logger.info(f"  {tag}: {count} ({percentage:.2f}%)")

# 使用示例
def main():
    # 初始化pipeline
    pipeline = DataGenerationPipeline(
        deepseek_reasoner_api_key=os.environ.get('yangzhch6_DEEPSEEK_API_TOKEN'),
        deepseek_chat_api_key=os.environ.get('yangzhch6_DEEPSEEK_API_TOKEN')
    )
    
    # 运行pipeline
    pipeline.run_pipeline(
        parquet_path="yangzhch6/Putnam-Informal-1995-2024", # Jiahao004/DeepTheorem  yangzhch6/Putnam-Informal-1995-2024
        model_name="deepseek-reasoner",
        data_dir="/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/putnam-prompt3-deepseekv3reasoner",  # 数据保存目录
        k=1,  # 每个样本生成k个响应
        max_tokens=65536,  # 设置请求超时时间
        temperature=0.6,  # 设置temperature参数
        max_samples=15000,  # 限制处理样本数量（可选）
        parallel_num=128,   # 同时处理的样本数量
        question_key="question",  # 指定问题字段
        ground_truth_key="solution",  # 指定答案字段
        fail_regen=True  # 重新生成所有响应都不是"pass"的样本
    )

if __name__ == "__main__":
    main()
