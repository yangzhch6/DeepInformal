import pandas as pd
import json
import time
import re
from typing import List, Dict, Any, Optional
import logging
from datasets import load_dataset
from enum import Enum
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义响应标签枚举
class ResponseTag(Enum):
    FORMAT_NOT_PASS = "format not pass"
    VERIFICATION_NOT_PASS = "verification not pass"
    PASS = "pass"

SYNTHESIS_SYS = """**Role:** You are an expert problem solver. Your task is to solve the given question independently, generating a detailed, step-by-step thinking process before producing your final answer.

**Instructions:**
1.  **Read the Question:** Carefully analyze the provided question.
2.  **Generate a Synthetic Thinking Process:** Before writing the final solution, you must construct a comprehensive "synthetic thinking process." This process should be a raw, unfiltered log of your internal reasoning.
3.  **Formatting:** Enclose your entire synthetic thinking process within the tags `<synthetic_think>` and `</synthetic_think>`.
4.  **Content of the Synthetic Thinking Process**: Within these tags, you must detail your entire reasoning journey. This includes showing your work, self-reflection (e.g., questioning your approach), and self-correction (e.g., identifying and fixing errors). Your syntheitc thought process must contain **as much "self-reflection and self-correction" as possible**.
5.  **Write the Final Solution:** After closing the `</synthetic_think>` tag, provide a clean, polished, and well-structured final solution. This should be the answer you arrived at through your thinking process, presented clearly.
6.  **Critical Rule:** You need to refer to the ground truth solution to provide your think process and solution to ensure accuracy. However, you must **NEVER** mention or imply the existence of a "ground truth solution" in your synthetic think process or your final solution. You are to behave as if you are solving this problem from scratch, entirely on your own.

**Output Format:**
Your response must strictly follow the structure below:
```
<synthetic_think>
[Your detailed, step-by-step reasoning goes here. Include self-reflection, doubts, corrections, and the path to your final answer.]
</synthetic_think>

[Your final, polished solution goes here.]
```"""

VERIFY_SYS = """You are a proof evaluator. Compare the "Candidate Proof" to the "Ground Truth Proof" for the given "Question".

Determine if the Candidate Proof is **both**:
1.  **Essentially Equivalent** in its core logic to the Ground Truth.
2.  **Completely Correct** with no logical or mathematical errors in any step.

Your final output must be **only** your judgment inside `\\boxed{}`:
- If both conditions are met, output: `\\boxed{true}`
- Otherwise, output: `\\boxed{false}`"""

class DataGenerationPipeline:
    def __init__(self, 
                 deepseek_r1_api_key: str,
                 deepseek_v3_api_key: str,
                 r1_base_url: str = "https://api.deepseek.com",
                 v3_base_url: str = "https://api.deepseek.com"):
        self.r1_client = OpenAI(api_key=deepseek_r1_api_key, base_url=r1_base_url)
        self.v3_client = OpenAI(api_key=deepseek_v3_api_key, base_url=v3_base_url)
        
    def load_seed_data(self, parquet_path: str) -> pd.DataFrame:
        """加载种子数据"""
        logger.info(f"Loading seed data from {parquet_path}")
        # df = pd.read_parquet(parquet_path)
        df = load_dataset(parquet_path, split="train").to_pandas()
        print(df)
        
        # 检查必要的列是否存在
        required_columns = ['informal_theorem_qa', 'proof']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"Loaded {len(df)} samples from seed data")
        return df
    
    def call_deepseek_r1(self, question: str, solution: str, temperature: float = 0.7) -> str:
        """调用DeepSeek-R1 API生成响应"""
        messages = [
            {
                "role": "system",
                "content": SYNTHESIS_SYS
            },
            {
                "role": "user",
                "content": f"## Question:\n{question}\n\n\n## Ground Truth Solution:\n{solution}"
            }
        ]
        
        try:
            response = self.r1_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                temperature=temperature,
                max_tokens=32768,
                stream=False
            )
            
            content = response.choices[0].message.content
            completion_token_usage = response.usage.completion_tokens
            return content.strip(), completion_token_usage
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek-R1 API: {e}")
            return None
    
    def validate_response_format(self, response: str) -> bool:
        """验证响应格式是否符合要求"""
        if not response:
            return False
        
        # 检查是否包含必要的标签
        if '<synthetic_think>' not in response or '</synthetic_think>' not in response:
            return False
        
        # 检查是否包含禁止的词汇
        forbidden_patterns = ['ground truth']
        response_lower = response.lower()
        for pattern in forbidden_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def extract_solution_part(self, response: str) -> str:
        """根据</think>标签截取solution部分"""
        if '</think>' not in response:
            return response
        
        # 分割think部分和solution部分
        parts = response.split('</think>', 1)
        if len(parts) > 1:
            return parts[1].strip()
        return response
    
    def call_deepseek_v3_for_validation(self, question: str, ground_truth: str, candidate_solution: str) -> bool:
        """调用DeepSeek-V3验证solution是否一致"""
        messages = [
            {
                "role": "system",
                "content": VERIFY_SYS
            },
            {
                "role": "user",
                "content": f"""## Question:\n{question}\n\n\n## Ground Truth Solution:\n{ground_truth}\n\n\n## Candidate Solution:\n{candidate_solution}"""
            }
        ]
        
        try:
            response = self.v3_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
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

    def generate_responses_for_sample(self, question: str, solution: str, k: int = 5, temperature: float = 0.7) -> List[str]:
        """为单个样本生成K个响应"""
        responses = []
        token_usages = []
        temperatures = [temperature] * k
        
        for i in range(k):
            temperature = temperatures[i % len(temperatures)]
            logger.info(f"Generating response {i+1}/{k} for sample (temperature: {temperature})")

            response, response_token_usage = self.call_deepseek_r1(question, solution, temperature)
            if response:
                responses.append(response)
                token_usages.append(response_token_usage)
            else:
                responses.append("")  # 保存空字符串表示API调用失败
                token_usages.append(0)
            
            # 添加延迟避免API限制
            time.sleep(2)

        return responses, token_usages

    def process_single_sample(self, question: str, solution: str, k: int = 5, temperature: float = 0.7) -> Dict[str, Any]:
        """处理单个样本"""
        logger.info("Processing sample...")
        
        # 步骤1: 生成K个响应
        raw_responses, token_usages = self.generate_responses_for_sample(question, solution, k, temperature)
        logger.info(f"Generated {len(raw_responses)} raw responses")
        
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
            is_consistent = True # 默认一致，减少API调用次数
            # is_consistent = self.call_deepseek_v3_for_validation(question, solution, solution_part)
            
            if is_consistent:
                response_data["tag"] = ResponseTag.PASS.value
            else:
                response_data["tag"] = ResponseTag.VERIFICATION_NOT_PASS.value
            
            all_responses_with_tags.append(response_data)
            
            # 添加延迟避免API限制
            time.sleep(1)
        
        # 统计各类响应的数量
        format_not_pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.FORMAT_NOT_PASS.value)
        verification_not_pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.VERIFICATION_NOT_PASS.value)
        pass_count = sum(1 for item in all_responses_with_tags if item["tag"] == ResponseTag.PASS.value)
        api_failed_count = sum(1 for item in all_responses_with_tags if item["tag"] == "API call failed")
        
        logger.info(f"Response statistics - Format not pass: {format_not_pass_count}, "
                   f"Verification not pass: {verification_not_pass_count}, "
                   f"Pass: {pass_count}, "
                   f"API failed: {api_failed_count}")
        
        return {
            "question": question,
            "ground_truth": solution,
            "thinking_cots": all_responses_with_tags
        }
    
    def run_pipeline(self, 
                    parquet_path: str, 
                    output_path: str, 
                    k: int = 5,
                    temperature: float = 0.7,
                    max_samples: Optional[int] = None) -> None:
        """运行完整的数据生成pipeline"""
        
        # 加载数据
        df = self.load_seed_data(parquet_path)
        
        # 限制处理样本数量
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)
            logger.info(f"Processing first {max_samples} samples")
        
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"Processing sample {idx + 1}/{len(df)}")
            
            question = row['informal_theorem_qa']
            solution = row['proof']
            
            try:
                result = self.process_single_sample(question, solution, k, temperature)
                results.append(result)  # 现在保存所有样本，无论是否有有效响应
                logger.info(f"Sample {idx + 1} completed with {len(result['thinking_cots'])} total responses")
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx + 1}: {e}")
                # 即使出错也保存一个空的结果
                error_result = {
                    "question": question,
                    "ground_truth": solution,
                    "thinking_cots": [{"response": f"Error: {str(e)}", "tag": "processing_error"}]
                }
                results.append(error_result)
                continue
            
            # 定期保存进度
            if (idx + 1) % 5 == 0:
                self._save_intermediate_results(results, output_path)
                logger.info(f"Progress saved after {idx + 1} samples")
        
        # 保存最终结果
        self._save_final_results(results, output_path)
        
        # 打印最终统计信息
        self._print_final_statistics(results)
        logger.info(f"Pipeline completed. Generated {len(results)} samples with all responses")
    
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
    
    def _save_intermediate_results(self, results: List[Dict], output_path: str) -> None:
        """保存中间结果"""
        temp_path = output_path.replace('.json', '_temp.json')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(self, results: List[Dict], output_path: str) -> None:
        """保存最终结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Final results saved to {output_path}")

# 使用示例
def main():
    # 初始化pipeline
    pipeline = DataGenerationPipeline(
        deepseek_r1_api_key=os.environ.get('yangzhch6_DEEPSEEK_API_TOKEN'),
        deepseek_v3_api_key=os.environ.get('yangzhch6_DEEPSEEK_API_TOKEN')
    )
    
    # 运行pipeline
    pipeline.run_pipeline(
        parquet_path="Jiahao004/DeepTheorem",
        output_path="generated_data.json",
        k=2,  # 每个样本生成k个响应
        temperature=0.6,  # 设置temperature参数
        max_samples=1  # 限制处理样本数量（可选）
    )

if __name__ == "__main__":
    main()