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
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOLVE_SYS = """You are a math AI assistant. You need to solve the given math problems. For calculations, show your work clearly. For proofs, provide a rigorous logical derivation. Ensure your solution is clearly stated. (You can only use natural language, not formal language.)"""

class DeepSeekAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
    
    def call_api(self, 
                question: str, 
                model_name: str = "deepseek-chat",
                temperature: float = 0.7,
                max_tokens: int = 8192) -> Dict[str, Any]:
        """
        调用DeepSeek API并返回响应
        
        Args:
            question: 问题文本
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            包含响应内容的字典
        """
        try:
            messages = self._format_messages(question)
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            content = response.choices[0].message.content
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = response.usage.total_tokens
            
            result = {
                "response_content": content,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": True
            }
            
            # 检查response中是否有reasoning content字段
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                reasoning_content = response.choices[0].message.reasoning_content
                result["reasoning_content"] = reasoning_content
                logger.info("检测到并保存了reasoning content")
            
            return result
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return {
                "response_content": "",
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": False,
                "error": str(e)
            }
    
    def _format_messages(self, question: str) -> List[Dict[str, str]]:
        """格式化消息列表以供模型使用"""
        messages = [
            {
                "role": "system",
                "content": SOLVE_SYS
            },
            {
                "role": "user",
                "content": question
            }
        ]
        return messages

class DatasetProcessor:
    def __init__(self, 
                 api_key: str,
                 model_name: str = "deepseek-chat",
                 responses_per_question: int = 8,
                 parallel_num: int = 5,
                 max_tokens: int = 8192,
                 temperature_list: Optional[List[float]] = None,
                 output_dir: str = "responses"):
        
        self.api_client = DeepSeekAPIClient(api_key)
        self.model_name = model_name
        self.responses_per_question = responses_per_question
        self.parallel_num = parallel_num
        self.max_tokens = max_tokens
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 使用提供的temperature列表，如果没有提供则使用默认值
        if temperature_list is not None:
            self.temperature_list = temperature_list
        else:
            # 默认的temperature范围
            self.temperature_list = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        # 确保responses_per_question不超过temperature_list的长度
        if self.responses_per_question > len(self.temperature_list):
            logger.warning(f"responses_per_question ({self.responses_per_question}) 大于temperature_list长度 ({len(self.temperature_list)})，将循环使用temperature值")
        
    def load_dataset(self):
        """加载DeepInformal-test数据集"""
        logger.info("正在加载DeepInformal-test数据集...")
        try:
            dataset = load_dataset("yangzhch6/DeepInformal-test")
            return dataset
        except Exception as e:
            logger.error(f"数据集加载失败: {str(e)}")
            raise
    
    def get_response_file_path(self, question_idx: int, response_idx: int) -> str:
        """获取单个响应的输出文件路径"""
        return os.path.join(self.output_dir, f"{question_idx}-{response_idx}.json")
    
    def is_response_completed(self, question_idx: int, response_idx: int) -> bool:
        """检查某个问题的某个响应是否已经完成"""
        file_path = self.get_response_file_path(question_idx, response_idx)
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查文件是否包含有效的响应数据
            return isinstance(data, dict) and "response_content" in data and data.get("api_success", False)
        except:
            return False
            
    def get_completed_responses_for_question(self, question_idx: int) -> List[int]:
        """获取某个问题已完成的响应索引列表"""
        completed_responses = []
        for response_idx in range(self.responses_per_question):
            if self.is_response_completed(question_idx, response_idx):
                completed_responses.append(response_idx)
        return completed_responses
    
    def get_remaining_responses_for_question(self, question_idx: int) -> List[int]:
        """获取某个问题需要处理的剩余响应索引"""
        completed_responses = self.get_completed_responses_for_question(question_idx)
        all_responses = list(range(self.responses_per_question))
        remaining_responses = [idx for idx in all_responses if idx not in completed_responses]
        return remaining_responses
    
    def is_question_completed(self, question_idx: int) -> bool:
        """检查某个问题是否已经完成所有响应"""
        completed_responses = self.get_completed_responses_for_question(question_idx)
        return len(completed_responses) >= self.responses_per_question
    
    def get_completed_questions(self, total_samples: int) -> List[int]:
        """获取所有已完成问题的索引"""
        completed_questions = []
        for question_idx in range(total_samples):
            if self.is_question_completed(question_idx):
                completed_questions.append(question_idx)
        return completed_questions
    
    def get_remaining_questions(self, total_samples: int) -> List[int]:
        """获取需要处理的剩余问题索引"""
        completed_questions = self.get_completed_questions(total_samples)
        all_questions = list(range(total_samples))
        remaining_questions = [idx for idx in all_questions if idx not in completed_questions]
        return remaining_questions
    
    def save_single_response(self, question_idx: int, response_idx: int, response_data: Dict[str, Any]):
        """保存单个响应到文件"""
        file_path = self.get_response_file_path(question_idx, response_idx)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
    
    def load_single_response(self, question_idx: int, response_idx: int) -> Optional[Dict[str, Any]]:
        """从文件加载单个响应"""
        file_path = self.get_response_file_path(question_idx, response_idx)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def load_all_responses_for_question(self, question_idx: int) -> List[Dict[str, Any]]:
        """加载某个问题的所有响应"""
        responses = []
        for response_idx in range(self.responses_per_question):
            response_data = self.load_single_response(question_idx, response_idx)
            if response_data:
                responses.append(response_data)
        return responses
    
    def process_single_response(self, 
                              question_data: Dict[str, Any], 
                              question_idx: int,
                              response_idx: int) -> Dict[str, Any]:
        """
        处理单个问题的单个响应
        
        Args:
            question_data: 问题数据
            question_idx: 问题索引
            response_idx: 响应索引
            
        Returns:
            响应数据
        """
        # 检查是否已经完成
        if self.is_response_completed(question_idx, response_idx):
            logger.info(f"问题 {question_idx} 的响应 {response_idx} 已完成，跳过")
            return self.load_single_response(question_idx, response_idx)
        
        logger.info(f"处理问题 {question_idx} 的第 {response_idx+1} 次响应")
        
        # 从temperature_list中获取temperature，循环使用
        temperature = self.temperature_list[response_idx % len(self.temperature_list)]
        
        question = question_data.get("question", "")
        
        # 调用API
        api_result = self.api_client.call_api(
            question=question,
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        
        # 构建结果记录
        result_record = {
            # 原始数据字段
            **question_data,
            # API响应信息
            "question_idx": question_idx,
            "response_idx": response_idx,
            "temperature": temperature,
            "api_success": api_result["success"],
            "response_content": api_result["response_content"],
            "completion_tokens": api_result["completion_tokens"],
            "prompt_tokens": api_result["prompt_tokens"],
            "total_tokens": api_result["total_tokens"],
            "model_name": api_result["model_name"]
        }
        
        # 只有当API结果中包含reasoning_content时才添加该字段
        if "reasoning_content" in api_result:
            result_record["reasoning_content"] = api_result["reasoning_content"]
        
        # 如果API调用失败，记录错误信息
        if not api_result["success"]:
            result_record["error"] = api_result.get("error", "")
        
        # 保存响应到文件
        self.save_single_response(question_idx, response_idx, result_record)
        
        # 添加延迟以避免API限制
        time.sleep(1)
        
        return result_record
    
    def process_single_question(self, 
                              question_data: Dict[str, Any], 
                              question_idx: int) -> List[Dict[str, Any]]:
        """
        处理单个问题的所有响应
        
        Args:
            question_data: 问题数据
            question_idx: 问题索引
            
        Returns:
            该问题的所有响应列表
        """
        # 获取需要处理的剩余响应
        remaining_responses = self.get_remaining_responses_for_question(question_idx)
        
        if not remaining_responses:
            logger.info(f"问题 {question_idx} 已完成所有 {self.responses_per_question} 个响应")
            return self.load_all_responses_for_question(question_idx)
        
        logger.info(f"问题 {question_idx}: 已有 {self.responses_per_question - len(remaining_responses)} 个响应，需要生成 {len(remaining_responses)} 个新响应")
        
        all_responses = []
        
        # 处理每个响应
        for response_idx in remaining_responses:
            response_data = self.process_single_response(question_data, question_idx, response_idx)
            all_responses.append(response_data)
        
        # 加载所有响应（包括之前完成的）
        all_responses = self.load_all_responses_for_question(question_idx)
        logger.info(f"问题 {question_idx} 已完成所有 {len(all_responses)} 个响应")
        
        return all_responses
    
    def combine_all_responses(self, final_output_file: str = "deepseek_responses.json"):
        """合并所有样本的响应到最终文件"""
        logger.info("正在合并所有样本的响应...")
        
        all_responses = []
        
        # 获取所有响应文件
        response_files = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json') and '-' in filename:
                try:
                    parts = filename.split('-')
                    question_idx = int(parts[0])
                    response_idx = int(parts[1].split('.')[0])
                    response_files.append((question_idx, response_idx, filename))
                except ValueError:
                    continue
        
        # 按question_idx和response_idx排序
        response_files.sort(key=lambda x: (x[0], x[1]))
        
        # 加载所有响应
        for question_idx, response_idx, filename in response_files:
            file_path = os.path.join(self.output_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                all_responses.append(response_data)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {str(e)}")
        
        # 保存最终文件
        with open(final_output_file, "w", encoding="utf-8") as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合并完成！总响应数: {len(all_responses)}，已保存到 {final_output_file}")
        return all_responses
    
    def process_dataset(self, final_output_file: str = "deepseek_responses.json"):
        """
        处理整个数据集，支持断点续传
        
        Args:
            final_output_file: 最终输出文件名
        """
        # 加载数据集
        dataset = self.load_dataset()
        test_data = dataset["train"]  # 假设使用train分割
        total_samples = len(test_data)
        
        # 获取需要处理的剩余问题
        remaining_questions = self.get_remaining_questions(total_samples)
        
        if not remaining_questions:
            logger.info("所有问题都已处理完成，直接合并最终文件")
            return self.combine_all_responses(final_output_file)
        
        logger.info(f"总问题数: {total_samples}, 已完成: {total_samples - len(remaining_questions)}, 剩余: {len(remaining_questions)}")
        
        # 处理剩余问题
        tasks = []
        for question_idx in remaining_questions:
            question_data = test_data[question_idx]
            tasks.append({
                "question_data": question_data,
                "question_idx": question_idx
            })
        
        logger.info(f"开始处理 {len(tasks)} 个问题，并行数: {self.parallel_num}")
        logger.info(f"使用的temperature列表: {self.temperature_list}")
        
        # 使用ThreadPoolExecutor进行并行处理
        with ThreadPoolExecutor(max_workers=self.parallel_num) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self.process_single_question, 
                    task["question_data"], 
                    task["question_idx"]
                ): task for task in tasks
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(tasks), desc="处理进度") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        future.result()  # 结果已经保存到文件，这里不需要处理
                    except Exception as e:
                        logger.error(f"问题 {task['question_idx']} 处理失败: {str(e)}")
                    pbar.update(1)
        
        # 合并所有响应到最终文件
        return self.combine_all_responses(final_output_file)

def main():
    """主函数"""
    # 从环境变量获取API密钥
    deepseek_api_key = os.environ.get('yangzhch6_DEEPSEEK_API_TOKEN')
    if not deepseek_api_key:
        raise ValueError("请设置环境变量 yangzhch6_DEEPSEEK_API_TOKEN")
    
    responses_per_question = 8  # 每个问题的响应数

    # 配置参数
    config = {
        "model_name": "deepseek-reasoner",  # 或 "deepseek-reasoner"
        "max_tokens": 62464,
        "responses_per_question": responses_per_question,
        "parallel_num": 512,  # 并行数，根据API限制调整
        "temperature_list": [0.6] * responses_per_question,  # 可以自定义temperature列表
        "output_dir": "/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/eval-deepseek-reasoner-64k-DeepInformal-test/responses",  # 响应文件目录
        "final_output_file": "/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/eval-deepseek-reasoner-64k-DeepInformal-test/deepseek_responses.json"  # 最终输出文件
    }
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 创建处理器
    processor = DatasetProcessor(
        api_key=deepseek_api_key,
        model_name=config["model_name"],
        responses_per_question=config["responses_per_question"],
        parallel_num=config["parallel_num"],
        temperature_list=config["temperature_list"],
        output_dir=config["output_dir"],
        max_tokens=config["max_tokens"]
    )
    
    # 处理数据集
    results = processor.process_dataset(final_output_file=config["final_output_file"])
    
    # 统计信息
    success_count = sum(1 for r in results if r.get("api_success", False))
    logger.info(f"处理完成！成功: {success_count}/{len(results)}")

def analyze_results(output_file: str = "deepseek_responses.json"):
    """分析结果文件"""
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 基本统计
    total_questions = len(set(r["question_idx"] for r in results))
    total_responses = len(results)
    success_responses = sum(1 for r in results if r.get("api_success", False))
    
    print(f"总问题数: {total_questions}")
    print(f"总响应数: {total_responses}")
    print(f"成功响应数: {success_responses}")
    print(f"成功率: {success_responses/total_responses*100:.2f}%")
    
    # Token使用统计
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    print(f"总Token使用: {total_tokens}")
    
    # Temperature使用统计
    used_temperatures = list(set(r.get("temperature", 0) for r in results))
    print(f"使用的temperature值: {sorted(used_temperatures)}")
    
    # Reasoning content统计
    reasoning_count = sum(1 for r in results if "reasoning_content" in r)
    print(f"包含reasoning content的响应数: {reasoning_count}")

if __name__ == "__main__":
    main()
    
    # 分析结果
    analyze_results("/mnt/weka/home/yongxin.wang/workspace/lark/DeepInformal/gen_deepseek/eval-deepseek-reasoner-64-DeepInformal-test/deepseek_responses.json")