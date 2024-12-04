# Copyright 2023 The OPRO Authors
# ...保留原有版权信息...

"""DeepSeek本地优化的核心工具函数"""

import collections
import json
import os
import pickle
import re
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import numpy as np
from opro.evaluation import eval_utils
import pandas as pd

# ====================== 修改1: 移除不必要的依赖 ======================

def extract_string_in_square_brackets(input_string):
    """从文本中提取方括号内容（适配DeepSeek输出格式）"""
    raw_result = re.findall(r"\[.*?\]", input_string)
    return raw_result[0][1:-1] if raw_result else ""

def parse_tag_content(text, prefix="<TEXT>", suffix="</TEXT>"):
    """解析自定义标签内容（保持原有逻辑）"""
    pattern = f"{prefix}(.*?){suffix}"
    return re.findall(pattern, text, re.DOTALL)

# ====================== 修改2: 简化评分分桶逻辑 ======================
def _bucketize_float(num, n_buckets=20):
    """将0-1的浮点数转换为分桶整数"""
    return round(min(max(num, 0), 1) * n_buckets)

def gen_ins_and_score_pairs_substr(
    old_instructions_and_scores,
    old_instruction_score_threshold=0.1,
    max_num_instructions=1000,
    return_str_only=False,
    num_score_buckets=20,  # 默认分桶数为20
):
    """生成指令-评分对字符串（适配DeepSeek）"""
    filtered_instructions = sorted(
        [ins for ins in old_instructions_and_scores if ins[1] >= old_instruction_score_threshold],
        key=lambda x: x[1]
    )[-max_num_instructions:]

    instructions_str = ""
    for ins, score, _ in filtered_instructions:
        score_display = _bucketize_float(score, num_score_buckets)
        instructions_str += f"\ntext:\n{ins}\nscore:\n{score_display}\n"
    
    return instructions_str if return_str_only else (instructions_str, filtered_instructions)

# ====================== 修改3: 重构元提示生成 ======================
def gen_meta_prompt(
    old_instructions_and_scores,
    instruction_pos,
    dataset_name="gsm8k",
    task_name="math",
    num_score_buckets=20,
    max_num_instructions=20,
    few_shot_index_list=None
):
    """生成DeepSeek专用的元提示"""
    assert instruction_pos in {"Q_begin", "Q_end", "A_begin"}, "指令位置需为Q_begin/Q_end/A_begin"
    
    # 生成指令-评分对
    instructions_str, _ = gen_ins_and_score_pairs_substr(
        old_instructions_and_scores=old_instructions_and_scores,
        num_score_buckets=num_score_buckets,
        max_num_instructions=max_num_instructions
    )
    
    # 构建基础提示
    meta_prompt = (
        "你是一个优化器，任务是生成改进的指令文本。以下是历史指令及其效果评分（分数越高越好）：\n"
        f"{instructions_str}\n\n"
        "请生成新的指令文本，要求：\n"
        "1. 用方括号[]包裹\n"
        "2. 与现有指令不同\n"
        "3. 适用于以下示例问题：\n"
    )
    
    # 添加示例问题
    if few_shot_index_list and dataset_name == "gsm8k":
        for idx in few_shot_index_list:
            question = old_instructions_and_scores[0][0]  # 假设第一个指令包含示例
            meta_prompt += f"\n问题：{question}\n"
    
    meta_prompt += "\n请直接输出优化后的指令："
    return meta_prompt

# ====================== 修改4: 优化进化算法主流程 ======================
def run_evolution(**kwargs):
    """DeepSeek专用的进化算法流程"""
    # 参数提取与校验
    num_search_steps = kwargs["num_search_steps"]
    call_scorer = kwargs["call_scorer_server_func"]
    call_optimizer = kwargs["call_optimizer_server_func"]
    dataset_name = kwargs.get("dataset_name", "gsm8k")
    
    # 初始化数据结构
    evolution_data = {
        "instructions": [],
        "scores": [],
        "meta_prompts": [],
        "best_score": 0.0
    }
    
    # 进化循环
    for step in range(num_search_steps):
        print(f"\n=== 进化步骤 {step+1}/{num_search_steps} ===")
        
        # 生成元提示
        meta_prompt = gen_meta_prompt(
            old_instructions_and_scores=zip(
                evolution_data["instructions"], 
                evolution_data["scores"],
                [step]*len(evolution_data["scores"])
            ),
            instruction_pos=kwargs["instruction_pos"],
            dataset_name=dataset_name
        )
        evolution_data["meta_prompts"].append(meta_prompt)
        
        # 调用DeepSeek生成新指令
        new_instructions = call_optimizer(meta_prompt)
        print(f"生成新指令: {new_instructions}")
        
        # 评估指令
        valid_instructions = []
        for ins in new_instructions:
            if 0 < len(ins) <= 500 and "答案" not in ins:
                score = np.mean(call_scorer(ins))
                if score >= kwargs["old_instruction_score_threshold"]:
                    valid_instructions.append((ins, score))
        
        # 更新进化数据
        for ins, score in valid_instructions:
            evolution_data["instructions"].append(ins)
            evolution_data["scores"].append(score)
            if score > evolution_data["best_score"]:
                evolution_data["best_score"] = score
                
        print(f"当前最佳得分: {evolution_data['best_score']:.2f}")
    
    # 保存最终结果
    with open(os.path.join(kwargs["save_folder"], "evolution_data.pkl"), "wb") as f:
        pickle.dump(evolution_data, f)
    
    return evolution_data

# ====================== 修改5: 移除多模型支持代码 ======================
# 删除所有与GPT/PaLM相关的条件判断和特殊处理逻辑
# 删除API密钥相关参数和校验代码
# 删除与多任务数据集相关的复杂处理逻辑
# 简化错误处理机制，仅保留基本重试逻辑