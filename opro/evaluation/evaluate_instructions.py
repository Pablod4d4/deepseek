# Copyright 2023 The OPRO Authors
# 保留核心版权声明

r"""DeepSeek本地评估主程序 (优化版)

使用方式:
python evaluate_instructions.py \
    --dataset=gsm8k \
    --instruction_pos=Q_begin
"""

import datetime
import functools
import json
import os
import sys

# ====================== 修复1：路径语法修正 ======================
OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import numpy as np
from opro import prompt_utils  # 修改后的prompt_utils
from opro.evaluation import eval_utils

# ====================== 配置优化 ======================
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset", "gsm8k",
    "支持的数据集: gsm8k")
flags.DEFINE_string(
    "instruction_pos", "Q_begin",
    "指令位置: Q_begin（问题前）或 Q_end（问题后）")
flags.DEFINE_integer(
    "max_samples", 10,
    "最大评估样本数（调试用）")

# ====================== 数据集路径映射 ======================
DATASET_PATHS = {
    "gsm8k": {
        "train": "train.jsonl",
        "test": "test.jsonl",
        "base_path": os.path.join(OPRO_ROOT_PATH, "data/gsm_data")
    }
}

def validate_environment():
    """环境校验"""
    try:
        import requests  # 本地调用必需
    except ImportError:
        raise RuntimeError("需安装requests库：pip install requests")

def load_dataset(dataset_name, split):
    """加载本地数据集"""
    base_path = DATASET_PATHS[dataset_name]["base_path"]
    file_path = os.path.join(base_path, DATASET_PATHS[dataset_name][split])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    samples = []
    with open(file_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= FLAGS.max_samples:
                break  # 限制样本数
    return samples

def build_prompt(question, instruction, position):
    """构建带指令的提示"""
    if position == "Q_begin":
        return f"{instruction}\n问题：{question}"
    elif position == "Q_end":
        return f"问题：{question}\n{instruction}"
    else:
        raise ValueError("无效的指令位置")

def main(_):
    validate_environment()
    
    # ====================== 模型初始化 ======================
    scorer = functools.partial(
        prompt_utils.call_ollama_deepseek,
        model="deepseek-r1:latest",
        temperature=0.3,  # 平衡创造性与稳定性
        max_decode_steps=512
    )
    
    # ====================== 指令配置 ======================
    candidate_instructions = [
        "请逐步分析并给出最终答案",
        "仔细思考后分步骤解答",
        "让我们详细拆解这个问题"
    ]
    
    # ====================== 数据加载 ======================
    dataset = FLAGS.dataset.lower()
    test_samples = load_dataset(dataset, "test")
    
    # ====================== 评估流程 ======================
    results = []
    for idx, sample in enumerate(test_samples):
        question = sample["question"]
        reference = sample["answer"].split("####")[-1].strip()
        
        for instr in candidate_instructions:
            prompt = build_prompt(question, instr, FLAGS.instruction_pos)
            
            # ====================== 模型调用 ======================
            try:
                response = scorer(prompt)[0]
                prediction = eval_utils.extract_last_number(response)
                is_correct = (str(prediction) == str(reference))
            except Exception as e:
                print(f"样本{idx}错误：{str(e)}")
                is_correct = False
            
            results.append({
                "instruction": instr,
                "question": question,
                "prediction": prediction,
                "reference": reference,
                "correct": is_correct
            })
    
    # ====================== 结果分析 ======================
    accuracy_report = {}
    for instr in candidate_instructions:
        correct = sum(r["correct"] for r in results if r["instruction"] == instr)
        total = len(test_samples)
        accuracy = correct / total if total > 0 else 0
        accuracy_report[instr] = f"{accuracy:.2%} ({correct}/{total})"
    
    # ====================== 输出保存 ======================
    output_dir = os.path.join(OPRO_ROOT_PATH, "outputs", 
        f"{dataset}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n===== 评估报告 =====")
    print(json.dumps(accuracy_report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app.run(main)