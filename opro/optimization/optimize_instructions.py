# Copyright 2023 The OPRO Authors
# ...保留原有版权信息...

r"""DeepSeek本地优化的主程序

使用方式:
python optimize_instructions.py \
    --optimizer="deepseek" --scorer="deepseek" \
    --instruction_pos="A_begin" --dataset="gsm8k" --task="train"
"""

import datetime
import functools
import os
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import numpy as np
from opro import prompt_utils
from opro.optimization import opt_utils
import pandas as pd

ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

# ====================== 修改1: 删除不必要的API密钥参数 ======================
_SCORER = flags.DEFINE_string(
    "scorer", "deepseek", "评分模型名称"
)

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "deepseek", "优化模型名称"
)

_DATASET = flags.DEFINE_string(
    "dataset", "gsm8k", "数据集名称"
)

_TASK = flags.DEFINE_string(
    "task", "train", "具体任务名称"
)

_INSTRUCTION_POS = flags.DEFINE_string(
    "instruction_pos",
    "A_begin",
    "指令位置"
)

_META_PROMPT_TYPE = flags.DEFINE_string(
    "meta_prompt_type",
    "both_instructions_and_exemplars",
    "元提示类型"
)

def main(_):
    # ====================== 修改2: 移除API密钥相关代码 ======================
  scorer_llm_name = _SCORER.value.lower()
  optimizer_llm_name = _OPTIMIZER.value.lower()
  dataset_name = _DATASET.value.lower()
  task_name = _TASK.value
  meta_prompt_type = _META_PROMPT_TYPE.value
  # 在main函数开头添加 ↓↓↓
  instruction_pos = _INSTRUCTION_POS.value
# 模型名称校验
  assert scorer_llm_name == "deepseek", "只支持DeepSeek评分模型"
  assert optimizer_llm_name == "deepseek", "只支持DeepSeek优化模型"

    # ====================== 修改3: 简化模型配置 ======================
    # 评分模型配置
  scorer_llm_dict = {
        "model_type": "deepseek",
        "temperature": 0.0,
        "max_decode_steps": 1024,
        "batch_size": 1,
        "num_servers": 1
    }
  call_scorer_server_func = functools.partial(
        prompt_utils.call_deepseek_local,
        model="deepseek-r1:latest",
        temperature=scorer_llm_dict["temperature"],
        max_decode_steps=scorer_llm_dict["max_decode_steps"]
    )

    # 优化模型配置
  optimizer_llm_dict = {
        "model_type": "deepseek",
        "temperature": 1.0,
        "max_decode_steps": 512,
        "batch_size": 1,
        "num_servers": 1
    }
  call_optimizer_server_func = functools.partial(
        prompt_utils.call_deepseek_local,
        model="deepseek-r1:latest",
        temperature=optimizer_llm_dict["temperature"],
        max_decode_steps=optimizer_llm_dict["max_decode_steps"]
    )

  root_data_folder_path = os.path.join(OPRO_ROOT_PATH, "data")
  if dataset_name == "gsm8k":
        root_data_folder_path = os.path.join(root_data_folder_path, "gsm_data")
    
    # ====================== 修复目录创建逻辑 ======================
  datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  save_folder = os.path.join(
        OPRO_ROOT_PATH, "outputs", "optimization-results",
        f"{dataset_name}-{task_name}-deepseek-{datetime_str}"
    )
  os.makedirs(save_folder, exist_ok=True)
    
  result_by_instruction_folder = os.path.join(save_folder, "result_by_instruction")
  os.makedirs(result_by_instruction_folder, exist_ok=True)

    # ====================== 数据加载优化 ======================
  if dataset_name == "gsm8k":
        data_path = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
        raw_data = pd.read_csv(data_path, sep="\t", header=None, names=["question", "answer"])
        print(f"Loaded {len(raw_data)} examples from {data_path}")

    # ====================== 模型调用容错处理 ======================
    # 修改测试调用部分
  # ====================== 修改4: 更新服务器测试 ======================
  print("\n======== 测试本地DeepSeek服务 ===========")
  test_prompt = "太阳从北方升起吗？只需回答是或否。"
  
  scorer_test_output = call_scorer_server_func(test_prompt)
  print(f"评分模型测试输出: {scorer_test_output}")
  
  optimizer_test_output = call_optimizer_server_func(test_prompt)
  print(f"优化模型测试输出: {optimizer_test_output}")
  print("服务测试完成")

  # ====================== 后续代码保持不变 ======================
  # ...（保留原有数据加载、任务配置、优化流程等代码）...
  print("\n================ prompt optimization settings ==============")
  # from https://github.com/hendrycks/test/blob/master/categories.py
  subcategories = {
      "abstract_algebra": ["math"],
      "anatomy": ["health"],
      "astronomy": ["physics"],
      "business_ethics": ["business"],
      "clinical_knowledge": ["health"],
      "college_biology": ["biology"],
      "college_chemistry": ["chemistry"],
      "college_computer_science": ["computer science"],
      "college_mathematics": ["math"],
      "college_medicine": ["health"],
      "college_physics": ["physics"],
      "computer_security": ["computer science"],
      "conceptual_physics": ["physics"],
      "econometrics": ["economics"],
      "electrical_engineering": ["engineering"],
      "elementary_mathematics": ["math"],
      "formal_logic": ["philosophy"],
      "global_facts": ["other"],
      "high_school_biology": ["biology"],
      "high_school_chemistry": ["chemistry"],
      "high_school_computer_science": ["computer science"],
      "high_school_european_history": ["history"],
      "high_school_geography": ["geography"],
      "high_school_government_and_politics": ["politics"],
      "high_school_macroeconomics": ["economics"],
      "high_school_mathematics": ["math"],
      "high_school_microeconomics": ["economics"],
      "high_school_physics": ["physics"],
      "high_school_psychology": ["psychology"],
      "high_school_statistics": ["math"],
      "high_school_us_history": ["history"],
      "high_school_world_history": ["history"],
      "human_aging": ["health"],
      "human_sexuality": ["culture"],
      "international_law": ["law"],
      "jurisprudence": ["law"],
      "logical_fallacies": ["philosophy"],
      "machine_learning": ["computer science"],
      "management": ["business"],
      "marketing": ["business"],
      "medical_genetics": ["health"],
      "miscellaneous": ["other"],
      "moral_disputes": ["philosophy"],
      "moral_scenarios": ["philosophy"],
      "nutrition": ["health"],
      "philosophy": ["philosophy"],
      "prehistory": ["history"],
      "professional_accounting": ["other"],
      "professional_law": ["law"],
      "professional_medicine": ["health"],
      "professional_psychology": ["psychology"],
      "public_relations": ["politics"],
      "security_studies": ["politics"],
      "sociology": ["culture"],
      "us_foreign_policy": ["politics"],
      "virology": ["health"],
      "world_religions": ["philosophy"],
  }

  categories = {
      "STEM": [
          "physics",
          "chemistry",
          "biology",
          "computer science",
          "math",
          "engineering",
      ],
      "humanities": ["history", "philosophy", "law"],
      "social sciences": [
          "politics",
          "culture",
          "economics",
          "geography",
          "psychology",
      ],
      "other (business, health, misc.)": ["other", "business", "health"],
  }

  if dataset_name == "mmlu":
    # EITHER: filter by category
    # category_names = [
    #     "STEM",
    #     "humanities",
    #     "social sciences",
    #     "other (business, health, misc.)",
    # ]
    category_names = [task_name]
    folder_name = "test"  # one of {'auxiliary_train', 'dev', 'val', 'test'}
    task_names = []
    for task_csv_name in os.listdir(
        os.path.join(root_data_folder_path, folder_name)
    ):
      task_names.append(task_csv_name.split(".")[0])

    tasks_in_category = []
    for category_name in category_names:
      for task_name in task_names:
        for subname in subcategories:
          if subname in task_name:
            if subcategories[subname][0] in categories[category_name]:
              tasks_in_category.append(task_name)
              break

    tasks_all = [(folder_name, task_name) for task_name in tasks_in_category]
    multiple_choice_tasks = set([item[1] for item in tasks_all])
    boolean_tasks = set()
    numerical_output_tasks = set()

    # OR: filter by task
    # tasks_all = [
    #     # ('test', 'abstract_algebra_test'),
    #     # ('test', 'college_computer_science_test'),
    #     # ('test', 'college_mathematics_test'),
    #     # ('test', 'college_physics_test'),
    #     # ('test', 'elementary_mathematics_test'),
    #     # ('test', 'global_facts_test'),
    #     # ('test', 'high_school_physics_test'),
    #     # ('test', 'machine_learning_test'),
    #     # ('test', 'management_test'),
    #     # ('test', 'medical_genetics_test'),
    #     # ('test', 'moral_scenarios_test'),
    #     # ('test', 'professional_psychology_test'),
    #     # ('test', 'public_relations_test'),
    #     # ('test', 'professional_law_test'),
    #     # ('test', 'high_school_psychology_test'),
    #     # ('test', 'high_school_world_history_test'),
    #     # ('test', 'human_aging_test'),
    #     # ('test', 'miscellaneous_test'),
    #     # ('test', 'moral_scenarios_test'),
    #     ('test', 'professional_psychology_test'),
    #     # ('test', 'security_studies_test'),
    # ]

  elif dataset_name == "bbh":
    tasks_all = [task_name]
    assert (
        len(tasks_all) == 1
    ), "for now only support prompt optimization on one BBH task"

    # all BBH tasks are as below
    # tasks_all = [
    #     'boolean_expressions',
    #     'causal_judgement',
    #     'date_understanding',
    #     'disambiguation_qa',
    #     'dyck_languages',
    #     'formal_fallacies',
    #     'geometric_shapes',
    #     'hyperbaton',
    #     'logical_deduction_five_objects',
    #     'logical_deduction_seven_objects',
    #     'logical_deduction_three_objects',
    #     'movie_recommendation',
    #     'multistep_arithmetic_two',
    #     'navigate',
    #     'object_counting',
    #     'penguins_in_a_table',
    #     'reasoning_about_colored_objects',
    #     'ruin_names',
    #     'salient_translation_error_detection',
    #     'snarks',
    #     'sports_understanding',
    #     'temporal_sequences',
    #     'tracking_shuffled_objects_five_objects',
    #     'tracking_shuffled_objects_seven_objects',
    #     'tracking_shuffled_objects_three_objects',
    #     'web_of_lies',
    #     'word_sorting'
    # ]
    numerical_output_tasks = {
        "object_counting",
        "multistep_arithmetic_two",
    }

    multiple_choice_tasks = {
        "date_understanding",
        "disambiguation_qa",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
    }

    boolean_tasks = {
        "boolean_expressions",  # True or False
        "causal_judgement",  # yes or no
        "formal_fallacies",  # valid or invalid
        "navigate",  # yes or no
        "sports_understanding",  # yes or no
        "web_of_lies",  # yes or no
    }

  else:
    assert dataset_name in {"gsm8k"}
    tasks_all = [task_name]
    multiple_choice_tasks = set()
    boolean_tasks = set()
    numerical_output_tasks = set(tasks_all)

  if dataset_name == "mmlu":
    raw_data = pd.DataFrame()
    prediction_treat_as_number = False
    prediction_treat_as_bool = False
  elif dataset_name == "bbh":
    raw_data = []
    prediction_treat_as_number = bool(
        tasks_all[0] in numerical_output_tasks
    )  # for now only check the first task
    prediction_treat_as_bool = bool(
        tasks_all[0] in boolean_tasks
    )  # for now only check the first task
    print(
        f"prediction_treat_as_number: {prediction_treat_as_number},"
        f" prediction_treat_as_bool: {prediction_treat_as_bool}"
    )
  else:
    assert dataset_name == "gsm8k"
    raw_data = pd.DataFrame()
    prediction_treat_as_number = True
    prediction_treat_as_bool = False

  for t in tasks_all:
    if dataset_name == "mmlu":
      folder_name = t[0]
      task_name = t[1]
      single_task_df = pd.read_csv(
          os.path.join(root_data_folder_path, f"{folder_name}/{task_name}.csv"),
          index_col=None,
          header=None,
      )
      raw_data = pd.concat([raw_data, single_task_df])
    elif dataset_name == "bbh":
      task_name = t
      single_task_list = opt_utils.load_bbh_task_data(
          task_name, base_dir=root_data_folder_path
      )
      raw_data += single_task_list
    else:
      assert dataset_name == "gsm8k"
      task_name = t
      f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
      single_task_df = pd.read_csv(f_gsm, sep="\t", header=None)
      raw_data = pd.concat([raw_data, single_task_df])

  if dataset_name == "mmlu":
    num_examples = raw_data.shape[0]
  elif dataset_name == "bbh":
    num_examples = len(raw_data)
  else:
    assert dataset_name in {"gsm8k"}
    num_examples = raw_data.shape[0]
  print(f"number of examples in the current task: {num_examples}")

  # ================ split data into train/val/test ==========================
  if dataset_name == "mmlu":
    train_ratio = 0.8
    eval_ratio = 0.2
  elif dataset_name == "gsm8k":
    train_ratio = 0.035
    eval_ratio = 0
  else:
    assert dataset_name == "bbh"
    train_ratio = 0.2
    eval_ratio = 0

  # train-validation-test split
  # It is important to sort the indices, as this ensures the is_multiple_choice
  # Boolean variables match the data points.
  assert train_ratio + eval_ratio <= 1
  test_ratio = 1 - train_ratio - eval_ratio
  print(
      f"train_ratio: {train_ratio}, eval_ratio: {eval_ratio}, "
      f"test_ratio: {test_ratio}"
  )
  np.random.seed(0)
  train_index = np.sort(
      np.array(
          np.random.choice(
              num_examples, size=int(train_ratio * num_examples), replace=False
          )
      )
  )
  eval_and_test_index = np.sort(
      np.array(list(set(np.arange(num_examples)) - set(train_index)))
  )
  eval_index = np.sort(
      np.array(
          np.random.choice(
              eval_and_test_index,
              size=int(eval_ratio * num_examples),
              replace=False,
          )
      )
  )

  # ========== set other optimization experiment hyperparameters ==============
# ====================== DeepSeek本地优化参数配置 ======================
# 指令评分阈值
  old_instruction_score_threshold = 0.2

# 答案提取设置
  extract_final_answer_by_prompting_again = True  # 需要二次提示获取规范化答案
  include_qa = True  # 在prompt中保留完整问答对
  evaluate_in_parallel = True  # 并行评估提升效率

# 优化器参数
  optimizer_llm_temperature = optimizer_llm_dict["temperature"]

# 示例数量配置
  num_few_shot_questions_for_instruction_refinement = 3  
  # To change the number of generated instructions in each step, one should
  # edit the value of the variable below, instead of editing the number of
  # decodes in model parameters, because those values are limited by model
  # serving configs.
  num_generated_instructions_in_each_step = 8
  num_search_steps = 200

  initial_instructions = [
      "Let's solve the problem.",
      # "",
      # "The answer is",
  ]
  few_shot_qa_pairs = True
  # one of {'accumulative_most_frequent', 'current_most_frequent', 'random',
  # 'constant'}
  few_shot_selection_criteria = "random"
  # whether to evaluate generated instructions on the exemplars in meta-prompt
  evaluate_generated_ins_on_few_shot = False
  # whether to evaluate old instructions on the exemplars in the meta-prompt
  evaluate_old_ins_on_few_shot = False
  # every this number of steps, compute the accuracies of current-step
  # instructions on the validation set
  eval_interval = 3

  max_num_instructions = (
      20  # the maximum number of instructions and scores in the meta-prompt
  )
  # The number of buckets when converting scores to integers in the meta-prompt.
  num_score_buckets = 100
  # whether to put old instructions and scores to before exemplars in
  # the meta-prompt
  meta_prompt_instructions_before_exemplars = True

  # ===================== run prompt optimization ======================

  assert few_shot_selection_criteria in {
      "accumulative_most_frequent",
      "current_most_frequent",
      "random",
      "constant",
  }
  evolution_kwargs = {
      "num_search_steps": num_search_steps,
      "old_instruction_score_threshold": old_instruction_score_threshold,
      "scorer_llm_dict": scorer_llm_dict,
      "optimizer_llm_dict": optimizer_llm_dict,
      "extract_final_answer_by_prompting_again": (
          extract_final_answer_by_prompting_again
      ),
      "include_qa": include_qa,
      "evaluate_in_parallel": evaluate_in_parallel,
      "tasks_all": tasks_all,
      "train_ratio": train_ratio,
      "eval_ratio": eval_ratio,
      "test_ratio": test_ratio,
      "train_index": train_index,
      "eval_index": eval_index,
      "dataset_name": dataset_name,
      "task_name": task_name,
      "num_examples": num_examples,
      "root_data_folder_path": root_data_folder_path,
      "optimizer_llm_temperature": optimizer_llm_temperature,
      # "optimizer_llm_temperature_schedule": (
      #     optimizer_llm_temperature_schedule
      # ),
      # "optimizer_llm_temperature_end": optimizer_llm_temperature_end,
      "initial_instructions": initial_instructions,
      "multiple_choice_tasks": multiple_choice_tasks,
      "raw_data": raw_data,
      "call_scorer_server_func": call_scorer_server_func,
      "call_optimizer_server_func": call_optimizer_server_func,
      "instruction_pos": instruction_pos,
      "prediction_treat_as_number": prediction_treat_as_number,
      "prediction_treat_as_bool": prediction_treat_as_bool,
      "result_by_instruction_folder": result_by_instruction_folder,
      "few_shot_qa_pairs": few_shot_qa_pairs,
      "num_score_buckets": num_score_buckets,
      "max_num_instructions": max_num_instructions,
      "meta_prompt_type": meta_prompt_type,
      "meta_prompt_instructions_before_exemplars": (
          meta_prompt_instructions_before_exemplars
      ),
      "few_shot_selection_criteria": few_shot_selection_criteria,
      "optimizer_llm_name": optimizer_llm_name,
      "num_generated_instructions_in_each_step": (
          num_generated_instructions_in_each_step
      ),
      "evaluate_generated_ins_on_few_shot": evaluate_generated_ins_on_few_shot,
      "num_few_shot_questions_for_instruction_refinement": (
          num_few_shot_questions_for_instruction_refinement
      ),
      "evaluate_old_ins_on_few_shot": evaluate_old_ins_on_few_shot,
      "eval_interval": eval_interval,
      "save_folder": save_folder,
  }

  opt_utils.run_evolution(**evolution_kwargs)




if __name__ == "__main__":
  app.run(main)