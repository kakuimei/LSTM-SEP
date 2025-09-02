from exp.exp_model import Exp_Model
import argparse
import torch
import numpy as np
import random
import os
from dotenv import load_dotenv

load_dotenv()
fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# general
parser.add_argument("--agent_name", type=str, default="brain_agent")
parser.add_argument("--log_dir", type=str, default="test_data/brain_output_log")
parser.add_argument("--log_filename", type=str, default="{symbol}_run.log")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/latest",help="Path to load/save BrainDB checkpoints")

# embedding
parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002")
parser.add_argument("--chunk_size", type=int, default=5000)
parser.add_argument("--verbose", action="store_true")   # bool 用 flag 更直观
parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))

# short memory
parser.add_argument("--short_jump_threshold_upper", type=float, default=70.0)
parser.add_argument("--short_importance_score_initialization", type=str, default="sample")
parser.add_argument("--short_recency_factor", type=float, default=10.0)
parser.add_argument("--short_importance_factor", type=float, default=0.98)
parser.add_argument("--short_recency_threshold", type=float, default=0.01)
parser.add_argument("--short_importance_threshold", type=float, default=10.0)

# mid memory
parser.add_argument("--mid_jump_threshold_upper", type=float, default=85.0)
parser.add_argument("--mid_jump_threshold_lower", type=float, default=50.0)
parser.add_argument("--mid_importance_score_initialization", type=str, default="sample")
parser.add_argument("--mid_recency_factor", type=float, default=12.0)
parser.add_argument("--mid_importance_factor", type=float, default=0.97)
parser.add_argument("--mid_recency_threshold", type=float, default=0.02)
parser.add_argument("--mid_importance_threshold", type=float, default=20.0)

# long memory
parser.add_argument("--long_jump_threshold_lower", type=float, default=70.0)
parser.add_argument("--long_importance_score_initialization", type=str, default="sample")
parser.add_argument("--long_recency_factor", type=float, default=15.0)
parser.add_argument("--long_importance_factor", type=float, default=0.95)
parser.add_argument("--long_recency_threshold", type=float, default=0.03)
parser.add_argument("--long_importance_threshold", type=float, default=30.0)

# reflection
parser.add_argument("--reflection_importance_score_initialization", type=str, default="sample")
parser.add_argument("--reflection_recency_factor", type=float, default=8.0)
parser.add_argument("--reflection_importance_factor", type=float, default=0.96)
parser.add_argument("--reflection_recency_threshold", type=float, default=0.01)
parser.add_argument("--reflection_importance_threshold", type=float, default=15.0)

# load data
parser.add_argument("--price_dir", type=str, default="test_data/price/")
parser.add_argument("--tweet_dir", type=str, default="test_data/tweet/")
parser.add_argument("--seq_len", type=int, default=5)

# ========== 通用 ==========
parser.add_argument("--base_model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--load_in_4bit", action="store_true", default=True)
parser.add_argument("--bf16", action="store_true", default=True)
parser.add_argument("--use_lora", action="store_true", default=True)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)

# ========== SFT ==========
parser.add_argument("--sft_data_path", type=str, default="./test_data/sft_samples.json")
parser.add_argument("--adapter_output_dir", type=str, default="./saved_models/lora-Vicuna-adapter")
parser.add_argument("--merged_output_dir", type=str, default="./saved_models/lora-Vicuna-adapter-merged")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--lr_sft", type=float, default=3e-4)
parser.add_argument("--batch_size_sft", type=int, default=128)
parser.add_argument("--micro_batch_size", type=int, default=4, help="单卡 micro batch size，用于梯度累积")
parser.add_argument("--cutoff_len", type=int, default=256, help="最大序列长度")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--resume_from", type=str, default=None, help="继续训练的checkpoint路径")
parser.add_argument("--ignore_data_skip", action="store_true", default=False)
parser.add_argument("--use_wandb", action="store_true")

# ===== Reward Model 阶段 =====
parser.add_argument("--rm_data_path", type=str, default="./test_data/grpo_samples.json")
parser.add_argument("--rm_model_name", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--rm_output_dir", type=str, default="./reward_model")
parser.add_argument("--epochs_rm", type=int, default=1)
parser.add_argument("--lr_rm", type=float, default=5e-6)
parser.add_argument("--batch_size_rm", type=int, default=4)
parser.add_argument("--grad_accum_rm", type=int, default=2)
parser.add_argument("--max_length_rm", type=int, default=1024)

# ========== GRPO ==========
parser.add_argument("--grpo_data_path", type=str, default="./test_data/grpo_samples.json")
parser.add_argument("--grpo_output_dir", type=str, default="./grpo_model")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lr_rl", type=float, default=5e-6, help="GRPO阶段学习率")
parser.add_argument("--batch_size_rl", type=int, default=1, help="GRPO阶段batch size")
parser.add_argument("--grad_accum", type=int, default=16, help="梯度累积步数")
parser.add_argument("--seed", type=int, default=42)

# 生成参数 (GRPO 必需采样)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--min_length", type=int, default=16)

# evaluation
parser.add_argument("--num_shots", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="results/")

args = parser.parse_args()
print('Args in experiment:')
print(args)

exp_model = Exp_Model(args)
exp_model.train()
exp_model.test()
