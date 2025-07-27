from data_load.dataloader import DataLoader
from explain_module.agents import PredictReflectAgent
from tqdm import tqdm
import argparse
import torch
import numpy as np
import random
from memory_module.memorydb import BrainDB
import os

fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# load data
parser.add_argument("--price_dir", type=str, default="data/price/preprocessed/")
parser.add_argument("--tweet_dir", type=str, default="data/tweet/raw/")
parser.add_argument("--seq_len", type=int, default=5)

# supervised finetuning
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="./data/merge_sample.json")
parser.add_argument("--output_path", type=str, default="./saved_models/lora-Vicuna")
parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--resume_from_supervised_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")

# training reward model
parser.add_argument("--num_reflect_trials", type=int, default=2)
parser.add_argument("--datasets_dir", type=str, default="./datasets/")
parser.add_argument('--local_rank', type=int, default=0, help="Used for multi-gpu")
parser.add_argument('--resume_from_reward_checkpoint', type=bool, default=False, help="If you want to resume training where it left off.")
parser.add_argument('--deepspeed', type=str, default=None, help="Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU.")
parser.add_argument('--per_device_train_batch_size', type=int, default=1)
parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
parser.add_argument('--reward_gradient_accumulation_steps', type=int, default=32)
parser.add_argument('--reward_learning_rate', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--reward_base_model', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.")
parser.add_argument('--bf16', type=bool, default=False, help="This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU.")
parser.add_argument('--num_train_epochs', type=int, default=1, help="The number of training epochs for the reward model.")
parser.add_argument('--train_subset', type=int, default=100000, help="The size of the subset of the training data to use")
parser.add_argument('--eval_subset', type=int, default=50000, help="The size of the subset of the eval data to use")
parser.add_argument('--gradient_checkpointing', type=bool, default=False, help="Enables gradient checkpointing.")
parser.add_argument('--optim', type=str, default="adamw_hf", help="Enables gradient checkpointing.")
parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="The lr scheduler")
parser.add_argument('--reward_adapter', type=str, default="./saved_models/reward_model_vicuna-7b")

# reinforcement learning
parser.add_argument('--rl_base_model', type=str, default="./saved_models/lora-Vicuna-adapter-merged", help="the model name")
parser.add_argument('--tokenizer_name', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="the tokenizer name")
parser.add_argument('--reward_model_name', type=str, default="./saved_models/reward_model_vicuna-7b-adapter-merged", help="the reward model name")
parser.add_argument('--log_with', type=str, default=None, help="use 'wandb' to log with wandb")
parser.add_argument('--rl_learning_rate', type=float, default=1.4e-5, help="the learning rate")
parser.add_argument('--output_max_length', type=int, default=128, help="maximum length for generation")
parser.add_argument('--mini_batch_size', type=int, default=1, help="the PPO minibatch size")
parser.add_argument('--batch_size', type=int, default=1, help="the batch size")
parser.add_argument('--ppo_epochs', type=int, default=4, help="the number of ppo epochs")
parser.add_argument('--rl_gradient_accumulation_steps', type=int, default=1, help="the number of gradient accumulation steps")
parser.add_argument('--adafactor', type=bool, default=False, help="whether to use the adafactor optimizer")
parser.add_argument('--early_stopping', type=bool, default=True, help="whether to early stop")
parser.add_argument('--target_kl', type=float, default=0.1, help="kl target for early stopping")
parser.add_argument('--reward_baseline', type=float, default=0, help="a baseline value that is subtracted from the reward")
parser.add_argument('--batched_gen', type=bool, default=True, help="whether to use the batched text gen")
parser.add_argument('--save_freq', type=int, default=None, help="n steps to save the model")
parser.add_argument('--output_dir', type=str, default="./saved_models/tuning_llama_rl_checkpoints/", help="directory to save the model")
parser.add_argument('--seed', type=int, default=0, help="the seed")

# evaluation
parser.add_argument("--num_shots", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="results/")

args = parser.parse_args()
print('Args in experiment:')
print(args)
config = {
    "general": {
        "agent_name": "stock_agent_1",
        "trading_symbol": "AAPL"
    },
    "agent": {
        "agent_1": {
            "embedding": {
                "detail": {
                    "embedding_model": "text-embedding-ada-002",  # Avoid using 'model'
                    "chunk_size": 5000,
                    "verbose": False,
                    "openai_api_key": os.environ.get("OPENAI_API_KEY")  # or replace with your key string
                }
            }
        }
    },
    "short": {
        "jump_threshold_upper": 70.0,
        "importance_score_initialization": "sample",
        "decay_params": {
            "recency_factor": 10.0,
            "importance_factor": 0.98
        },
        "clean_up_threshold_dict": {
            "recency_threshold": 0.01,
            "importance_threshold": 10.0
        }
    },
    "mid": {
        "jump_threshold_upper": 85.0,
        "jump_threshold_lower": 50.0,
        "importance_score_initialization": "sample",
        "decay_params": {
            "recency_factor": 12.0,
            "importance_factor": 0.97
        },
        "clean_up_threshold_dict": {
            "recency_threshold": 0.02,
            "importance_threshold": 20.0
        }
    },
    "long": {
        "jump_threshold_lower": 70.0,
        "importance_score_initialization": "sample",
        "decay_params": {
            "recency_factor": 15.0,
            "importance_factor": 0.95
        },
        "clean_up_threshold_dict": {
            "recency_threshold": 0.03,
            "importance_threshold": 30.0
        }
    },
    "reflection": {
        "importance_score_initialization": "sample",
        "decay_params": {
            "recency_factor": 8.0,
            "importance_factor": 0.96
        },
        "clean_up_threshold_dict": {
            "recency_threshold": 0.01,
            "importance_threshold": 15.0
        }
    }
}


# === 测试DeepSeekSummarizer调用 ===
if __name__ == "__main__":

    brain_db = BrainDB.from_config(config)
    agent_cls = PredictReflectAgent
    dataloader = DataLoader(args=args,brain_db=brain_db)
    sample_generator = dataloader.load("train")

    agents = []
    for sample in tqdm(sample_generator, desc="Training agents"):
        agent = agent_cls(ticker=sample['ticker'], date=sample['date'], summary=sample['summary'], target=sample['target'], brain_db=dataloader.brain_db)
        agent.finished = True
        agent.run()
        # agents.append(agent)
        
        if agent.is_correct():
            prompt = agent._build_agent_prompt()
            response = agent.scratchpad.split('Price Movement: ')[-1]

            sample = {
                "instruction": prompt,
                "input": "",
                "output": response
            }
    # summarizer = DeepSeekSummarizer()

    # sample_tweets = [
    #     "NVIDIA just announced record earnings, revenue up 50% YoY!",
    #     "Gaming GPUs see strong demand, but supply chain remains tight.",
    #     "CEO says AI adoption is driving unprecedented growth.",
    # ]
    # summary = summarizer.get_summary("NVDA", sample_tweets)
    # print("\nSummary:")
    # print(summary)

    # if summarizer.is_informative(summary):
    #     print("\n✅ Summary is informative.")
    # else:
    #     print("\n❌ Summary is not informative.")

# import argparse
# import os
# import shutil
# import json
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from summarize_module.summarizer import DeepSeekSummarizer
# from data_load.dataloader import DataLoader

# def create_dummy_data(base_dir, tickers, start_date, num_days):
#     # 创建临时数据目录
#     price_dir = os.path.join(base_dir, "prices")
#     tweet_dir = os.path.join(base_dir, "tweets")
#     os.makedirs(price_dir, exist_ok=True)
#     os.makedirs(tweet_dir, exist_ok=True)

#     for ticker in tickers:
#         dates = [start_date + timedelta(days=i) for i in range(num_days)]
#         price_data = [[d.strftime("%Y-%m-%d"), f"{np.random.uniform(-1,1):.2f}"] for d in dates]
#         price_path = os.path.join(price_dir, f"{ticker}.csv")

#         # 强制确保写入是二维 CSV，无 header，无 index
#         df_price = pd.DataFrame(price_data)
#         df_price.to_csv(price_path, index=False, header=False, sep=' ')

#         ticker_tweet_dir = os.path.join(tweet_dir, ticker)
#         os.makedirs(ticker_tweet_dir, exist_ok=True)
#         for d in dates:
#             day_str = d.strftime("%Y-%m-%d")
#             tweet_path = os.path.join(ticker_tweet_dir, day_str)
#             with open(tweet_path, "w") as f:
#                 json.dump({"text": f"Sample tweet for {ticker} on {day_str}"}, f)
#                 f.write("\n")

#     return price_dir, tweet_dir

# if __name__ == "__main__":
#     # 1. 创建测试数据
#     base_dir = "temp_test_data"
#     if os.path.exists(base_dir):
#         shutil.rmtree(base_dir)
#     tickers = ["TEST1", "TEST2"]
#     start_date = datetime(2025, 1, 1)
#     num_days = 12
#     price_dir, tweet_dir = create_dummy_data(base_dir, tickers, start_date, num_days)

#     # 2. 构造参数对象
#     class Args:
#         def __init__(self):
#             self.price_dir = price_dir
#             self.tweet_dir = tweet_dir
#             self.seq_len = 3
#     print("=== Price File Preview ===")
#     with open(os.path.join(price_dir, "TEST1.csv")) as f:
#         print(f.read())

#     print("=== np.genfromtxt Check ===")
#     import numpy as np
#     test_data = np.genfromtxt(os.path.join(price_dir, "TEST1.csv"), dtype=str, delimiter=",")
#     print("Shape:", test_data.shape)
#     print("Sample:", test_data[:3])
#     args = Args()
#     loader = DataLoader(args)

#     # 3. 测试加载
#     print("=== TRAIN SET ===")
#     df_train = loader.load(flag="train")
#     print(df_train.head(), "\n")

#     print("=== TEST SET ===")
#     df_test = loader.load(flag="test")
#     print(df_test.head(), "\n")
