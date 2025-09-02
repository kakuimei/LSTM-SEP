from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results#, save_agents
from explain_module.agents import PredictReflectAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.train_reward_model import train_reward_model
from predict_module.tuning_lm_with_rl import train_grpo_with_rm
from transformers import LlamaTokenizer, pipeline #, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import os, json
from tqdm import tqdm
from memory_module.memorydb import BrainDB
from utils.config import args_to_config
from summarize_module.summarizer import DeepSeekSummarizer
from utils.config import load_existing_keys

class Exp_Model:
    def __init__(self, args):
        self.args = args
        # Initialize BrainDB
        if  os.path.exists(self.args.ckpt_dir):
            print(f"\n[Info] Loading BrainDB from checkpoint: {self.args.ckpt_dir}")
            self.brain_db = BrainDB.load_checkpoint(self.args.ckpt_dir)
        else:
            print("\n[Info] No checkpoint found. Building BrainDB from config.")
            self.brain_db = BrainDB.from_config(args_to_config(args))

        self.dataloader = DataLoader(args, brain_db=self.brain_db, summarizer=DeepSeekSummarizer())

    def train(self):
        # print("[Info] Streaming Train Agents with progress tracking...")
        # agent_cls = PredictReflectAgent
        # sample_generator = self.dataloader.load("train")

        # seen_sft  = load_existing_keys(self.args.sft_data_path)
        # seen_grpo = load_existing_keys(self.args.grpo_data_path)
        # seen_all  = seen_sft | seen_grpo

        # for sample in sample_generator:
        #     unique_key = f"{sample['ticker']}|{sample['date']}"
        #     if unique_key in seen_all:
        #         print(f"[Info] Skip existing sample: {unique_key}")
        #         continue
        #     agent = agent_cls(ticker=sample['ticker'], date=sample['date'], summary=sample['summary'], target=sample['target'], brain_db=self.brain_db)
        #     agent.run()
        #     # agents.append(agent)
        #     if agent.is_correct():
        #         prompt = agent._build_agent_prompt()
        #         response = agent.scratchpad.split('Price Movement: ')[-1]
        #         record = {
        #             "prompt": prompt,
        #             "responses": response,
        #             "rewards": [ 1.0 ],
        #             "meta": {
        #                 "ticker": sample['ticker'],
        #                 "date": sample['date'],
        #                 "label": sample['target']
        #             }
        #         }
        #         os.makedirs(os.path.dirname(self.args.sft_data_path), exist_ok=True)
        #         with open(self.args.sft_data_path, 'a') as f:
        #             f.write(json.dumps(record) + "\n")
        #     else:
        #         print(f"[Info] Agent for {sample['ticker']} on {sample['date']} was incorrect. Initiating reflection trials...")
        #         responses = []
        #         rewards = []
        #         for trial in range(self.args.num_reflect_trials):
        #             print(f"[Info] Trial {trial+1} for {sample['ticker']} on {sample['date']}")
        #             prev_response = agent.scratchpad.split('Price Movement: ')[-1]
        #             responses.append(prev_response)
        #             rewards.append(-1.0)

        #             agent.run()
        #             if agent.is_correct():
        #                 print(agent._build_agent_prompt(), "\n\n\n")
        #                 prompt = remove_reflections(agent._build_agent_prompt())
        #                 response = agent.scratchpad.split('Price Movement: ')[-1]
        #                 responses.append(response)
        #                 rewards.append(1.0)
        #                 record = {
        #                     "prompt": prompt,
        #                     "responses": responses,
        #                     "rewards": rewards,
        #                     "meta": {
        #                         "ticker": sample['ticker'],
        #                         "date": sample['date'],
        #                         "label": sample['target']
        #                     }
        #                 }
        #                 os.makedirs(os.path.dirname(self.args.grpo_data_path), exist_ok=True)
        #                 with open(self.args.grpo_data_path, 'a') as f:
        #                     f.write(json.dumps(record) + "\n")
        #                 print(f"[Info] Agent for {sample['ticker']} on {sample['date']} corrected itself in trial {trial+1}.")
        #                 break
        #     seen_all.add(unique_key)
        # print("[Info] Finished streaming training agents.")

        # Train supervised policy
        # print("\n[Info] Starting Supervised Fine-Tuning...")
        # supervised_finetune(self.args)
        # print("\n[Info] starting merging PEFT adapter...")
        # merge_peft_adapter(adapter_dir=self.args.output_path, base_model_id=self.args.model_path, output_dir=self.args.rl_base_model)
        # print("\n[Info] Starting Reward Model Training...")
        # train_reward_model(self.args)
        # Optimize using reinforcement learning
        train_grpo_with_rm(self.args)
        merge_peft_adapter(model_name=self.args.output_dir+"step_saved", output_name="./saved_models/sep_model")


    def test(self):
        print("Loading Test Agents...")
        data = self.dataloader.load(flag="test")

        agent_cls = PredictReflectAgent
        test_agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in data.iterrows()]
        print("Loaded Test Agents.")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "./saved_models/sep_model",
            load_in_4bit=True,
            device_map="auto"
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.args.output_dir+"step_saved")
        reward_model = pipeline(
            "sentiment-analysis",
            model=self.args.reward_model_name,
            device_map="auto",
            model_kwargs={"load_in_4bit": True},
            tokenizer=tokenizer
        )

        for agent in test_agents:
            agent.run_n_shots(
                              model=model,
                              tokenizer=tokenizer,
                              reward_model=reward_model,
                              num_shots=self.args.num_shots
                              )

        correct, incorrect = summarize_trial(test_agents)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        save_results(test_agents, self.args.save_dir)
