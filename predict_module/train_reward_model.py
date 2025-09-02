# rm_grpo_pipeline.py
import os, random
from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GenerationConfig
from trl import RewardTrainer, RewardConfig, GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model

# =============== utils ===============
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_pairs_from_responses(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从一条样本里自动构造 (chosen, rejected) 多个对：
    - 找 rewards 最大的一个或多个作为 chosen
    - 找 rewards 最小的一个或多个作为 rejected
    - 按笛卡尔积组合所有 (chosen, rejected)
    """
    rsps = ex.get("responses") or []
    rws  = ex.get("rewards")  or []
    if not rsps or len(rsps) < 2 or len(rsps) != len(rws):
        return []

    max_reward = max(rws)
    min_reward = min(rws)

    chosen_indices   = [i for i, r in enumerate(rws) if r == max_reward]
    rejected_indices = [i for i, r in enumerate(rws) if r == min_reward]

    prompt = (ex.get("prompt") or "").strip()

    pairs = []
    for ci in chosen_indices:
        for ri in rejected_indices:
            if ci != ri:
                pairs.append({
                    "prompt": prompt,
                    "chosen": rsps[ci].strip(),
                    "rejected": rsps[ri].strip(),
                })
    return pairs

# =============== 1) Reward Model 训练 ===============
def train_reward_model(args):
    """
    期望 args：
      rm_data_path, rm_output_dir, rm_model_name(或与SFT同底座), seed, bf16
      （可选）use_lora, lora_r/lora_alpha/lora_dropout, lr_rm, batch_size_rm, epochs_rm, grad_accum_rm, max_length_rm
    数据需要包含: prompt, responses(list), rewards(list)
    """
    os.makedirs(args.rm_output_dir, exist_ok=True)
    set_seed(args.seed)

    # 读取 json/jsonl
    print(f"[Info] Loading Reward Model training data from: {args.rm_data_path}")
    ds = load_dataset("json", data_files=args.rm_data_path, split="train")
    # 构 pair 数据
    pairs = []
    for ex in ds:
        pair = build_pairs_from_responses(ex)
        if pair: pairs.extend(pair)
    if not pairs:
        raise ValueError("没有可用的 (chosen, rejected) 对。请确认每条样本至少有2个 responses 且 rewards 不相同。")
    pair_ds = Dataset.from_list(pairs)
    print(f"[Info] Constructed {len(pair_ds)} (chosen, rejected) pairs for Reward Model training.")

    # 分词器
    print(f"[Info] Loading tokenizer from: {args.rm_model_name}")
    tok = AutoTokenizer.from_pretrained(args.rm_model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # RM = 序列分类 1 维输出（标量回报）
    print(f"[Info] Loading Reward Model from: {args.rm_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name,
        num_labels=1,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        use_safetensors=True,
    )

    # 可选：对 RM 也用 LoRA（通常不需要）
    if getattr(args, "use_lora", False):
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj","v_proj"], bias="none", task_type="SEQ_CLS"
        )
        model = get_peft_model(model, peft_cfg)
    print(f"[Info] Reward Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    rm_cfg = RewardConfig(
        output_dir=args.rm_output_dir,
        learning_rate=getattr(args, "lr_rm", 5e-6),
        per_device_train_batch_size=getattr(args, "batch_size_rm", 4),
        gradient_accumulation_steps=getattr(args, "grad_accum_rm", 2),
        num_train_epochs=getattr(args, "epochs_rm", 1),
        max_length=getattr(args, "max_length_rm", 1024),
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )
    
    model.config.use_cache = False
    trainer = RewardTrainer(
        model=model,
        args=rm_cfg,
        train_dataset=pair_ds,
        eval_dataset=None,
        processing_class=tok,
        # RewardTrainer 期望字段名：prompt/chosen/rejected
    )
    print("[Info] Starting Reward Model training...")
    trainer.train()
    print("[Info] Reward Model training completed.")
    model.save_pretrained(args.rm_output_dir, safe_serialization=True)
    tok.save_pretrained(args.rm_output_dir)
    print(f"✅ Reward Model saved to: {args.rm_output_dir}")
