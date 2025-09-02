# rm_grpo_pipeline.py
import os, random
from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GenerationConfig
from trl import RewardTrainer, RewardConfig, GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from dataclasses import replace

# =============== utils ===============
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def simple_format_bonus(text: str) -> float:
    """ 两行 + 前缀 + 禁括号 + 合理长度 → 最多 +1.0 """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    two = (len(lines)==2)
    pref = len(lines)>=2 and lines[0].startswith("Price Movement: ") and lines[1].startswith("Explanation: ")
    no_par = "(" not in text and ")" not in text
    exp_len = len(lines[1][len("Explanation: "):]) if len(lines)>=2 else 0
    lng = 20 <= exp_len <= 600
    return 0.25*two + 0.25*pref + 0.25*no_par + 0.25*lng

# ===============  用 RM 做 GRPO 在线优化 ===============
def train_grpo_with_rm(args):
    """
    期望 args：
      grpo_data_path, grpo_output_dir, base_model_name, rm_path, seed, bf16, load_in_4bit
      epochs, steps, lr_rl, batch_size_rl, grad_accum
      max_new_tokens, temperature, top_p, top_k, min_length
      （可选）use_lora, lora_r/lora_alpha/lora_dropout
    grpo 数据需要包含: prompt （label可选，不强制）
    """
    os.makedirs(args.grpo_output_dir, exist_ok=True)
    set_seed(args.seed)

    # 读 prompt-only 数据
    print(f"[Info] Loading GRPO dataset from: {args.grpo_data_path}")
    ds = load_dataset("json", data_files=args.grpo_data_path, split="train")
    def _map(ex):
        return {"prompt": (ex.get("prompt") or "").strip()}
    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c!="prompt"])
    if len(ds) > 50:
        split = ds.train_test_split(test_size=0.02, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # 分词器
    print("[Info] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # 加载策略模型（被优化的 LLM）
    quant_kwargs = {}
    if getattr(args, "load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
    print("[Info] Loading policy model...")
    pol = AutoModelForCausalLM.from_pretrained(
        args.merged_output_dir,
        device_map="auto",
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="sdpa",
        **quant_kwargs,
    )
    if getattr(args, "use_lora", False):
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj","v_proj"], bias="none", task_type="CAUSAL_LM"
        )
        print("[Info] Applying LoRA...")
        pol = get_peft_model(pol, peft_cfg)
    pol.config.use_cache = False

    # 加载 RM（打分器）
    rm_tok = tok  # 直接复用同 tokenizer（也可单独加载）
    print("[Info] Loading reward model...")
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm_output_dir,
        num_labels=1,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    rm.eval()

    # 采样参数（GRPO 必须采样）
    print("[Info] Configuring generation...")
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_new_tokens=args.min_length,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # 在线奖励：RM 分数 + 轻量格式奖励（兼容 GRPO 的调用方式）
    def reward_fn(
        prompts=None,
        completions=None,
        completion_ids=None,
        **kwargs,
    ) -> list[float]:
        assert completions is not None and len(completions) > 0

        # === 选其一 ===
        texts = completions                       # 只用 completion（推荐，干净）
        # texts = [p + c for p, c in zip(prompts, completions)]   # 想用 prompt+completion 时启用

        with torch.no_grad():
            toks = rm_tok(
                texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            # 确保与 RM 在同一设备（rm.to(device) 已在外部完成的话，也可以用 .to(device)）
            toks = {k: v.to(rm.device) for k, v in toks.items()}
            out = rm(**toks)
            rm_score = out.logits.squeeze(-1).float().cpu().tolist()

        fmt_bonus = [simple_format_bonus(s) for s in completions]
        alpha, beta = 1.0, 0.3
        return [alpha * rm_score[i] + beta * fmt_bonus[i] for i in range(len(completions))]
    
    grpo_cfg = GRPOConfig(
        output_dir=args.grpo_output_dir,
        learning_rate=args.lr_rl,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size_rl,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to="none",
        beta=0.04,
        generation_kwargs=gen_cfg.__dict__,
    )
    grpo_cfg = replace(grpo_cfg,
    steps_per_generation=16,
    generation_batch_size=None,
    num_generations=8,
    )
    print("[DBG] steps_per_generation=", grpo_cfg.steps_per_generation,
        "generation_batch_size=", grpo_cfg.generation_batch_size,
        "num_generations=", grpo_cfg.num_generations)
    
    print("\n[Info] Starting GRPO training...")
    trainer = GRPOTrainer(
        args=grpo_cfg,                # ← 用 config，不要用 args
        model=pol,
        processing_class=tok,                  # ← 用 tokenizer，不要用 processing_class
        reward_funcs=[reward_fn],   # ← 用 reward_functions，不要用 reward_funcs
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # generation_config=gen_cfg,      # 新版接口
    )
    trainer.train()
    print("\n[Info] Saving GRPO (with RM)...")
    pol.save_pretrained(args.grpo_output_dir, safe_serialization=True)
    tok.save_pretrained(args.grpo_output_dir)
    print(f"✅ GRPO (with RM) saved to: {args.grpo_output_dir}")
