from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def supervised_finetune(args):
    # ===== 1) Tokenizer =====
    print("\n[Info] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 常见 Llama/Vicuna 需要设置 pad
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ===== 2) 量化配置（可选） =====
    bnb_cfg = None
    if args.load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
        )

    # ===== 3) 加载基础模型 =====
    print("\n[Info] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
        use_safetensors=True,
    )

    # ===== 4) LoRA 配置（最小必要就够） =====
    print("\n[Info] Configuring LoRA...")
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 也可用 ["q_proj","k_proj","v_proj","o_proj"]
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ===== 5) 数据集（jsonl/json 自动读）=====
    # 期望每条样本有 "text" 字段；若不是，下面 select_columns / formatting_func 可自定义
    print("\n[Info] Loading dataset...")
    ds = load_dataset("json", data_files=args.sft_data_path, split="train")
    def build_text(example):
        prompt = (example.get("prompt") or "").strip()
        resp = (example.get("responses") or "").strip()
        text = f"{prompt}\n{resp}".strip()
        return {"text": text}
    ds = ds.map(build_text, remove_columns=[c for c in ds.column_names if c != "text"])
    val_pct = 0.0 if len(ds) <= 10 else 0.1
    if val_pct > 0:
        split = ds.train_test_split(test_size=val_pct, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    # ===== 6) 训练参数 =====
    print("\n[Info] Preparing training...")
    grad_accum = max(args.batch_size // args.micro_batch_size, 1)
    sft_args = SFTConfig(
        # === 你的原超参（等价映射） ===
        output_dir=args.output_path,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_steps=100,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=20,
        eval_strategy=("steps" if eval_ds is not None else "no"),   # 你的环境用 eval_strategy
        eval_steps=(args.eval_steps if eval_ds is not None else None),
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        load_best_model_at_end=bool(eval_ds is not None),
        ddp_find_unused_parameters=False,
        report_to=("wandb" if args.use_wandb else "none"),
        optim=("paged_adamw_32bit" if args.load_in_4bit else "adamw_torch"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_first_step=True,

        # === SFT 数据处理相关（新接口放这里） ===
        max_length=args.cutoff_len,          # 旧的 max_seq_length -> max_length
        packing=False,                        # 如不想 pack 就改为 False
        dataset_text_field="text",           # 你 map 后用于训练的字段
        dataset_num_proc=1,                  # 可按机器提升
        assistant_only_loss=False,           # 你是 LM 文本，不是对话掩码
        padding_free=False,                  # 非必要不启
    )

    # ===== 7) SFTTrainer：一行搞定数据到训练 =====
    print("\n[Info] Initializing trainer...")
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=sft_args,  
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # 可选：继续训练（断点恢复）
    print("\n[Info] Training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 只保存 LoRA 适配器（最轻量）
    print("\n[Info] Saving LoRA adapters...")
    trainer.model.save_pretrained(args.output_path)
    tok.save_pretrained(args.output_path)
    print("✅ Done. LoRA adapters saved to:", args.output_path)
