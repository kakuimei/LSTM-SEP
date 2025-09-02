from typing import Optional
import os
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_peft_adapter(
    adapter_dir: str = "./lora-alpaca_default_config",
    output_dir: Optional[str] = None,
    base_model_id: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype: str = "auto",          # 可选: "auto" / "bfloat16" / "float16" / "float32"
) -> str:
    """
    将 LoRA 适配器与基座模型合并并导出为 safetensors。
    - adapter_dir: 训练得到的 LoRA 目录（包含 adapter_model.* / adapter_config.json）
    - output_dir: 结果保存目录（默认 adapter_dir + '-merged'）
    - base_model_id: 覆盖使用的基座模型名/路径（默认从适配器配置中读取）
    - trust_remote_code: 对需要自定义代码的模型设为 True（如某些 ChatGLM 等）
    - dtype: 加载合并时的精度；"auto" 会按模型权重自动选择
    """
    if output_dir is None:
        output_dir = f"{adapter_dir}-merged"
    os.makedirs(output_dir, exist_ok=True)

    # 1) 解析基座模型 id
    if base_model_id is None:
        peft_cfg = PeftConfig.from_pretrained(adapter_dir)
        base_model_id = peft_cfg.base_model_name_or_path

    # 2) 加载基座（不量化，避免合并出错）
    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch_dtype,
        device_map="auto",            # 显存不足时可改为 None + .to("cpu")
        trust_remote_code=trust_remote_code,
        use_safetensors=True,
    )

    # 3) 加载并合并 LoRA
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()  # 将 LoRA 权重写回基座并丢弃适配器结构
    gc = model.generation_config
    gc.do_sample = True 

    # 4) 保存为 safetensors（安全/通用）
    model.save_pretrained(output_dir, safe_serialization=True)

    # 5) 保存 tokenizer（优先基座同款）
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=trust_remote_code)
    tok.save_pretrained(output_dir,safe_serialization=True)

    print(f"✅ Merged model saved to: {output_dir}")
    return output_dir
