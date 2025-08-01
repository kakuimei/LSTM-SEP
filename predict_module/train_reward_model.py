import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig

from predict_module import rm_dataloader

# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"


def train_reward_model(args):
    script_args = args

    dataset_name = script_args.datasets_dir
    print("dataset_name: ", dataset_name)

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    # model_name_split = script_args.reward_base_model.split("/")[-1]
    # output_name = (
    # f"{model_name_split}_peft_stack-exchange-paired_rmts__{script_args.train_subset}_{script_args.reward_learning_rate}"
    # )
    output_name = script_args.reward_adapter

    # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.reward_learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=200,  # 500,
        save_strategy="steps",
        save_steps=200,  # 500,
        save_total_limit=2,
        gradient_accumulation_steps=script_args.reward_gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        # local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        # bf16=script_args.bf16,
        # fp16=True,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        report_to="none"
    )

    # Load the value-head model and tokenizer.
    if "llama" in script_args.reward_base_model or "vicuna" in script_args.reward_base_model or "Vicuna" in script_args.reward_base_model:
        tokenizer = LlamaTokenizer.from_pretrained(script_args.reward_base_model)
        config = LlamaConfig.from_pretrained(script_args.reward_base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.reward_base_model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(script_args.reward_base_model, trust_remote_code=True)

    # if "llama" in script_args.reward_base_model or "vicuna" in script_args.reward_base_model or "Vicuna" in script_args.reward_base_model:
    #     # required for llama
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #             "pad_token": DEFAULT_PAD_TOKEN,
    #         }
    #     )
    # else:
    #     # required for gpt2
    #     tokenizer.pad_token = tokenizer.eos_token


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    print("device_map: ", device_map)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #    script_args.reward_base_model, num_labels=1, torch_dtype=torch.bfloat16
    # )

    if "llama" in script_args.reward_base_model or "vicuna" in script_args.reward_base_model or "Vicuna" in script_args.reward_base_model:
        model = LlamaForSequenceClassification.from_pretrained(
            script_args.reward_base_model,
            num_labels=1,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_base_model,
            num_labels=1,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,  # 32,
        lora_dropout=0.05,  # 0.1,
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not script_args.gradient_checkpointing
    num_proc = 1  # Can adjust to be higher if you have more processors.


    reward_dataloder = rm_dataloader.RewardDataLoader(dataset_name, script_args.train_subset, script_args.eval_subset, num_proc, tokenizer)
    train_dataset, eval_dataset = reward_dataloder.load_data()

    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            features_j = []
            features_k = []
            for feature in features:
                features_j.append(
                    {
                        "input_ids": feature["input_ids_j"],
                        "attention_mask": feature["attention_mask_j"],
                    }
                )
                features_k.append(
                    {
                        "input_ids": feature["input_ids_k"],
                        "attention_mask": feature["attention_mask_k"],
                    }
                )
            batch_j = self.tokenizer.pad(
                features_j,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch_k = self.tokenizer.pad(
                features_k,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch = {
                "input_ids_j": batch_j["input_ids"],
                "attention_mask_j": batch_j["attention_mask"],
                "input_ids_k": batch_k["input_ids"],
                "attention_mask_k": batch_k["attention_mask"],
                "return_loss": True,
            }
            return batch


    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)


    class RewardTrainer(Trainer):
        # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
        def compute_loss(self, model, inputs, return_outputs=False):
            rewards_j = model(
                input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
            rewards_k = model(
                input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss


    # Train the model, woohoo.
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=512, pad_to_multiple_of=8),
    )

    model.config.use_cache = False

    trainer.train(script_args.resume_from_reward_checkpoint)

    print("Saving last checkpoint of the model")
    # model.save_pretrained(output_name + "_peft_last_checkpoint")
    model.save_pretrained(output_name)
