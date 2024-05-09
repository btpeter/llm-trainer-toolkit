import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from aim.hugging_face import AimCallback
from aim import Text

import torch
import pandas as pd
import datasets
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer
from aim.hugging_face import AimCallback
from aim import Text

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help":"The train data file folder."})
    valid_file_dir: Optional[str] = field(default=None, metadata={"help":"An optional input evaluation data file to evaluate the perplexity on text file folder."})

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    num_train_epochs: Optional[int] = field(default=1)
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    target_modules: Optional[str] = field(default="all")

    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    save_total_limit: Optional[int] = field(default=5)

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 200 samples"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    # aim report params
    experiment_name: Optional[str] = field(default="default-experiment")
    run_name: Optional[str] = field(default="default-run")


class MyAimCallback(AimCallback):

    @staticmethod
    def create_callback(
        exp_name: str,
        repo: str,
        run_name: Optional[str] = None,
        run_tags: Optional[str] = None,
        log_system_params: bool = False,
        pdb: bool = False,
        **kwargs
    ) -> AimCallback:
        """ 构造自定义的`AimCallback`
        - exp_name:  str; Experniments Name
        - repo:      str; Path of `.aim` default is os.environ['AIM_HOME']
        - run_name:  str; Run Name
        - run_tags:  str; Run Tags join by ' '
        - log_system_params: bool; default=False
        **kwargs:    dict[str, str]; 需额外记录的自定义信息, 例如`argv, trainable_parameters, ...`
        """
        _callback = MyAimCallback(
            repo=repo,
            experiment=exp_name,
            log_system_params=log_system_params
        )
        if run_name is not None:
            _callback._my_name = run_name
        if run_tags is not None:
            _callback._my_tags = run_tags.split()
        _callback._my_note = kwargs
        # _callback._my_note['argv'] = _parse_argv()[-1]
        for key, val in _callback._my_note.items():
            assert isinstance(val, str), f"type({key}) is not str; but {type(val)}"
        return _callback

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        if not self._run:
            return
        if hasattr(self, '_my_tags'):
            for tag in self._my_tags:
                self._run.add_tag(tag)
        if hasattr(self, '_my_name'):
            self._run.props.name = self._my_name


def get_format_paired_data(data_path, tokenizer, sanity_check=False, cache_dir=None, num_proc=24):
    # Loading dataframe from pandas file
    df = pd.read_csv(data_path)
    # Convert to Huggingface datasets
    dataset = Dataset.from_pandas(df)

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 200)))
    
    # Add special tokens
    
    # IGNORE_INDEX = tokenizer.pad_token_id
    IGNORE_INDEX = -100
    # Question token index [195] : <reserved_106>
    BAICHUAN2_QUESTION_TOKEN_INDEX = [195]
    BAICHUAN2_QUESTION_TOKEN = tokenizer.decode(BAICHUAN2_QUESTION_TOKEN_INDEX)
    # Anwser token index [196] : <reserved_107>
    BAICHUAN2_ANWSER_TOKEN_INDEX = [196]
    BAICHUAN2_ANWSER_TOKEN = tokenizer.decode(BAICHUAN2_ANWSER_TOKEN_INDEX)

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [BAICHUAN2_QUESTION_TOKEN + p + BAICHUAN2_ANWSER_TOKEN for p in samples['prompt']],
            "chosen": samples['chosen'], 
            "rejected": samples['rejected'],
        }

    return dataset.map(
        return_prompt_and_responses, 
        batched=True, 
        num_proc=num_proc, 
        remove_columns=original_columns,
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=script_args.load_in_4bit,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=script_args.load_in_4bit,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)

    train_dataset = get_format_paired_data(
        data_path=script_args.train_file_dir, 
        tokenizer=tokenizer, 
        sanity_check=script_args.sanity_check,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    eval_dataset = get_format_paired_data(
        data_path=script_args.valid_file_dir, 
        tokenizer=tokenizer, 
        sanity_check=script_args.sanity_check,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # initalize training arguments

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        save_total_limit=script_args.save_total_limit,
    )

    # format lora target modules param
    target_modules = script_args.target_modules.split(",") if script_args.target_modules else None
    if target_modules and "all" in target_modules:
        target_modules = find_all_linear_names(model, int4=script_args.load_in_4bit, int8=script_args.load_in_8bit)

    peft_config = LoraConfig(
        r=script_args.lora_r, 
        lora_alpha=script_args.lora_alpha, 
        lora_dropout=script_args.lora_dropout, 
        target_modules=target_modules,
        bias="none", 
        task_type="CAUSAL_LM",
    )

    aim_callback = MyAimCallback.create_callback(repo='/data2/aim_home/.aim/', exp_name=script_args.experiment_name, run_name=script_args.run_name)

    dpo_trainer = DPOTrainer(
        model, 
        model_ref, 
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer, 
        peft_config=peft_config, 
        max_prompt_length=script_args.max_prompt_length, 
        max_length=script_args.max_length,
        callbacks=([aim_callback])
    )

    # training
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # Final save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == '__main__':
    main()

