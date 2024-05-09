import json
import math
import os
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping, Sequence

# import comet_ml
from aim.hugging_face import AimCallback
from aim import Text

from collections import Counter
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler

import numpy as np
import torch
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import accuracy_score
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    AutoConfig,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    set_seed,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
    deepspeed,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import DataLoader, Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_int8_training

logger.add('./logs/log-{time}.log', encoding='utf-8')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8bit mode or not."}
    )

    load_in_bits: Optional[int] = field(default=16)

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )

    template_name: Optional[str] = field(
        default="dxy",
        metadata={"help": "The instruction alignment template format"}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(default=None)

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    max_source_length: Optional[int] = field(
        default=2048, metadata={"help": "Max length of prompt input text"}
    )

    max_target_length: Optional[int] = field(
        default=1024, metadata={"help": "Max length of output text"}
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined"}
    )


@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    experiment_name: Optional[str] = field(default="default-experiment")
    run_name: Optional[str] = field(default="default-run")


@dataclass
class Conversation:
    """A class that task prompt templates"""

    # The name of this template
    name: str
    # The instruction prompt
    system_prompt: str
    # The prompt template
    template: str
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Separator
    sep: str

    def _format_example(
            self,
            instruction: Optional[str] = "",
            source: Optional[str] = ""

    ) -> str:
        """ Note:  Here we just format single-turn instruction dataset.
                  For multi-turn dialog dataset, need refactor in feature.
        """
        prompt = self.system_prompt + instruction
        if instruction != "":
            prompt += self.sep
        prompt += source
        return self.template.format(query=prompt)

    def get_item(
            self,
            instruction: Optional[str] = "",
            source: Optional[str] = ""
    ) -> str:
        return self._format_example(instruction, source)


# Global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """Register conversation template"""
    conv_templates[template.name] = template


""" Baichuan-13B-Chat template """
register_conv_template(
    Conversation(
        name="baichuan-chat",
        system_prompt="",
        template=" <reserved_102> {query} <reserved_103> ",
        roles=(" <reserved_102> ", " <reserved_103> "),
        sep="</s>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template"""
    return conv_templates[name]


class SavePeftModelCallback(TrainerCallback):
    """
    Lora model callback saver
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            print(" ========= peft model save call back ========= ")
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            print(" ========= peft model save Done ========= ")
            return control

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
        # self.experiment['_my_note'] = self._my_note

# class AimCustomCallback(AimCallback):
#     def on_log(self, args, state, control,
#                model=None, logs=None, **kwargs):
#         super().on_log(args, state, control, model, logs, **kwargs)

#         context = {
#             'subset': self._current_shift,
#         }
#         for log_name, log_value in logs.items():
#             if isinstance(log_value, str):
#                 self.experiment.track(Text(log_value), name=log_name, context=context)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # init seed
    set_seed(training_args.seed)

    # Datasets preparing
    data_files = {}
    if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
        train_data_files = glob(f'{data_args.train_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"train files: {train_data_files}")
        # Train data files must be same type, e.g. all txt or all jsonl
        types = [f.split('.')[-1] for f in train_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
        data_files["train"] = train_data_files

    if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
        eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"eval files: {eval_data_files}")
        data_files["validation"] = eval_data_files
        # Train data files must be same type, e.g. all txt or all jsonl
        types = [f.split('.')[-1] for f in eval_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")

    extension = "text" if data_files["train"][0].endswith('txt') else 'json'

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    if "validation" not in raw_datasets.keys():
        # if accelerator.is_main_process:
        #     logger.warning(f"Validation is not in dataset file dir, split from training sets.")
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )

    # Loading tokenizer & Define data item processor
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # IGNORE_INDEX = tokenizer.pad_token_id
    IGNORE_INDEX = -100
    PATIENT_TOKEN = [195]
    DOCTOR_TOKEN = [196]

    max_length = data_args.max_source_length + data_args.max_target_length

    def preprocess_function(examples):

        all_input_ids = list()
        all_labels = list()
        all_attention_mask = list()
        
        batch_conversation = examples["conversations"]

        for conversation in batch_conversation:
            input_ids = list()
            labels = list()
            for message in conversation:
                from_ = message["from"]
                value = message["value"]
                value_ids = tokenizer.encode(value)

                if from_ == "human":
                    input_ids += PATIENT_TOKEN + value_ids
                    labels += [tokenizer.eos_token_id] + [IGNORE_INDEX] * len(value_ids) 
                else:
                    input_ids += DOCTOR_TOKEN + value_ids
                    labels += [IGNORE_INDEX] + value_ids
        
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)

            # trunck of max length
            input_ids = input_ids[: data_args.max_source_length]
            labels = labels[: data_args.max_target_length]

            # padding
            input_ids += [tokenizer.pad_token_id] * (data_args.max_source_length - len(input_ids))
            labels += [IGNORE_INDEX] * (data_args.max_target_length - len(labels))

            # format tensor
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)

        results = {'input_ids': all_input_ids, 'labels': all_labels, "attention_mask": all_attention_mask}
        return results
        

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    # Tokenize all raw datasets & format example pairs
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Number of train_dataset: {len(train_dataset)}")
        # logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = train_dataset.shuffle().map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            train_dataset = train_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            # logger.debug(f"Num train_samples: {len(train_dataset)}")
            # logger.debug("Tokenized training example:")
            # logger.debug(f"Decode example [0]: {train_dataset[0].keys()}")
            # logger.debug(f"Decode input_ids [0]: {tokenizer.decode(train_dataset[0]['input_ids'])}")
            # logger.debug(f"Decode labels [0]: {tokenizer.decode(train_dataset[0]['labels'])}")
            # replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
            #                    for label in list(train_dataset[0]['labels'])]
            # logger.debug(f"Decode labels[0]: {tokenizer.decode(replaced_labels)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            # logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            # logger.debug("Tokenized eval examples: ")
            # logger.debug(f"Decode input_ids[0]: {tokenizer.decode(eval_dataset[0]['input_ids'])}")

    # Model Loading
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}

        if training_args.qlora and (len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=model_args.cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            load_in_8bit=model_args.load_in_8bit,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            ) if training_args.qlora else None,
        )
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if training_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            target_modules = training_args.target_modules.split(",") if training_args.target_modules else None
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {training_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)
    logger.debug(f"Model : {model}")

    # Initialize Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)

    # metric tracker callback
    # aim_callback = AimCallback(experiment_name="baichuan2-sft1")
    # aim_callback = AimCallback(repo='/data2/aim_home/.aim/', experiment=training_args.experiment_name)
    aim_callback = MyAimCallback.create_callback(repo='/data2/aim_home/.aim/', exp_name=training_args.experiment_name, run_name=training_args.run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=([SavePeftModelCallback, aim_callback] if isinstance(model, PeftModel) else None)
    )

    # Training
    if training_args.do_train:
        logger.info("##### Training #####")
        sample = next(iter(trainer.get_train_dataloader()))
        # logger.debug(f"Train dataloader example: {sample}")
        # logger.debug(f"Detail input_ids: {list(sample['input_ids'])[:3]}, \nlabels: {list(sample['labels'])[:3]}")
        # logger.debug(f"Decode input_ids[0]: {tokenizer.decode(sample['input_ids'][0])}")
        replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in sample['labels'][0]]
        # logger.debug(f"Decode labels[0]: {tokenizer.decode(replaced_labels)}")

        checkpoint = None

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        model.config.use_cache = True
        trainer.save_state()
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")

    # Evaluation
    if training_args.do_eval:
        logger.info("##### Evaluate #####")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples

        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        logger.debug(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()
