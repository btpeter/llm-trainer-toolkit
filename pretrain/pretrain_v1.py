import json
import math
import os
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping

import comet_ml
# from aim import Run

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
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import DataLoader, Dataset

# os.environ['HF_EVALUATE_OFFLINE'] = '1'

logger.add('./logs/log-{time}.log', encoding='utf-8')
# for aim metrics tracking
# run = Run(experiment="bloomz-560m")


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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
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


# def accuracy(predictions, references, normalize=True, sample_weight=None):
#     return {
#         "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
#     }

# metric = evaluate.load("accuracy")
#
#
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # preds have the same shape as the labels, after the argmax(-1) has been calculated
#     # by preprocess_logits_for_metrics, we need to shift the labels
#     labels = labels[:, 1:].reshape(-1)
#     preds = preds[:, :-1].reshape(-1)
#     # return accuracy(predictions=preds, references=labels)
#     return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    #                           log_with="wandb")
    # accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    #                           dispatch_batches=training_args.dispatch_batches,
    #                           log_with="comet_ml")
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                              log_with="comet_ml")
    # accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    # Check output dir
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Detecting last checkpoint
    set_seed(training_args.seed)

    # Datasets preparing
    data_files = {}
    dataset_args = {}
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
    if extension == "text":
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        **dataset_args,
    )

    if "validation" not in raw_datasets.keys():
        if accelerator.is_main_process:
            logger.warning(f"Validation is not in dataset file dir, split from training sets.")
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )

    if accelerator.is_main_process:
        logger.info(f"Raw datasets: {raw_datasets}")

    # Loading model configs
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # Loading tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    def tokenize_function(sample):
        return tokenizer(sample["text"])

    # Check block_size validity
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 4096. If you would like to use a longer `block_size`, you can override with --block_size xxx"
            )
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        try:
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError:  # quick fix by simply take the first example
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))

        return batch

    # Concatenate all texts from our dataset and generate chunks of block_size
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # [WARNING] here drop the small remainder, later could add padding if we need
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Loading pretrained model
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # check embedding size to avoid index errors
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     if accelerator.is_main_process:
    #         logger.warning("Tokenizer length mismatch embedding size, reshape token embedding!")
    #     model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # For distributed environment, tokenize all raw datasets and generate chunks
    # with training_args.main_process_first(desc="Dataset tokenization and grouping"):
    with accelerator.main_process_first():
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    # truncate dataset as max_train_samples
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # logger.info(f"Num train samples: {len(train_dataset)}")
        # logger.debug("Tokenized training example: ")
        # logger.debug(tokenizer.decode(train_dataset[0]["input_ids"]))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        # logger.debug("Tokenized eval example:")
        # logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # For accelerator distribute training, init Data Loader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=fault_tolerance_data_collator,
        batch_size=training_args.per_device_train_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        collate_fn=fault_tolerance_data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )

    # For accelerator distribute training, init Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # For accelerate + deepspeed, change optimizer to dummy optimizer
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    optimizer_dummy = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in
        accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_dummy(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = 1
    if training_args.max_steps == -1:
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # For accelerate + deepspeed, change scheduler to dummy scheduler

    # lr_scheduler = get_scheduler(
    #     name=training_args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
    #     num_training_steps=max_train_steps * training_args.gradient_accumulation_steps,
    # )

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
            num_training_steps=max_train_steps * training_args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            total_num_steps=max_train_steps * training_args.gradient_accumulation_steps,
            warmup_num_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps
        )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate the total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    # Calculate total number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Init accelerator trackers
    accelerator_env_config = {**vars(model_args), **vars(training_args), **vars(data_args)}
    accelerator.init_trackers("baichuan2", accelerator_env_config)

    # Training ..
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    # ---- training steps & epoch arguments check ----
    if accelerator.is_main_process:
        logger.info(" **** Training **** ")
        logger.info(f" Num Examples = {len(train_dataset)}")
        logger.info(f" Num Epochs = {num_train_epochs}")
        logger.info(f" Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f" Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f" Total optimization steps = {max_train_steps}")
    # --------------------------------------------------

    # Only show the progress bar once on each machine
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Detecting last checkpoint, reload model if we set resume path
    resume_step = None
    if training_args.resume_from_checkpoint is not None:
        if training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
        else:
            raise ValueError("--resume_from_checkpoint requires a legal path")
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.basename(checkpoint_path)

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # ---- debug step count ----
    if accelerator.is_main_process:
        logger.debug("Here for step & epoch count debug ...")
        logger.debug(f"resume_step: {resume_step}")
        logger.debug(f"starting_epoch: {starting_epoch}")
        logger.debug(f"completed_steps: {completed_steps}")
        logger.debug(f"num_train_epochs: {num_train_epochs}")
    # -----------------------------

    # training step
    best_loss = float("inf")
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()

        total_loss = 0

        if training_args.resume_from_checkpoint is not None and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Check if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps % training_args.eval_steps == 0:
                    model.eval()
                    losses = []
                    for s, b in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**b)
                        loss = outputs.loss
                        losses.append(
                            accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size))
                        )
                    losses = torch.cat(losses)
                    eval_loss = float('inf')
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    train_loss = total_loss.item() / completed_steps
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": train_loss,
                        },
                        step=completed_steps
                    )
                    if accelerator.is_main_process:
                        logger.info(
                            f"[Evaluation on STEP] Step {completed_steps} : perplexity : {perplexity}"
                            f" train_loss:{train_loss} eval_loss : {eval_loss}"
                        )

                    # run.track(perplexity, name='perplexity', step=completed_steps)
                    # run.track(eval_loss, name='eval_loss', step=completed_steps)
                    # run.track(train_loss, name='train_loss', step=completed_steps)

                    # check current loss is lowest & save best model
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        accelerator.wait_for_everyone()
                        best_output_dir = os.path.join(training_args.output_dir, "best")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            best_output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(best_output_dir)

                    torch.cuda.empty_cache()

                    model.train()

                # Save checkpoint by --save_steps
                if completed_steps % training_args.save_steps == 0:
                    # save last 5 group step checkpoints
                    output_dir = f"last_step_{int(completed_steps/training_args.save_steps)%5}"
                    if output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    if accelerator.is_main_process:
                        logger.info(f"Save checkpoint on step : {completed_steps}")

                if completed_steps >= max_train_steps:
                    break

        # Final save
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)

        eval_loss = float('inf')
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        if accelerator.is_main_process:
            logger.info(f"[Evaluation on epoch] Epoch {epoch}: perplexity: {perplexity} valid_loss: {eval_loss}")

        # accelerator.log(
        #     {
        #         "perplexity": perplexity,
        #         "eval_loss": eval_loss,
        #         "train_loss": total_loss.item() / len(train_dataloader),
        #         "epoch": epoch,
        #         "step": completed_steps,
        #     },
        #     step=completed_steps
        # )
        torch.cuda.empty_cache()
    # Save Final model
    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        output_dir_final = os.path.join(training_args.output_dir, "final")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir_final,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir, "final")
            with open(os.path.join(training_args.output_dir, "all_results.json"), 'w') as f:
                json.dump({"perplexity": perplexity}, f)

    accelerator.end_training()


if __name__ == '__main__':
    main()
