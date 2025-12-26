import logging
import os
import random

import hydra
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTConfig, SFTTrainer

import wandb
from src.dataset import get_addition_datasets
from src.helpers import log_config
from src.huggingface_callbacks import (
    GreedyDecodeOnce,
    GreedyEvaluation,
    SaveCheckpointCallback,
)


def create_digit_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal tokenizer with 14 tokens: 0-9, +, |, =, [PAD]"""
    vocab = {str(i): i for i in range(10)}  # 0-9 -> IDs 0-9
    vocab["+"] = 10
    vocab["|"] = 11
    vocab["="] = 12

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=None))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
    )

    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

    return tokenizer


def global_setup(args, wandb_run_id=None):
    """
    Rank-0: real wandb.init; writes {id, name} to wandb_meta.json + exports env.
    Workers: DO NOT init; they poll the file and set WANDB_RUN_NAME/ID locally.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.resume:
        run = wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            id=wandb_run_id,
            # tags=[],
            resume="must",
            group=str(args.wandb.group),
            config=vars(args)["_content"],
            settings=wandb.Settings(start_method="thread", _service_wait=240),
        )
        run_id = run.id
        run_name = run.name
    else:
        run = wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            id=wandb_run_id,
            # tags=[],
            resume="allow",
            group=str(args.wandb.group),
            config=vars(args)["_content"],
            settings=wandb.Settings(start_method="thread", _service_wait=240),
        )
        run_id = wandb.run.id
        run_name = wandb.run.name

    return run_id, run_name


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args):
    log_config(args)

    if not args.resume:
        logging.info(f"Loading model from scratch: {args.model_pars.hf_model_id}")
        wandb_run_id, wandb_run_name = global_setup(args)
        output_dir = os.path.join(args.save_dir, f"{wandb_run_name}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logging.info("Creating digit tokenizer")
        tokenizer = create_digit_tokenizer()
        logging.info(f"Tokenizer vocab size: {len(tokenizer)}")

    else:
        model_dir = args.model_pars.model_dir
        output_dir = os.path.dirname(model_dir)
        resume_step = int(model_dir.split("-")[-1])
        logging.info(
            f"Loading trained model from step {resume_step} and resuming: {args.model_pars.model_dir}"
        )
        wandb_run_id = None
        with open(os.path.join(model_dir, "wandb_run_id.txt"), "r") as f:
            wandb_run_id = f.read().strip()
        global_setup(args, wandb_run_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir)

    train_dataset, eval_dataset = get_addition_datasets(
        args, tokenizer, k=args.finetuning_pars.num_digits
    )

    effective_batch_size = (
        args.finetuning_pars.per_device_train_batch_size
        * max(1, args.finetuning_pars.gradient_accumulation_steps)
        * torch.cuda.device_count()
    )
    steps_per_epoch = len(train_dataset) // effective_batch_size
    if len(train_dataset) < effective_batch_size:
        steps_per_epoch = 1
    save_every_n_steps = (
        steps_per_epoch // args.finetuning_pars.save_n_per_epoch
        if args.finetuning_pars.save_n_per_epoch > 0
        else args.finetuning_pars.save_every_n_steps
    )
    eval_every_n_steps = (
        steps_per_epoch // args.finetuning_pars.eval_n_per_epoch
        if args.finetuning_pars.eval_n_per_epoch > 0
        else args.finetuning_pars.eval_every_n_steps
    )
    logging_steps = (
        max(steps_per_epoch // args.finetuning_pars.log_n_per_epoch, 1)
        if args.finetuning_pars.log_n_per_epoch > 0
        else args.finetuning_pars.logging_steps
    )
    logging.info(
        f"Effective batch size: {effective_batch_size}, steps per epoch: {steps_per_epoch}, saving every {save_every_n_steps} steps, evaluating every {eval_every_n_steps} steps, logging steps {logging_steps}."
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=logging_steps,
        num_train_epochs=args.finetuning_pars.num_train_epochs,
        completion_only_loss=True,
        per_device_train_batch_size=args.finetuning_pars.per_device_train_batch_size,
        save_steps=save_every_n_steps,
        save_strategy="steps",
        save_total_limit=args.finetuning_pars.save_total_limit,
        eval_steps=eval_every_n_steps,
        eval_strategy="steps" if eval_every_n_steps > 0 else "no",
        gradient_accumulation_steps=args.finetuning_pars.gradient_accumulation_steps,
        learning_rate=args.finetuning_pars.reference_learning_rate,
        lr_scheduler_type=args.finetuning_pars.lr_scheduler_type,
        weight_decay=args.finetuning_pars.weight_decay,
        warmup_ratio=args.finetuning_pars.warmup_ratio,
        max_grad_norm=args.finetuning_pars.max_grad_norm,
        packing=args.finetuning_pars.packing,
        label_smoothing_factor=args.finetuning_pars.label_smoothing_factor,
        deepspeed=args.finetuning_pars.deepspeed
        if args.finetuning_pars.deepspeed != ""
        else None,
        ddp_find_unused_parameters=args.finetuning_pars.ddp_find_unused_parameters,
        bf16=args.finetuning_pars.bf16,
        fp16=args.finetuning_pars.fp16,
        prediction_loss_only=True,
        optim=args.finetuning_pars.optimiser,
        gradient_checkpointing=args.finetuning_pars.gradient_checkpointing,
        dataloader_num_workers=args.finetuning_pars.dataloader_num_workers,
        dataloader_pin_memory=args.finetuning_pars.dataloader_pin_memory,
        dataloader_prefetch_factor=args.finetuning_pars.dataloader_prefetch_factor,
        dataloader_persistent_workers=args.finetuning_pars.dataloader_persistent_workers,
    )
    if args.method == "scratch":
        config = AutoConfig.from_pretrained(
            args.model_pars.hf_model_id,
            num_hidden_layers=args.model_pars.num_hidden_layers,
            hidden_size=args.model_pars.hidden_size,
            num_attention_heads=args.model_pars.num_attention_heads,
            intermediate_size=args.model_pars.intermediate_size,
        )
        config.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(config)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
    elif args.method == "SFT":
        model = args.model_pars.hf_model_id
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
    else:
        raise ValueError(
            f"Unknown training method: {args.method}. Options: ['scratch', 'SFT']"
        )

    check_greedy_decoding = GreedyDecodeOnce(
        trainer, tokenizer, prompt="0+0|1+2|3+0|4+3|=", max_new_tokens=16
    )
    greedy_evaluation = GreedyEvaluation(
        trainer,
        tokenizer,
        eval_dataset,
        max_new_tokens=16,
    )
    save_checkpoint_callback = SaveCheckpointCallback(trainer)
    trainer.add_callback(check_greedy_decoding)
    trainer.add_callback(save_checkpoint_callback)
    trainer.add_callback(greedy_evaluation)

    # add length generalization evals
    min_len_eval = max(args.finetuning_pars.num_digits - 2, 1)
    max_len_eval = min(args.finetuning_pars.num_digits + 2, 5)
    for num_digits in range(min_len_eval, max_len_eval + 1):
        if num_digits == args.finetuning_pars.num_digits:
            continue

        # sample 1000 data points for length generalization evaluation
        len_eval_dataset = get_addition_datasets(
            args, tokenizer, k=num_digits, train_ratio=-1000
        )[1]
        len_eval_greedy_evaluation = GreedyEvaluation(
            trainer,
            tokenizer,
            len_eval_dataset,
            wandb_group=str(num_digits),
            max_new_tokens=16,
        )
        trainer.add_callback(len_eval_greedy_evaluation)

    logging.info(
        f"Starting training on {args.model_pars.hf_model_id} using {args.method} (output dir {output_dir})."
    )
    if not args.resume:
        trainer.train()
    else:
        logging.info(
            f"Resuming training from checkpoint at {model_dir} (output dir {output_dir})."
        )
        trainer.train(resume_from_checkpoint=model_dir)


if __name__ == "__main__":
    main()
