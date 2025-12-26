import logging
import os
import random

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback

import wandb


class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.tokenizer = trainer.processing_class
        self._checkpoint_prefix = "checkpoint-"

    def on_save(self, args, state, control, **kwargs):
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(
            args.output_dir, f"{self._checkpoint_prefix}{state.global_step}"
        )
        if getattr(wandb, "run", None) is not None and getattr(
            self, "_is_main_process", True
        ):
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"{self._checkpoint_prefix}{state.global_step}"})
        with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logging.info("Training ended.")
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(
            args.output_dir, f"{self._checkpoint_prefix}{state.global_step}"
        )
        if getattr(wandb, "run", None) is not None and getattr(
            self, "_is_main_process", True
        ):
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"{self._checkpoint_prefix}{state.global_step}"})
        with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control


class GreedyDecodeOnce(TrainerCallback):
    """
    One-time manual greedy decode at training start (no .generate()).
    - Works with ZeRO-3: all ranks run forward; only rank-0 prints.
    - Uses use_cache=False to avoid KV/cache issues.
    - Keeps decode short (few tokens) to avoid memory spikes.
    """

    def __init__(
        self, trainer, tokenizer, prompt="0+0|1+2|3+0|4+3|=", max_new_tokens=16
    ):
        self.trainer = trainer
        self.tok = tokenizer
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.done = False

    def _is_main(self, trainer):
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "is_local_main_process"):
            return bool(acc.is_local_main_process)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _run_once(self, trainer):
        if self.done:
            return
        self.done = True

        model = trainer.model
        device = getattr(
            trainer.accelerator,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        enc = self.tok(self.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = (
            attn.to(device)
            if isinstance(attn, torch.Tensor)
            else torch.ones_like(input_ids, device=device)
        )

        was_training = model.training
        model.eval()

        # Manual greedy decode: append one token at a time; no caches or extra kwargs.
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # keep attention mask in lockstep
                one = torch.ones_like(next_token, device=device)
                attn = torch.cat([attn, one], dim=1)

        # sync & print only on rank-0
        try:
            trainer.accelerator.wait_for_everyone()
        except Exception:
            pass
        if self._is_main(trainer):
            text = self.tok.decode(input_ids[0], skip_special_tokens=True)
            print("[Greedy Decoding Test] prompt:", repr(self.prompt))
            print("[Greedy Decoding Test] output:", text)

        # restore training state
        model.train(was_training)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def on_train_begin(self, args, state, control, **kw):
        tr = getattr(self, "trainer", None)
        self._run_once(tr)
        return control


class GreedyEvaluation(TrainerCallback):
    def __init__(
        self,
        trainer,
        tokenizer,
        eval_dataset: Dataset,
        wandb_group: str = "default",
        max_new_tokens: int = 16,
        eval_batch_size: int = 1000,
    ):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_new_tokens = max_new_tokens
        self.wandb_group = wandb_group
        self.eval_batch_size = eval_batch_size

    @torch.no_grad()
    def greedy_decode(self, batch: list[str], max_new_tokens: int = 16) -> list[str]:
        self.trainer.model.eval()

        encoded_batch = self.tokenizer(batch, return_tensors="pt")
        input_ids = encoded_batch["input_ids"].to(self.trainer.model.device)
        attn = encoded_batch.get("attention_mask").to(input_ids.device)
        starting_length = input_ids.shape[1]
        finished_samples = torch.zeros(
            len(batch), dtype=torch.bool, device=input_ids.device
        )
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            out = self.trainer.model(
                input_ids=input_ids, attention_mask=attn, use_cache=False
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            just_finished = next_token.squeeze(dim=-1) == eos_token_id
            finished_samples |= just_finished

            next_token[finished_samples] = eos_token_id

            input_ids = torch.cat([input_ids, next_token], dim=1)
            # keep attention mask in lockstep
            one = torch.ones_like(next_token, device=attn.device)
            attn = torch.cat([attn, one], dim=1)

            if finished_samples.all():
                break

        raw_responses = self.tokenizer.batch_decode(
            input_ids[:, starting_length:], skip_special_tokens=True
        )

        # because of the WordLevel tokenizer, the characters in the response are padded with spaces
        responses_characters = [
            response.split()
            for response in tqdm(
                raw_responses, desc="Splitting up response characters", leave=False
            )
        ]
        responses = [
            "".join(characters)
            for characters in tqdm(
                responses_characters, desc="Joining response characters", leave=False
            )
        ]
        return responses

    def on_evaluate(self, args, state, control, **kwargs):
        for batch in tqdm(
            self.eval_dataset.iter(batch_size=self.eval_batch_size),
            desc="Evaluating",
            leave=False,
        ):
            predictions = self.greedy_decode(batch["prompt"], self.max_new_tokens)
            completions = batch["completion"]

            correct = [
                prediction == completion
                for prediction, completion in zip(predictions, completions, strict=True)
            ]
            accuracy = sum(correct) / len(correct)

            # log sample response
            sample_idx = random.randint(0, len(predictions) - 1)
            logging.info(f"sample prompt: {batch['prompt'][sample_idx]}")
            logging.info(f"sample response: {predictions[sample_idx]}")
            logging.info(f"sample completion: {completions[sample_idx]}")
            wandb.log({f"eval/greedy_acc_{self.wandb_group}": accuracy})

        return control
