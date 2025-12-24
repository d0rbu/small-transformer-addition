from transformers import TrainerCallback
import torch

import logging
import wandb
import os


class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.tokenizer = trainer.processing_class
        self._checkpoint_prefix = "checkpoint-"

    def on_save(self, args, state, control, **kwargs):
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"{self._checkpoint_prefix}{state.global_step}")
        if getattr(wandb, "run", None) is not None and getattr(self, "_is_main_process", True): 
            wandb.save(f"{output_dir}/pytorch_model.bin")
            wandb.log({"checkpoint": f"{self._checkpoint_prefix}{state.global_step}"})
        with open(os.path.join(output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logging.info(f"Saved checkpoint at step {state.global_step} to {output_dir}.")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        logging.info("Training ended.")
        logging.info(f"Saving checkpoint at step {state.global_step}.")
        output_dir = os.path.join(args.output_dir, f"{self._checkpoint_prefix}{state.global_step}")
        if getattr(wandb, "run", None) is not None and getattr(self, "_is_main_process", True): 
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
    def __init__(self, trainer, tokenizer, prompt="The quick brown fox", max_new_tokens=16):
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
        device = getattr(trainer.accelerator, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        enc = self.tok(self.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if isinstance(attn, torch.Tensor) else torch.ones_like(input_ids, device=device)

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