from typing import Dict, List
import logging
import hydra
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
import ray

from src.dataset import get_dataset


class LLMInference:
    def __init__(self, num_datapoints, vocab_size: int, tokenizer, model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", 
                 max_tokens=1024, n=16, temperature=0.8, lora_path=None, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1):
        self.llm = LLM(model=model_name_or_path, enable_lora=lora_path != None, 
                       tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size, data_parallel_size=data_parallel_size,
                       max_logprobs=vocab_size)
        self.llm.set_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.lora_path = lora_path
        self.sampling_params = SamplingParams(max_tokens=max_tokens, n=n, temperature=temperature, logprobs=vocab_size)
        self.num_datapoints = num_datapoints

    def __call__(self, batch: Dict[str, torch.LongTensor]) -> Dict[str, list]:
        outputs = self.llm.generate(batch["prompt"], self.sampling_params)
        # One per input in the batch
        prompt: List[str] = []
        # Multiple per input in the batch (num_samples)
        generated_text: List[List[str]] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append([self.tokenizer.decode(o.token_ids) for o in output.outputs])
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "example_id": batch["example_id"],
            "ground_truth": batch["ground_truth"]
        }

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_pars.model_dir, fix_mistral_regex=True)
    dataset = get_dataset(args, tokenizer)

    if args.method == "eval":
        logging.info(f"Evaluating on max {args.eval_pars.max_to_eval} datapoints.")
        num_datapoints = min(len(dataset), args.eval_pars.max_to_eval)
        # Do inference
        ray_ds = ray.data.from_pandas(dataset.to_pandas()[:num_datapoints], override_num_blocks=args.eval_pars.num_workers)
        # Get the number of available gpus
        num_gpus = ray.cluster_resources().get("GPU", 0)
        # Print cluster resources
        logging.info(f"Ray cluster resources: {ray.cluster_resources()}")
        ray_ds = ray_ds.map_batches(
            LLMInference,
            concurrency=args.eval_pars.num_workers,
            batch_size=args.eval_pars.num_samples,
            num_gpus=num_gpus,
            fn_constructor_kwargs={"num_datapoints": num_datapoints, "tokenizer": tokenizer, "vocab_size": len(tokenizer), "model_name_or_path": args.model_pars.model_dir, "max_tokens": args.eval_pars.max_tokens,
                                    "n": args.eval_pars.num_samples, "temperature": args.eval_pars.temperature, 
                                    "tensor_parallel_size": args.eval_pars.tensor_parallel_size, "pipeline_parallel_size": args.eval_pars.pipeline_parallel_size,
                                    "data_parallel_size": args.eval_pars.data_parallel_size}
        )

        outputs = ray_ds.take_all()
        output_path = os.path.join(args.model_pars.model_dir, f"generated_outputs.json")
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(outputs, f, indent=4)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
