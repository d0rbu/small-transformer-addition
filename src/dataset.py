import regex
from datasets import Dataset

from src.helpers import compute_pass_at_k


class FinetuningDataset:
    def __init__(self, seed, tokenizer, apply_chat_template):
        output_pattern = r"""\\boxed\{
            (?P<content>
                (?:
                    [^{}]
                | (?P<brace>
                        \{
                            (?: [^{}] | (?&brace) )*
                        \}
                    )
                )*
            )
        \}"""
        self.output_pattern = output_pattern
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.seed = seed

    def extract_output(self, completion: str):
        last = None
        for m in regex.finditer(
            self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE
        ):
            last = m.group("content")
        return last

    def extract_output_first(self, completion: str):
        m = regex.search(
            self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE
        )
        if m:
            return m.group("content")
        return None

    def eval_outputs(self, outputs, pass_at_k: int, is_base_model: bool, scratch):
        accuracy = 0
        pass_at_k_scores = {k: 0 for k in range(1, pass_at_k + 1)}
        all_predictions = []

        for i, out in enumerate(outputs):
            print(f"Evaluating output {i + 1}/{len(outputs)}")

            # Get the predictions
            texts = [t for t in out["generated_text"]]
            example_id = out["example_id"]
            ground_truth = out["ground_truth"]
            preds = []
            num_correct = 0

            for text in texts:
                if not scratch:
                    if is_base_model:
                        pred = self.extract_output_first(text)
                    else:
                        pred = self.extract_output(text)
                else:
                    pred = text.strip()
                preds.append(pred)
                if pred is not None and pred == ground_truth[-1]:
                    num_correct += 1
            example_accuracy = num_correct / len(texts)
            accuracy += example_accuracy
            current_pass_at_k = {k: 0 for k in range(1, pass_at_k + 1)}
            for k in range(1, pass_at_k + 1):
                p_at_k = compute_pass_at_k(len(texts), num_correct, k)
                pass_at_k_scores[k] += p_at_k
                current_pass_at_k[k] = p_at_k
            all_predictions.append(
                {
                    "example_id": example_id,
                    "ground_truth": ground_truth,
                    "accuracy": example_accuracy,
                    "pass_at_k": {k: p_at_k for k, p_at_k in current_pass_at_k.items()},
                    "predictions": preds,
                    "prompt": out["prompt"],
                    "texts": texts,
                }
            )
        accuracy /= len(outputs)
        for k in pass_at_k_scores:
            pass_at_k_scores[k] /= len(outputs)
        return accuracy, pass_at_k_scores, all_predictions

    def generate_data(self):
        messages = {
            "chat": [
                [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm fine, thank you!"},
                ],
                [
                    {"role": "user", "content": "What's the weather like today?"},
                    {"role": "assistant", "content": "It's sunny and warm."},
                ],
            ],
            "example_id": [0, 1],
            "ground_truth": ["I'm fine, thank you!", "It's sunny and warm."],
        }

        train_dataset = Dataset.from_dict(messages)
        train_dataset = train_dataset.shuffle(seed=self.seed)
        test_dataset = Dataset.from_dict(messages)
        if self.apply_chat_template:
            train_dataset = train_dataset.map(
                lambda x: {
                    "prompt": self.tokenizer.apply_chat_template(
                        [x["chat"][0]], tokenize=False, add_generation_prompt=False
                    ),
                    "completion": self.tokenizer.apply_chat_template(
                        [x["chat"][1]], tokenize=False, add_generation_prompt=True
                    ),
                }
            )
            test_dataset = test_dataset.map(
                lambda x: {
                    "prompt": self.tokenizer.apply_chat_template(
                        [x["chat"][0]], tokenize=False, add_generation_prompt=False
                    ),
                    "completion": self.tokenizer.apply_chat_template(
                        [x["chat"][1]], tokenize=False, add_generation_prompt=True
                    ),
                }
            )
        else:
            train_dataset = train_dataset.map(
                lambda x: {
                    "prompt": x["chat"][0]["content"],
                    "completion": x["chat"][1]["content"],
                }
            )
            test_dataset = test_dataset.map(
                lambda x: {
                    "prompt": x["chat"][0]["content"],
                    "completion": x["chat"][1]["content"],
                }
            )
        return train_dataset, test_dataset


def get_datasets(args, tokenizer) -> tuple[Dataset, Dataset]:
    return FinetuningDataset(
        args.seed, tokenizer, args.dataset_pars.apply_chat_template
    ).generate_data()
