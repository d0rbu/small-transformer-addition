import random
from copy import deepcopy

import regex
import torch as th
from datasets import Dataset
from tqdm import tqdm

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


def get_finetuning_datasets(args, tokenizer) -> tuple[Dataset, Dataset]:
    return FinetuningDataset(
        args.seed, tokenizer, args.dataset_pars.apply_chat_template
    ).generate_data()


class AdditionDataset:
    def __init__(self, seed, tokenizer, k: int = 4, num_operands: int = 2):
        self.tokenizer = tokenizer
        self.seed = seed
        self.k = k
        self.num_operands = num_operands

    def _get_all_combinations(self) -> th.Tensor:
        start = 10 ** (self.k - 1)
        end = 10**self.k
        all_numbers = th.arange(start, end)

        combinations = th.combinations(
            all_numbers, self.num_operands, with_replacement=True
        )

        return combinations

    def _uniform_split_combinations(
        self, seed: int, train_ratio: int | float
    ) -> tuple[th.Tensor, th.Tensor]:
        th.manual_seed(seed)

        all_combinations = self._get_all_combinations()
        num_combinations = all_combinations.shape[0]
        if isinstance(train_ratio, float):
            num_train = int(num_combinations * train_ratio)
        else:
            num_train = train_ratio

        random_shuffle = th.randperm(num_combinations)
        shuffled_combinations = all_combinations[random_shuffle]

        train_combinations = shuffled_combinations[:num_train]
        test_combinations = shuffled_combinations[num_train:]

        return train_combinations, test_combinations

    def _format_solution(
        self, summands: th.Tensor, solution: th.Tensor, leading_zeroes: int = 0
    ) -> dict[str, str]:
        summand_strs = ["0" * leading_zeroes + str(s.item()) for s in summands]
        solution_digits = str(solution.item())

        prompt_parts = []

        for digits_at_pos in zip(*summand_strs, strict=True):
            prompt_parts.append("+".join(digits_at_pos) + "|")

        prompt_parts.append("=")

        prompt = "".join(prompt_parts)
        solution = "|".join(solution_digits) + "|"

        return {
            "prompt": prompt,
            "completion": solution,
        }

    def _format_combinations(self, combinations: th.Tensor) -> list[dict]:
        solutions = combinations.sum(dim=1)
        dataset_iterator = tqdm(
            zip(combinations, solutions),
            total=combinations.shape[0],
            desc="Formatting combinations",
            leave=False,
        )

        formatted = [
            self._format_solution(combo, solution)
            for combo, solution in dataset_iterator
        ]

        return formatted

    # this is needed to more efficiently get eval data for larger k and num_samples
    def generate_eval_data_uniform(self, num_samples: int) -> Dataset:
        random.seed(self.seed)
        th.manual_seed(self.seed)

        start = 10 ** (self.k - 1)
        end = 10**self.k
        num_unique_values = end - start

        total_combinations = num_unique_values**self.num_operands
        num_samples_to_draw = min(num_samples, total_combinations)

        sampled_combinations = set()
        while len(sampled_combinations) < num_samples_to_draw:
            combo = tuple(random.choices(range(start, end), k=self.num_operands))
            sampled_combinations.add(combo)

        eval_combinations = th.tensor(list(sampled_combinations))

        return Dataset.from_list(self._format_combinations(eval_combinations))

    def generate_data(self, train_ratio: int | float = 0.8) -> tuple[Dataset, Dataset]:
        train_combos, test_combos = self._uniform_split_combinations(
            self.seed, train_ratio
        )

        train_dataset = Dataset.from_list(self._format_combinations(train_combos))
        test_dataset = Dataset.from_list(self._format_combinations(test_combos))

        return train_dataset, test_dataset

    def eval_outputs(self, outputs: list[dict]) -> tuple[float, int, list[dict]]:
        num_correct: int = 0

        modified_outputs = deepcopy(outputs)

        for output in tqdm(
            modified_outputs,
            total=len(outputs),
            desc="Evaluating outputs",
            leave=False,
        ):
            assert "prediction" in output, "Prediction is required"
            assert "completion" in output, "Completion is required"

            prediction = output["prediction"]
            ground_truth = output["completion"]

            correct = prediction == ground_truth
            num_correct += correct
            output["correct"] = correct

        accuracy = num_correct / len(outputs)
        return accuracy, num_correct, modified_outputs


def get_addition_datasets(
    args, tokenizer, k: int = 4, num_operands: int = 2, train_ratio: int | float = 0.9
) -> tuple[Dataset, Dataset]:
    return AdditionDataset(
        args.seed, tokenizer, k=k, num_operands=num_operands
    ).generate_data(train_ratio=train_ratio)


def get_addition_eval_dataset_uniform(
    args, tokenizer, k: int = 4, num_operands: int = 2, num_samples: int = 1000
) -> Dataset:
    return AdditionDataset(
        args.seed, tokenizer, k=k, num_operands=num_operands
    ).generate_eval_data_uniform(num_samples)
