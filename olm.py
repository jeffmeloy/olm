import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Iterator, Optional
import logging
from pathlib import Path
import yaml
import os
import json
import shutil
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s > %(message)s")
logger = logging.getLogger(__name__)


def download_model(model_name: str, models_dir: str) -> Optional[str]:
    """Download model from Hugging Face Hub."""
    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} to {local_path}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                revision="main",
            )
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            return None
    else:
        print(f"Model {model_name} already exists at {local_path}")

    return local_path


def load_model(
    model_path: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model from local path and return model and tokenizer."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def save_best_model(
    model_ranks: dict[str, list[int]],
    layer_metrics: dict[str, dict[str, float]],
    layer_idx: int,
    models_dir: str,
    output_dir: str,
    working_model: AutoModelForCausalLM,
) -> None:
    """Save the best model for the current layer."""
    best_model = min(model_ranks.items(), key=lambda x: sum(x[1]))[0]
    try:
        best_source = load_model(f"{models_dir}/{best_model.replace('/', '_')}", "cpu")[
            0
        ]
        replace_layer(working_model, get_layer(best_source, layer_idx), layer_idx)
        save_layer_state(
            working_model, layer_metrics, layer_idx, output_dir, best_model
        )
        del best_source
        torch.cuda.empty_cache()

        avg_rank = (
            sum(model_ranks[best_model]) / len(model_ranks[best_model])
            if isinstance(model_ranks[best_model], list)
            else model_ranks[best_model]
        )

        logger.info(
            f"Applied layer {layer_idx} from {best_model} (avg rank: {avg_rank:.2f})"
        )
    except Exception as e:
        logger.error(f"Error applying best layer from {best_model}: {e}")
        return


def save_layer_state(
    model: AutoModelForCausalLM,
    results: dict,
    layer_idx: int,
    output_dir: str,
    best_model: str,
) -> None:
    """Save model state and merge report for current layer."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    report = json.load(open(os.path.join(output_dir, "merge_report.json")))
    report["layers"][str(layer_idx)] = {
        "metrics": results,
        "best_model": best_model,
    }
    json.dump(
        report, open(os.path.join(output_dir, "merge_report.json"), "w"), indent=4
    )


def get_context(conversation: dict) -> str:
    """extract dataset context from conversation."""
    context = ""
    for msg in conversation["conversation"]:
        if msg["from"] == "system":
            context += f"<|im_start|>system\n{msg['value']}<|im_end|>\n"
        elif msg["from"] == "human":
            context += f"<|im_start|>user\n{msg['value']}<|im_end|>\n"
        else:
            context += "<|im_start|>assistant\n"

    return context


def compute_response_quality(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    max_length: int = 2048,
) -> float:
    prompt = get_context(conversation)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    ).to(model.device)

    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_length * 2,
        num_return_sequences=1,
        num_beams=1,
        do_sample=False,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens_ids = generated_ids[:, inputs.input_ids.size(1) :]
    response_text = tokenizer.decode(new_tokens_ids[0], skip_special_tokens=True)
    logger.info(f"Response: {response_text}")

    if any("\u4e00" <= char <= "\u9fff" for char in response_text):
        chinese_chars = {char for char in response_text if "\u4e00" <= char <= "\u9fff"}
        for char in chinese_chars:
            logger.info(f"Chinese character detected: {char} (U+{ord(char):04X})")
        logger.info("Chinese characters detected in response.")
        return max_length

    words = response_text.split()
    total_words = len(words)
    if total_words < 100:
        return max_length

    num_words = len(words)
    num_unique_words = len(set(words))
    length_penalty = min(num_words, max_length) / max_length
    repitition_penalty = num_unique_words / num_words
    quality_score = 1 / (length_penalty * repitition_penalty**2)

    logger.info(
        f"Words: {total_words}, Unique: {len(set(words))}, Score: {quality_score:.4f}"
    )

    return quality_score


def compute_response_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
) -> float:
    context = get_context(conversation)
    expected_response = f"{conversation['conversation'][-1]['value']}"

    context_ids = tokenizer(context, return_tensors="pt").to(model.device)
    response_ids = tokenizer(expected_response, return_tensors="pt").to(model.device)
    full_ids = torch.cat([context_ids.input_ids, response_ids.input_ids], dim=1).to(
        model.device
    )
    outputs = model(full_ids)

    shift_logits = outputs.logits[
        :, context_ids.input_ids.size(1) - 1 : -2, :
    ].contiguous()
    next_logits = outputs.logits[:, context_ids.input_ids.size(1) : -1, :].contiguous()
    shift_labels = response_ids.input_ids[:, :-1].contiguous()
    next_labels = response_ids.input_ids[:, 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")
    current_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    next_loss = loss_fct(
        next_logits.view(-1, next_logits.size(-1)), next_labels.view(-1)
    )
    average_loss = torch.mean(current_loss + next_loss)
    return torch.exp(average_loss).item()


def compute_exact_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
) -> float:
    """Compute avg losss for the exact response."""
    context = get_context(conversation)
    expected_response = f"{conversation['conversation'][-1]['value']}"
    context_ids = tokenizer(context, return_tensors="pt").to(model.device)
    expected_ids = tokenizer(expected_response, return_tensors="pt").to(model.device)
    current_ids = context_ids.input_ids

    all_logits = []
    for i in range(len(expected_ids.input_ids[0])):
        outputs = model(current_ids)
        next_token_logits = outputs.logits[:, -1:, :]
        all_logits.append(next_token_logits)

    response_logits = torch.cat(all_logits, dim=1)
    labels = expected_ids.input_ids
    logits_view = response_logits.view(-1, response_logits.size(-1))
    labels_view = labels.view(-1)
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.eos_token_id)
    token_losses = loss_fct(logits_view, labels_view)
    total_loss = token_losses.sum()
    num_tokens = len(token_losses)
    avg_loss = (total_loss / num_tokens).item()

    return avg_loss


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict],
    mode: str,
    samples: int = 200,
) -> float:
    """get the average metric for the model on the dataset."""
    total_metric = 0.0
    max_conversations = min(len(dataset), samples)
    dataset = dataset[:max_conversations]
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.bfloat16)

    for conversation in dataset:
        if mode == "exact":
            metric = compute_exact_match(model, tokenizer, conversation)
        elif mode == "bigram":
            metric = compute_response_perplexity(model, tokenizer, conversation)
        elif mode == "quality":
            metric = compute_response_quality(model, tokenizer, conversation)
        total_metric += metric

    model = model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return total_metric / len(dataset)


def get_layer(model: AutoModelForCausalLM, layer_idx: int):
    return getattr(model.model.layers, str(layer_idx))


def replace_layer(target_model: AutoModelForCausalLM, source_layer, layer_idx: int):
    setattr(target_model.model.layers, str(layer_idx), source_layer)


def layer_sequence(num_layers: int, direction: str) -> Iterator[int]:
    return reversed(range(num_layers)) if direction == "backward" else range(num_layers)


def get_last_layer_idx(merge_report: dict) -> Optional[int]:
    """Get index of most recently merged layer."""
    if not merge_report.get("layers"):
        return None
    return max(int(layer_num) for layer_num in merge_report["layers"].keys())


def get_layer_metrics(merge_report: dict, layer_idx: int) -> dict:
    """Get metrics for specified layer index."""
    return merge_report["layers"][str(layer_idx)]["metrics"]


def compute_model_ranks(
    model_name: str, layer_metrics: dict, valid_models: set
) -> list:
    """Compute rank of model across all datasets."""
    ranks = []
    for dataset_scores in layer_metrics.values():
        valid_scores = {m: s for m, s in dataset_scores.items() if m in valid_models}
        sorted_models = sorted(valid_scores.items(), key=lambda x: x[1])
        rank = next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
        ranks.append(rank)
    return ranks


def compute_layer_ranks(
    layer_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute rank aggregation across datasets for each model."""
    model_names = set().union(*[models.keys() for models in layer_metrics.values()])
    model_ranks = {}
    for model_name in model_names:
        ranks = []
        for dataset_name, dataset_scores in layer_metrics.items():
            sorted_models = sorted(dataset_scores.items(), key=lambda x: x[1])
            rank = (
                next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
            )
            ranks.append(rank)
        model_ranks[model_name] = ranks
    return model_ranks


def get_datasets(config: dict) -> dict:
    """Load datasets from configuration."""
    datasets = {}
    try:
        for dataset_name, dataset_config in config["dataset"].items():
            with open(
                Path(config["dataset_dir"]) / dataset_name, "r", encoding="utf-8"
            ) as f:
                datasets[dataset_name] = {
                    "data": json.load(f),
                    "mode": dataset_config["mode"],
                }
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

    return datasets


def get_base_metrics(
    base_path, output_dir, base_model_name, datasets, working_model, tokenizer
) -> None:
    """Copy base model files and establish baseline metrics."""
    logger.info(f"Copying base model files to {output_dir}")
    for file in os.listdir(base_path):
        try:
            if not file.startswith("."):
                shutil.copy2(
                    os.path.join(base_path, file), os.path.join(output_dir, file)
                )
        except Exception as e:
            logger.error(f"Error copying {file}: {e}")

    logger.info(f"Evaluating {base_model_name} on datasets")
    base_metrics = {}
    for dataset_name, dataset in datasets.items():
        try:
            metric = evaluate_model_on_dataset(
                working_model, tokenizer, dataset["data"], dataset["mode"]
            )
            base_metrics[dataset_name] = metric
            logger.info(f"Base model {dataset_name}: {metric:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating base model on {dataset_name}: {e}")
            base_metrics[dataset_name] = float("inf")

    report_path = os.path.join(output_dir, "merge_report.json")
    report = {}
    report["base_model"] = {
        "name": base_model_name,
        "metrics": base_metrics,
    }
    report["layers"] = {}

    json.dump(report, open(report_path, "w"), indent=4)
    logger.info(f"Base metrics saved to {report_path}")


def select_base_model(models_dir: str, models: list[str], datasets: dict) -> str:
    """Pick best base model using existing ranking functions."""
    layer_metrics = {dataset_name: {} for dataset_name in datasets}

    for model_name in models:
        logger.info(f"Evaluating base candidate: {model_name}")

        try:
            model, tokenizer = load_model(
                Path(models_dir) / model_name.replace("/", "_"), "cpu"
            )
            for dataset_name, dataset in datasets.items():
                # if dataset["mode"] == "quality" then get metric, else remove dataset_name from layer_metrics
                if dataset["mode"] == "quality":
                    metric = evaluate_model_on_dataset(
                        model, tokenizer, dataset["data"], dataset["mode"]
                    )
                    # Store metrics in expected structure
                    layer_metrics[dataset_name][model_name] = metric
                    logger.info(f"{model_name}: {dataset_name}: {metric}")
                else:
                    layer_metrics[dataset_name][model_name] = float("inf")
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    valid_models = set(models)
    model_ranks = {
        model: compute_model_ranks(model, layer_metrics, valid_models)
        for model in valid_models
        if all(model in scores for scores in layer_metrics.values())
    }

    if not model_ranks:
        raise RuntimeError("No valid base models found")

    return min(model_ranks, key=model_ranks.get)


def process_model(
    model_name: str,
    layer_metrics: dict,
    datasets: dict,
    working_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_metrics: dict,
    merge_report: dict,
    layer_idx: int,
    improve_all: str = None,
) -> dict:
    """Process model with proper historical metric comparison."""
    if improve_all:
        prev_metrics = None
        last_idx = get_last_layer_idx(merge_report)
        if last_idx is not None:
            best_model = merge_report["layers"][str(last_idx)]["best_model"]
            prev_metrics = {
                dataset: scores[best_model]
                for dataset, scores in merge_report["layers"][str(last_idx)][
                    "metrics"
                ].items()
            }

    for dataset_name, dataset in datasets.items():
        if dataset_name not in layer_metrics:
            layer_metrics[dataset_name] = {}

        try:
            metric = evaluate_model_on_dataset(
                working_model, tokenizer, dataset["data"], dataset["mode"]
            )
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
            metric = float("inf")

        layer_metrics[dataset_name][model_name] = metric
        logger.info(f"{model_name}: {dataset_name}: {metric}")

        # Only enforce improvement on specified metric type(s)
        if dataset["mode"] == improve_all or improve_all == "all":
            if metric > base_metrics[dataset_name] or (
                prev_metrics and metric > prev_metrics[dataset_name]
            ):
                logger.info("Layer degradation detected, skip layer")
                for remaining in datasets:
                    if remaining not in layer_metrics:
                        layer_metrics[remaining] = {}
                    layer_metrics[remaining][model_name] = float("inf")
                return layer_metrics

    return layer_metrics


def olm(config: dict) -> str:
    models_dir = config["models_dir"]
    output_dir = config["output_dir"]
    datasets = get_datasets(config)
    improve_all = config.get("improve_all", None)

    # download all models to models_dir
    for model_name in config["models"]:
        download_model(model_name, models_dir)

    base_model_name = config.get("base_model_name")
    if not base_model_name:
        base_model_name = select_base_model(models_dir, config["models"], datasets)

    logger.info(f"Selected base model: {base_model_name}")
    base_path = Path(models_dir) / base_model_name.replace("/", "_")
    working_model, tokenizer = load_model(str(base_path), "cpu")
    num_layers = working_model.config.num_hidden_layers

    report_path = os.path.join(output_dir, "merge_report.json")
    get_base_metrics(
        base_path, output_dir, base_model_name, datasets, working_model, tokenizer
    )
    merge_report = json.load(open(report_path))
    base_metrics = merge_report["base_model"]["metrics"]

    for layer_idx in layer_sequence(num_layers, config.get("direction", "forward")):
        merge_report = json.load(open(report_path))
        logger.info(f"\nOptimizing layer {layer_idx}")
        layer_metrics = {dataset: {} for dataset in datasets}

        for model_name in config["models"]:
            logger.info(f"Testing {model_name} layer {layer_idx}")
            try:
                source = load_model(
                    f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                )[0]
                replace_layer(working_model, get_layer(source, layer_idx), layer_idx)
                del source

                layer_metrics = process_model(
                    model_name=model_name,
                    layer_metrics=layer_metrics,
                    datasets=datasets,
                    working_model=working_model,
                    tokenizer=tokenizer,
                    base_metrics=base_metrics,
                    merge_report=merge_report,
                    layer_idx=layer_idx,
                    improve_all=improve_all,
                )

            except Exception as e:
                logger.error(f"Error testing layer from {model_name}: {e}")
                continue

        model_ranks = compute_layer_ranks(layer_metrics)
        if not model_ranks:
            logger.error(
                f"No valid candidates for layer {layer_idx}, skipping optimization"
            )
            continue

        save_best_model(
            model_ranks, layer_metrics, layer_idx, models_dir, output_dir, working_model
        )

    return output_dir


@torch.inference_mode()
def main(config_path: str):
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        output_path = olm(config)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error during merge: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="olm_config.yaml")
    args = parser.parse_args()
    main(args.config)
