import os
import re
import json
import math
import yaml
import torch
import shutil
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Optional, Tuple
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s > %(message)s")
logger = logging.getLogger(__name__)

# --- Model and Dataset Handling ---


def get_model_from_merge_report(
    merge_report_path: str,
    models_dir: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Reconstruct a model from a merge report by loading the base model and swapping in specified layers."""
    with open(merge_report_path) as f:
        merge_report = json.load(f)

    base_model_name = merge_report["base_model"]["name"]
    base_model_path = Path(models_dir) / base_model_name.replace("/", "_")
    logger.info(f"Loading base model {base_model_name}")
    model, tokenizer = load_model(str(base_model_path), "cpu")

    if "boundary_layers" in merge_report and merge_report["boundary_layers"]["name"]:
        logger.info(
            f"Applying boundary layers from {merge_report['boundary_layers']['name']}"
        )
        boundary_source_path = Path(models_dir) / merge_report["boundary_layers"][
            "name"
        ].replace("/", "_")
        boundary_source, _ = load_model(str(boundary_source_path), "cpu")

        model.model.embed_tokens.load_state_dict(
            boundary_source.model.embed_tokens.state_dict()
        )
        model.model.norm.load_state_dict(boundary_source.model.norm.state_dict())
        model.lm_head.load_state_dict(boundary_source.lm_head.state_dict())
        del boundary_source

    if "layers" in merge_report:
        for layer_idx, layer_info in merge_report["layers"].items():
            layer_source_model = layer_info.get("best_model")
            if layer_source_model:
                logger.info(f"Applying layer {layer_idx} from {layer_source_model}")
                try:
                    layer_source_path = Path(models_dir) / layer_source_model.replace(
                        "/", "_"
                    )
                    layer_source, _ = load_model(str(layer_source_path), "cpu")
                    layer_idx = int(layer_idx)
                    replace_layer(model, get_layer(layer_source, layer_idx), layer_idx)
                    del layer_source
                except Exception as e:
                    logger.error(
                        f"Error applying layer {layer_idx} from {layer_source_model}: {e}"
                    )
                    logger.info("Keeping base model layer")

    return model, tokenizer


def download_model(model_name: str, models_dir: str) -> Optional[str]:
    """Download model from Hugging Face Hub."""
    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if not os.path.exists(local_path):
        logger.info(f"Downloading {model_name} to {local_path}")
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
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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


# --- Evaluation Metrics ---


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


@torch.no_grad()
def compute_response_quality(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    max_length: int = 4096,
    cache: Optional[dict] = None,
) -> float:
    """Compute the quality of a generated response."""
    if cache:
        # Use cached context
        context_ids = cache["context_ids"].to(model.device)
        inputs = context_ids
    else:
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

    new_tokens_ids = generated_ids[0, inputs.input_ids.size(1) :]
    response_text = tokenizer.decode(new_tokens_ids, skip_special_tokens=True)
    logger.info(f"Long Response: {response_text}")

    if any("\u4e00" <= char <= "\u9fff" for char in response_text):
        chinese_chars = {char for char in response_text if "\u4e00" <= char <= "\u9fff"}
        for char in chinese_chars:
            logger.info(f"Chinese character detected: {char} (U+{ord(char):04X})")
        return max_length

    words = response_text.split()
    total_words = len(words)
    if total_words < 100:
        return max_length

    if max(len(word) for word in words) > 20:
        return max_length

    num_words = len(words)
    num_unique_words = len(set(words))
    length_penalty = min(num_words, max_length) / max_length
    repitition_penalty = num_unique_words / num_words
    quality_score = 1 / (length_penalty * repitition_penalty**2)

    # logger.info(
    #    f"Words: {total_words}, Unique: {len(set(words))}, Score: {quality_score}"
    # )

    return quality_score


@torch.no_grad()
def compute_uncertain_bigram(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    cache: Optional[dict] = None,
) -> float:
    """Compute bigram loss that rewards correct responses and distribution preservation."""
    if cache:
        context_ids = cache["context_ids"].to(model.device)
        response_ids = cache["response_ids"].to(model.device)
        full_ids = cache["full_ids"].to(model.device)
        context_end_idx = cache["context_end_idx"]
    else:
        context = get_context(conversation)
        expected_response = f"{conversation['conversation'][-1]['value']}"
        context_ids = tokenizer(context, return_tensors="pt").to(model.device)
        response_ids = tokenizer(expected_response, return_tensors="pt").to(
            model.device
        )
        full_ids = torch.cat([context_ids.input_ids, response_ids.input_ids], dim=1).to(
            model.device
        )
        context_end_idx = context_ids.input_ids.size(1)

    outputs = model(full_ids)
    shift_logits = outputs.logits[:, context_end_idx - 1 : -2, :]
    next_logits = outputs.logits[:, context_end_idx:-1, :]
    shift_probs = torch.softmax(shift_logits, dim=-1)
    next_probs = torch.softmax(next_logits, dim=-1)
    shift_labels = response_ids.input_ids[:, :-1]
    next_labels = response_ids.input_ids[:, 1:]
    shift_correct = torch.gather(shift_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    next_correct = torch.gather(next_probs, 2, next_labels.unsqueeze(-1)).squeeze(-1)
    shift_max, _ = torch.max(shift_probs, dim=-1)
    next_max, _ = torch.max(next_probs, dim=-1)
    shift_wrong = shift_correct < shift_max
    next_wrong = next_correct < next_max
    shift_score = torch.where(shift_wrong, 1.0 + shift_max, shift_max)
    next_score = torch.where(next_wrong, 1.0 + next_max, next_max)
    combined_score = (shift_score + next_score).mean().item()
    return combined_score


@torch.no_grad()
def compute_exact_uncertain_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    cache: Optional[dict] = None,
) -> float:
    """Lower scores are better: (correct + uncertain) < (correct + certain) < (incorrect uncertain) < (incorrect certain)"""
    if cache:
        context_ids = cache["context_ids"].to(model.device)
        expected_ids = cache["response_ids"].to(model.device)
    else:
        context = get_context(conversation)
        expected_response = f"{conversation['conversation'][-1]['value']}"
        context_ids = tokenizer(context, return_tensors="pt").to(model.device)
        expected_ids = tokenizer(expected_response, return_tensors="pt").to(
            model.device
        )

    current_ids = context_ids.input_ids
    all_logits = []

    for i in range(len(expected_ids.input_ids[0])):
        outputs = model(current_ids)
        all_logits.append(outputs.logits[:, -1:, :].detach())

    response_logits = torch.cat(all_logits, dim=1)
    predicted_tokens = response_logits.argmax(dim=-1)
    correct_mask = predicted_tokens == expected_ids.input_ids
    probs = torch.softmax(response_logits, dim=-1)

    max_probs = probs.max(dim=-1).values
    base_incorrect_penalty = 1.0
    result = torch.where(correct_mask, max_probs, base_incorrect_penalty + max_probs)
    return float(result.mean().item())


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict],
    mode: str,
    samples: int = 200,
    cache: Optional[dict] = None,
) -> float:
    """Evaluate a model on a dataset using a specified evaluation mode."""
    total_metric = 0.0
    max_conversations = min(len(dataset), samples)
    dataset = dataset[:max_conversations]
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model.device.type != device:
        model.to(device, dtype=torch.bfloat16)

    for idx, conversation in enumerate(dataset):
        # Extract conversation-specific cache if available
        conv_cache = cache.get(idx) if cache else None

        if mode == "quality":
            metric = compute_response_quality(
                model, tokenizer, conversation, 4096, conv_cache
            )
        elif mode == "uncertain_bigram":
            metric = compute_uncertain_bigram(
                model, tokenizer, conversation, conv_cache
            )
        elif mode == "exact_uncertain":
            metric = compute_exact_uncertain_match(
                model, tokenizer, conversation, conv_cache
            )
        total_metric += metric

    model.to("cpu")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return total_metric / len(dataset)


# --- Layer Operations ---


def get_layer(model: AutoModelForCausalLM, layer_idx: int) -> torch.nn.Module:
    """Retrieve a specific layer from a model."""
    return model.model.layers[layer_idx]


def replace_layer(
    target_model: AutoModelForCausalLM, source_layer: torch.nn.Module, layer_idx: int
) -> None:
    """Replace a layer in the target model with a layer from a source model."""
    target_model.model.layers[layer_idx].load_state_dict(source_layer.state_dict())


def layer_sequence(num_layers: int, direction: str) -> Iterator[int]:
    """Generate a sequence of layer indices."""
    return reversed(range(num_layers)) if direction == "backward" else range(num_layers)


def get_last_layer_idx(merge_report: dict) -> Optional[int]:
    """Get the index of the last optimized layer from the merge report."""
    if not merge_report.get("layers"):
        return None
    return max(int(layer_num) for layer_num in merge_report["layers"].keys())


def get_layer_metrics(merge_report: dict, layer_idx: int) -> dict:
    """Retrieve the performance metrics for a specific layer from the merge report."""
    return merge_report["layers"][str(layer_idx)]["metrics"]


# --- Merge Report, Ranking and Selection ---


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
    report_path = os.path.join(output_dir, "merge_report.json")

    if os.path.exists(report_path):
        report = json.load(open(report_path))
    else:
        report = {}

    if "layers" not in report:
        report["layers"] = {}

    report["layers"][str(layer_idx)] = {
        "metrics": results,
        "best_model": best_model,
    }
    json.dump(report, open(report_path, "w"), indent=4)


def compute_model_ranks(
    model_name: str, layer_metrics: dict, valid_models: list[str]
) -> list[int]:
    """Compute the ranks of a model across datasets for a given layer."""
    ranks = []
    for dataset_scores in layer_metrics.values():
        valid_scores = {m: s for m, s in dataset_scores.items() if m in valid_models}
        sorted_models = sorted(valid_scores.items(), key=lambda x: x[1])
        rank = next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
        ranks.append(rank)
    return ranks


def compute_layer_ranks(
    layer_metrics: dict[str, dict[str, float]],
) -> dict[str, list[int]]:
    """Compute rank aggregation across datasets for each model."""
    model_names = set().union(*[models.keys() for models in layer_metrics.values()])
    model_ranks = {}
    for model_name in model_names:
        ranks = []
        for dataset_name, dataset_scores in layer_metrics.items():
            if model_name not in dataset_scores:
                continue
            sorted_models = sorted(dataset_scores.items(), key=lambda x: x[1])
            rank = (
                next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
            )
            ranks.append(rank)
        model_ranks[model_name] = ranks
    return model_ranks


def select_best_model(
    model_ranks: dict[str, list[int]],
    layer_metrics: dict[str, dict[str, float]] = None,
    selection: str = "ranks",
) -> str:
    """Select the best model based on all rank aggregation"""
    return min(model_ranks.items(), key=lambda x: sum(x[1]))[0]


def calculate_normalized_effective_rank(tensor: torch.Tensor) -> float:
    """Calculate the Normalized Effective Rank (NER) of a matrix using PyTorch."""
    try:
        # Get device and prepare tensor in one step
        device = "cuda" if torch.cuda.is_available() else "cpu"
        A = tensor.to(device=device, dtype=torch.float32)
        A /= max(A.abs().max().item(), 1e-10)  # Normalize tensor

        if A.dim() == 1:
            A = A.unsqueeze(0)
        if 1 in A.shape:
            S = A.abs().view(-1)
        else:
            S = torch.linalg.svdvals(A)

        # Filter near-zero values and compute entropy
        mask = S > 1e-10
        if not mask.any():
            return 1.0

        S = S[mask]
        s_sum = S.sum()
        S.div_(s_sum)  # Normalize in-place

        # Compute entropy and NER
        H = -(S * torch.log2(S)).sum()
        H_max = torch.log2(torch.tensor(S.numel(), dtype=torch.float32, device=device))
        ner = 1.0 if H_max <= 0 else (1 - (H / H_max)).item()

        del A, S, H
        if device == "cuda":
            torch.cuda.empty_cache()
        return ner
    except Exception as e:
        logger.error(f"Error calculating NER: {e}")
        return 1.0


# --- Base Model Handling and Initialization ---


def _evaluate_model_candidates(
    models: list[str],
    models_dir: str,
    datasets: dict,
    eval_func,
    *args,
    **kwargs,
) -> dict:
    """Helper function to evaluate multiple model candidates."""
    layer_metrics = {dataset_name: {} for dataset_name in datasets}
    for model_name in models:
        logger.info(f"Evaluating candidate: {model_name}")
        try:
            model, tokenizer = load_model(
                Path(models_dir) / model_name.replace("/", "_"), "cpu"
            )
            for dataset_name, dataset in datasets.items():
                metric = eval_func(model, tokenizer, dataset, *args, **kwargs)
                layer_metrics[dataset_name][model_name] = metric
                logger.info(f"{model_name}: {dataset_name}: {metric}")
            del model
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    return layer_metrics


def get_base_metrics(
    base_path: Path,
    output_dir: str,
    base_model_name: str,
    datasets: dict,
    working_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_cache: dict = None,
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
            # Use evaluation cache if available
            dataset_cache = (
                evaluation_cache.get(dataset_name) if evaluation_cache else None
            )

            metric = evaluate_model_on_dataset(
                working_model,
                tokenizer,
                dataset["data"],
                dataset["mode"],
                cache=dataset_cache,
            )
            base_metrics[dataset_name] = metric
            logger.info(f"Base model {dataset_name}: {metric}")
        except Exception as e:
            logger.error(f"Error evaluating base model on {dataset_name}: {e}")
            base_metrics[dataset_name] = float("inf")

    report_path = os.path.join(output_dir, "merge_report.json")
    report = {}
    report["base_model"] = {
        "name": base_model_name,
        "metrics": base_metrics,
    }

    json.dump(report, open(report_path, "w"), indent=4)
    logger.info(f"Base metrics saved to {report_path}")


# --- Processing and Degradation Handling ---


def get_ner_metrics(working_model, layer_idx, model_name, layer_metrics):
    """Calculate NER for the current model while it's already loaded."""
    if "ner" not in layer_metrics:
        layer_metrics["ner"] = {}
    if isinstance(layer_idx, int):  # For layer-wise optimization
        layer = get_layer(working_model, layer_idx)
        weight_matrices = [
            p for n, p in layer.named_parameters() if "weight" in n and p.dim() > 1
        ]
        ner_values = [calculate_normalized_effective_rank(w) for w in weight_matrices]
        ner = sum(ner_values) / len(ner_values)
    else:  # For tensor-wise optimization
        tensor = dict(working_model.named_parameters())[layer_idx]
        ner = calculate_normalized_effective_rank(tensor)
    layer_metrics["ner"][model_name] = ner
    logger.info(f"{model_name}: NER: {ner}")
    return layer_metrics


def process_model(
    model_name: str,
    layer_metrics: dict,
    datasets: dict,
    working_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_metrics: dict,
    merge_report: dict,
    layer_idx: int,
    samples: int = 200,
    improve_all: str = None,
    evaluation_cache: Optional[dict] = None,
) -> dict:
    """Process a candidate model for a layer, evaluate performance, and handle degradation."""
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

    # Evaluate model against each dataset
    for dataset_name, dataset in datasets.items():
        if dataset_name not in layer_metrics:
            layer_metrics[dataset_name] = {}

        try:
            # Get dataset-specific cache if available
            dataset_cache = None
            if evaluation_cache and dataset_name in evaluation_cache:
                dataset_cache = evaluation_cache[dataset_name]

            # Run the evaluation
            metric = evaluate_model_on_dataset(
                working_model,
                tokenizer,
                dataset["data"],
                dataset["mode"],
                samples,
                cache=dataset_cache,
            )
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
            metric = float("inf")

        # Store the metric result
        layer_metrics[dataset_name][model_name] = metric
        logger.info(f"{model_name}: {dataset_name}: {metric}")

        # Check for degradation based on improve_all setting
        should_skip = False
        if dataset["mode"] == improve_all or improve_all == "all":
            if metric > base_metrics[dataset_name] or (
                prev_metrics
                and dataset_name in prev_metrics
                and metric > prev_metrics[dataset_name]
            ):
                logger.info(
                    f"Layer degradation on {dataset_name}, skipping {model_name}"
                )
                should_skip = True

        elif improve_all == "base":
            if metric > base_metrics[dataset_name]:
                should_skip = True

        # If degradation detected, mark all remaining datasets as inf and bail
        if should_skip:
            logger.info(f"DEGRADATION DETECTED for {model_name} on {dataset_name}")
            for remaining in datasets:
                if remaining not in layer_metrics:
                    layer_metrics[remaining] = {}
                layer_metrics[remaining][model_name] = float("inf")

            # Just set NER to inf for failed models - no need to calculate
            if "ner" not in layer_metrics:
                layer_metrics["ner"] = {}
            layer_metrics["ner"][model_name] = float("inf")
            return layer_metrics

    # Model passed all dataset checks - calculate NER
    layer_metrics = get_ner_metrics(working_model, layer_idx, model_name, layer_metrics)

    return layer_metrics


# --- Boundary Layer Optimization ---


def _apply_boundary_components(
    working_model: AutoModelForCausalLM, source_model: AutoModelForCausalLM
) -> None:
    """Helper Function to apply boundary components to the working model"""
    working_model.model.embed_tokens.load_state_dict(
        source_model.model.embed_tokens.state_dict()
    )
    working_model.model.norm.load_state_dict(source_model.model.norm.state_dict())
    working_model.lm_head.load_state_dict(source_model.lm_head.state_dict())


# --- layer and tensor selection ---


def replace_sublayer(
    target_model: AutoModelForCausalLM,
    source_sublayer: torch.nn.Module,
    layer_idx: int,
    sublayer_name: str,
) -> None:
    """Replace a specific sublayer in the target model with a sublayer from the source model."""
    target_layer = target_model.model.layers[layer_idx]
    target_sublayer = getattr(target_layer, sublayer_name)
    target_sublayer.load_state_dict(source_sublayer.state_dict())


def save_tensor_state(
    model: AutoModelForCausalLM,
    tensor_metrics: dict,
    tensor_name: str,
    output_dir: str,
    best_model: str,
) -> None:
    """Save model state and merge report for tensor-wise optimization."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    report_path = os.path.join(output_dir, "merge_report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    if "tensors" not in report:
        report["tensors"] = {}
    report["tensors"][tensor_name] = {
        "metrics": tensor_metrics,
        "best_model": best_model,
    }

    json.dump(report, open(report_path, "w"), indent=4)


def _get_target_parameter_units(model: AutoModelForCausalLM) -> dict[str, list[str]]:
    """Groups parameter names by their functional unit (e.g., q_proj, mlp.gate_proj, input_layernorm)."""
    units = defaultdict(list)
    pattern = re.compile(r"^(.*?)\.(weight|bias)$")

    for name, param in model.named_parameters():
        if not param.requires_grad:  # Skip non-trainable
            continue

        match = pattern.match(name)
        if match:
            unit_name = match.group(1)
            units[unit_name].append(name)

    final_units = {}
    for unit_name, param_names in units.items():
        if param_names:
            final_units[unit_name] = param_names

    logger.info(f"Identified {len(final_units)} parameter units for optimization.")
    return final_units


def select_component(
    component_type: str,  # "base", "boundary", or "layer"
    models_dir: str,
    model_pool: list[str],
    datasets: dict,
    config: dict,
    working_model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    evaluation_cache: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Unified selection for any model component (base, boundary layers, or regular layers)."""
    logger.info(f"Selecting optimal {component_type}...")

    selection_metric = config.get(f"{component_type}_select", "uncertain_bigram")
    selected_datasets = {
        name: dataset
        for name, dataset in datasets.items()
        if dataset["mode"] == selection_metric or selection_metric == "all"
    }

    if not selected_datasets:
        selected_datasets = datasets  # Fallback to all datasets

    # For base model evaluate each model directly
    if component_type == "base":
        component_metrics = _evaluate_model_candidates(
            model_pool,
            models_dir,
            selected_datasets,
            lambda model, tokenizer, dataset: evaluate_model_on_dataset(
                model,
                tokenizer,
                dataset["data"],
                dataset["mode"],
                cache=evaluation_cache.get(dataset["name"])
                if evaluation_cache
                else None,
            ),
        )

    # For boundary layers swap them into the working model and evaluate
    elif component_type == "boundary":
        component_metrics = {dataset_name: {} for dataset_name in selected_datasets}
        original_states = {}

        # Backup original states if we're modifying a working model
        if working_model:
            original_states = {
                "embed_tokens": working_model.model.embed_tokens.state_dict(),
                "norm": working_model.model.norm.state_dict(),
                "lm_head": working_model.lm_head.state_dict(),
            }

        for model_name in model_pool:
            try:
                if working_model:  # Apply boundary components to working model
                    source, _ = load_model(
                        f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                    )
                    _apply_boundary_components(working_model, source)
                    test_model, test_tokenizer = working_model, tokenizer
                    del source
                else:  # load the model directly if no working model
                    test_model, test_tokenizer = load_model(
                        f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                    )

                # Evaluate on selected datasets
                for dataset_name, dataset in selected_datasets.items():
                    # Use cache if available
                    dataset_cache = (
                        evaluation_cache.get(dataset_name) if evaluation_cache else None
                    )

                    metric = evaluate_model_on_dataset(
                        test_model,
                        test_tokenizer,
                        dataset["data"],
                        dataset["mode"],
                        cache=dataset_cache,
                    )
                    component_metrics[dataset_name][model_name] = metric
                    logger.info(f"{model_name}: {dataset_name}: {metric}")

                # Restore original boundary layers
                if working_model and original_states:
                    for component_name, state in original_states.items():
                        getattr(
                            working_model.model
                            if component_name != "lm_head"
                            else working_model,
                            component_name,
                        ).load_state_dict(state)

                if component_type == "base" and test_model != working_model:
                    del test_model

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

    # Calculate ranks and select best model
    valid_models = set(
        model_name
        for model_name in model_pool
        if any(model_name in metrics for metrics in component_metrics.values())
    )

    model_ranks = {}
    for model_name in valid_models:
        ranks = []
        for dataset_name, dataset_scores in component_metrics.items():
            if model_name not in dataset_scores:
                continue
            sorted_models = sorted(dataset_scores.items(), key=lambda x: x[1])
            rank = (
                next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
            )
            ranks.append(rank)
        if ranks:
            model_ranks[model_name] = ranks

    if not model_ranks:
        raise RuntimeError(f"No valid models found for {component_type} selection")

    best_model = min(model_ranks.items(), key=lambda x: sum(x[1]))[0]
    avg_rank = sum(model_ranks[best_model]) / len(model_ranks[best_model])
    logger.info(f"Selected {component_type}: {best_model} (avg rank: {avg_rank})")

    return best_model, component_metrics


def layer_wise_optimization(
    working_model: AutoModelForCausalLM,
    model_pool: list[str],
    datasets: dict,
    tokenizer: AutoTokenizer,
    config: dict,
    models_dir: str,
    output_dir: str,
    base_metrics: dict,
    merge_report: dict,
    evaluation_cache: Optional[dict] = None,
) -> AutoModelForCausalLM:
    """Perform layer-wise optimization. Neural network surgery, one layer at a time."""
    num_layers = working_model.config.num_hidden_layers
    direction = config.get("direction", "forward")

    for layer_idx in layer_sequence(num_layers, direction):
        logger.info(f"\nOptimizing layer {layer_idx}")
        layer_metrics = {dataset: {} for dataset in datasets}

        for model_name in model_pool:
            try:
                source, _ = load_model(
                    f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                )
                replace_layer(working_model, get_layer(source, layer_idx), layer_idx)
                del source
                torch.cuda.empty_cache()

                layer_metrics = process_model(
                    model_name=model_name,
                    layer_metrics=layer_metrics,
                    datasets=datasets,
                    working_model=working_model,
                    tokenizer=tokenizer,
                    base_metrics=base_metrics,
                    merge_report=merge_report,
                    layer_idx=layer_idx,
                    improve_all=config.get("improve_all"),
                    evaluation_cache=evaluation_cache,
                )
            except Exception as e:
                logger.error(f"Layer optimization failed for {model_name}: {e}")
                continue

        model_ranks = compute_layer_ranks(layer_metrics)
        if not model_ranks:
            logger.error(f"No valid layer candidates for layer {layer_idx}")
            continue

        selection = config.get("selection", "ranks")
        best_model = select_best_model(model_ranks, selection=selection)

        try:
            best_source, _ = load_model(
                f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
            )
            working_model.model.layers[layer_idx].load_state_dict(
                get_layer(best_source, layer_idx).state_dict()
            )
            save_layer_state(
                working_model, layer_metrics, layer_idx, output_dir, best_model
            )
            del best_source

            avg_rank = (
                sum(model_ranks[best_model]) / len(model_ranks[best_model])
                if isinstance(model_ranks[best_model], list)
                else model_ranks[best_model]
            )
            logger.info(
                f"Applied layer {layer_idx} from {best_model} (avg rank: {avg_rank})"
            )

            working_model.save_pretrained(
                output_dir, safe_serialization=True, push_to_hub=False, exist_ok=True
            )
        except Exception as e:
            logger.error(f"Failed to apply best layer from {best_model}: {e}")
            continue

    return working_model


def tensor_wise_optimization(
    working_model: AutoModelForCausalLM,
    model_pool: list[str],
    datasets: dict,
    tokenizer: AutoTokenizer,
    config: dict,
    models_dir: str,
    output_dir: str,
    base_metrics: dict,
    merge_report: dict,
    evaluation_cache: Optional[dict] = None,
) -> AutoModelForCausalLM:
    """Unit-wise optimization: functional parameters treated as coherent units."""
    logger.info("Starting unit-wise optimization...")
    parameter_units = _get_target_parameter_units(working_model)
    unit_names = list(parameter_units.keys())

    # Reverse if needed
    if config.get("direction", "forward") == "backward":
        unit_names.reverse()

    skip_norm = config.get("skip_norm", False)
    working_params_dict = dict(working_model.named_parameters())

    for unit_name in tqdm(unit_names, desc="Optimizing Units"):
        # Skip norm layers if configured
        if skip_norm and any("norm" in part for part in unit_name.split(".")):
            continue

        logger.info(f"\n--- Optimizing: {unit_name} ---")
        unit_param_names = parameter_units[unit_name]
        unit_metrics = {dataset_name: {} for dataset_name in datasets}

        # Get representative weight tensor for NER
        weight_tensor_name = next(
            (p for p in unit_param_names if "weight" in p),
            next(iter(unit_param_names), None),
        )
        if not weight_tensor_name:
            continue

        # Save originals
        orig_states = {
            name: param.data.clone()
            for name, param in working_params_dict.items()
            if name in unit_param_names
        }
        if not orig_states:
            continue

        # Test each candidate
        best_model = config.get("base_model", None)
        if config.get("tensor_swap", True):
            for model_name in model_pool:
                try:
                    # Load and apply candidate
                    candidate, _ = load_model(
                        f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                    )
                    candidate_params = dict(candidate.named_parameters())

                    # Apply unit params
                    for param_name in unit_param_names:
                        if (
                            param_name in working_params_dict
                            and param_name in candidate_params
                        ):
                            working_params_dict[param_name].data.copy_(
                                candidate_params[param_name].data
                            )

                    # Evaluate
                    unit_metrics = process_model(
                        model_name=model_name,
                        layer_metrics=unit_metrics,
                        datasets=datasets,
                        working_model=working_model,
                        tokenizer=tokenizer,
                        base_metrics=base_metrics,
                        merge_report=merge_report,
                        layer_idx=weight_tensor_name,
                        improve_all=config.get("improve_all"),
                        evaluation_cache=evaluation_cache,
                    )
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {unit_name}: {e}")
                    # Mark as bad
                    for ds in unit_metrics:
                        unit_metrics[ds][model_name] = float("inf")
                finally:
                    # Restore original state
                    for name, data in orig_states.items():
                        if name in working_params_dict:
                            working_params_dict[name].data.copy_(data)

                    if "candidate" in locals():
                        del candidate
                        torch.cuda.empty_cache()

            # Find best model
            model_ranks = compute_layer_ranks(unit_metrics)
            if not model_ranks:
                continue

            best_model = select_best_model(
                model_ranks, selection=config.get("selection", "ranks")
            )

        # Apply best model's unit params permanently
        best_source, _ = load_model(
            f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
        )
        best_params = dict(best_source.named_parameters())

        applied = 0
        for param_name in unit_param_names:
            if param_name in working_params_dict and param_name in best_params:
                working_params_dict[param_name].data.copy_(best_params[param_name].data)
                applied += 1

        best_metrics = base_metrics.copy()
        for dataset_name, dataset in best_metrics.items():
            score = unit_metrics[dataset_name][best_model]
            if score == float("inf"):
                score = base_metrics[dataset_name]
            best_metrics[dataset_name] = score

        if config.get("es_swap", False):
            updated_metrics = fine_tune_tensor_1plambdaES(
                working_model,
                weight_tensor_name,
                datasets,
                tokenizer,
                evaluation_cache,
                best_metrics,
                samples=config.get("samples", 200),
                generations=config.get("es_generations", 1),
                initial_sigma=config.get("es_sigma", 0.5),
            )

        logger.info(f"Unit {unit_name}: updated metrics: {updated_metrics}")
        for dataset_name in datasets:
            unit_metrics[dataset_name][best_model] = updated_metrics[dataset_name]

        save_tensor_state(
            working_model,
            unit_metrics,
            weight_tensor_name,
            output_dir,
            best_model,
        )
        logger.info(f"Unit {unit_name}: applied {applied} params from {best_model}")

        del best_source
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Checkpoint
        working_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            push_to_hub=False,
            exist_ok=True,
        )

    return working_model


# --- Evolutionary based fine-tuning ---


@torch.no_grad()
def calculate_full_svd(
    tensor: torch.Tensor, epsilon: float = 1e-10
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Calculate the full SVD of a matrix after normalization."""

    # Early exit for invalid dimensions
    if tensor.dim() == 1 or 1 in tensor.shape:
        return None, None, None

    # compute and returnSVD
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        A = tensor.to(device=device, dtype=torch.float32)
        max_val = A.abs().max().item()
        if max_val > epsilon:
            A /= max_val
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            mask = S > epsilon
            if mask.any():
                return U[:, mask].to("cpu"), S[mask].to("cpu"), Vh[mask, :].to("cpu")

    except Exception as e:
        logger.warning(f"SVD computation failed: {e}")

    return None, None, None


@torch.no_grad()
def generate_noise_from_svd_coeffs(
    original_tensor_shape: torch.Size,
    U: Optional[torch.Tensor],
    S_original: torch.Tensor,
    Vh: Optional[torch.Tensor],
    target_s_indices: torch.Tensor,
    perturbation_coeffs_on_target_s: torch.Tensor,
    epsilon: float = 1e-10,
    noise_clamp_value: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate multiplicative noise tensor from SVD coefficient perturbations."""

    # Get target subset values and prepare scaled perturbations
    device = S_original.device
    S_target_subset = S_original.index_select(0, target_s_indices)
    max_s = S_target_subset.abs().max()

    # Calculate scaling factors and normalize in-place
    if max_s > epsilon:
        scaling = torch.sqrt(max_s / (S_target_subset.abs() + epsilon))
        scaling /= scaling.mean()
        scaled_perturb = perturbation_coeffs_on_target_s * scaling
    else:
        scaled_perturb = perturbation_coeffs_on_target_s

    # Create full dimension perturbation tensor and apply to target indices
    s_additive = torch.zeros_like(S_original)
    s_additive.index_copy_(0, target_s_indices, scaled_perturb)

    # Reconstruct the additive noise through SVD components
    additive_noise = (
        torch.matmul(torch.matmul(U, torch.diag(s_additive)), Vh)
        .reshape(original_tensor_shape)
        .flatten()
    )

    # Generate multiplicative noise by exponentiating clamped additive noise
    additive_noise.clamp_(-noise_clamp_value, noise_clamp_value)
    multiplicative_noise = torch.exp(additive_noise)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return multiplicative_noise, additive_noise


@torch.no_grad()
def fine_tune_tensor_1plambdaES(
    working_model: AutoModelForCausalLM,
    tensor_name: str,
    datasets: dict[str, dict[str, any]],
    tokenizer: AutoTokenizer,
    evaluation_cache: dict[str, dict[int, dict[str, any]]],
    baseline_metrics: dict[str, float],
    samples: int = 200,
    generations: int = 1,
    initial_sigma: float = 0.5,
    noise_clamp_val: float = 3.0,
    epsilon: float = 1e-10,
) -> dict[str, float]:
    """Fine-tunes a tensor using an Estimation of Distribution Algorithm (EDA) approach."""
    # Quick validation
    params = dict(working_model.named_parameters())
    if tensor_name not in params or not torch.is_floating_point(
        params[tensor_name].data
    ):
        logger.warning(f"Tensor {tensor_name} not found or not float. Skipping.")
        return baseline_metrics

    original_model_tensor = params[tensor_name]
    if original_model_tensor.dim() == 1 or 1 in original_model_tensor.shape:
        logger.info(
            f"Tensor {tensor_name} is 1D (shape: {original_model_tensor.shape}). SVD opt skipped."
        )
        return baseline_metrics

    # Setup
    original_tensor_dtype = original_model_tensor.dtype
    original_tensor_shape = original_model_tensor.shape
    compute_device = original_model_tensor.device
    current_best_flat = (
        original_model_tensor.data.to(compute_device, dtype=torch.float32)
        .flatten()
        .clone()
    )
    current_best_metrics = baseline_metrics.copy()

    # SVD decomposition
    U_svd, S_svd, V_svd = calculate_full_svd(
        current_best_flat.reshape(original_tensor_shape), epsilon
    )
    if U_svd is None or S_svd is None or V_svd is None or S_svd.numel() == 0:
        logger.warning(
            f"SVD failed for tensor {tensor_name} (shape {original_tensor_shape}). Skipping."
        )
        return baseline_metrics

    # Initialize parameters
    dim_s = S_svd.numel()
    target_s_indices = torch.arange(dim_s, device=compute_device)
    sigma_s = initial_sigma * max(S_svd.std().item(), epsilon)

    # Standard CME-ES parameter
    _lambda = max(4, int(4 + 3 * math.log(dim_s)))

    # Log initial stats
    tensor_stats = original_model_tensor.data.float()
    logger.info(
        f"Tensor {tensor_name} initial stats: mean={tensor_stats.mean().item()}, std={tensor_stats.std().item()}, "
        f"min={tensor_stats.min().item()}, max={tensor_stats.max().item()}"
    )
    logger.info(
        f"Optimizing {tensor_name} (SVD dim: {dim_s}) with pop: {_lambda}, for {generations} generations. Init sigma: {sigma_s}"
    )

    # Preallocate buffers
    delta_s_batch = torch.empty(
        (_lambda, dim_s), device=compute_device, dtype=torch.float32
    )
    avg_rel_changes = torch.full(
        (_lambda,), float("inf"), device=compute_device, dtype=torch.float32
    )

    offspring_metrics_list = [{} for _ in range(_lambda)]

    for gen in range(generations):
        genstring = f"  Generation {gen + 1}/{generations}"
        logger.info(f"\n--- {genstring}, Tensor: {tensor_name}, Sigma: {sigma_s} ---")

        # Generate random perturbations and scale by sigma
        delta_s_batch.normal_(0, 1)
        delta_s_batch.mul_(sigma_s)

        # Create and evaluate offsprings that apply noise perturbations
        avg_rel_changes.fill_(float("inf"))
        for i in range(_lambda):
            current_eval = f"{genstring} Offspring {i + 1}/{_lambda}:"

            # Generate and apply noise
            noise_mult, _ = generate_noise_from_svd_coeffs(
                original_tensor_shape,
                U_svd,
                S_svd,
                V_svd,
                target_s_indices,
                delta_s_batch[i],
                epsilon,
                noise_clamp_val,
            )

            # Apply noise directly to model tensor
            original_model_tensor.data.copy_(
                (current_best_flat * noise_mult)
                .reshape(original_tensor_shape)
                .to(original_tensor_dtype)
            )
            # Log noise stats
            logger.info(
                f"{current_eval}Noise: mean={noise_mult.mean().item()}, std={noise_mult.std().item()}, "
                f"min={noise_mult.min().item()}, max={noise_mult.max().item()}"
            )

            # Evaluate candidate
            metrics, valid = {}, True
            for ds_name, ds_conf in datasets.items():
                try:
                    eval_samples = min(samples, len(ds_conf["data"]))
                    if eval_samples > 0:
                        ds_cache = (
                            evaluation_cache.get(ds_name) if evaluation_cache else None
                        )
                        metrics[ds_name] = evaluate_model_on_dataset(
                            working_model,
                            tokenizer,
                            ds_conf["data"][:eval_samples],
                            ds_conf["mode"],
                            eval_samples,
                            cache=ds_cache,
                        )
                except Exception as e:
                    valid = False
                    logger.warning(f"{current_eval} fail on {ds_name}: {e}")
                    return baseline_metrics

            # compute relative changes and update offspring metrics
            if valid and metrics:
                offspring_metrics_list[i] = metrics.copy()
                logger.info(f"{current_eval} Metrics: {metrics}")

        # Compute ranks for each offspring across all datasets
        aggregate_ranks = torch.full(
            (_lambda,), float("inf"), device=compute_device, dtype=torch.float32
        )

        # Build matrix of scores for ranking
        valid_offspring = [i for i in range(_lambda) if offspring_metrics_list[i]]
        if not valid_offspring:
            logger.warning(f"{genstring} No valid offspring.")
            original_model_tensor.data.copy_(
                current_best_flat.reshape(original_tensor_shape).to(
                    original_tensor_dtype
                )
            )
            continue

        # Find offspring with best aggregate rank
        dataset_names = list(datasets.keys())
        ranks_sum = {i: 0 for i in valid_offspring}
        for ds_name in dataset_names:
            scores = [
                (i, offspring_metrics_list[i].get(ds_name, float("inf")))
                for i in valid_offspring
            ]
            scores.sort(key=lambda x: x[1])
            for rank, (idx, _) in enumerate(scores, 1):
                ranks_sum[idx] += rank
        best_idx = min(ranks_sum.items(), key=lambda x: x[1])[0]
        aggregate_ranks[best_idx] = ranks_sum[best_idx]
        current_best_metrics = offspring_metrics_list[best_idx].copy()
        logger.info(
            f"{genstring} Selected rank-1 offspring (idx={best_idx}, aggregate_rank={ranks_sum[best_idx]}). "
            f"Metrics: {current_best_metrics}"
        )

        # Apply winning noise and update in-place
        winning_noise_mult, _ = generate_noise_from_svd_coeffs(
            original_tensor_shape,
            U_svd,
            S_svd,
            V_svd,
            target_s_indices,
            delta_s_batch[best_idx],
            epsilon,
            noise_clamp_val,
        )
        current_best_flat.mul_(winning_noise_mult)

        # Update SVD for next generation if needed
        if gen < generations - 1:
            U_new, S_new, V_new = calculate_full_svd(
                current_best_flat.reshape(original_tensor_shape), epsilon
            )
            U_svd, S_svd, V_svd = U_new, S_new, V_new
            sigma_s = initial_sigma * max(S_svd.std().item(), epsilon)
            logger.info(f"{genstring} SVD updated. New sigma: {sigma_s}")

        # Update model tensor with current best
        original_model_tensor.data.copy_(
            current_best_flat.reshape(original_tensor_shape).to(original_tensor_dtype)
        )

    logger.info(f"ES for {tensor_name} done. Generations: {generations}")
    logger.info(f"Final metrics for {tensor_name}: {current_best_metrics}")
    return current_best_metrics


# --- OLM Function ---


def get_evaluation_cache(
    tokenizer: AutoTokenizer,
    datasets: dict,
    samples: int = 200,
) -> dict:
    """Pre-compute all static parts of evaluation to avoid redundant work."""
    cache = {}
    for dataset_name, dataset in datasets.items():
        cache[dataset_name] = {}
        data_samples = dataset["data"][:samples]

        for conv_idx, conv in enumerate(data_samples):
            context = get_context(conv)
            expected_response = conv["conversation"][-1]["value"]

            # Tokenize once, use many times
            context_ids = tokenizer(context, return_tensors="pt")
            response_ids = tokenizer(expected_response, return_tensors="pt")

            # Pre-compute full sequences
            full_ids = torch.cat([context_ids.input_ids, response_ids.input_ids], dim=1)

            # Calculate boundary indices for slicing operations
            context_end_idx = context_ids.input_ids.size(1)

            cache[dataset_name][conv_idx] = {
                "context": context,
                "expected_response": expected_response,
                "context_ids": context_ids,
                "response_ids": response_ids,
                "full_ids": full_ids,
                "context_end_idx": context_end_idx,
                "mode": dataset["mode"],
            }

    return cache


def olm(config: dict, merge_report: str) -> str:
    """Main Optimal Layer Merging (OLM) function with unified selection."""
    models_dir = config["models_dir"]
    output_dir = config["output_dir"]
    datasets = get_datasets(config)
    layer_swap = config.get("layer_swap", False)
    tensor_swap = config.get("tensor_swap", False)
    base_select = config.get("base_select", False)
    boundary_select = config.get("boundary_select", False)
    load_merge_report = config.get("load_merge_report", False)
    samples = config.get("samples", 200)

    logger.info("Downloading required models...")
    for model_name in config["models"]:
        download_model(model_name, models_dir)

    if base_select:
        base_model_name, base_metrics_dict = select_component(
            "base", models_dir, config["models"], datasets, config
        )
    else:
        base_model_name = config.get("base_model_name")
        if not base_model_name:
            logger.info(
                "Base model not provided and OLM selection disabled. Using first model..."
            )
            base_model_name = config["models"][0]
    logger.info(f"Selected base model: {base_model_name}")
    base_path = Path(models_dir) / base_model_name.replace("/", "_")

    if merge_report:
        logger.info("Loading working model from merge report...")
        working_model, tokenizer = get_model_from_merge_report(
            merge_report,
            models_dir,
        )
    else:
        if not os.path.exists(base_path):
            download_model(base_model_name, models_dir)

        logger.info("Loading base model and tokenizer...")
        working_model, tokenizer = load_model(
            str(base_path),
            "cpu",
        )

    use_cache = True
    evaluation_cache = None
    if use_cache:
        logger.info("Building evaluation cache...")
        evaluation_cache = get_evaluation_cache(tokenizer, datasets, samples)

    report_path = os.path.join(output_dir, "merge_report.json")
    if os.path.exists(report_path) and load_merge_report:
        logger.info("Loading existing merge report...")
        merge_report = json.load(open(report_path))
    else:
        logger.info("Initializing new merge report...")
        merge_report = {}

    if "base_model" not in merge_report:
        get_base_metrics(
            base_path,
            output_dir,
            base_model_name,
            datasets,
            working_model,
            tokenizer,
            evaluation_cache=evaluation_cache,
        )
        merge_report = json.load(open(report_path))

    base_metrics = merge_report["base_model"]["metrics"]

    if boundary_select:
        boundary_model, boundary_metrics = select_component(
            "boundary",
            models_dir,
            config["models"],
            datasets,
            config,
            working_model,
            tokenizer,
            evaluation_cache=evaluation_cache,
        )
        boundary_source, _ = load_model(
            f"{models_dir}/{boundary_model.replace('/', '_')}", "cpu"
        )
        _apply_boundary_components(working_model, boundary_source)
        del boundary_source

        merge_report["boundary_layers"] = {
            "name": boundary_model,
            "metrics": {
                dataset_name: scores[boundary_model]
                for dataset_name, scores in boundary_metrics.items()
                if boundary_model in scores
            },
        }

        working_model.save_pretrained(output_dir)
        json.dump(merge_report, open(report_path, "w"), indent=4)

    if layer_swap:
        logger.info("Starting layer-wise optimization...")
        working_model = layer_wise_optimization(
            working_model=working_model,
            model_pool=config["models"],
            datasets=datasets,
            tokenizer=tokenizer,
            config=config,
            models_dir=models_dir,
            output_dir=output_dir,
            base_metrics=base_metrics,
            merge_report=merge_report,
            evaluation_cache=evaluation_cache,
        )

    if tensor_swap:
        logger.info("Starting tensor-wise optimization...")
        working_model = tensor_wise_optimization(
            working_model=working_model,
            model_pool=config["models"],
            datasets=datasets,
            tokenizer=tokenizer,
            config=config,
            models_dir=models_dir,
            output_dir=output_dir,
            base_metrics=base_metrics,
            merge_report=merge_report,
            evaluation_cache=evaluation_cache,
        )

    return output_dir


# --- Main Function ---


@torch.inference_mode()
def main(config_path: str, merge_report: str = None) -> None:
    """Main entry point."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    try:
        output_path = olm(config, merge_report)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error during merge: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="olm_config.yaml")
    parser.add_argument("--merge_report", default=None)
    args = parser.parse_args()
    main(args.config, args.merge_report)
