import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Iterator, Optional, Dict, List, Tuple
import logging
from pathlib import Path
import yaml
import os
import json
import shutil
from huggingface_hub import snapshot_download
import sqlite3
import uuid
from datetime import datetime

from get_model_metric import ModelLoaderFromDatabase, ModelDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s > %(message)s")
logger = logging.getLogger(__name__)


# --- Model and Dataset Handling ---


def save_merge_report_to_database(merge_report: Dict, database_dir: str) -> None:
    """Store just the merge report in the database."""
    if not database_dir:
        return

    db = ModelDatabase(database_dir)
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()

        # Add a merge_reports table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS merge_reports (
                report_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)

        report_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cursor.execute(
            "INSERT INTO merge_reports (report_id, timestamp, report_json) VALUES (?, ?, ?)",
            (report_id, timestamp, json.dumps(merge_report)),
        )
        conn.commit()


def get_merge_report_from_database(report_id: str, database_dir: str) -> Optional[Dict]:
    """Retrieve a merge report from the database."""
    db = ModelDatabase(database_dir)
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT report_json FROM merge_reports WHERE report_id = ?", (report_id,)
        )
        result = cursor.fetchone()

        if result:
            return json.loads(result[0])
        return None


def get_merge_report_model_from_database(
    merge_report: dict, database_dir: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Reconstruct a model from a merge report using the database."""
    model_loader = ModelLoaderFromDatabase(database_dir)

    # Load base model
    base_model_name = merge_report["base_model"]["name"]
    logger.info(f"Loading base model {base_model_name} from database")
    model = model_loader.get_model_from_db(base_model_name)
    tokenizer = model_loader.get_tokenizer_from_db(base_model_name)

    if model is None or tokenizer is None:
        raise ValueError(f"Failed to load base model {base_model_name} from database")

    # Apply boundary layers if specified
    if "boundary_layers" in merge_report and merge_report["boundary_layers"]["name"]:
        boundary_model_name = merge_report["boundary_layers"]["name"]
        logger.info(f"Loading boundary layers from {boundary_model_name}")

        try:
            boundary_source = model_loader.get_model_from_db(boundary_model_name)
            if boundary_source is None:
                raise ValueError(f"Failed to load boundary model {boundary_model_name}")

            # Apply boundary layers
            model.model.embed_tokens.load_state_dict(
                boundary_source.model.embed_tokens.state_dict()
            )
            model.model.norm.load_state_dict(boundary_source.model.norm.state_dict())
            model.lm_head.load_state_dict(boundary_source.lm_head.state_dict())
            del boundary_source

        except Exception as e:
            logger.error(
                f"Error applying boundary layers from {boundary_model_name}: {e}"
            )
            logger.info("Keeping base model boundary layers")

    # Apply individual layers from merge report
    if "layers" in merge_report:
        for layer_idx, layer_info in merge_report["layers"].items():
            layer_source_model = layer_info.get("best_model")
            if layer_source_model:
                logger.info(f"Loading layer {layer_idx} from {layer_source_model}")
                try:
                    layer_source = model_loader.get_model_from_db(layer_source_model)
                    if layer_source is None:
                        raise ValueError(
                            f"Failed to load source model {layer_source_model}"
                        )

                    layer_idx = int(layer_idx)
                    replace_layer(model, get_layer(layer_source, layer_idx), layer_idx)
                    del layer_source

                except Exception as e:
                    logger.error(
                        f"Error applying layer {layer_idx} from {layer_source_model}: {e}"
                    )
                    logger.info("Keeping base model layer")

    return model, tokenizer


def get_model_from_merge_report(
    merge_report_path: str, models_dir: str, database_dir: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Reconstruct a model from a merge report by loading the base model and swapping in specified layers."""
    with open(merge_report_path) as f:
        merge_report = json.load(f)

    if database_dir:
        model, tokenizer = get_merge_report_model_from_database(
            merge_report, database_dir
        )
        return model, tokenizer

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


def load_model_from_database(
    model_path: str, database_dir: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model from database."""
    try:
        model_name = Path(model_path).name.replace("_", "/", 1)
        model_loader = ModelLoaderFromDatabase(database_dir)
        model = model_loader.get_model_from_db(model_name)
        tokenizer = model_loader.get_tokenizer_from_db(model_name)

        if model is None or tokenizer is None:
            raise ValueError(
                f"Failed to load model or tokenizer for {model_name} from database"
            )
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model from database: {e}")
        raise


def load_model(
    model_path: str, device: str, database_dir: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model from local path and return model and tokenizer."""
    if database_dir:
        model, tokenizer = load_model_from_database(model_path, "database_dir")
        return model, tokenizer

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


def get_datasets(config: Dict) -> Dict:
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


def save_layer_to_database(
    db: ModelDatabase,
    cursor: sqlite3.Cursor,
    model_id: str,
    layer_idx: int,
    best_model: str,
    metrics: Dict,
) -> None:
    """Save layer optimization results to the database."""
    if db is None:
        return
    layer_id = str(uuid.uuid4())
    db.store_layer_data(cursor, layer_id, model_id, str(layer_idx), "merged")
    db.store_model_reference(cursor, layer_id, best_model)
    for dataset, dataset_metrics in metrics.items():
        db.store_layer_metrics(cursor, layer_id, dataset, dataset_metrics)


def load_model_with_db(
    model_name: str, models_dir: str, db_dir: str = None, device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, prioritizing the database if available."""
    if db_dir:
        model_loader = ModelLoaderFromDatabase(db_dir)
        model = model_loader.get_model_from_db(model_name, device)
        tokenizer = model_loader.get_tokenizer_from_db(model_name)
        if model is None or tokenizer is None:
            raise ValueError(f"Failed to load {model_name} from database")
        return model, tokenizer
    else:
        model_path = Path(models_dir) / model_name.replace("/", "_")
        model, tokenizer = load_model(str(model_path), device)
        return model, tokenizer


# --- Evaluation Metrics ---


def get_context(conversation: Dict) -> str:
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
    conversation: Dict,
    max_length: int = 4096,
) -> float:
    """Compute the quality of a generated response."""
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
    conversation: Dict,
) -> float:
    """Compute the bigram loss of a response."""
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
    conversation: Dict,
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
        next_token_logits = outputs.logits[:, -1:, :].detach()
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
    dataset: List[Dict],
    mode: str,
    samples: int = 200,
) -> float:
    """Evaluate a model on a dataset using a specified evaluation mode."""
    total_metric = 0.0
    max_conversations = min(len(dataset), samples)
    dataset = dataset[:max_conversations]
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model.device.type != device:
        model.to(device, dtype=torch.bfloat16)

    for conversation in dataset:
        if mode == "exact":
            metric = compute_exact_match(model, tokenizer, conversation)
        elif mode == "bigram":
            metric = compute_response_perplexity(model, tokenizer, conversation)
        elif mode == "quality":
            metric = compute_response_quality(model, tokenizer, conversation)
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


def get_last_layer_idx(merge_report: Dict) -> Optional[int]:
    """Get the index of the last optimized layer from the merge report."""
    if not merge_report.get("layers"):
        return None
    return max(int(layer_num) for layer_num in merge_report["layers"].keys())


def get_layer_metrics(merge_report: Dict, layer_idx: int) -> Dict:
    """Retrieve the performance metrics for a specific layer from the merge report."""
    return merge_report["layers"][str(layer_idx)]["metrics"]


# --- Merge Report, Ranking and Selection ---


def save_results_to_database(
    report: Dict,
    model: AutoModelForCausalLM,
    layer_idx: int,
    best_model: str,
    database_dir: str,
) -> None:
    """Store layer optimization results in SQLite database."""
    db = ModelDatabase(database_dir)
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        layer_id = str(uuid.uuid4())
        db.store_layer_data(
            cursor, layer_id, model.config._name_or_path, str(layer_idx), "merged"
        )
        db.store_model_reference(cursor, layer_id, best_model)
        metrics = report["layers"][str(layer_idx)]["metrics"]
        for dataset, dataset_metrics in metrics.items():
            db.store_layer_metrics(cursor, layer_id, dataset, dataset_metrics)

        conn.commit()


def save_layer_state(
    model: AutoModelForCausalLM,
    results: Dict,
    layer_idx: int,
    output_dir: str,
    best_model: str,
    database_dir: str = None,
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

    if database_dir:
        save_results_to_database(report, model, layer_idx, best_model, database_dir)


def compute_model_ranks(
    model_name: str, layer_metrics: Dict, valid_models: List[str]
) -> List[int]:
    """Compute the ranks of a model across datasets for a given layer."""
    ranks = []
    for dataset_scores in layer_metrics.values():
        valid_scores = {m: s for m, s in dataset_scores.items() if m in valid_models}
        sorted_models = sorted(valid_scores.items(), key=lambda x: x[1])
        rank = next(i for i, (m, _) in enumerate(sorted_models) if m == model_name) + 1
        ranks.append(rank)
    return ranks


def compute_layer_ranks(
    layer_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, List[int]]:
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


def select_best_model(model_ranks: Dict[str, List[int]]) -> str:
    """Select the best model based on rank aggregation."""
    return min(model_ranks.items(), key=lambda x: sum(x[1]))[0]


# --- Base Model Handling and Initialization ---


def select_base_model(
    models_dir: str, models: List[str], datasets: Dict, selected_metric: str
) -> str:
    """Select the best base model based on a specific metric."""
    selected_datasets = {
        name: dataset
        for name, dataset in datasets.items()
        if dataset["mode"] == selected_metric
    }
    layer_metrics = {dataset_name: {} for dataset_name in selected_datasets}
    for model_name in models:
        logger.info(f"Evaluating base candidate: {model_name}")
        try:
            model, tokenizer = load_model(
                Path(models_dir) / model_name.replace("/", "_"), "cpu"
            )
            for dataset_name, dataset in selected_datasets.items():
                metric = evaluate_model_on_dataset(
                    model, tokenizer, dataset["data"], dataset["mode"]
                )
                layer_metrics[dataset_name][model_name] = metric
                logger.info(f"{model_name}: {dataset_name}: {metric}")
            del model
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

    return select_best_model(model_ranks)


def get_base_metrics(
    base_path: Path,
    output_dir: str,
    base_model_name: str,
    datasets: Dict,
    working_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
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

    json.dump(report, open(report_path, "w"), indent=4)
    logger.info(f"Base metrics saved to {report_path}")


# --- Layer Processing and Degradation Handling ---


def process_model(
    model_name: str,
    layer_metrics: Dict,
    datasets: Dict,
    working_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_metrics: Dict,
    merge_report: Dict,
    layer_idx: int,
    samples: int = 200,
    improve_all: str = None,
    skip_quality: bool = False,
) -> Dict:
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

    if skip_quality:
        layer_metrics = {
            name: metrics
            for name, metrics in layer_metrics.items()
            if datasets[name]["mode"] != "quality"
        }
    for dataset_name, dataset in datasets.items():
        if dataset_name not in layer_metrics:
            layer_metrics[dataset_name] = {}

        try:
            metric = evaluate_model_on_dataset(
                working_model, tokenizer, dataset["data"], dataset["mode"], samples
            )
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
            metric = float("inf")

        layer_metrics[dataset_name][model_name] = metric
        logger.info(f"{model_name}: {dataset_name}: {metric}")

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
        elif improve_all == "base":  # Moved out to be independent check
            if metric > base_metrics[dataset_name]:
                logger.info("Performance below base model detected, skip layer")
                for remaining in datasets:
                    if remaining not in layer_metrics:
                        layer_metrics[remaining] = {}
                    layer_metrics[remaining][model_name] = float("inf")
                return layer_metrics

    return layer_metrics


# --- Boundary Layer Optimization ---


def get_boundary_layers(
    working_model: AutoModelForCausalLM,
    config: Dict,
    datasets: Dict,
    tokenizer: AutoTokenizer,
) -> Dict:
    """Select boundary components based on dataset performance."""
    models_dir = config["models_dir"]
    output_dir = config["output_dir"]
    report_path = os.path.join(output_dir, "merge_report.json")
    boundary_metric = config.get("boundary_select", "quality")

    if os.path.exists(report_path):
        merge_report = json.load(open(report_path))
    else:
        merge_report = {}

    logger.info(f"Selecting boundary components for best {boundary_metric} performance")
    component_metrics = {dataset: {} for dataset in datasets}

    original_states = {
        "embed_tokens": working_model.model.embed_tokens.state_dict(),
        "norm": working_model.model.norm.state_dict(),
        "lm_head": working_model.lm_head.state_dict(),
    }

    for model_name in config["models"]:
        try:
            source, _ = load_model(
                f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
            )

            working_model.model.embed_tokens.load_state_dict(
                source.model.embed_tokens.state_dict()
            )
            working_model.model.norm.load_state_dict(source.model.norm.state_dict())
            working_model.lm_head.load_state_dict(source.lm_head.state_dict())

            for dataset_name, dataset in datasets.items():
                if dataset["mode"] == boundary_metric:
                    metric = evaluate_model_on_dataset(
                        working_model, tokenizer, dataset["data"], dataset["mode"]
                    )
                    component_metrics[dataset_name][model_name] = metric
                    logger.info(f"{model_name}: {dataset_name}: {metric}")
                else:
                    component_metrics[dataset_name][model_name] = float("inf")

            del source

        except Exception as e:
            logger.error(f"Error testing components from {model_name}: {e}")
            continue

        for component_name, state in original_states.items():
            getattr(
                working_model.model if component_name != "lm_head" else working_model,
                component_name,
            ).load_state_dict(state)

    selected_datasets = [
        name for name, ds in datasets.items() if ds["mode"] == boundary_metric
    ]
    model_scores = {}

    for model_name in config["models"]:
        try:
            scores = [component_metrics[ds][model_name] for ds in selected_datasets]
            if all(score != float("inf") for score in scores):
                model_scores[model_name] = sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"Error calculating scores for {model_name}: {e}")
            continue

    if model_scores:
        best_model = min(model_scores.items(), key=lambda x: x[1])[0]

        source, _ = load_model(f"{models_dir}/{best_model.replace('/', '_')}", "cpu")

        working_model.model.embed_tokens.load_state_dict(
            source.model.embed_tokens.state_dict()
        )
        working_model.model.norm.load_state_dict(source.model.norm.state_dict())
        working_model.lm_head.load_state_dict(source.lm_head.state_dict())

        logger.info(f"Applied boundary components from {best_model}")

        if "boundary_layers" not in merge_report:
            merge_report["boundary_layers"] = {}

        merge_report["boundary_layers"] = {
            "name": best_model,
            "metrics": {
                dataset_name: component_metrics[dataset_name][best_model]
                for dataset_name in selected_datasets
            },
        }

        del source

    working_model.save_pretrained(output_dir)
    json.dump(merge_report, open(report_path, "w"), indent=4)

    return merge_report, working_model


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
    tensor_metrics: Dict,
    tensor_name: str,  # One name to rule them all
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


def layer_wise_optimization(
    working_model: AutoModelForCausalLM,
    model_pool: List[str],
    datasets: Dict,
    tokenizer: AutoTokenizer,
    config: Dict,
    models_dir: str,
    output_dir: str,
    base_metrics: Dict,
    merge_report: Dict,
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
                    skip_quality=config.get("skip_quality", False),
                )
            except Exception as e:
                logger.error(f"Layer optimization failed for {model_name}: {e}")
                continue

        model_ranks = compute_layer_ranks(layer_metrics)
        if not model_ranks:
            logger.error(f"No valid layer candidates for layer {layer_idx}")
            continue

        best_model = select_best_model(model_ranks)
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
                f"Applied layer {layer_idx} from {best_model} (avg rank: {avg_rank:.2f})"
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
    model_pool: List[str],
    datasets: Dict,
    tokenizer: AutoTokenizer,
    config: Dict,
    models_dir: str,
    output_dir: str,
    base_metrics: Dict,
    merge_report: Dict,
) -> AutoModelForCausalLM:
    """Tensor optimization: Computational natural selection, one tensor at a time."""
    tensor_map = {
        name: param
        for name, param in working_model.named_parameters()
        if "weight" in name  # That's all we need, fam
    }
    logger.info(f"Found {len(tensor_map)} tensors ready for optimization")

    for tensor_name, tensor in tensor_map.items():
        logger.info(f"\nOptimizing tensor: {tensor_name}")
        layer_metrics = {dataset_name: {} for dataset_name in datasets}
        original_weights = tensor.data.clone()

        for model_name in model_pool:
            try:
                source_model, _ = load_model(
                    f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                )

                try:
                    source_tensor = dict(source_model.named_parameters())[tensor_name]
                    tensor.data.copy_(source_tensor.data)

                    layer_metrics = process_model(
                        model_name=model_name,
                        layer_metrics=layer_metrics,
                        datasets=datasets,
                        working_model=working_model,
                        tokenizer=tokenizer,
                        base_metrics=base_metrics,
                        merge_report=merge_report,
                        layer_idx=tensor_name,
                        improve_all=config.get("improve_all"),
                        skip_quality=config.get("skip_quality", False),
                    )

                    if any(
                        model_name in ds and ds[model_name] == float("inf")
                        for ds in layer_metrics.values()
                    ):
                        tensor.data.copy_(original_weights)

                except KeyError:
                    logger.warning(f"No matching tensor in {model_name}")

                del source_model

            except Exception as e:
                logger.error(f"Tensor optimization failed for {model_name}: {e}")
                tensor.data.copy_(original_weights)
                continue

        model_ranks = compute_layer_ranks(layer_metrics)
        if not model_ranks:
            continue

        best_model = select_best_model(model_ranks)
        try:
            best_source_model, _ = load_model(
                f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
            )
            best_tensor = dict(best_source_model.named_parameters())[tensor_name]
            tensor.data.copy_(best_tensor.data)

            save_tensor_state(
                working_model, layer_metrics, tensor_name, output_dir, best_model
            )
            logger.info(f"Tensor {tensor_name} upgraded with parts from {best_model}")

            del best_source_model

            working_model.save_pretrained(
                output_dir, safe_serialization=True, push_to_hub=False, exist_ok=True
            )

        except Exception as e:
            logger.error(f"Failed to apply best tensor from {best_model}: {e}")
            tensor.data.copy_(original_weights)

    return working_model


# --- Main OLM Function ---


def olm(config: Dict, merge_report: str) -> str:
    """Main Optimal Layer Merging (OLM) function."""

    models_dir = config["models_dir"]
    output_dir = config["output_dir"]
    datasets = get_datasets(config)
    layer_swap = config.get("layer_swap", False)
    tensor_swap = config.get("tensor_swap", False)
    boundary_select = config.get("boundary_select", None)

    logger.info("Downloading required models...")
    for model_name in config["models"]:
        download_model(model_name, models_dir)

    base_model_name = config.get("base_model_name")
    if not base_model_name:
        logger.info("Base model not provided. Selecting the best base model...")
        base_model_name = select_base_model(
            models_dir, config["models"], datasets, config.get("base_select", "quality")
        )
    logger.info(f"Selected base model: {base_model_name}")
    base_path = Path(models_dir) / base_model_name.replace("/", "_")

    if merge_report:
        logger.info("Loading working model from merge report...")
        working_model, tokenizer = get_model_from_merge_report(
            merge_report, models_dir, config.get("database_dir", None)
        )
    else:
        logger.info("Loading base model and tokenizer...")
        working_model, tokenizer = load_model(
            str(base_path), "cpu", config.get("database_dir", None)
        )

    if boundary_select:
        logger.info("Optimizing boundary layers...")
        merge_report, working_model = get_boundary_layers(
            working_model, config, datasets, tokenizer
        )

    logger.info("Computing base metrics...")
    report_path = os.path.join(output_dir, "merge_report.json")
    get_base_metrics(
        base_path, output_dir, base_model_name, datasets, working_model, tokenizer
    )

    if os.path.exists(report_path):
        logger.info("Loading existing merge report...")
        merge_report = json.load(open(report_path))
    else:
        logger.info("Initializing new merge report...")
        merge_report = {}
    base_metrics = merge_report["base_model"]["metrics"]

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
        )

    db_dir = config.get("db_dir", None)
    if db_dir:
        save_merge_report_to_database(merge_report, db_dir)

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
