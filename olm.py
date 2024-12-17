import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Dict, Iterator
import logging
from pathlib import Path
import yaml
import os
import json
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s > %(message)s")
logger = logging.getLogger(__name__)


def load_model(
    model_path: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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


def get_context(conversation: dict) -> str:
    context = ""
    for msg in conversation["conversation"]:
        role = (
            "System: "
            if msg["from"] == "system"
            else "Human: "
            if msg["from"] == "human"
            else "Assistant: "
        )
        context += f"{role}{msg['value']}\n"
    return context


def get_context2(conversation: dict) -> str:
    context = ""
    for msg in conversation["conversation"]:
        if msg["from"] == "system":
            context += f"<|im_start|>system\n{msg['value']}<|im_end|>\n"
        elif msg["from"] == "human":
            context += f"<|im_start|>human\n{msg['value']}<|im_end|>\n"
        else:
            context += "<|im_start|>assistant\n"

    return context


def compute_response_quality(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    max_length: int = 2048,
) -> float:
    prompt = get_context2(conversation)

    # Keep the efficient generation setup
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

    # Check for Chinese characters
    if any("\u4e00" <= char <= "\u9fff" for char in response_text):
        chinese_chars = {char for char in response_text if "\u4e00" <= char <= "\u9fff"}
        for char in chinese_chars:
            logger.info(f"Chinese character detected: {char} (U+{ord(char):04X})")
        logger.info("Chinese characters detected in response.")
        return max_length  # Assign a high metric to penalize

    words = response_text.split()
    total_words = len(words)
    if total_words < 2:
        return max_length
    num_words = len(words)
    num_unique_words = len(set(words))
    length_penalty = min(num_words, max_length) / max_length
    repitition_penalty = num_unique_words / num_words

    # penelize more for repitition
    quality_score = 1 / (length_penalty * repitition_penalty**2)

    logger.info(
        f"Words: {total_words}, Unique: {len(set(words))}, Score: {quality_score:.4f}"
    )

    return quality_score


def compute_exact_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
) -> float:
    context = get_context(conversation)
    reference_response = (  # format response with EOS token for proper loss calc
        f"Assistant: {conversation['conversation'][-1]['value']}{tokenizer.eos_token}"
    )

    context_ids = tokenizer(context, return_tensors="pt").to(model.device)
    response_ids = tokenizer(reference_response, return_tensors="pt").to(model.device)
    full_ids = torch.cat(
        [context_ids.input_ids, response_ids.input_ids], dim=1
    )  # combine for full sequence
    outputs = model(full_ids)

    response_start_index = context_ids.input_ids.size(
        1
    )  # where response begins in sequence
    shift_logits = outputs.logits[
        :, response_start_index:-1, :
    ].contiguous()  # align for loss calc
    shift_labels = response_ids.input_ids[
        :, 1:
    ].contiguous()  # shift for next-token prediction

    loss_fct = CrossEntropyLoss(reduction="mean")  # standard language modeling loss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return torch.exp(loss).item()  # convert loss to perplexity


def compute_response_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
) -> float:
    context = get_context(conversation)
    response = (  # include EOS token for proper sequence termination
        f"Assistant: {conversation['conversation'][-1]['value']}{tokenizer.eos_token}"
    )

    context_ids = tokenizer(context, return_tensors="pt").to(model.device)
    response_ids = tokenizer(response, return_tensors="pt").to(model.device)
    full_ids = torch.cat([context_ids.input_ids, response_ids.input_ids], dim=1).to(
        model.device
    )
    outputs = model(full_ids)

    shift_logits = outputs.logits[  # setup for bigram loss calculation
        :, context_ids.input_ids.size(1) - 1 : -2, :
    ].contiguous()
    next_logits = outputs.logits[:, context_ids.input_ids.size(1) : -1, :].contiguous()
    shift_labels = response_ids.input_ids[:, :-1].contiguous()  # current tokens
    next_labels = response_ids.input_ids[:, 1:].contiguous()  # next tokens

    loss_fct = CrossEntropyLoss(reduction="none")  # need per-token loss for bigram
    current_loss = loss_fct(  # loss for current token predictions
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    next_loss = loss_fct(  # loss for next token predictions
        next_logits.view(-1, next_logits.size(-1)), next_labels.view(-1)
    )
    average_loss = torch.mean(current_loss + next_loss)  # combine for bigram perplexity
    return torch.exp(average_loss).item()


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: list[dict],
    mode: str,
) -> float:
    total_perplexity = 0.0
    max_conversations = min(len(dataset), 200)
    dataset = dataset[:max_conversations]
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.bfloat16)

    for conversation in dataset:
        if mode == "exact":
            perplexity = compute_exact_perplexity(model, tokenizer, conversation)
        if mode == "bigram":
            perplexity = compute_response_perplexity(model, tokenizer, conversation)
        if mode == "quality":
            perplexity = compute_response_quality(model, tokenizer, conversation)
        total_perplexity += perplexity

    return total_perplexity / len(dataset)


def get_layer(model: AutoModelForCausalLM, layer_idx: int):
    return getattr(model.model.layers, str(layer_idx))


def replace_layer(target_model: AutoModelForCausalLM, source_layer, layer_idx: int):
    setattr(target_model.model.layers, str(layer_idx), source_layer)


def layer_sequence(num_layers: int, direction: str) -> Iterator[int]:
    return reversed(range(num_layers)) if direction == "backward" else range(num_layers)


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

    report = (
        {"layers": {}}
        if not os.path.exists(f"{output_dir}/merge_report.json")
        else json.load(open(f"{output_dir}/merge_report.json"))
    )
    report["layers"][str(layer_idx)] = results
    report["layers"][str(layer_idx)]["best_model"] = best_model
    json.dump(report, open(f"{output_dir}/merge_report.json", "w"), indent=4)


def compute_layer_ranks(layer_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute average rank across datasets for each model."""
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
        model_ranks[model_name] = sum(ranks) / len(ranks)

    return model_ranks


def get_base_model_name(models_dir: str, models: list[str], datasets) -> str:
    """Select the model with the lowest average perplexity across all datasets."""
    base_model_name = None
    base_model_perplexity = float("inf")

    for model_name in models:
        try:
            model_path = Path(models_dir) / model_name.replace("/", "_")
            model, tokenizer = load_model(str(model_path), "cpu")
            total_perplexity = 0.0

            for dataset_name, dataset in datasets.items():
                perplexity = evaluate_model_on_dataset(
                    model, tokenizer, dataset["data"], dataset["mode"]
                )
                total_perplexity += perplexity

            if total_perplexity < base_model_perplexity:
                base_model_name = model_name
                base_model_perplexity = total_perplexity

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    return base_model_name


def olm(config: dict) -> str:
    """Optimized Layer Merge"""
    models_dir = config["models_dir"]
    output_dir = config["output_dir"]

    logger.info("Loading datasets...")
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

    base_model_name = config.get("base_model_name", None)

    if not base_model_name:
        base_model_name = get_base_model_name(models_dir, config["models"], datasets)

    logger.info(f"Selected base model: {base_model_name}")

    base_path = Path(models_dir) / base_model_name.replace("/", "_")
    working_model, tokenizer = load_model(str(base_path), "cpu")
    num_layers = working_model.config.num_hidden_layers
    logger.info("Preserving base model tokenizer configuration...")
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    for file in tokenizer_files:
        src_path = os.path.join(base_path, file)
        dst_path = os.path.join(output_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            logger.info(f"Preserved {file}")
        else:
            logger.warning(f"Missing expected tokenizer file: {file}")

    working_model, tokenizer = load_model(str(base_path), "cpu")
    num_layers = working_model.config.num_hidden_layers

    # Layer-wise optimization
    for layer_idx in layer_sequence(num_layers, config.get("direction", "forward")):
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
                torch.cuda.empty_cache()

                for dataset_name, dataset in datasets.items():
                    perplexity = evaluate_model_on_dataset(
                        working_model,
                        tokenizer,
                        dataset["data"],
                        dataset["mode"],
                    )
                    layer_metrics[dataset_name][model_name] = perplexity
                    logger.info(f"{dataset_name}: {perplexity:.4f}")

            except Exception as e:
                logger.error(f"Error testing layer from {model_name}: {e}")
                continue

        model_ranks = compute_layer_ranks(layer_metrics)
        if not model_ranks:
            logger.error(
                f"No valid candidates for layer {layer_idx}, skipping optimization"
            )
            continue

        best_model = min(model_ranks.items(), key=lambda x: x[1])[0]
        try:
            best_source = load_model(
                f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
            )[0]
            replace_layer(working_model, get_layer(best_source, layer_idx), layer_idx)
            save_layer_state(
                working_model, layer_metrics, layer_idx, output_dir, best_model
            )
            del best_source
            torch.cuda.empty_cache()
            logger.info(
                f"Applied layer {layer_idx} from {best_model} (avg rank: {model_ranks[best_model]:.2f})"
            )
        except Exception as e:
            logger.error(f"Error applying best layer from {best_model}: {e}")
            continue

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
