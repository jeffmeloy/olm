import argparse
import json
import os
import sqlite3
import bz2
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transfer_files_to_temp_dir(
    model_name: str, database_dir: str, temp_dir: Path, file_keys: list[str]
) -> bool:
    """Transfers specified files from the database to a temporary directory."""
    model_data = load_model_data(model_name, database_dir)

    for key in file_keys:
        data = model_data.get(key)
        if not data:
            logger.error(f"Data for {key} not found in the database for {model_name}.")
            return False

        file_path = temp_dir / f"{key}.json"
        with open(file_path, "w") as f:
            f.write(data)

    return True


def clear_temp_directory(temp_dir: Path) -> None:
    """Clears all files and directories within the temporary directory."""
    for item in temp_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def get_model_from_db(
    model_name: str, database_dir: str, device: str = "cpu"
) -> Optional[AutoModelForCausalLM]:
    """Loads a model from the database to the specified device (default: CPU)"""
    temp_dir = Path(database_dir) / "temp_model_files"
    temp_dir.mkdir(exist_ok=True)
    clear_temp_directory(temp_dir)
    if not transfer_files_to_temp_dir(model_name, database_dir, temp_dir, ["config"]):
        return None  # Transfer failed

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(temp_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device},
        )
    except Exception as e:
        logger.error(f"Error loading model from temporary directory: {e}")
        return None

    tensors_data = load_tensors_and_metrics(model_name, database_dir)
    for name, tensor_data in tensors_data.items():
        with torch.no_grad():
            param = model.get_parameter(name)
            if param is not None:
                # Ensure tensor is on CPU before copying
                param.copy_(tensor_data["tensor"].to("cpu"))

    return model


def get_tokenizer_from_db(
    model_name: str, database_dir: str
) -> Optional[AutoTokenizer]:
    """Loads a tokenizer from the database."""
    temp_dir = Path(database_dir) / "temp_model_files"
    temp_dir.mkdir(exist_ok=True)
    clear_temp_directory(temp_dir)  # Clear temp directory before transferring files

    if not transfer_files_to_temp_dir(
        model_name, database_dir, temp_dir, ["tokenizer_config", "tokenizer"]
    ):
        return None  # Transfer failed

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(temp_dir))
    except Exception as e:
        logger.error(f"Error loading tokenizer from temporary directory: {e}")
        return None

    return tokenizer


def init_database(database_dir: str) -> None:
    """Initializes the SQLite database with improved schema."""
    db_path = Path(database_dir) / "models.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,  -- UUID string
                model_name TEXT UNIQUE NOT NULL,
                config_json TEXT,
                tokenizer_json TEXT,
                tokenizer_config_json TEXT,
                vocab_json TEXT,
                generation_config_json TEXT,
                added_tokens_json TEXT,
                special_tokens_map_json TEXT
            );

            CREATE TABLE IF NOT EXISTS tensors (
                tensor_id TEXT PRIMARY KEY,  -- UUID string
                model_id TEXT NOT NULL,
                name TEXT NOT NULL,
                tensor_data BLOB NOT NULL,
                tensor_shape TEXT NOT NULL,
                tensor_dtype TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(model_id),
                UNIQUE(model_id, name)
            );

            CREATE TABLE IF NOT EXISTS tensor_metrics (
                tensor_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(tensor_id, metric_name)
            );
        """)


def load_model_data(model_name: str, database_dir: str) -> dict:
    """Loads model configuration data from the database with new schema."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model_id, config_json, tokenizer_json, tokenizer_config_json,
                   vocab_json, generation_config_json, added_tokens_json,
                   special_tokens_map_json
            FROM models 
            WHERE model_name = ?
            """,
            (model_name,),
        )

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Model {model_name} not found in database")

        return {
            "model_id": row[0],
            "config": json.loads(row[1]) if row[1] else None,
            "tokenizer": json.loads(row[2]) if row[2] else None,
            "tokenizer_config": json.loads(row[3]) if row[3] else None,
            "vocab": json.loads(row[4]) if row[4] else None,
            "generation_config": json.loads(row[5]) if row[5] else None,
            "added_tokens": json.loads(row[6]) if row[6] else None,
            "special_tokens_map": json.loads(row[7]) if row[7] else None,
        }


def load_tensors_and_metrics(model_name: str, database_dir: str) -> dict:
    """Loads tensors and their metrics from the database with new schema."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # First get model_id
        cursor.execute(
            "SELECT model_id FROM models WHERE model_name = ?", (model_name,)
        )
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found in database")

        model_id = result[0]

        # Get tensors with their metrics using a more efficient query
        cursor.execute(
            """
            SELECT 
                t.name,
                t.tensor_data,
                t.tensor_shape,
                t.tensor_dtype,
                GROUP_CONCAT(m.metric_name || ':' || m.metric_value) as metrics
            FROM tensors t
            LEFT JOIN tensor_metrics m ON t.tensor_id = m.tensor_id
            WHERE t.model_id = ?
            GROUP BY t.tensor_id, t.name
            """,
            (model_id,),
        )

        tensors_and_metrics = {}
        for name, tensor_data, shape_str, dtype_str, metrics_str in cursor.fetchall():
            # Reconstruct tensor on CPU
            shape = json.loads(shape_str)
            tensor = (
                torch.frombuffer(
                    tensor_data, dtype=getattr(torch, dtype_str.split(".")[-1])
                )
                .reshape(shape)
                .clone()
                .to("cpu")
            )

            # Parse metrics
            metrics = {}
            if metrics_str:
                for metric_item in metrics_str.split(","):
                    metric_name, value = metric_item.split(":")
                    metrics[metric_name] = float(value)

            tensors_and_metrics[name] = {"tensor": tensor, "metrics": metrics}

    return tensors_and_metrics


def load_model_and_metrics(model_name: str, database_dir: str) -> tuple[dict, dict]:
    """Convenience function to load both model data and tensors/metrics."""
    model_data = load_model_data(model_name, database_dir)
    tensors_data = load_tensors_and_metrics(model_name, database_dir)
    return model_data, tensors_data


def _store_tensor(
    cursor: sqlite3.Cursor,
    tensor_id: str,
    model_id: str,
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Store a single tensor in the database."""
    cursor.execute(
        """
        INSERT INTO tensors (tensor_id, model_id, name, tensor_data, tensor_shape, tensor_dtype)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            tensor_id,
            model_id,
            name,
            tensor.numpy().tobytes(),
            json.dumps(list(tensor.shape)),
            str(tensor.dtype),
        ),
    )


def _calculate_metrics(tensor: torch.Tensor, device: torch.device) -> dict[str, float]:
    """Calculate all metrics for a tensor."""
    metrics = {}
    # Only move tensor to device for metric calculation
    tensor_device = tensor.to(device)

    # Define all metrics to calculate
    all_metrics = [
        "snr",
        "normalized_effective_rank",
        "svd_skewness",
        "bzip2",
        "kurtosis",
        "skewness",
        "sparsity",
        "spectral_norm",
        "frobenius_norm",
        "weight_entropy",
        "stable_rank",
    ]

    for metric_name in all_metrics:
        try:
            metrics[metric_name] = calculate_metric(tensor_device, metric_name)
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {e}")
            metrics[metric_name] = float("nan")

    # Move tensor back to CPU after calculations
    del tensor_device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def _store_metrics(
    cursor: sqlite3.Cursor, tensor_id: str, metrics: dict[str, float]
) -> None:
    """Store calculated metrics for a tensor."""
    cursor.executemany(
        """
        INSERT INTO tensor_metrics (tensor_id, metric_name, metric_value)
        VALUES (?, ?, ?)
        """,
        [(tensor_id, name, value) for name, value in metrics.items()],
    )


def _load_model_configs(model_path: str) -> dict[str, str]:
    """Load all configuration files from model directory."""
    model_path = Path(model_path)
    configs = {}

    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "generation_config.json",
        "added_tokens.json",
    ]

    for filename in config_files:
        filepath = model_path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                configs[filename.replace(".json", "_json")] = f.read()
        else:
            configs[filename.replace(".json", "_json")] = None

    return configs


def calculate_metric(layer: torch.Tensor, metric_name: str) -> float:
    """Calculates a specified metric for a layer."""
    # In-place conversion to float32 for calculations
    layer = layer.float()

    if metric_name == "snr":
        return _calculate_snr(layer)
    elif metric_name == "normalized_effective_rank":
        return _calculate_normalized_effective_rank(layer)
    elif metric_name == "svd_skewness":
        return _calculate_svd_skewness(layer)
    elif metric_name == "bzip2":
        return _calculate_bzip2(layer)
    elif metric_name == "kurtosis":
        return _calculate_weight_kurtosis(layer)
    elif metric_name == "skewness":
        return _calculate_weight_skewness(layer)
    elif metric_name == "sparsity":
        return _calculate_weight_sparsity(layer)
    elif metric_name == "spectral_norm":
        return _calculate_weight_spectral_norm(layer)
    elif metric_name == "frobenius_norm":
        return _calculate_frobenius_norm(layer)
    elif metric_name == "weight_entropy":
        return _calculate_weight_entropy(layer)
    elif metric_name == "stable_rank":
        return _calculate_stable_rank(layer)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _calculate_snr(A: torch.Tensor) -> float:
    """Calculate signal-to-noise ratio (SNR) metric, higher = more signal."""
    S = torch.linalg.svdvals(A)
    max_singular_value = S[0]
    sigma_estimated = _estimate_sigma_with_full_iqr(S)
    n, m = A.shape[-2:]
    mp_threshold = _marchenko_pastur_threshold(sigma_estimated, n, m)
    signal = torch.sum(S[S > mp_threshold])  # More efficient masking
    noise = torch.sum(S[S <= mp_threshold])
    snr = signal / max(noise, 1e-10)
    return (snr / max_singular_value).item()


def _estimate_sigma_with_full_iqr(S: torch.Tensor) -> float:
    """Estimate sigma using full interquartile range (IQR) of singular values."""
    q75, q25 = torch.quantile(S, torch.tensor([0.75, 0.25], device=S.device))
    return ((q75 - q25) / 1.349).item()


def _marchenko_pastur_threshold(sigma: float, n: int, m: int) -> float:
    """Calculate Marchenko-Pastur threshold."""
    beta = min(n, m) / max(n, m)
    return sigma * (1 + np.sqrt(beta)) ** 2


def _calculate_svd_skewness(A: torch.Tensor) -> float:
    """Calculates the skewness of the singular value distribution."""
    S = torch.linalg.svdvals(A)
    return (1 - (torch.mean(S) / S[0])).item()


def _calculate_normalized_effective_rank(A: torch.Tensor) -> float:
    """Calculates the normalized effective rank."""
    S = torch.linalg.svdvals(A)
    S_norm = S / S.sum()
    effective_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm)))
    actual_rank = torch.linalg.matrix_rank(A)
    return (effective_rank / actual_rank).item()


def _calculate_bzip2(A: torch.Tensor) -> float:
    """Calculates the bzip2 compression ratio."""
    A_bytes = A.cpu().contiguous().numpy().tobytes()
    return len(bz2.compress(A_bytes)) / len(A_bytes)


def _calculate_weight_kurtosis(A: torch.Tensor) -> float:
    """Calculates the kurtosis of the weight distribution."""
    return torch.kurtosis(A.flatten()).item()


def _calculate_weight_skewness(A: torch.Tensor) -> float:
    """Calculates the skewness of the weight distribution."""
    return torch.skew(A.flatten()).item()


def _calculate_weight_sparsity(A: torch.Tensor) -> float:
    """Calculates the sparsity of the weights (proportion of near-zero values)."""
    return (torch.abs(A) < 1e-5).float().mean().item()


def _calculate_weight_spectral_norm(A: torch.Tensor) -> float:
    """Calculates the spectral norm (largest singular value)."""
    return torch.linalg.norm(A, ord=2).item()


def _calculate_frobenius_norm(A: torch.Tensor) -> float:
    """Calculates the Frobenius norm."""
    return torch.linalg.norm(A).item()


def _calculate_weight_entropy(A: torch.Tensor) -> float:
    """Calculates the entropy of the weight distribution."""
    A = A.flatten()
    num_bins = int(np.sqrt(min(A.shape))) if A.numel() > 1 else 2
    num_bins = max(num_bins, 2)
    hist = torch.histogram(A, bins=num_bins)
    probs = hist.hist / hist.hist.sum()
    return (-torch.sum(probs * torch.log2(probs + 1e-12))).item()


def _calculate_stable_rank(A: torch.Tensor) -> float:
    """Calculates the stable rank."""
    S = torch.linalg.svdvals(A)
    return (torch.sum(S**2) / S[0] ** 2).item()


def _normalize_tensor(A: torch.Tensor) -> torch.Tensor:
    """Normalizes tensor magnitude while preserving signs."""
    signs = A.sign()
    A = A.abs()
    A /= A.max().clamp(min=1e-8)
    return A * signs


def download_model(model_name: str, models_dir: str) -> str:
    """Downloads a model from the Hugging Face Hub."""
    local_path = Path(models_dir) / model_name.replace("/", "_")
    if not local_path.exists():
        logger.info(f"Downloading {model_name} to {local_path}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                revision="main",
            )
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")
            raise
    else:
        logger.info(f"Model {model_name} already exists at {local_path}")
    return str(local_path)


def load_model(model_path: str) -> AutoModelForCausalLM:
    """Loads a model from a local path to CPU."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # You can still load in bfloat16 on CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cpu"},  # Explicitly load to CPU
        )
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def process_and_store_model(
    model_name: str,
    model_path: str,
    database_dir: str,
    device: torch.device,
) -> None:
    """Process and store model data and tensors for Qwen models, calculating metrics on the specified device."""
    db_path = Path(database_dir) / "models.db"
    model_id = str(uuid.uuid4())
    model_configs = _load_model_configs(model_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO models (
                model_id, model_name, config_json, tokenizer_json,
                tokenizer_config_json, vocab_json, generation_config_json,
                added_tokens_json, special_tokens_map_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                model_name,
                *[
                    model_configs.get(k)
                    for k in [
                        "config_json",
                        "tokenizer_json",
                        "tokenizer_config_json",
                        "vocab_json",
                        "generation_config_json",
                        "added_tokens_json",
                        "special_tokens_map_json",
                    ]
                ],
            ),
        )

        # Load model strictly to CPU
        model = load_model(model_path)
        if model is None:
            return

        # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # num_layers = config.num_hidden_layers

        # Iterate through Qwen layers and specific parameters
        qwen_layers = model.transformer.h
        for layer_idx, layer in enumerate(qwen_layers):
            logger.info(f"Processing layer {layer_idx}")
            layer_params = {
                f"model.layers.{layer_idx}.self_attn.q_proj": layer.attn.q_proj,
                f"model.layers.{layer_idx}.self_attn.k_proj": layer.attn.k_proj,
                f"model.layers.{layer_idx}.self_attn.v_proj": layer.attn.v_proj,
                f"model.layers.{layer_idx}.self_attn.o_proj": layer.attn.o_proj,
                f"model.layers.{layer_idx}.mlp.gate_proj": layer.mlp.gate_proj,
                f"model.layers.{layer_idx}.mlp.up_proj": layer.mlp.up_proj,
                f"model.layers.{layer_idx}.mlp.down_proj": layer.mlp.down_proj,
                f"model.layers.{layer_idx}.input_layernorm": layer.ln_1,
                f"model.layers.{layer_idx}.post_attention_layernorm": layer.ln_2,
            }

            for name, param in layer_params.items():
                # bias parameters do not have a .weight attribute in qwen,
                # only process the weight parameters
                if hasattr(param, "weight") and param.weight is not None:
                    tensor = param.weight.data.clone().cpu()
                elif hasattr(param, "bias") and param.bias is not None:
                    tensor = param.bias.data.clone().cpu()
                else:
                    continue

                tensor_id = str(uuid.uuid4())
                normalized_tensor = _normalize_tensor(tensor)
                metrics = _calculate_metrics(normalized_tensor, device)

                _store_tensor(cursor, tensor_id, model_id, name, tensor)
                _store_metrics(cursor, tensor_id, metrics)

        # Handle other parameters outside of the layers
        other_params = {
            "model.embed_tokens": model.transformer.wte,
            "model.norm": model.transformer.ln_f,
            "lm_head": model.lm_head,
        }

        for name, param in other_params.items():
            if hasattr(param, "weight") and param.weight is not None:
                tensor = param.weight.data.clone().cpu()
            elif hasattr(param, "bias") and param.bias is not None:
                tensor = param.bias.data.clone().cpu()
            else:
                continue

            tensor_id = str(uuid.uuid4())
            normalized_tensor = _normalize_tensor(tensor)
            metrics = _calculate_metrics(normalized_tensor, device)

            _store_tensor(cursor, tensor_id, model_id, name, tensor)
            _store_metrics(cursor, tensor_id, metrics)

        conn.commit()


def ensure_model_files(model_name: str, models_dir: str) -> Optional[str]:
    """Ensures model files exist locally, downloading if necessary."""
    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if os.path.exists(local_path):
        if all(os.path.exists(os.path.join(local_path, f)) for f in ["config.json"]):
            return local_path

    logger.info(f"Downloading or completing download of {model_name} to {local_path}")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            revision="main",
        )
        return local_path
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return None


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


@torch.inference_mode()
def main(config_path: str):
    """Main function to run the model analysis process."""
    config = load_config(config_path)
    database_dir = config["database_dir"]
    models_dir = config["models_dir"]

    # Determine the device for metric calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for metric calculations")

    init_database(database_dir)

    os.makedirs(models_dir, exist_ok=True)

    models = [config["base_model"]] + config["fine_tuned_models"]
    for model_name in models:
        local_model_path = ensure_model_files(model_name, models_dir)
        if not local_model_path:
            logger.error(f"Failed to get model files for {model_name}")
            continue
        process_and_store_model(model_name, local_model_path, database_dir, device)

    logger.info("Model processing and metric calculation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified model management and analysis tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mastermerge_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args.config)
