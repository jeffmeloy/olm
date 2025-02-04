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


class ModelDatabase:
    """Handles interactions with the SQLite database for storing and retrieving model data."""

    def __init__(self, database_dir: str):
        self.db_path = Path(database_dir) / "models.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initializes the SQLite database with the required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT UNIQUE NOT NULL,
                    config_json TEXT,
                    tokenizer_json TEXT,
                    tokenizer_config_json TEXT,
                    vocab_json TEXT,
                    generation_config_json TEXT,
                    added_tokens_json TEXT,
                    special_tokens_map_json TEXT
                );

                CREATE TABLE IF NOT EXISTS layers (
                    layer_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    layer_index TEXT NOT NULL,
                    layer_name TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(model_id),
                    UNIQUE(model_id, layer_index, layer_name)
                );

                CREATE TABLE IF NOT EXISTS tensors (
                    tensor_id TEXT PRIMARY KEY,
                    layer_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    tensor_data BLOB NOT NULL,
                    tensor_shape TEXT NOT NULL,
                    tensor_dtype TEXT NOT NULL,
                    FOREIGN KEY(layer_id) REFERENCES layers(layer_id)
                );

                CREATE TABLE IF NOT EXISTS tensor_metrics (
                    tensor_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                    UNIQUE(tensor_id, metric_name)
                );

                CREATE TABLE IF NOT EXISTS layer_compositions (
                    layer_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    layer_index TEXT NOT NULL,
                    source_model TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(model_id)
                );

                CREATE TABLE IF NOT EXISTS layer_metrics (
                    layer_id TEXT NOT NULL, 
                    dataset TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY(layer_id) REFERENCES layer_compositions(layer_id)
                );
            """
            )

    def load_model_data(self, model_name: str) -> dict:
        """Loads model configuration data from the database."""
        with sqlite3.connect(self.db_path) as conn:
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

    def load_tensors_and_metrics(self, model_name: str) -> dict:
        """Loads tensors and their metrics from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT model_id FROM models WHERE model_name = ?", (model_name,)
            )
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Model {model_name} not found in database")
            model_id = result[0]

            cursor.execute(
                "SELECT layer_id, layer_index, layer_name FROM layers WHERE model_id = ?",
                (model_id,),
            )
            layers = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

            tensors_and_metrics = {}
            for layer_id, (layer_index, layer_name) in layers.items():
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
                    WHERE t.layer_id = ?
                    GROUP BY t.tensor_id, t.name
                    """,
                    (layer_id,),
                )

                layer_tensors = {}
                for (
                    name,
                    tensor_data,
                    shape_str,
                    dtype_str,
                    metrics_str,
                ) in cursor.fetchall():
                    shape = json.loads(shape_str)
                    tensor = (
                        torch.frombuffer(
                            tensor_data,
                            dtype=getattr(torch, dtype_str.split(".")[-1]),
                        )
                        .reshape(shape)
                        .clone()
                        .to("cpu")
                    )

                    metrics = {}
                    if metrics_str:
                        for metric_item in metrics_str.split(","):
                            metric_name, value = metric_item.split(":")
                            metrics[metric_name] = float(value)

                    layer_tensors[name] = {"tensor": tensor, "metrics": metrics}

                if layer_index.isdigit():
                    layer_key = f"model.layers.{layer_index}.{layer_name}"
                else:
                    layer_key = f"{layer_index}.{layer_name}"
                tensors_and_metrics[layer_key] = layer_tensors

        return tensors_and_metrics

    def store_model_data(
        self,
        model_id: str,
        model_name: str,
        model_configs: dict,
    ) -> None:
        """Stores model configuration data into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
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
                conn.commit()
            except sqlite3.IntegrityError as e:
                logger.error(f"Error storing model data for {model_name}: {e}")
                raise

    def store_layer_data(
        self,
        cursor: sqlite3.Cursor,
        layer_id: str,
        model_id: str,
        layer_idx: int | str,
        component_name: str,
    ) -> None:
        """Stores layer metadata."""
        try:
            cursor.execute(
                "INSERT INTO layers (layer_id, model_id, layer_index, layer_name) VALUES (?, ?, ?, ?)",
                (layer_id, model_id, layer_idx, component_name),
            )
        except sqlite3.IntegrityError as e:
            logger.error(
                f"Error storing layer data for layer {layer_idx}-{component_name}: {e}"
            )
            raise

    def store_tensor_data(
        self,
        cursor: sqlite3.Cursor,
        tensor_id: str,
        layer_id: str,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Stores tensor data."""
        try:
            cursor.execute(
                """
                INSERT INTO tensors (tensor_id, layer_id, name, tensor_data, tensor_shape, tensor_dtype)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tensor_id,
                    layer_id,
                    name,
                    tensor.numpy().tobytes(),
                    json.dumps(list(tensor.shape)),
                    str(tensor.dtype),
                ),
            )
        except sqlite3.Error as e:
            logger.error(f"Error storing tensor data for tensor {name}: {e}")
            raise

    def store_tensor_metrics(
        self, cursor: sqlite3.Cursor, tensor_id: str, metrics: dict[str, float]
    ) -> None:
        """Stores tensor metrics."""
        try:
            cursor.executemany(
                """
                INSERT INTO tensor_metrics (tensor_id, metric_name, metric_value)
                VALUES (?, ?, ?)
                """,
                [(tensor_id, name, value) for name, value in metrics.items()],
            )
        except sqlite3.Error as e:
            logger.error(f"Error storing metrics for tensor ID {tensor_id}: {e}")
            raise


class ModelLoader:
    """Handles downloading and loading models from Hugging Face Hub or local paths."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def ensure_model_files(self, model_name: str) -> str:
        """Ensures model files exist locally, downloading if necessary."""
        local_path = os.path.join(self.models_dir, model_name.replace("/", "_"))
        if os.path.exists(local_path):
            if all(
                os.path.exists(os.path.join(local_path, f)) for f in ["config.json"]
            ):
                return local_path

        logger.info(
            f"Downloading or completing download of {model_name} to {local_path}"
        )
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
            raise

    def load_model(self, model_path: str) -> AutoModelForCausalLM:
        """Loads a model from a local path to CPU."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": "cpu"},
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise


class TensorAnalyzer:
    """Calculates and stores metrics for model tensors."""

    def __init__(self, device: torch.device):
        self.device = device

    def calculate_metrics(self, tensor: torch.Tensor) -> dict[str, float]:
        """Calculate all metrics for a tensor."""
        metrics = {}
        tensor_device = tensor.to(self.device)

        all_metrics = {
            "snr": self._calculate_snr,
            "normalized_effective_rank": self._calculate_normalized_effective_rank,
            "svd_skewness": self._calculate_svd_skewness,
            "bzip2": self._calculate_bzip2,
            "kurtosis": self._calculate_weight_kurtosis,
            "skewness": self._calculate_weight_skewness,
            "sparsity": self._calculate_weight_sparsity,
            "spectral_norm": self._calculate_weight_spectral_norm,
            "frobenius_norm": self._calculate_frobenius_norm,
            "weight_entropy": self._calculate_weight_entropy,
            "stable_rank": self._calculate_stable_rank,
        }

        for metric_name, metric_func in all_metrics.items():
            try:
                metrics[metric_name] = metric_func(tensor_device)
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                metrics[metric_name] = float("nan")

        del tensor_device
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return metrics

    def _calculate_snr(self, A: torch.Tensor) -> float:
        """Calculate signal-to-noise ratio (SNR) metric, higher = more signal."""
        S = torch.linalg.svdvals(A)
        max_singular_value = S[0]
        sigma_estimated = self._estimate_sigma_with_full_iqr(S)
        n, m = A.shape[-2:]
        mp_threshold = self._marchenko_pastur_threshold(sigma_estimated, n, m)
        signal = torch.sum(S[S > mp_threshold])  # More efficient masking
        noise = torch.sum(S[S <= mp_threshold])
        snr = signal / max(noise, 1e-10)
        return (snr / max_singular_value).item()

    def _estimate_sigma_with_full_iqr(self, S: torch.Tensor) -> float:
        """Estimate sigma using full interquartile range (IQR) of singular values."""
        q75, q25 = torch.quantile(S, torch.tensor([0.75, 0.25], device=S.device))
        return ((q75 - q25) / 1.349).item()

    def _marchenko_pastur_threshold(self, sigma: float, n: int, m: int) -> float:
        """Calculate Marchenko-Pastur threshold."""
        beta = min(n, m) / max(n, m)
        return sigma * (1 + np.sqrt(beta)) ** 2

    def _calculate_svd_skewness(self, A: torch.Tensor) -> float:
        """Calculates the skewness of the singular value distribution."""
        S = torch.linalg.svdvals(A)
        return (1 - (torch.mean(S) / S[0])).item()

    def _calculate_normalized_effective_rank(self, A: torch.Tensor) -> float:
        """Calculates the normalized effective rank."""
        S = torch.linalg.svdvals(A)
        S_norm = S / S.sum()
        effective_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm)))
        actual_rank = torch.linalg.matrix_rank(A)
        return (effective_rank / actual_rank).item()

    def _calculate_bzip2(self, A: torch.Tensor) -> float:
        """Calculates the bzip2 compression ratio."""
        A_bytes = A.cpu().contiguous().numpy().tobytes()
        return len(bz2.compress(A_bytes)) / len(A_bytes)

    def _calculate_weight_kurtosis(self, A: torch.Tensor) -> float:
        """Calculates the kurtosis of the weight distribution."""
        return torch.kurtosis(A.flatten()).item()

    def _calculate_weight_skewness(self, A: torch.Tensor) -> float:
        """Calculates the skewness of the weight distribution."""
        return torch.skew(A.flatten()).item()

    def _calculate_weight_sparsity(self, A: torch.Tensor) -> float:
        """Calculates the sparsity of the weights (proportion of near-zero values)."""
        return (torch.abs(A) < 1e-5).float().mean().item()

    def _calculate_weight_spectral_norm(self, A: torch.Tensor) -> float:
        """Calculates the spectral norm (largest singular value)."""
        return torch.linalg.norm(A, ord=2).item()

    def _calculate_frobenius_norm(self, A: torch.Tensor) -> float:
        """Calculates the Frobenius norm."""
        return torch.linalg.norm(A).item()

    def _calculate_weight_entropy(self, A: torch.Tensor) -> float:
        """Calculates the entropy of the weight distribution."""
        A = A.flatten()
        num_bins = int(np.sqrt(min(A.shape))) if A.numel() > 1 else 2
        num_bins = max(num_bins, 2)
        hist = torch.histogram(A, bins=num_bins)
        probs = hist.hist / hist.hist.sum()
        return (-torch.sum(probs * torch.log2(probs + 1e-12))).item()

    def _calculate_stable_rank(self, A: torch.Tensor) -> float:
        """Calculates the stable rank."""
        S = torch.linalg.svdvals(A)
        return (torch.sum(S**2) / S[0] ** 2).item()

    def _normalize_tensor(self, A: torch.Tensor) -> torch.Tensor:
        """Normalizes tensor magnitude while preserving signs."""
        signs = A.sign()
        A = A.abs()
        A /= A.max().clamp(min=1e-8)
        return A * signs


class QwenModelProcessor:
    """Processes Qwen models, handling layer iteration and data storage."""

    def __init__(
        self,
        model_loader: ModelLoader,
        tensor_analyzer: TensorAnalyzer,
        database: ModelDatabase,
    ):
        self.model_loader = model_loader
        self.tensor_analyzer = tensor_analyzer
        self.database = database

    def process_and_store_model(
        self,
        model_name: str,
        model_path: str,
    ) -> None:
        """Process and store model data and tensors for Qwen models."""
        model_id = str(uuid.uuid4())
        model_configs = self._load_model_configs(model_path)

        self.database.store_model_data(model_id, model_name, model_configs)

        model = self.model_loader.load_model(model_path)
        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()

            qwen_layers = model.transformer.h
            for layer_idx, layer in enumerate(qwen_layers):
                logger.info(f"Processing layer {layer_idx}")
                self._process_layer(cursor, model_id, layer_idx, layer)

            # these are actually layers but treated differently because they are not in the layer loop
            boundary_layers = {
                "embed_tokens": model.transformer.wte,
                "norm": model.transformer.ln_f,
                "lm_head": model.lm_head,
            }

            for component_name, param in boundary_layers.items():
                self._process_boundary_layer(cursor, model_id, component_name, param)

            conn.commit()

    def _process_layer(
        self, cursor: sqlite3.Cursor, model_id: str, layer_idx: int, layer
    ) -> None:
        """Processes a single Qwen layer."""
        layer_params = {
            "self_attn.q_proj": layer.attn.q_proj,
            "self_attn.k_proj": layer.attn.k_proj,
            "self_attn.v_proj": layer.attn.v_proj,
            "self_attn.o_proj": layer.attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
            "input_layernorm": layer.ln_1,
            "post_attention_layernorm": layer.ln_2,
        }

        for component_name, param in layer_params.items():
            layer_id = str(uuid.uuid4())
            self.database.store_layer_data(
                cursor, layer_id, model_id, layer_idx, component_name
            )

            tensor_name, tensor = self._get_tensor(param)
            if tensor_name is not None:
                self._process_tensor(cursor, layer_id, tensor_name, tensor)

    def _process_boundary_layer(
        self, cursor: sqlite3.Cursor, model_id: str, component_name: str, param
    ) -> None:
        """Processes boundary layers (embedding, final norm, lm_head)."""
        layer_id = str(uuid.uuid4())
        self.database.store_layer_data(
            cursor, layer_id, model_id, component_name, "weight"
        )

        tensor_name, tensor = self._get_tensor(param)
        if tensor_name is not None:
            self._process_tensor(cursor, layer_id, tensor_name, tensor)

    def _get_tensor(self, param) -> tuple[Optional[str], Optional[torch.Tensor]]:
        """Retrieves the weight or bias tensor from a parameter."""
        if hasattr(param, "weight") and param.weight is not None:
            return "weight", param.weight.data.clone().cpu()
        elif hasattr(param, "bias") and param.bias is not None:
            return "bias", param.bias.data.clone().cpu()
        else:
            return None, None

    def _process_tensor(
        self,
        cursor: sqlite3.Cursor,
        layer_id: str,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> None:
        """Calculates metrics, normalizes, and stores a tensor."""
        tensor_id = str(uuid.uuid4())
        normalized_tensor = self.tensor_analyzer._normalize_tensor(tensor)
        metrics = self.tensor_analyzer.calculate_metrics(normalized_tensor)

        self.database.store_tensor_data(
            cursor, tensor_id, layer_id, tensor_name, tensor
        )
        self.database.store_tensor_metrics(cursor, tensor_id, metrics)

    def _load_model_configs(self, model_path: str) -> dict[str, str]:
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
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        configs[filename.replace(".json", "_json")] = f.read()
                except Exception as e:
                    logger.error(f"Error loading config file {filename}: {e}")
                    configs[filename.replace(".json", "_json")] = None
            else:
                configs[filename.replace(".json", "_json")] = None

        return configs


class ModelLoaderFromDatabase:
    """Handles loading models and tokenizers from the database."""

    def __init__(self, database_dir: str):
        self.database = ModelDatabase(database_dir)
        self.temp_dir = Path(database_dir) / "temp_model_files"

    def transfer_files_to_temp_dir(self, model_name: str, file_keys: list[str]) -> bool:
        """Transfers specified files from the database to a temporary directory."""
        self.temp_dir.mkdir(exist_ok=True)
        self._clear_temp_directory()

        try:
            model_data = self.database.load_model_data(model_name)
        except ValueError as e:
            logger.error(f"Error loading model data for {model_name}: {e}")
            return False

        for key in file_keys:
            data = model_data.get(key)
            if not data:
                logger.error(
                    f"Data for {key} not found in the database for {model_name}."
                )
                return False

            file_path = self.temp_dir / f"{key}.json"
            try:
                with open(file_path, "w") as f:
                    f.write(data)
            except Exception as e:
                logger.error(f"Error writing data to {file_path}: {e}")
                return False

        return True

    def _clear_temp_directory(self) -> None:
        """Clears all files and directories within the temporary directory."""
        for item in self.temp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def get_model_from_db(
        self, model_name: str, device: str = "cpu"
    ) -> Optional[AutoModelForCausalLM]:
        """Loads a model from the database to the specified device (default: CPU)"""
        if not self.transfer_files_to_temp_dir(model_name, ["config"]):
            return None  # Transfer failed

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.temp_dir),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": device},
            )
        except Exception as e:
            logger.error(f"Error loading model from temporary directory: {e}")
            return None

        try:
            tensors_data = self.database.load_tensors_and_metrics(model_name)
        except ValueError as e:
            logger.error(f"Error loading tensors and metrics for {model_name}: {e}")
            return None

        for layer_key, tensor_dict in tensors_data.items():
            for tensor_name, tensor_data in tensor_dict.items():
                with torch.no_grad():
                    param_name = f"{layer_key}.{tensor_name}"
                    param = model.get_parameter(param_name)
                    if param is not None:
                        param.copy_(tensor_data["tensor"].to("cpu"))

        return model

    def get_tokenizer_from_db(self, model_name: str) -> Optional[AutoTokenizer]:
        """Loads a tokenizer from the database."""
        if not self.transfer_files_to_temp_dir(
            model_name, ["tokenizer_config", "tokenizer"]
        ):
            return None  # Transfer failed

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.temp_dir))
        except Exception as e:
            logger.error(f"Error loading tokenizer from temporary directory: {e}")
            return None

        return tokenizer


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


@torch.inference_mode()
def main(config_path: str):
    """Main function to run the model analysis process."""
    config = load_config(config_path)
    database_dir = config["database_dir"]
    models_dir = config["models_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for metric calculations")

    os.makedirs(models_dir, exist_ok=True)

    database = ModelDatabase(database_dir)
    model_loader = ModelLoader(models_dir)
    tensor_analyzer = TensorAnalyzer(device)
    model_processor = QwenModelProcessor(model_loader, tensor_analyzer, database)

    models = [config["base_model"]] + config["fine_tuned_models"]
    for model_name in models:
        try:
            local_model_path = model_loader.ensure_model_files(model_name)
            model_processor.process_and_store_model(model_name, local_model_path)
        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")

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
