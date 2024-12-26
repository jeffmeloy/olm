import argparse
import json
import os
import sqlite3
import bz2
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import tempfile


"""
1. Load a complete Hugging Face model (architecture and weights) from an SQLite database 
    created by the `save_model_to_database` and `save_model_artifacts` functions.
2. Load a Hugging Face tokenizer from the database, reconstructing the necessary files.

The database is assumed to have the following table structure (created by the original code):

-   `models`: Stores general model information, including configuration files as TEXT.
    -   `model_id` (INTEGER PRIMARY KEY)
    -   `model_name` (TEXT UNIQUE NOT NULL)
    -   `config_json` (TEXT)
    -   `tokenizer_json` (TEXT)
    -   `tokenizer_config_json` (TEXT)
    -   `special_tokens_map_json` (TEXT)
    -   `vocab_json` (TEXT)
    -   ... (other columns)
-   `layers`: Stores layer-specific information, including weights as BLOBs.
    -   `layer_id` (INTEGER PRIMARY KEY)
    -   `model_id` (INTEGER, FOREIGN KEY referencing `models.model_id`)
    -   `layer_name` (TEXT)
    -   `layer_tensor` (BLOB)
    -   `tensor_shape` (TEXT)
    -   ... (other columns)
-   `positional_embeddings`: Stores positional embedding data (if available).
    -   `embedding_id` (INTEGER PRIMARY KEY)
    -   `model_id` (INTEGER, FOREIGN KEY referencing `models.model_id`)
    -   `embedding_data` (BLOB)
    -   ... (other columns)

"""


def load_model_from_database(
    model_name: str, database_dir: str
) -> Optional[AutoModelForCausalLM]:
    """Loads a model from the database.

    Args:
        model_name: The name of the model to load.
        database_dir: The directory where the database is located.

    Returns:
        The loaded Hugging Face model object, or None if an error occurred.
    """
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1. Get Model Configuration
        cursor.execute(
            "SELECT model_type, config_json FROM models WHERE model_name = ?",
            (model_name,),
        )
        result = cursor.fetchone()
        if not result:
            print(f"Model {model_name} not found in database")
            return None

        model_type, config_json = result
        try:
            config = AutoConfig.from_dict(json.loads(config_json))
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.bfloat16, device_map="auto"
            )
        except Exception as e:
            print(f"Error initializing model from config: {e}")
            return None

        # 2. Get Model ID
        model_id = _get_model_id(cursor, model_name)

        # 3. Load Layer Weights and Metadata
        cursor.execute(
            """
            SELECT layer_name, layer_tensor, tensor_shape, dtype, metadata_json
            FROM layers
            WHERE model_id = ?
        """,
            (model_id,),
        )
        for layer_name, tensor_bytes, shape_json, dtype_str, metadata_json in cursor.fetchall():
            try:
                tensor_shape = json.loads(shape_json)
                tensor_dtype = getattr(torch, dtype_str.split(".")[-1])
                tensor = torch.from_numpy(
                    np.frombuffer(tensor_bytes, dtype=np.float32).reshape(tensor_shape)
                ).to(dtype=tensor_dtype)

                layer_module = _get_module_by_name(model, layer_name)
                if layer_module:
                    layer_module.weight.data = tensor
                    metadata = json.loads(metadata_json)
                    if metadata.get("bias") is not None:
                        layer_module.bias.data = torch.tensor(
                            metadata["bias"], dtype=tensor_dtype, device=tensor.device
                        )
            except Exception as e:
                print(f"Error loading layer {layer_name}: {e}")
                return None

        # 4. Load Positional Embeddings (if present)
        cursor.execute(
            "SELECT embedding_data, embedding_config_json FROM positional_embeddings WHERE model_id = ?",
            (model_id,),
        )
        pos_emb_row = cursor.fetchone()
        if pos_emb_row and hasattr(model, "set_position_embeddings"):
            emb_data, emb_config_json = pos_emb_row
            try:
                pos_emb = torch.from_numpy(np.frombuffer(emb_data, dtype=np.float32))
                model.set_position_embeddings(pos_emb)
            except Exception as e:
                print(f"Error loading positional embeddings: {e}")
                return None

        return model

def load_tokenizer_from_database(
    model_name: str, database_dir: str, output_dir: Optional[str] = None
) -> Tuple[Optional[AutoTokenizer], Optional[Path]]:
    """Loads a tokenizer from the database.

    Args:
        model_name: The name of the model whose tokenizer to load.
        database_dir: The directory where the database is located.
        output_dir: Optional directory to save the tokenizer files to. 
                    If None, a temporary directory will be used.

    Returns:
        A tuple containing:
        - The loaded Hugging Face tokenizer object, or None if an error occurred.
        - The path to the directory containing the tokenizer files.
    """
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                tokenizer_json,
                tokenizer_config_json,
                special_tokens_map_json,
                vocab_json
            FROM models
            WHERE model_name = ?
        """,
            (model_name,),
        )
        row = cursor.fetchone()
        if not row:
            print(f"Tokenizer data not found for model {model_name} in database")
            return None, None

        tokenizer_json, tokenizer_config_json, special_tokens_map_json, vocab_json = row

        # Create a temporary or specified directory for tokenizer files
        if output_dir:
            tokenizer_dir = Path(output_dir) / f"{model_name.replace('/', '_')}_tokenizer"
        else:
            tokenizer_dir = Path(tempfile.mkdtemp()) / f"{model_name.replace('/', '_')}_tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        # Write tokenizer files to the directory
        try:
            if tokenizer_json:
                with open(tokenizer_dir / "tokenizer.json", "w", encoding="utf-8") as f:
                    f.write(tokenizer_json)
            if tokenizer_config_json:
                with open(tokenizer_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
                    f.write(tokenizer_config_json)
            if special_tokens_map_json:
                with open(tokenizer_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
                    f.write(special_tokens_map_json)
            if vocab_json:
                with open(tokenizer_dir / "vocab.json", "w", encoding="utf-8") as f:
                    f.write(vocab_json)

        except Exception as e:
            print(f"Error writing tokenizer files to {tokenizer_dir}: {e}")
            return None, None

        # Load the tokenizer from the directory
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            return tokenizer, tokenizer_dir
        except Exception as e:
            print(f"Error loading tokenizer from {tokenizer_dir}: {e}")
            return None, None

# --- Helper Functions ---

def _get_model_id(cursor: sqlite3.Cursor, model_name: str) -> int:
    """Helper function to get model_id from the database."""
    cursor.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,))
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found in database")
    return result[0]

def _get_module_by_name(
    model: torch.nn.Module, module_name: str
) -> Optional[torch.nn.Module]:
    """Helper function to get a module by its hierarchical name."""
    for name, module in model.named_modules():
        if name == module_name:
            return module
    return None




def init_database(database_dir: str) -> None:
    """Initializes the SQLite database."""
    db_path = Path(database_dir) / "models.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                tokenizer_json TEXT,
                tokenizer_config_json TEXT,
                special_tokens_map_json TEXT,
                vocab_json TEXT,
                generation_config_json TEXT,
                model_card_json TEXT,
                version TEXT NOT NULL,
                creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS layers (
                layer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                layer_name TEXT NOT NULL,
                layer_type TEXT NOT NULL,
                layer_tensor BLOB NOT NULL,
                tensor_shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                device TEXT NOT NULL,
                requires_grad BOOLEAN NOT NULL,
                metadata_json TEXT,
                layer_config_json TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id),
                UNIQUE(model_id, layer_name)
            );

            CREATE TABLE IF NOT EXISTS attention_configs (
                attention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_id INTEGER NOT NULL,
                num_attention_heads INTEGER NOT NULL,
                attention_config_json TEXT NOT NULL,
                FOREIGN KEY (layer_id) REFERENCES layers(layer_id)
            );

            CREATE TABLE IF NOT EXISTS positional_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                embedding_type TEXT NOT NULL,
                embedding_data BLOB NOT NULL,
                embedding_config_json TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (layer_id) REFERENCES layers(layer_id),
                UNIQUE(layer_id, metric_name)
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                version_number TEXT NOT NULL,
                commit_hash TEXT,
                version_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );
        """
        )

def _insert_or_update_model_record(
    cursor: sqlite3.Cursor, model_name: str, model_data: dict
):
    """Inserts or updates a model record in the database."""
    cursor.execute(
        """
        INSERT OR REPLACE INTO models (
            model_name, model_type, config_json, tokenizer_json, tokenizer_config_json,
            special_tokens_map_json, vocab_json, generation_config_json, model_card_json, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            model_name,
            model_data.get("model_type"),
            model_data.get("config_json"),
            model_data.get("tokenizer_json"),
            model_data.get("tokenizer_config_json"),
            model_data.get("special_tokens_map_json"),
            model_data.get("vocab_json"),
            model_data.get("generation_config_json"),
            model_data.get("model_card_json"),
            model_data.get("version", "1.0"),
        ),
    )


def save_model_to_database(
    model: AutoModelForCausalLM, model_name: str, database_dir: str
):
    """Saves model information to the database."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            model_data = {
                "model_type": model.__class__.__name__,
                "config_json": json.dumps(model.config.to_dict()),
                "version": "1.0",
            }
            _insert_or_update_model_record(cursor, model_name, model_data)
            model_id = _get_model_id(cursor, model_name)

            for layer_name, module in model.named_modules():
                if hasattr(module, "weight"):
                    _save_layer_data(cursor, model_id, layer_name, module)
                    if hasattr(module, "num_attention_heads"):
                        _save_attention_config(cursor, cursor.lastrowid, module)

            if hasattr(model, "get_position_embeddings"):
                _save_positional_embeddings(cursor, model_id, model)

        except Exception as e:
            print(f"Error saving model {model_name} to database: {e}")
            conn.rollback()
            raise


def _save_layer_data(
    cursor: sqlite3.Cursor,
    model_id: int,
    layer_name: str,
    module: torch.nn.Module,
):
    """Saves layer data to the database."""
    tensor = module.weight.data
    metadata = {
        "bias": module.bias.data.cpu().numpy().tolist()
        if hasattr(module, "bias") and module.bias is not None
        else None,
        "in_features": getattr(module, "in_features", None),
        "out_features": getattr(module, "out_features", None),
        "activation_function": getattr(
            module, "activation_function", None
        ).__class__.__name__
        if hasattr(module, "activation_function")
        else None,
    }
    layer_config = _get_layer_config(module)
    cursor.execute(
        """
        INSERT OR REPLACE INTO layers (
            model_id, layer_name, layer_type, layer_tensor, tensor_shape, dtype,
            device, requires_grad, metadata_json, layer_config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            model_id,
            layer_name,
            module.__class__.__name__,
            tensor.cpu().numpy().tobytes(),
            json.dumps(list(tensor.shape)),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            json.dumps(metadata),
            json.dumps(layer_config),
        ),
    )


def _get_layer_config(module: torch.nn.Module) -> dict:
    """Extracts layer configuration based on layer type."""
    if hasattr(module, "config"):
        return module.config.to_dict()
    elif isinstance(module, torch.nn.Linear):
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    elif isinstance(module, torch.nn.LayerNorm):
        return {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
        }
    return {}


def _save_attention_config(
    cursor: sqlite3.Cursor, layer_id: int, module: torch.nn.Module
):
    """Saves attention configuration to the database."""
    attention_config = {
        "num_attention_heads": module.num_attention_heads,
        "attention_head_size": getattr(module, "attention_head_size", None),
        "attention_dropout": getattr(module, "attention_dropout", None),
        "attention_type": getattr(module, "attention_type", "default"),
    }
    cursor.execute(
        """
        INSERT OR REPLACE INTO attention_configs (
            layer_id, num_attention_heads, attention_config_json
        ) VALUES (?, ?, ?)
    """,
        (layer_id, module.num_attention_heads, json.dumps(attention_config)),
    )


def _save_positional_embeddings(
    cursor: sqlite3.Cursor, model_id: int, model: AutoModelForCausalLM
):
    """Saves positional embeddings to the database."""
    pos_emb = model.get_position_embeddings()
    if pos_emb is not None:
        emb_config = {
            "max_position_embeddings": getattr(
                model.config, "max_position_embeddings", None
            ),
            "position_embedding_type": getattr(
                model.config, "position_embedding_type", "absolute"
            ),
        }
        cursor.execute(
            """
            INSERT OR REPLACE INTO positional_embeddings (
                model_id, embedding_type, embedding_data, embedding_config_json
            ) VALUES (?, ?, ?, ?)
        """,
            (
                model_id,
                emb_config["position_embedding_type"],
                pos_emb.cpu().numpy().tobytes(),
                json.dumps(emb_config),
            ),
        )


def save_model_artifacts(model_path: str, model_name: str, database_dir: str):
    """Saves model artifacts (config files) to the database."""
    db_path = Path(database_dir) / "models.db"
    model_path = Path(model_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            artifacts = {}
            for filename in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "generation_config.json",
                "README.md",
            ]:
                filepath = model_path / filename
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        artifacts[filename.replace(".", "_")] = f.read()
                else:
                    artifacts[filename.replace(".", "_")] = None

            # Determine model type
            model_type = "AutoModelForCausalLM"  # Default, can be refined
            if artifacts["config_json"]:
                config_data = json.loads(artifacts["config_json"])
                model_type = config_data.get("_name_or_path", model_type)

            model_data = {
                "model_type": model_type,
                "config_json": artifacts["config_json"],
                "tokenizer_json": artifacts["tokenizer_json"],
                "tokenizer_config_json": artifacts["tokenizer_config_json"],
                "special_tokens_map_json": artifacts["special_tokens_map_json"],
                "vocab_json": artifacts["vocab_json"],
                "generation_config_json": artifacts["generation_config_json"],
                "model_card_json": artifacts["README_md"],
                "version": "1.0",
            }
            _insert_or_update_model_record(cursor, model_name, model_data)

        except Exception as e:
            print(f"Error saving model artifacts for {model_name} to database: {e}")
            conn.rollback()
            raise


def load_model_artifacts(model_name: str, output_dir: str, database_dir: str) -> Path:
    """Loads model artifacts from the database and reconstructs the model directory."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                config_json, tokenizer_json, tokenizer_config_json,
                special_tokens_map_json, vocab_json, generation_config_json,
                model_card_json
            FROM models WHERE model_name = ?
        """,
            (model_name,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Model {model_name} not found in database")

        model_dir = Path(output_dir) / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "config.json": row[0],
            "tokenizer.json": row[1],
            "tokenizer_config.json": row[2],
            "special_tokens_map.json": row[3],
            "vocab.json": row[4],
            "generation_config.json": row[5],
            "README.md": row[6],
        }
        for filename, content in artifacts.items():
            if content:
                with open(model_dir / filename, "w", encoding="utf-8") as f:
                    f.write(content)

        return model_dir


def save_metrics_to_database(model_name: str, layer_metrics: dict, database_dir: str):
    """Saves metrics for each layer to the database."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            model_id = _get_model_id(cursor, model_name)
            for layer_name, metrics in layer_metrics.items():
                cursor.execute(
                    "SELECT layer_id FROM layers WHERE model_id = ? AND layer_name = ?",
                    (model_id, layer_name),
                )
                result = cursor.fetchone()
                if not result:
                    print(
                        f"Warning: Layer {layer_name} not found for model {model_name}"
                    )
                    continue
                layer_id = result[0]

                for metric_name, metric_value in metrics.items():
                    try:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO metrics (
                                layer_id, metric_name, metric_value, calculation_date
                            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                            (layer_id, metric_name, float(metric_value)),
                        )
                    except Exception as e:
                        print(
                            f"Error saving metric {metric_name} for layer {layer_name}: {e}"
                        )
        except Exception as e:
            print(f"Error saving metrics for {model_name} to database: {e}")
            conn.rollback()
            raise


def get_metrics_from_database(
    model_name: str, database_dir: str, metric_names: Optional[list[str]] = None
) -> dict:
    """Retrieves metrics for a model from the database."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        model_id = _get_model_id(cursor, model_name)

        query = """
            SELECT l.layer_name, m.metric_name, m.metric_value, m.calculation_date
            FROM layers l
            JOIN metrics m ON l.layer_id = m.layer_id
            WHERE l.model_id = ?
        """
        params = [model_id]
        if metric_names:
            query += f" AND m.metric_name IN ({','.join(['?']*len(metric_names))})"
            params.extend(metric_names)

        cursor.execute(query, params)
        results = {}
        for layer_name, metric_name, metric_value, calc_date in cursor.fetchall():
            results.setdefault(layer_name, {}).update(
                {metric_name: {"value": metric_value, "calculation_date": calc_date}}
            )

        return results


def download_model(model_name: str, models_dir: str) -> str:
    """Downloads a model from the Hugging Face Hub."""
    local_path = Path(models_dir) / model_name.replace("/", "_")
    if not local_path.exists():
        print(f"Downloading {model_name} to {local_path}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                revision="main",
            )
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            raise
    else:
        print(f"Model {model_name} already exists at {local_path}")
    return str(local_path)


def load_model(model_path: str) -> AutoModelForCausalLM:
    """Loads a model from a local path."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


def get_model_layer_metrics(model_name: str, database_dir: str) -> dict:
    """Gets all metrics for all layers of a specific model."""
    return get_metrics_from_database(model_name, database_dir)


def compare_model_metrics(
    model_names: list[str], database_dir: str, metric_names: Optional[list[str]] = None
) -> dict:
    """Compares metrics between multiple models."""
    return {
        model_name: get_metrics_from_database(model_name, database_dir, metric_names)
        for model_name in model_names
    }


def calculate_metric(
    layer: torch.Tensor, metric_name: str, all_layers: Optional[list] = None
) -> float:
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
    elif metric_name == "weight_kurtosis":
        return _calculate_weight_kurtosis(layer)
    elif metric_name == "weight_skewness":
        return _calculate_weight_skewness(layer)
    elif metric_name == "weight_sparsity":
        return _calculate_weight_sparsity(layer)
    elif metric_name == "weight_spectral_norm":
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
    A /= A.max().clamp(min=1e-8)  # In-place division with clamp
    return A * signs


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def ensure_model_files(model_name: str, models_dir: str) -> Optional[str]:
    """Ensures model files exist locally, downloading if necessary."""
    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if os.path.exists(local_path):
        # Check for essential files
        if all(
            os.path.exists(os.path.join(local_path, f))
            for f in ["config.json", "pytorch_model.bin"]
        ):
            return local_path

    # Download if not found or incomplete
    print(f"Downloading or completing download of {model_name} to {local_path}")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            revision="main",
        )
        return local_path
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return None


def save_model_complete(
    model: AutoModelForCausalLM, model_name: str, model_path: str, database_dir: str
):
    """Saves complete model (artifacts and weights) to the database."""
    save_model_artifacts(model_path, model_name, database_dir)
    save_model_to_database(model, model_name, database_dir)


def _get_module_by_name(
    model: torch.nn.Module, module_name: str
) -> Optional[torch.nn.Module]:
    """Helper function to get a module by its hierarchical name."""
    for name, module in model.named_modules():
        if name == module_name:
            return module
    return None


def model_exists_in_database(model_name: str, database_dir: str) -> bool:
    """Checks if a model exists in the database."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM models WHERE model_name = ?", (model_name,)
        )
        return cursor.fetchone()[0] > 0


def metrics_exist_in_database(model_name: str, database_dir: str) -> bool:
    """Checks if metrics exist for a model in the database."""
    db_path = Path(database_dir) / "models.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM models m
            JOIN layers l ON m.model_id = l.model_id
            JOIN metrics mt ON l.layer_id = mt.layer_id
            WHERE m.model_name = ?
        """,
            (model_name,),
        )
        return cursor.fetchone()[0] > 0


def get_model_metrics(config: dict):
    """Main function to process models and calculate metrics."""
    models_dir = config["models_dir"]
    database_dir = config["database_dir"]
    os.makedirs(models_dir, exist_ok=True)

    models = [config["base_model"]] + config["fine_tuned_models"]
    metrics_to_calculate = [
        metric for metric, enabled in config["metrics"].items() if enabled
    ]

    for model_name in models:
        try:
            local_model_path = ensure_model_files(model_name, models_dir)
            if not local_model_path:
                print(f"Failed to get model files for {model_name}")
                continue

            if model_exists_in_database(model_name, database_dir):
                print(f"Model {model_name} exists in database")
                if metrics_exist_in_database(model_name, database_dir):
                    print(f"Metrics exist for {model_name}. Skipping...")
                    continue
                model = load_model_from_database(model_name, database_dir)
            else:
                model = load_model(local_model_path)
                if not model:
                    print(f"Failed to load model: {model_name}")
                    continue
                save_model_complete(model, model_name, local_model_path, database_dir)

            all_layers, layer_names = _collect_and_normalize_weights(model)
            layer_metrics = _calculate_metrics_for_layers(
                layer_names, all_layers, metrics_to_calculate
            )
            save_metrics_to_database(model_name, layer_metrics, database_dir)

            # Cleanup
            del model, all_layers
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue


def _collect_and_normalize_weights(
    model: AutoModelForCausalLM,
) -> tuple[list[torch.Tensor], list[str]]:
    """Collects and normalizes all layers from the model."""
    all_layers = []
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            layer = module.weight.data
            if layer.ndim < 2:
                layer = layer.unsqueeze(0)
            # Normalize and convert to bfloat16, store in list
            all_layers.append(_normalize_tensor(layer).to(torch.bfloat16))
            layer_names.append(name)
    return all_layers, layer_names


def _calculate_metrics_for_layers(
    layer_names: list[str], all_layers: list[torch.Tensor], metrics: list[str]
) -> dict:
    """Calculates metrics for each layer."""
    layer_metrics = {}
    for metric in metrics:
        print(f"Calculating {metric} for all layers")
        for i, layer in enumerate(all_layers):
            name = layer_names[i]
            try:
                result = calculate_metric(layer, metric)
                layer_metrics.setdefault(name, {})[metric] = result
            except Exception as e:
                print(f"Error calculating {metric} for layer {name}: {e}")
                layer_metrics.setdefault(name, {})[metric] = float("nan")
    return layer_metrics


def normalize_metrics(metrics: dict) -> dict:
    """Normalizes each metric to be between 0 and 1."""
    metric_values = {
        metric: torch.tensor([layer[metric] for layer in metrics.values()])
        for metric in next(iter(metrics.values()))
    }
    normalized = {
        metric: (values - values.min()) / (values.max() - values.min()).clamp(min=1e-8)
        for metric, values in metric_values.items()
    }
    return {k: v.tolist() for k, v in normalized.items()}


@torch.inference_mode()
def main(config_path: str):
    """Main function to run the model analysis process."""
    config = load_config(config_path)
    init_database(config["database_dir"])
    get_model_metrics(config)
    print("Metric calculation completed and saved to database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mastermerge: Advanced model merging and analysis tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mastermerge_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(args.config)
