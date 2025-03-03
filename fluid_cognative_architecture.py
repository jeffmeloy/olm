import argparse
import json
import os
import sqlite3
import bz2
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set, Iterator
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid
import logging
import shutil
import tempfile
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, deque
import math
from torch.utils.data import Dataset
import random
import time

"""
Project Goal: Develop a system for adaptive tensor analysis, model optimization, and reinforcement learning-driven tensor selection for transformer models. The system integrates multi-framework tensor analysis (fluid dynamics, spectral, statistical, topological, and information theory metrics) with hyperdimensional computing (HDC) and reinforcement learning to optimize tensor selection and model performance.

Key Components:

1. **Database Layer**:
   - Stores tensor metadata, metrics, and relationships using UUID-based indexing.
   - Manages model configurations, tensor loading orders, and HDC signatures.
   - Supports cross-tensor metric computation and model reconstruction from tensor combinations.

2. **Tensor Analysis**:
   - Computes multi-framework metrics (e.g., fluid dynamics, spectral, topological).
   - Tracks temporal evolution of tensor properties.
   - Correlates tensor metrics with model performance.

3. **Hyperdimensional Computing (HDC)**:
   - Converts tensor metrics into high-dimensional signatures for fast similarity search.
   - Maps tensor relationships in HDC space for pattern recognition and clustering.
   - Enables adaptive signature refinement based on performance feedback.

4. **Model Management**:
   - Downloads, processes, and stores models and their tensors.
   - Reconstructs models from stored tensors and configurations.
   - Validates model integrity and tensor consistency.

5. **Reinforcement Learning (RL)**:
   - Uses swarm intelligence and evolutionary strategies to optimize tensor selection.
   - Integrates Q-learning and HDC for adaptive decision-making.
   - Dynamically adjusts exploration and exploitation strategies based on performance.

6. **Tensor Optimization**:
   - Automatically discovers functionally similar tensors across architectures.
   - Recommends tensor substitutions based on HDC signatures and task performance.
   - Dynamically maps tensor relationships for online model optimization.

Key Features:
- **Adaptive Learning**: Refines tensor selection strategies based on performance feedback.
- **Multi-Framework Analysis**: Combines fluid dynamics, spectral, and topological metrics for comprehensive tensor evaluation.
- **HDC Integration**: Uses hyperdimensional computing for efficient tensor relationship mapping and similarity search.
- **Reinforcement Learning**: Optimizes tensor selection through swarm intelligence and evolutionary algorithms.
- **Model Reconstruction**: Reconstructs models from stored tensors and configurations for inference and further analysis.

TODO:
1. Implement adaptive signature refinement based on performance feedback.
2. Add reinforcement learning loops for tensor selection optimization.
3. Enhance cross-tensor relationship analysis for better model composition.
4. Integrate xLSTM for learning transformer model outputs based on inputs and context.
5. Improve fluid dynamics and topological metrics for tensor analysis.
6. Add support for dynamic task-tensor mapping based on downstream task performance.
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_tensor_to_tempfile(tensor: torch.Tensor) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        tensor_path = temp_file.name
        torch.save(tensor, tensor_path)  # Save the tensor to the temporary file
    return tensor_path


def load_tensor_from_tempfile(tensor_path: str) -> torch.Tensor:
    return torch.load(tensor_path)


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


class _DummyLock:
    """No-op lock for single-threaded mode"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DatabaseSchema:
    """Central definition of all database schemas and relationships."""

    @staticmethod
    def create_all_tables(conn):
        """Create all database tables if they don't exist."""
        conn.executescript("""
            -- Base model tables
            CREATE TABLE IF NOT EXISTS base_models (
                base_model_id TEXT PRIMARY KEY,  -- UUID for model
                model_name TEXT UNIQUE NOT NULL,
                config_json TEXT NOT NULL,
                tokenizer_json TEXT,
                tokenizer_config_json TEXT,
                special_tokens_map_json TEXT,
                added_tokens_json TEXT,
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            -- Track derived models and their tensor composition
            CREATE TABLE IF NOT EXISTS derived_models (
                model_id TEXT PRIMARY KEY,  -- UUID for derived model
                model_name TEXT UNIQUE NOT NULL,
                base_model_id TEXT NOT NULL,  -- Link to base model
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(base_model_id) REFERENCES base_models(base_model_id)
            );

            -- Store tensors with their source info (No BLOB, just metadata)
            CREATE TABLE IF NOT EXISTS tensors (
                tensor_id TEXT PRIMARY KEY,  -- UUID for tensor
                model_id TEXT NOT NULL,  -- Link to derived model
                tensor_path TEXT NOT NULL,     -- Path of the tensor in model
                tensor_shape TEXT NOT NULL,    -- Shape (JSON string or text)
                tensor_dtype TEXT NOT NULL,    -- Type of tensor data (e.g., float32, bfloat16)
                source_model_id TEXT NOT NULL, -- Track source model
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(model_id) REFERENCES derived_models(model_id),
                UNIQUE(model_id, tensor_path)  -- Enforce uniqueness for tensor path
            );

            -- Define tensor loading order for model assembly
            CREATE TABLE IF NOT EXISTS tensor_loading_order (
                model_id TEXT NOT NULL,
                tensor_id TEXT NOT NULL,
                load_order INTEGER NOT NULL,
                FOREIGN KEY(model_id) REFERENCES derived_models(model_id),
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(model_id, load_order),
                UNIQUE(model_id, tensor_id)
            );

            -- Tensor metrics tables (Store each metric with UUID for easy querying)
            CREATE TABLE IF NOT EXISTS tensor_metrics (
                metric_id TEXT PRIMARY KEY,  -- UUID for each metric
                tensor_id TEXT NOT NULL,     -- Link to tensor
                metric_name TEXT NOT NULL,   -- Metric name (e.g., "viscosity")
                metric_value REAL NOT NULL,  -- Metric value
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(tensor_id, metric_name)  -- Enforce uniqueness for each metric of a tensor
            );

            CREATE TABLE IF NOT EXISTS cross_tensor_metrics (
                source_tensor_id TEXT NOT NULL,  -- Source tensor
                target_tensor_id TEXT NOT NULL,  -- Target tensor
                metric_name TEXT NOT NULL,       -- Metric name (e.g., "cosine_similarity")
                metric_value REAL NOT NULL,      -- Metric value
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_tensor_id) REFERENCES tensors(tensor_id),
                FOREIGN KEY(target_tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(source_tensor_id, target_tensor_id, metric_name)  -- Enforce uniqueness for cross-tensor metrics
            );

            -- Model merging tables (No changes needed)
            CREATE TABLE IF NOT EXISTS merge_reports (
                report_id TEXT PRIMARY KEY,  -- UUID for merge report
                base_model_name TEXT NOT NULL,
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT,
                metrics_json TEXT
            );
            
            CREATE TABLE IF NOT EXISTS merge_tensor_sources (
                report_id TEXT NOT NULL,  -- Link to merge report
                tensor_path TEXT NOT NULL,
                source_model TEXT NOT NULL,
                metrics_json TEXT,
                FOREIGN KEY(report_id) REFERENCES merge_reports(report_id),
                UNIQUE(report_id, tensor_path)
            );

            -- HDC signature tables (Store signature as a series of components, not as a BLOB)
            CREATE TABLE IF NOT EXISTS hdc_signatures (
                signature_id TEXT PRIMARY KEY,  -- UUID for each HDC signature
                tensor_id TEXT NOT NULL,        -- Link to tensor
                signature_components TEXT,      -- Store components of the signature (JSON string or serialized list)
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id)
            );

            CREATE TABLE IF NOT EXISTS hdc_relations (
                source_id TEXT NOT NULL,        -- Source signature
                target_id TEXT NOT NULL,        -- Target signature
                relation_type TEXT NOT NULL,    -- Type of relation (e.g., "similarity")
                similarity REAL NOT NULL,       -- Similarity score between tensors
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_id) REFERENCES hdc_signatures(signature_id),
                FOREIGN KEY(target_id) REFERENCES hdc_signatures(signature_id),
                UNIQUE(source_id, target_id, relation_type)  -- Enforce uniqueness for relations between signatures
            );

            -- SFT dataset tables (No significant changes)
            CREATE TABLE IF NOT EXISTS sft_datasets (
                dataset_id TEXT PRIMARY KEY,  -- UUID for dataset
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                metadata_json TEXT,
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS sft_examples (
                example_id TEXT PRIMARY KEY,  -- UUID for example
                dataset_id TEXT NOT NULL,     -- Link to dataset
                prompt TEXT NOT NULL,
                completion TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY(dataset_id) REFERENCES sft_datasets(dataset_id)
            );

            -- Task-tensor mapping tables
            CREATE TABLE IF NOT EXISTS task_tensor_mappings (
                mapping_id TEXT PRIMARY KEY,  -- UUID for mapping
                task_vector_id TEXT NOT NULL, -- Link to task vector (HDC)
                tensor_id TEXT NOT NULL,      -- Link to tensor
                importance_score REAL NOT NULL,  -- Importance score of the tensor for the task
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(task_vector_id) REFERENCES hdc_signatures(signature_id),
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(task_vector_id, tensor_id)  -- Enforce uniqueness for task-tensor mapping
            );

            -- Task-tensor mapping tables
            CREATE TABLE IF NOT EXISTS task_tensor_mappings (
                mapping_id TEXT PRIMARY KEY,  -- UUID for mapping
                task_vector_id TEXT NOT NULL, -- Link to task vector (HDC)
                tensor_id TEXT NOT NULL,      -- Link to tensor
                importance_score REAL NOT NULL,  -- Importance score of the tensor for the task
                creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(task_vector_id) REFERENCES hdc_signatures(signature_id),
                FOREIGN KEY(tensor_id) REFERENCES tensors(tensor_id),
                UNIQUE(task_vector_id, tensor_id)  -- Enforce uniqueness for task-tensor mapping
            );
        """)


class ModelDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(
            db_path
        )  # Keep this open for the lifetime of the instance
        self.conn.row_factory = sqlite3.Row  # For accessing rows by name
        DatabaseSchema.create_all_tables(self.conn)

    def close(self):
        """Close the database connection when done."""
        self.conn.close()

    def store_base_model(
        self, model_name: str, config: dict, tokenizer_files: dict
    ) -> str:
        """Stores base model configuration and tokenizer files."""
        base_model_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO base_models (
                    base_model_id, model_name, config_json,
                    tokenizer_json, tokenizer_config_json,
                    special_tokens_map_json, added_tokens_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    base_model_id,
                    model_name,
                    json.dumps(config),
                    tokenizer_files.get("tokenizer_json"),
                    tokenizer_files.get("tokenizer_config_json"),
                    tokenizer_files.get("special_tokens_map_json"),
                    tokenizer_files.get("added_tokens_json"),
                ),
            )
            conn.commit()
        return base_model_id

    def store_tensors_and_hdc(self, tensor_map: Dict[str, torch.Tensor], model_id: str):
        """Store tensors and their HDC signatures in a batch."""
        tensor_data = []
        hdc_signatures = []

        # Generate tensor data and HDC signatures
        for tensor_name, tensor_data in tensor_map.items():
            tensor_id = str(uuid.uuid4())  # Generate unique tensor_id
            tensor_shape = tensor_data.shape
            tensor_dtype = str(tensor_data.dtype)
            tensor_path = tensor_name

            tensor_data.append(
                (
                    tensor_id,
                    model_id,
                    tensor_path,
                    json.dumps(tensor_shape),
                    tensor_dtype,
                    model_id,
                )
            )

            # Compute HDC signature
            hdc_signature = self.compute_hdc_signature(
                tensor_id
            )  # Add HDC computation logic
            hdc_signatures.append(
                (
                    str(uuid.uuid4()),
                    tensor_id,
                    json.dumps(hdc_signature),
                )
            )

        # Insert tensors and HDC signatures in batch
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO tensors (tensor_id, model_id, tensor_path, tensor_shape, tensor_dtype, source_model_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                tensor_data,
            )
            self.conn.executemany(
                """
                INSERT INTO hdc_signatures (signature_id, tensor_id, signature_components)
                VALUES (?, ?, ?)
                """,
                hdc_signatures,
            )

    def compute_hdc_signature(self, tensor_id: str) -> List[float]:
        """Compute the HDC signature for a tensor based on its metrics."""
        # Placeholder: Implement HDC signature computation based on tensor's metrics
        # Example: return [0.5, 0.7, 0.9] as dummy signature components
        return [0.5, 0.7, 0.9]

    def create_derived_model(
        self, model_name: str, base_model_name: str, tensor_specs: List[Dict[str, Any]]
    ) -> str:
        """Creates a new model from specified tensors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT base_model_id FROM base_models WHERE model_name = ?",
                (base_model_name,),
            )
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Base model {base_model_name} not found")
            base_model_id = result[0]
            model_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO derived_models (model_id, model_name, base_model_id)
                VALUES (?, ?, ?)
                """,
                (model_id, model_name, base_model_id),
            )

            for order, spec in enumerate(tensor_specs):
                tensor_id = str(uuid.uuid4())  # Generate unique tensor_id
                cursor.execute(
                    """
                    INSERT INTO tensors (tensor_id, model_id, tensor_path, tensor_shape, tensor_dtype, source_model_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tensor_id,
                        model_id,
                        spec["tensor_path"],
                        json.dumps(spec["tensor_shape"]),
                        spec["tensor_dtype"],
                        spec["source_model_id"],
                    ),
                )

                # Store tensor loading order
                cursor.execute(
                    """
                    INSERT INTO tensor_loading_order (model_id, tensor_id, load_order)
                    VALUES (?, ?, ?)
                    """,
                    (model_id, tensor_id, order),
                )

                # Store HDC signature for this tensor (if applicable)
                hdc_signature = self.compute_hdc_signature(
                    spec["tensor_id"]
                )  # Add HDC computation logic
                cursor.execute(
                    """
                    INSERT INTO hdc_signatures (signature_id, tensor_id, signature_components)
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        tensor_id,
                        json.dumps(hdc_signature),  # Store components of HDC signature
                    ),
                )

            conn.commit()
            return model_id

    def reconstruct_model(
        self, model_name: str, device: str = "cpu"
    ) -> Tuple[Any, Any]:
        """Reconstructs a working model from stored tensors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dm.model_id, bm.base_model_id, bm.config_json,
                       bm.tokenizer_json, bm.tokenizer_config_json,
                       bm.special_tokens_map_json, bm.added_tokens_json
                FROM derived_models dm
                JOIN base_models bm ON dm.base_model_id = bm.base_model_id
                WHERE dm.model_name = ?
                """,
                (model_name,),
            )
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Model {model_name} not found")

            model_id, base_model_id, config_json = result[0:3]
            tokenizer_files = {
                "tokenizer_json": result[3],
                "tokenizer_config_json": result[4],
                "special_tokens_map_json": result[5],
                "added_tokens_json": result[6],
            }

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                config = json.loads(config_json)
                with open(temp_dir / "config.json", "w") as f:
                    json.dump(config, f)

                for filename, content in tokenizer_files.items():
                    if content:
                        with open(
                            temp_dir / f"{filename.replace('_json', '.json')}", "w"
                        ) as f:
                            f.write(content)

                cursor.execute(
                    """
                    SELECT t.tensor_path, t.tensor_shape, t.tensor_dtype
                    FROM tensor_loading_order tlo
                    JOIN tensors t ON tlo.tensor_id = t.tensor_id
                    WHERE tlo.model_id = ?
                    ORDER BY tlo.load_order
                    """,
                    (model_id,),
                )

                model = AutoModelForCausalLM.from_config(config)
                model = model.to(device)
                for path, shape, dtype in cursor.fetchall():
                    tensor = torch.empty(
                        json.loads(shape), dtype=getattr(torch, dtype.split(".")[-1])
                    ).to(device)
                    module_path, param_name = path.rsplit(".", 1)
                    if module_path:
                        module = model.get_submodule(module_path)
                    else:
                        module = model
                    setattr(module, param_name, nn.Parameter(tensor))

                tokenizer = AutoTokenizer.from_pretrained(temp_dir)
                return model, tokenizer

    def validate_model_integrity(
        self, model_name: str, required_tensors: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Validates that all necessary tensors are present and correctly ordered."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT t.tensor_path, t.tensor_shape, tlo.load_order
                FROM tensor_loading_order tlo
                JOIN tensors t ON tlo.tensor_id = t.tensor_id
                JOIN derived_models dm ON tlo.model_id = dm.model_id
                WHERE dm.model_name = ?
                ORDER BY tlo.load_order
                """,
                (model_name,),
            )

            tensors = cursor.fetchall()
            if not tensors:
                raise ValueError(f"No tensors found for model {model_name}")

            tensor_paths = {t[0] for t in tensors}
            if required_tensors and not required_tensors.issubset(tensor_paths):
                missing = required_tensors - tensor_paths
                raise ValueError(f"Missing required tensors: {missing}")

            orders = [t[2] for t in tensors]
            if set(orders) != set(range(len(orders))):
                raise ValueError("Gaps detected in tensor loading order")

            return {
                "tensor_count": len(tensors),
                "tensors": {
                    path: {"shape": json.loads(shape), "load_order": order}
                    for path, shape, order in tensors
                },
            }


class ModelLoader:
    """Downloads models and extracts their tensor goodness."""

    def __init__(self, models_dir: str, database: ModelDatabase):
        self.models_dir = models_dir
        self.database = database  # We need this for the tensor extraction magic
        Path(models_dir).mkdir(parents=True, exist_ok=True)

    def ensure_model_files(self, model_name: str) -> str:
        """Grabs model files, but now also extracts tokenizer configs."""
        local_path = Path(self.models_dir) / model_name.replace("/", "_")

        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)

        logger.info(f"Yoinking {model_name} from the hub...")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                revision="main",
            )

            # Snag all the tokenizer goodies
            tokenizer_files = {}
            for filename in [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ]:
                file_path = local_path / filename
                if file_path.exists():
                    tokenizer_files[f"{filename.split('.')[0]}_json"] = (
                        file_path.read_text()
                    )

            # Store base model info right away
            with open(local_path / "config.json") as f:
                config = json.load(f)

            self.database.store_base_model(
                model_name=model_name, config=config, tokenizer_files=tokenizer_files
            )

            return str(local_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise

    def load_model(
        self, model_path: str, extract_tensors: bool = True, device: str = "cpu"
    ) -> Tuple[AutoModelForCausalLM, Optional[Dict[str, torch.Tensor]]]:
        """Loads a model and optionally extracts all its juicy tensors."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": device},
            )

            if not extract_tensors:
                return model, None

            # Extract ALL the tensors!
            tensor_map = {}
            for name, param in model.named_parameters():
                if param.requires_grad:  # Only grab the trainable ones
                    tensor_map[name] = param.data.clone().cpu()

            # Store tensors and their metadata in the database
            for tensor_name, tensor_data in tensor_map.items():
                tensor_id = str(uuid.uuid4())  # Generate a unique tensor_id
                tensor_shape = tensor_data.shape
                tensor_dtype = str(tensor_data.dtype)
                tensor_path = tensor_name

                # Insert tensor metadata into the database
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO tensors (tensor_id, model_id, tensor_path, tensor_shape, tensor_dtype, source_model_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tensor_id,
                            model_path,
                            tensor_path,
                            json.dumps(tensor_shape),
                            tensor_dtype,
                            model_path,
                        ),
                    )
                    conn.commit()

                # Store HDC signature for this tensor (Placeholder function)
                hdc_signature = self.compute_hdc_signature(
                    tensor_id
                )  # Add HDC computation logic
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO hdc_signatures (signature_id, tensor_id, signature_components)
                        VALUES (?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            tensor_id,
                            json.dumps(
                                hdc_signature
                            ),  # Store components of HDC signature
                        ),
                    )
                    conn.commit()

            return model, tensor_map

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def extract_tensor_specs(
        self, source_model: str, tensor_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Gets specs for requested tensors from a source model."""
        specs = []

        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()

            for path in tensor_paths:
                cursor.execute(
                    """
                    SELECT tensor_id, tensor_shape, tensor_dtype
                    FROM tensors t
                    JOIN derived_models dm ON t.model_id = dm.model_id
                    WHERE dm.model_name = ? AND t.tensor_path = ?
                    """,
                    (source_model, path),
                )

                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Tensor {path} not found in model {source_model}")

                specs.append(
                    {
                        "tensor_id": result[0],
                        "path": path,
                        "shape": json.loads(result[1]),
                        "dtype": result[2],
                    }
                )

        return specs

    def compute_hdc_signature(self, tensor_id: str) -> List[float]:
        """Compute the HDC signature for a tensor based on its metrics."""
        # Placeholder: Implement HDC signature computation based on tensor's metrics
        # Example: return [0.5, 0.7, 0.9] as dummy signature components
        return [0.5, 0.7, 0.9]


class ModelLoaderFromDatabase:
    """Reassembles models from their tensor essence and uses Hugging Face Transformers for inference."""

    def __init__(self, database_dir: str):
        self.database = ModelDatabase(database_dir)
        self.temp_dir = Path(database_dir) / "temp_model_files"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_model_from_db(
        self, model_name: str, device: str = "cpu", strict_loading: bool = True
    ) -> Optional[AutoModelForCausalLM]:
        """Reassembles a model from its constituent tensors."""
        if not self._prepare_model_files(model_name):
            return None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.temp_dir),  # Load model from the temporary directory
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map={"": device},
            )

            # Load tensors and match them with model parameters
            tensors_data = self.database.load_tensors_and_metrics(model_name)
            missing_tensors = []
            unexpected_tensors = []
            expected_params = set(n for n, _ in model.named_parameters())
            available_tensors = set(tensors_data.keys())
            missing_tensors = expected_params - available_tensors
            unexpected_tensors = available_tensors - expected_params

            if missing_tensors and strict_loading:
                raise ValueError(f"Missing tensors for {model_name}: {missing_tensors}")
            if unexpected_tensors:
                logger.warning(
                    f"Found unexpected tensors in database: {unexpected_tensors}"
                )

            # Reassign tensors to model parameters
            for tensor_path, tensor_data in tensors_data.items():
                if tensor_path in expected_params:
                    param = model.get_parameter(tensor_path)
                    if param is not None:
                        tensor = tensor_data["tensor"].to(device)
                        if param.shape == tensor.shape:
                            param.copy_(tensor)
                        else:
                            raise ValueError(
                                f"Shape mismatch for {tensor_path}: expected {param.shape}, got {tensor.shape}"
                            )
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_tokenizer_from_db(
        self, model_name: str, required_files: Optional[List[str]] = None
    ) -> Optional[AutoTokenizer]:
        """Loads a tokenizer with validation."""
        required_files = required_files or ["tokenizer_config", "tokenizer"]

        if not self._prepare_model_files(model_name, required_files):
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.temp_dir), trust_remote_code=True
            )
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            return None

    def _prepare_model_files(
        self, model_name: str, required_files: Optional[List[str]] = None
    ) -> bool:
        """Sets up necessary files with validation."""
        self._clear_temp_directory()

        try:
            model_data = self.database.load_model_data(model_name)
            required_files = required_files or ["config"]
            for key in required_files:
                data = model_data.get(key)
                if not data:
                    raise ValueError(f"Missing required file: {key}")
                file_path = self.temp_dir / f"{key}.json"
                with open(file_path, "w") as f:
                    f.write(data)
            return True
        except Exception as e:
            logger.error(f"Failed to prepare files for {model_name}: {e}")
            return False

    def _clear_temp_directory(self) -> None:
        """Keeps our workspace clean."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)

    def validate_model_tensors(
        self, model_name: str, check_metrics: bool = True
    ) -> Dict[str, Any]:
        """Validates tensor consistency and optionally checks metric sanity."""
        validation_results = {
            "missing_tensors": [],
            "shape_mismatches": [],
            "metric_anomalies": [],
            "is_valid": False,
        }

        try:
            self._prepare_model_files(model_name)
            base_model = AutoModelForCausalLM.from_config(str(self.temp_dir))
            expected_params = dict(base_model.named_parameters())
            tensors_data = self.database.load_tensors_and_metrics(model_name)

            for param_name, param in expected_params.items():
                if param_name not in tensors_data:
                    validation_results["missing_tensors"].append(param_name)
                else:
                    stored_tensor = tensors_data[param_name]["tensor"]
                    if stored_tensor.shape != param.shape:
                        validation_results["shape_mismatches"].append(
                            {
                                "param": param_name,
                                "expected": param.shape,
                                "got": stored_tensor.shape,
                            }
                        )

            if check_metrics:
                for param_name, tensor_data in tensors_data.items():
                    metrics = tensor_data.get("metrics", {})
                    for metric_name, value in metrics.items():
                        if not self._is_metric_sane(metric_name, value):
                            validation_results["metric_anomalies"].append(
                                {
                                    "param": param_name,
                                    "metric": metric_name,
                                    "value": value,
                                }
                            )

            validation_results["is_valid"] = (
                not validation_results["missing_tensors"]
                and not validation_results["shape_mismatches"]
                and (not check_metrics or not validation_results["metric_anomalies"])
            )
            return validation_results
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            validation_results["error"] = str(e)
            return validation_results

    def _is_metric_sane(self, metric_name: str, value: float) -> bool:
        """Placeholder: Implement sanity check for metric values"""
        return not (value != value or value == float("inf") or value == -float("inf"))


class MergeReportManager:
    """Handles the sacred texts of model merging."""

    def __init__(self, database: ModelDatabase):
        self.database = database

    def save_merge_report(self, merge_report: Dict) -> str:
        """Preserve the sacred knowledge of how we built this monster."""
        report_id = str(uuid.uuid4())

        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO merge_reports (
                    report_id, base_model_name, config_json, metrics_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    report_id,
                    merge_report["base_model"]["name"],
                    json.dumps(merge_report.get("config", {})),
                    json.dumps(merge_report["base_model"].get("metrics", {})),
                ),
            )

            tensor_sources = []
            if "boundary_layers" in merge_report:
                boundary = merge_report["boundary_layers"]
                if boundary.get("name"):
                    tensor_sources.extend(
                        [
                            (
                                report_id,
                                "model.embed_tokens",
                                boundary["name"],
                                json.dumps(boundary.get("metrics", {})),
                            ),
                            (
                                report_id,
                                "model.norm",
                                boundary["name"],
                                json.dumps(boundary.get("metrics", {})),
                            ),
                            (
                                report_id,
                                "lm_head",
                                boundary["name"],
                                json.dumps(boundary.get("metrics", {})),
                            ),
                        ]
                    )

            if "layers" in merge_report:
                for layer_idx, layer_info in merge_report["layers"].items():
                    source_model = layer_info.get("best_model")
                    if source_model:
                        tensor_sources.extend(
                            [
                                (
                                    report_id,
                                    f"model.layers.{layer_idx}",
                                    source_model,
                                    json.dumps(layer_info.get("metrics", {})),
                                )
                            ]
                        )

            cursor.executemany(
                """
                INSERT INTO merge_tensor_sources (
                    report_id, tensor_path, source_model, metrics_json
                ) VALUES (?, ?, ?, ?)
                """,
                tensor_sources,
            )

            conn.commit()

        return report_id

    def load_merge_report(self, report_id: str) -> Dict:
        """Resurrect the ancient knowledge."""
        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT base_model_name, config_json, metrics_json
                FROM merge_reports WHERE report_id = ?
                """,
                (report_id,),
            )

            result = cursor.fetchone()
            if not result:
                raise ValueError(f"No merge report found for ID {report_id}")

            base_model, config_json, metrics_json = result

            report = {
                "base_model": {
                    "name": base_model,
                    "metrics": json.loads(metrics_json) if metrics_json else {},
                },
                "config": json.loads(config_json) if config_json else {},
            }

            cursor.execute(
                """
                SELECT tensor_path, source_model, metrics_json
                FROM merge_tensor_sources
                WHERE report_id = ?
                ORDER BY tensor_path
                """,
                (report_id,),
            )
            for path, source, metrics in cursor.fetchall():
                metrics = json.loads(metrics) if metrics else {}
                if path in ["model.embed_tokens", "model.norm", "lm_head"]:
                    if "boundary_layers" not in report:
                        report["boundary_layers"] = {"name": source, "metrics": metrics}
                elif path.startswith("model.layers."):
                    layer_idx = path.split(".")[2]
                    if "layers" not in report:
                        report["layers"] = {}
                    report["layers"][layer_idx] = {
                        "best_model": source,
                        "metrics": metrics,
                    }
            return report

    def get_model_from_report(
        self, report_id: str, device: str = "cpu"
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Resurrect the monster from our database."""
        report = self.load_merge_report(report_id)
        base_model_name = report["base_model"]["name"]
        model = self.database.load_model(base_model_name, device)
        tokenizer = self.database.load_tokenizer(base_model_name)

        if model is None or tokenizer is None:
            raise ValueError(f"Failed to load base model {base_model_name}")

        if "boundary_layers" in report and report["boundary_layers"]["name"]:
            try:
                boundary_model = report["boundary_layers"]["name"]
                boundary_tensors = self.database.load_tensors(boundary_model)

                for tensor_name in ["model.embed_tokens", "model.norm", "lm_head"]:
                    if tensor_name in boundary_tensors:
                        self._set_tensor(
                            model, tensor_name, boundary_tensors[tensor_name]
                        )
            except Exception as e:
                logger.error(f"Failed to apply boundary layers: {e}")

        if "layers" in report:
            for layer_idx, layer_info in report["layers"].items():
                source_model = layer_info["best_model"]
                try:
                    layer_tensors = self.database.load_tensors(
                        source_model, f"model.layers.{layer_idx}"
                    )
                    for tensor_path, tensor in layer_tensors.items():
                        self._set_tensor(model, tensor_path, tensor)
                except Exception as e:
                    logger.error(f"Failed to apply layer {layer_idx}: {e}")

        return model, tokenizer

    @staticmethod
    def _set_tensor(model: nn.Module, tensor_path: str, tensor: torch.Tensor):
        """Carefully place tensor in its new home."""
        try:
            module_path, param_name = tensor_path.rsplit(".", 1)
            if module_path:
                module = model.get_submodule(module_path)
            else:
                module = model
            param = getattr(module, param_name)
            with torch.no_grad():
                param.copy_(tensor)
        except Exception as e:
            raise ValueError(f"Failed to set tensor {tensor_path}: {e}")


class TensorAnalyzer:
    """Analyzes tensors through multiple conceptual lenses with unified caching."""

    def __init__(self, device: torch.device):
        self.device = device
        self.current_cache = None
        self.cross_cache = None
        self._setup_metrics()

    def _setup_metrics(self):
        """Define unified metric arsenal spanning multiple conceptual frameworks."""
        self.metrics = {
            # Fluid dynamics metrics
            "viscosity": self._calc_viscosity,
            "surface_tension": self._calc_surface_tension,
            "vorticity": self._calc_vorticity,
            "stability": self._calc_stability,
            "phase_coherence": self._calc_phase_coherence,
            "reynolds": self._calc_reynolds,
            # Information theory metrics
            "weight_entropy": self._calculate_entropy,
            "bzip2_compression": self._calculate_compression,
            "permutation_entropy": self._calculate_permutation_entropy,
            # Spectral metrics
            "svd_skewness": self._calculate_svd_skewness,
            "stable_rank": self._calculate_stable_rank,
            "normalized_effective_rank": self._calculate_normalized_effective_rank,
            "weight_spectral_norm": self._calculate_spectral_norm,
            # Statistical metrics
            "snr": self._calculate_snr,
            "weight_kurtosis": self._calculate_kurtosis,
            "weight_skewness": self._calculate_skewness,
            "weight_sparsity": self._calculate_sparsity,
            "outlier_influence": self._calculate_outliers,
            # Topological metrics
            "weight_clustering": self._calculate_clustering,
            "mode_collapse": self._calculate_mode_collapse,
            "persistence": self._calculate_persistence,
            # Dynamical systems metrics
            "lyapunov_estimate": self._calculate_lyapunov,
            "zipf_deviation": self._calculate_zipf_deviation,
            "weight_memorization": self._calculate_memorization,
            "phase_space": self._calculate_phase_space,
            "hurst_exponent": self._calculate_hurst_exponent,
            # Multifractal metrics
            "multifractal_width": lambda: self._calculate_multifractal_width,
            "correlation_dimension": lambda: self._calculate_multifractal_width(
                ret="dimension"
            ),
        }

        # Cross-tensor comparison metrics
        self.cross_metrics = {
            # Statistical distances
            "mahalanobis_distance": self._calculate_mahalanobis,
            "earth_mover": self._calculate_earth_mover,
            "cosine_similarity": self._calculate_cosine_sim,
            # Information theory
            "mutual_information": self._calculate_mutual_info,
            "distribution_overlap": self._calculate_distribution_overlap,
            # Geometric distances
            "hyperbolic_distance": self._hyperbolic_distance,
            "cucconi": self._calculate_cucconi,
            "cvd": self._calculate_cvd,
            # Fluid dynamics interactions
            "energy_transfer": self._calculate_energy_transfer,
            "information_flux": self._calculate_information_flux,
            "pattern_persistence": self._calculate_pattern_persistence,
        }

    def _build_cache(self, tensor: torch.Tensor) -> dict:
        """Build optimized cache with preprocessed values for all metrics."""
        cache = {}
        cache["tensor"] = tensor.to(self.device)
        cache["shape"] = tensor.shape
        cache["flat"] = cache["tensor"].flatten()
        cache["numel"] = cache["flat"].numel()
        cache["mean"] = torch.mean(cache["flat"])
        cache["std"] = torch.std(cache["flat"])
        cache["var"] = torch.var(cache["flat"])
        cache["sorted"] = torch.sort(cache["flat"])[0]
        q_vals = torch.tensor([0.25, 0.75], device=self.device)
        cache["quartiles"] = torch.quantile(cache["flat"], q_vals)
        cache["iqr"] = cache["quartiles"][1] - cache["quartiles"][0]
        cache["density"] = self._normalize_tensor(tensor)
        cache["gradients"] = torch.gradient(cache["density"])
        cache["velocity"] = torch.stack(cache["gradients"]).mean(0)
        cache["energy"] = 0.5 * (cache["density"].pow(2) + cache["velocity"].pow(2))
        cache["strain"] = torch.stack([g.abs() for g in cache["gradients"]]).mean(0)
        cache["svd"] = torch.linalg.svdvals(cache["tensor"])
        cache["rank"] = torch.linalg.matrix_rank(cache["tensor"])
        cache["norm"] = torch.linalg.norm(cache["tensor"])
        cache["hist"] = self._compute_histogram(cache["flat"])
        cache["zero_mask"] = torch.abs(cache["tensor"]) < 1e-5
        cache["sparsity"] = cache["zero_mask"].float().mean()
        cache["ranks"] = torch.argsort(cache["flat"].float()).argsort().float()
        cache["angles"] = torch.angle(cache["flat"][1:] + 1j * cache["flat"][:-1])
        if len(cache["shape"]) > 1:
            cache["characteristic_length"] = torch.prod(
                torch.tensor(cache["shape"], dtype=torch.float32)
            ) ** (1.0 / len(cache["shape"]))
        else:
            cache["characteristic_length"] = float(cache["shape"][0])
        cache["viscosity"] = float(1.0 / (1.0 + cache["strain"].mean()))
        cache["stability"] = float(
            1.0
            / (
                1.0
                + cache["velocity"].abs().std()
                / (cache["velocity"].abs().mean() + 1e-8)
            )
        )
        cache["gradient_magnitude"] = float(
            sum(g.pow(2).mean() for g in cache["gradients"]).sqrt()
        )

        return cache

    @torch.inference_mode()
    def analyze(
        self,
        tensor: torch.Tensor,
        prev_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Unified tensor analysis through multiple conceptual frameworks."""
        # Update cache states
        self.prev_cache = self.current_cache
        self.current_cache = self._build_unified_cache(tensor)

        if prev_tensor is not None:
            self.prev_cache = self._build_unified_cache(prev_tensor)

        metrics_to_run = self.metrics
        results = {}
        for name, func in metrics_to_run.items():
            try:
                if hasattr(self.current_cache, name):
                    results[name] = getattr(self.current_cache, name)
                else:
                    results[name] = float(func())
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                results[name] = float("nan")

        if self.prev_cache is not None:
            temporal_results = self._calc_temporal_metrics()
            results.update(temporal_results)

        # Add derived/compound metrics
        if "normalized_effective_rank" in results and "bzip2_compression" in results:
            results["rank_compression"] = (
                results["normalized_effective_rank"] * results["bzip2_compression"]
            )

        if "normalized_effective_rank" in results and "stable_rank" in results:
            results["rank_stability"] = (
                results["normalized_effective_rank"] * results["stable_rank"]
            )

        return results

    def compare(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> Dict[str, float]:
        """Compare two tensors."""
        self.current_cache = self._build_cache(tensor_a)
        self.cross_cache = self._build_cache(tensor_b)
        results = {}
        for name, func in self.cross_metrics.items():
            try:
                results[name] = func()
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                results[name] = float("nan")

        return results

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize while preserving signs."""
        signs = tensor.sign()
        tensor = tensor.abs()
        tensor /= tensor.max().clamp(min=1e-8)
        return tensor * signs

    def _compute_histogram(
        self, values: torch.Tensor, bins: int = None
    ) -> torch.Tensor:
        """Adaptive histogram computation."""
        if bins is None:
            bins = int(values.numel() ** 0.5)
            bins = max(bins, 2)

        edges = torch.quantile(
            values, torch.linspace(0, 1, bins + 1, device=self.device)
        )
        hist = torch.histogram(values, edges)
        return hist.hist / hist.hist.sum()

    def _marchenko_pastur_threshold(self, sigma: float, n: int, m: int) -> float:
        """Compute M-P threshold for singular values."""
        beta = n / m
        return sigma * (1 + torch.sqrt(torch.tensor(beta)))

    def _calculate_snr(self) -> float:
        """Signal-to-noise ratio using M-P threshold."""
        S = self.current_cache["svd"]
        max_sv = S[0]
        sigma = S[-len(S) // 4 :].std()  # Use last quartile for noise estimation
        n, m = self.current_cache["shape"][-2:]
        threshold = self._marchenko_pastur_threshold(sigma, n, m)
        signal = torch.sum(S[S > threshold])
        noise = torch.sum(S[S <= threshold])
        snr = signal / max(noise, 1e-10)
        return float(snr / max_sv)

    def _calculate_svd_skewness(self) -> float:
        """Skewness of singular value distribution."""
        S = self.current_cache["svd"]
        return float(1 - (torch.mean(S) / S[0]))

    def _calculate_stable_rank(self) -> float:
        """Stable rank from SVD."""
        S = self.current_cache["svd"]
        return float(torch.sum(S**2) / S[0] ** 2)

    def _calculate_normalized_effective_rank(self) -> float:
        """Normalized effective rank."""
        S = self.current_cache["svd"]
        S_norm = S / S.sum()
        effective_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm)))
        return float(effective_rank / self.current_cache["rank"])

    def _calculate_spectral_norm(self) -> float:
        """Largest singular value."""
        return float(self.current_cache["svd"][0])

    def _calculate_frobenius_norm(self) -> float:
        """Direct from cache."""
        return float(self.current_cache["norm"])

    def _calculate_kurtosis(self) -> float:
        """Direct from flat values."""
        return float(torch.kurtosis(self.current_cache["flat"]))

    def _calculate_skewness(self) -> float:
        """Direct from flat values."""
        return float(torch.skew(self.current_cache["flat"]))

    def _calculate_sparsity(self) -> float:
        """Direct from cache."""
        return float(self.current_cache["sparsity"])

    def _calculate_entropy(self) -> float:
        """From cached histogram."""
        hist = self.current_cache["hist"]
        return float(-torch.sum(hist * torch.log2(hist + 1e-12)))

    def _calculate_outliers(self) -> float:
        """Using cached quartiles."""
        flat = self.current_cache["flat"]
        q75 = self.current_cache["quartiles"][1]
        iqr = self.current_cache["iqr"]
        threshold = q75 + 1.5 * iqr
        outliers = flat[flat > threshold]
        return (
            float(outliers.sum().abs() / flat.abs().sum()) if len(outliers) > 0 else 0.0
        )

    def _calculate_clustering(self) -> float:
        """Peak detection in histogram."""
        hist = self.current_cache["hist"]
        peaks = ((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])).sum()
        return float(peaks) / (len(hist) // 2)

    def _calculate_mode_collapse(self) -> float:
        """Using cached SVD."""
        S = self.current_cache["svd"]
        squared_sv = S**2
        pr = squared_sv.sum() ** 2 / (squared_sv**2).sum()
        return float(pr / len(S))

    def _calculate_zipf_deviation(self) -> float:
        """Compare to ideal Zipf distribution."""
        sorted_vals = self.current_cache["sorted"]
        ranks = torch.arange(
            1, len(sorted_vals) + 1, device=self.device, dtype=torch.float32
        )
        ideal_zipf = 1 / ranks
        ideal_zipf = ideal_zipf / ideal_zipf.sum()
        actual_dist = sorted_vals / sorted_vals.sum()
        kl_div = torch.sum(actual_dist * torch.log2(actual_dist / ideal_zipf + 1e-10))
        return float(1 / (1 + kl_div))

    def _calculate_compression(self) -> float:
        """bzip2 compression ratio."""
        tensor_bytes = self.current_cache["tensor"].cpu().numpy().tobytes()
        return len(bz2.compress(tensor_bytes)) / len(tensor_bytes)

    def _calculate_memorization(self) -> float:
        """Based on compression ratio."""
        return 1 - self._calculate_compression()

    def _calculate_lyapunov(self) -> float:
        """Estimate largest Lyapunov exponent."""
        diffs = torch.diff(self.current_cache["flat"])
        div_rates = torch.log(torch.abs(diffs) + 1e-10)
        positive_rates = div_rates[div_rates > 0]
        return float(positive_rates.mean()) if len(positive_rates) > 0 else 0.0

    def _calculate_permutation_entropy(self) -> float:
        """Ordinal pattern complexity."""
        flat = self.current_cache["flat"]
        n = min(4, len(flat) - 1)
        patterns = []
        for i in range(len(flat) - n):
            pattern = flat[i : i + n]
            perm = torch.argsort(pattern)
            patterns.append(tuple(perm.tolist()))
        pattern_counts = Counter(patterns)
        probs = torch.tensor(
            [count / len(patterns) for count in pattern_counts.values()]
        )
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        return float(entropy / np.log2(math.factorial(n)))

    def _calculate_temperature(self) -> float:
        """Normalized variance-based temperature."""
        var = self.current_cache["var"]
        max_val = self.current_cache["sorted"][-1]
        temp = var / (max_val**2 + 1e-10)
        return float(2 / (1 + torch.exp(-5 * temp)) - 1)

    def _calculate_phase_coherence(self) -> float:
        """Using cached angles."""
        angles = self.current_cache["angles"]
        return float(torch.abs(torch.mean(torch.exp(1j * angles))))

    def _calculate_wasserstein(self) -> float:
        """Distance to normal distribution."""
        sorted_tensor = self.current_cache["sorted"]
        normal_samples = torch.sort(torch.randn_like(sorted_tensor))[0]
        return float(torch.abs(sorted_tensor - normal_samples).mean())

    def _calculate_phase_space(self) -> float:
        """Using delay embedding."""
        x = (
            self.current_cache["flat"] - self.current_cache["mean"]
        ) / self.current_cache["std"]
        tau = max(1, len(x) // 100)
        x1 = x[: -2 * tau]
        x2 = x[tau:-tau]
        x3 = x[2 * tau :]
        points = torch.stack([x1, x2, x3])
        dists = torch.cdist(points.T, points.T)
        return float(torch.exp(-dists).mean())

    def _calculate_persistence(self) -> float:
        """Topological persistence from histogram."""
        hist = self.current_cache["hist"]
        peaks = []
        valleys = []
        for i in range(1, len(hist) - 1):
            if hist[i - 1] < hist[i] > hist[i + 1]:
                peaks.append((i, hist[i]))
            elif hist[i - 1] > hist[i] < hist[i + 1]:
                valleys.append((i, hist[i]))
        persistence = 0.0
        if peaks and valleys:
            for peak_idx, peak_val in peaks:
                closest_valley = min(valleys, key=lambda x: abs(x[0] - peak_idx))
                persistence += abs(peak_val - closest_valley[1])
        return persistence / (len(peaks) + 1e-6)

    def _calculate_mahalanobis(self) -> float:
        """Calculate the Mahalanobis distance between two tensors."""
        diff = self.current_cache["flat"] - self.cross_cache["flat"]
        cov_matrix = torch.cov(
            torch.stack([self.current_cache["flat"], self.cross_cache["flat"]], dim=1).T
        )
        inv_cov_matrix = torch.linalg.inv(cov_matrix)
        mahalanobis_dist = torch.sqrt(
            torch.matmul(torch.matmul(diff.T, inv_cov_matrix), diff)
        )
        return float(mahalanobis_dist)

    def _hyperbolic_distance(self) -> float:
        """Calculate hyperbolic distance between tensor manifolds"""
        x = self.current_cache["flat"]
        y = self.cross_cache["flat"]
        # Project tensors onto Poincaré ball
        x_norm = torch.norm(x)
        y_norm = torch.norm(y)
        x_p = x / (1 + torch.sqrt(1 + x_norm**2))
        y_p = y / (1 + torch.sqrt(1 + y_norm**2))
        inner_term = -2 * torch.sum(x_p * y_p)
        return torch.acosh(1 + inner_term)

    def _calculate_mutual_info(self) -> float:
        """Using joint histogram."""
        hist_2d = self._compute_histogram(
            torch.stack([self.current_cache["flat"], self.cross_cache["flat"]], dim=1)
        )
        h1 = self._calculate_entropy()
        self.current_cache, self.cross_cache = self.cross_cache, self.current_cache
        h2 = self._calculate_entropy()
        self.current_cache, self.cross_cache = self.cross_cache, self.current_cache
        h12 = float(-torch.sum(hist_2d * torch.log2(hist_2d + 1e-12)))
        return h1 + h2 - h12

    def _calculate_cosine_sim(self) -> float:
        """Direct cosine similarity."""
        return float(
            F.cosine_similarity(
                self.current_cache["flat"], self.cross_cache["flat"], dim=0
            )
        )

    def _calculate_cvd(self) -> float:
        """Cramer-von Mises with cached ranks."""
        xr = self.current_cache["ranks"]
        yr = self.cross_cache["ranks"]
        combined_values = torch.cat([xr, yr]).unique(sorted=True)
        cdf_x = torch.searchsorted(xr, combined_values, right=True).float() / len(xr)
        cdf_y = torch.searchsorted(yr, combined_values, right=True).float() / len(yr)
        return float((cdf_x - cdf_y).square().sum())

    def _calculate_cucconi(self) -> float:
        """Cucconi statistic using cached ranks."""
        xr = self.current_cache["ranks"]
        yr = self.cross_cache["ranks"]
        n, m = len(xr), len(yr)
        N = n + m
        combined = torch.cat([xr, yr])
        R = combined.argsort().argsort()[:n].float() + 1
        j = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        j_term = (j / (n + 1) - 0.5).square()
        U_term = torch.sub(float(N + 1), R).square()
        V_term = R.square()
        U = torch.sum(U_term * j_term)
        V = torch.sum(V_term * j_term)
        N_squared_minus_4 = N * N - 4
        rho = 2 * N_squared_minus_4 / (5 * N_squared_minus_4 + 16)
        rho_term = 1 - rho * rho
        return float((U + V - 2 * rho * torch.sqrt(U * V)) / (2 * rho_term))

    def _calculate_earth_mover(self) -> float:
        """EMD using cached sorted values."""
        a_sorted = self.current_cache["sorted"]
        b_sorted = self.cross_cache["sorted"]
        if len(a_sorted) != len(b_sorted):
            if len(a_sorted) > len(b_sorted):
                indices = torch.linspace(0, len(b_sorted) - 1, len(a_sorted))
                b_sorted = torch.tensor([b_sorted[int(i)] for i in indices])
            else:
                indices = torch.linspace(0, len(a_sorted) - 1, len(b_sorted))
                a_sorted = torch.tensor([a_sorted[int(i)] for i in indices])
        return float(torch.abs(a_sorted - b_sorted).mean())

    def _calculate_distribution_overlap(self) -> float:
        """Histogram intersection from cached histograms."""
        return float(
            torch.minimum(self.current_cache["hist"], self.cross_cache["hist"]).sum()
        )

    def _calc_viscosity(self) -> float:
        """Calculate effective viscosity using strain rate (use cached value if available)"""
        if self.current_cache.viscosity is not None:
            return self.current_cache.viscosity
        cache = self.current_cache
        return float(1.0 / (1.0 + cache.strain.mean()))

    def _calc_surface_tension(self) -> float:
        """Surface tension from density gradients (use cached value if available)"""
        if self.current_cache.gradient_magnitude is not None:
            return self.current_cache.gradient_magnitude
        cache = self.current_cache
        grad_mag = sum(g.pow(2).mean() for g in cache.gradients)
        return float(grad_mag.sqrt())

    def _calc_vorticity(self) -> float:
        """Calculate vorticity with proper curl approximation when possible"""
        cache = self.current_cache
        shape = cache.shape
        if len(shape) < 2:
            return 0.0
        if len(shape) == 2 and len(cache.gradients) == 2:
            try:
                curl = cache.gradients[1].mean(dim=1) - cache.gradients[0].mean(dim=0)
                return float(curl.abs().mean())
            except Exception:
                pass

        vorticity = 0.0
        for i, gi in enumerate(cache.gradients[:-1]):
            for gj in cache.gradients[i + 1 :]:
                vorticity += float((gi - gj).abs().mean())

        return vorticity

    def _calc_turbulence(self) -> float:
        """Turbulence from velocity fluctuations with improved epsilon"""
        cache = self.current_cache
        epsilon = max(1e-8, cache.tensor.abs().max() * 1e-5)
        return float(cache.velocity.std() / (cache.velocity.abs().mean() + epsilon))

    def _calc_flow_coherence(self) -> float:
        """Spatial coherence through autocorrelation with downsampling for large tensors"""
        cache = self.current_cache
        flat_density = cache.density.flatten()
        if flat_density.numel() > 10000:
            stride = max(1, flat_density.numel() // 10000)
            flat_density = flat_density[::stride]
        fft = torch.fft.rfft(flat_density)
        power = torch.abs(fft).pow(2)
        corr = torch.fft.irfft(power)
        epsilon = max(1e-8, corr.abs().max() * 1e-5)
        return float(corr[0] / (corr.max() + epsilon))

    def _calc_phase_coherence(self) -> float:
        """Phase coherence across field with multi-scale analysis"""
        cache = self.current_cache
        flat = cache.density.flatten()
        angles_short = torch.angle(flat[1:] + 1j * flat[:-1])
        coherence_short = torch.abs(torch.exp(1j * angles_short).mean())
        if len(flat) > 100:
            stride = max(5, len(flat) // 50)
            angles_medium = torch.angle(flat[stride:] + 1j * flat[:-stride])
            coherence_medium = torch.abs(torch.exp(1j * angles_medium).mean())
            return float(0.7 * coherence_short + 0.3 * coherence_medium)
        return float(coherence_short)

    def _calc_stability(self, cache=None) -> float:
        """Stability from spectral properties (use cached value if available)"""
        if cache is None:
            cache = self.current_cache
            if cache.stability is not None:
                return cache.stability
        velocity_mag = cache.velocity.abs()
        epsilon = max(1e-8, cache.tensor.abs().max() * 1e-5)
        return float(1.0 / (1.0 + velocity_mag.std() / (velocity_mag.mean() + epsilon)))

    def _calc_energy_density(self) -> float:
        """Total energy density"""
        return float(self.current_cache.energy.mean())

    def _calc_pressure_gradient(self) -> float:
        """Pressure gradient using density gradient"""
        cache = self.current_cache
        return float(2.0 * (cache.density * torch.stack(cache.gradients)).abs().mean())

    def _calc_reynolds(self) -> float:
        """Reynolds number analog using geometric mean characteristic length"""
        cache = self.current_cache
        return (
            float(cache.velocity.abs().mean() * cache.characteristic_length)
            / self._calc_viscosity()
        )

    def _calc_temporal_metrics(self) -> Dict[str, float]:
        """Calculate metrics between current and previous state"""
        curr, prev = self.current_cache, self.prev_cache
        if curr is None or prev is None:
            return {}
        current_stability = self._calc_stability(curr)
        prev_stability = self._calc_stability(prev)
        return {
            "information_flux": float((curr.density - prev.density).abs().mean()),
            "stability_trend": float(current_stability - prev_stability),
            "energy_transfer": float((curr.energy - prev.energy).mean()),
            "velocity_change": float((curr.velocity - prev.velocity).abs().mean()),
            "pattern_persistence": float(
                torch.cosine_similarity(
                    curr.density.flatten(), prev.density.flatten(), dim=0
                )
            ),
        }

    def compute_flow(self, tensor: Tensor, steps: int = 1) -> Tensor:
        """Compute natural flow field for tensor evolution"""
        self.analyze(tensor)
        cache = self.current_cache
        flow = torch.zeros_like(tensor)
        energy_grads = torch.gradient(cache.energy)
        for i, grad in enumerate(energy_grads):
            slices = [slice(None)] * len(cache.shape)
            slices[i] = slice(None)
            flow[tuple(slices)] -= grad * (1.0 - cache.viscosity)

        if len(cache.shape) >= 2:
            for i, gi in enumerate(cache.gradients[:-1]):
                for j, gj in enumerate(cache.gradients[i + 1 :], i + 1):
                    rotational = gi - gj
                    slices_i = [slice(None)] * len(cache.shape)
                    slices_i[j] = slice(None)
                    flow[tuple(slices_i)] += rotational * 0.2
                    slices_j = [slice(None)] * len(cache.shape)
                    slices_j[i] = slice(None)
                    flow[tuple(slices_j)] -= rotational * 0.2
        flow_norm = flow.norm()
        if flow_norm > 0:
            flow = flow / flow_norm
        return flow

    def _calculate_hurst_exponent(self) -> float:
        """Calculate Hurst exponent to detect fractal scaling properties."""
        x = self.current_cache["flat"]
        max_lag = min(100, len(x) // 4)  # Practical limit
        lags = range(2, max_lag)
        tau = [torch.sqrt(torch.var(x[lag:] - x[:-lag])) for lag in lags]
        m = torch.tensor([torch.log(t) for t in tau])
        x_vals = torch.tensor([torch.log(torch.tensor(float(lag))) for lag in lags])
        slope = (m.mean() * x_vals.mean() - (m * x_vals).mean()) / (
            x_vals.mean() ** 2 - (x_vals**2).mean()
        )

        return float(slope)

    def _calculate_multifractal_width(self, ret: str = "") -> float:
        """Estimate width of multifractal spectrum from gradient distribution."""
        gradients = self.current_cache["gradients"]
        tensor_dim = self.current_cache["shape"][-1]
        min_scale = max(
            8, min(tensor_dim // 100, 2)
        )  # At least 2, ideally ~1% of dimension
        max_scale = min(tensor_dim // 4, 256)  # Don't exceed 25% of dimension
        log_steps = 4  # How many scales to sample
        scales = torch.logspace(
            math.log2(min_scale), math.log2(max_scale), log_steps, base=2
        ).int()

        alpha_min = float("inf")
        alpha_max = float("-inf")
        correlation_dims = []
        for scale in scales:
            coarse_grads = [g[::scale] for g in gradients]
            grad_mag = torch.stack([g.abs() for g in coarse_grads]).mean(0)
            alpha = torch.log(grad_mag + 1e-8) / torch.log(torch.tensor(1.0 / scale))
            alpha_min = min(alpha_min, float(alpha.min()))
            alpha_max = max(alpha_max, float(alpha.max()))
            pairs = torch.cdist(grad_mag.unsqueeze(0), grad_mag.unsqueeze(0)).flatten()
            below_threshold = (pairs < scale).float().mean()
            if below_threshold > 0:
                corr_dim = torch.log(below_threshold) / torch.log(
                    torch.tensor(float(scale))
                )
                correlation_dims.append(float(corr_dim))

        if ret == "dimension":
            return (
                sum(correlation_dims) / len(correlation_dims)
                if correlation_dims
                else 0.0
            )
        else:
            return alpha_max - alpha_min


class SFTDataset(Dataset):
    def __init__(self, sft_data_list):
        self.sft_data = sft_data_list

    def __len__(self):
        return len(self.sft_data)

    def __getitem__(self, idx):
        return self.sft_data[idx]


class HDCore:
    """Core hyperdimensional computing engine with natural scaling properties"""

    def __init__(self, config):
        self.dim = config.get("hdc_dim", 10000)
        self.seed = config.get("random_seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self._pos_vectors = self.create_orthogonal_set(
            int(np.sqrt(self.dim))  # Natural scale - sqrt(dim) positions
        )
        self._base_vectors = self.create_orthogonal_set(
            int(np.log2(self.dim))  # log2(dim) is the natural capacity
        )
        self._random_similarity_stats = []

    def binarize(self, vector):
        """Convert to {-1,1} binary representation"""
        return np.sign(vector + 1e-10)

    def create_random_vector(self):
        """Create normalized random binary vector"""
        return self.binarize(self.rng.normal(size=self.dim))

    def similarity(self, v1, v2):
        """Normalized dot product similarity"""
        sim = np.dot(v1, v2) / self.dim
        if (
            len(self._random_similarity_stats) < int(np.log2(self.dim))
            and random.random() < 0.01
        ):
            self._random_similarity_stats.append(abs(sim))
        return sim

    def bind(self, v1, v2):
        """Bind two vectors (element-wise multiplication)"""
        return v1 * v2

    def bundle(self, vectors, weights=None):
        """Combine multiple vectors with optional weighting"""
        if not vectors:
            return np.zeros(self.dim)

        if weights is None:
            result = np.sum(vectors, axis=0)
        else:
            result = np.average(vectors, axis=0, weights=weights)

        return self.binarize(result)

    def permute(self, vector, shift=1):
        """Apply permutation for creating dissimilar vectors"""
        return np.roll(vector, shift)

    def create_orthogonal_set(self, n):
        """Create a set of approximately orthogonal vectors"""
        base = self.create_random_vector()
        return [self.permute(base, i * (self.dim // n)) for i in range(n)]

    def position_encode(self, idx):
        """Get position encoding vector - scales naturally with dimension"""
        return self._pos_vectors[idx % len(self._pos_vectors)]

    def hash_encode(self, data):
        """Create stable vector from any hashable data - HDC primitive"""
        data_hash = hash(str(data))
        base_idx = data_hash % len(self._base_vectors)
        base_vector = self._base_vectors[base_idx]
        perm_levels = abs(data_hash) % int(np.log2(self.dim)) + 1
        return self.permute(base_vector, perm_levels)

    def similarity_matrix(self, query, vectors):
        """Efficiently compute similarities for multiple vectors"""
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        return np.dot(vectors, query.T).flatten() / self.dim

    def cleanup(self, noisy_vector, attractors):
        """Clean up a noisy vector using HD attractors"""
        similarities = self.similarity_matrix(noisy_vector, np.array(attractors))
        best_idx = np.argmax(similarities)
        return attractors[best_idx]

    def get_adaptive_threshold(self):
        """Calculate similarity threshold based on HD dimension"""
        if len(self._random_similarity_stats) >= int(np.log2(self.dim) / 2):
            mean_sim = np.mean(self._random_similarity_stats)
            return mean_sim + np.std(self._random_similarity_stats)
        else:
            return 0.5 * (1 + 1 / np.sqrt(self.dim))


class HDMemory:
    """Hyperdimensional memory system riding the natural flows of HD space"""

    def __init__(self, hd_core, config):
        self.hdc = hd_core
        self.memory = {}
        self.cache = {}
        self.config = config
        self.cache_size = self._resolve_auto_param("cache_size", self.hdc.dim)
        if config.get("concurrency", "none") == "threadsafe":
            from threading import RLock

            self.lock = RLock()
        else:
            self.lock = _DummyLock()
        self.stats_keeper = {
            "lookups": 0,
            "cache_hits": 0,
            "similarities": deque(maxlen=int(np.sqrt(self.hdc.dim))),
            "last_reset": time.time(),
        }

    def _resolve_auto_param(self, param_name, default_value):
        """Resolve 'auto' config values to concrete numbers"""
        value = self.config.get(param_name, "auto")
        if value == "auto":
            return default_value
        return value

    def encode(self, key, data):
        """Encode anything to binary vector using natural HDC patterns"""
        with self.lock:
            if key in self.cache:
                self.stats_keeper["cache_hits"] += 1
                return self.cache[key]
            self.stats_keeper["lookups"] += 1
            if isinstance(data, (int, float)):
                vector = self._encode_numeric(data)
            elif isinstance(data, str):
                vector = self._encode_text(data)
            elif isinstance(data, dict):
                vector = self._encode_dict(data)
            elif isinstance(data, (list, tuple)):
                vector = self._encode_sequence(data)
            else:
                vector = self.hdc.hash_encode(data)
            self.memory[key] = vector
            self._update_cache(key, vector)
            return vector

    def _encode_numeric(self, num):
        """Encode numbers using logarithmic HD space mapping"""
        if num == 0:
            return self.hdc.hash_encode(0)
        sign = 1 if num >= 0 else -1
        log_val = np.log10(abs(num) + 1e-10)
        magnitude = int(log_val)
        precision = log_val - magnitude
        sign_vec = (
            self.hdc.create_random_vector()
            if sign > 0
            else -self.hdc.create_random_vector()
        )
        magnitude_vec = self.hdc.hash_encode(magnitude)
        precision_vec = self.hdc.hash_encode(int(precision * self.hdc.dim))

        return self.hdc.bind(sign_vec, self.hdc.bind(magnitude_vec, precision_vec))

    def _encode_text(self, text):
        """Encode text using HD-native multi-scale features"""
        if not text:
            return np.zeros(self.hdc.dim)

        if len(text) <= int(np.log2(self.hdc.dim)):
            return self.hdc.hash_encode(text)

        vectors = []
        words = text.lower().split()

        for i, word in enumerate(words):
            word_vec = self.hdc.hash_encode(word)
            pos_vec = self.hdc.position_encode(i)
            vectors.append(self.hdc.bind(word_vec, pos_vec))

        n = max(2, min(5, int(np.log2(len(text)) / np.log2(self.hdc.dim / 10) + 2)))
        for i in range(len(text) - n + 1):
            ngram = text[i : i + n]
            vectors.append(self.hdc.hash_encode(ngram))

        return self.hdc.bundle(vectors)

    def _encode_dict(self, data):
        """Encode dictionary using natural information-weighted bundling"""
        if not data:
            return np.zeros(self.hdc.dim)

        vectors = []
        weights = []
        for k, v in data.items():
            key_vec = self.hdc.hash_encode(k)
            val_key = f"{k}_{hash(str(v)) % self.hdc.dim}"
            val_vec = self.encode(val_key, v)
            pair_vec = self.hdc.bind(key_vec, val_vec)
            vectors.append(pair_vec)
            if isinstance(v, str):
                weights.append(np.log1p(len(v)))
            elif isinstance(v, dict):
                weights.append(np.log1p(len(v)))
            else:
                weights.append(1.0)

        if weights:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = None
        else:
            weights = None

        return self.hdc.bundle(vectors, weights)

    def _encode_sequence(self, items):
        """Encode sequence with pure information distribution weighting"""
        if not items:
            return np.zeros(self.hdc.dim)
        if len(items) == 1:
            return self.encode(f"item_{hash(str(items[0]))}", items[0])
        item_ids = []
        item_vectors = []
        for i, item in enumerate(items):
            item_id = hash(str(item))
            item_ids.append(item_id)
            item_key = f"seq_item_{item_id % self.hdc.dim}"
            item_vec = self.encode(item_key, item)
            pos_vec = self.hdc.position_encode(i)
            item_vectors.append(self.hdc.bind(item_vec, pos_vec))
        counter = Counter(item_ids)
        total_items = len(items)
        weights = [np.log(total_items / counter[item_id]) for item_id in item_ids]
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]
        return self.hdc.bundle(item_vectors, weights)

    def _update_cache(self, key, vector):
        """Cache management with dimension-scaled similarity eviction"""
        if len(self.cache) >= self.cache_size:
            evict_key = self._find_least_similar(vector)
            del self.cache[evict_key]

        self.cache[key] = vector

    def _find_least_similar(self, vector):
        """Find least similar vector with natural sampling rate"""
        sample_ratio = np.sqrt(np.log2(self.hdc.dim) / self.hdc.dim)
        sample_size = max(1, int(len(self.cache) * sample_ratio))
        if sample_size < len(self.cache):
            sample_keys = random.sample(list(self.cache.keys()), sample_size)
        else:
            sample_keys = list(self.cache.keys())
        return min(
            sample_keys,
            key=lambda k: self.hdc.similarity(vector, self.cache[k]),
            default=next(iter(self.cache)),
        )

    def batch_encode(self, items):
        """Encode multiple items in one go"""
        with self.lock:
            results = {}
            for key, data in items.items():
                results[key] = self.encode(key, data)
            return results

    def retrieve(self, key):
        """Get vector from memory"""
        return self.memory.get(key, np.zeros(self.hdc.dim))

    def find_similar(self, query_vector, threshold=None, limit=None):
        """Find similar vectors with dimension-appropriate approach"""
        with self.lock:
            limit = (
                limit
                if limit is not None
                else self.config.get("result_limit", int(np.log2(self.hdc.dim)))
            )
            if threshold is None:
                threshold = self.hdc.get_adaptive_threshold()
                similarities = list(self.stats_keeper["similarities"])
                if len(similarities) >= int(np.log2(self.hdc.dim)):
                    sims = np.array(similarities)
                    std_scale = np.log2(self.hdc.dim) / self.hdc.dim
                    threshold = max(threshold, sims.mean() - std_scale * sims.std())

            if len(self.memory) > self.hdc.dim:
                return self._find_similar_matrix(query_vector, threshold, limit)
            else:
                return self._find_similar_iterative(query_vector, threshold, limit)

    def _find_similar_matrix(self, query_vector, threshold, limit):
        """Optimized matrix-based similarity search"""
        keys = list(self.memory.keys())
        vectors = np.array(list(self.memory.values()))
        similarities = self.hdc.similarity_matrix(query_vector, vectors)
        sample_size = int(np.log2(self.hdc.dim))
        if len(similarities) > sample_size:
            sample_idx = np.random.choice(len(similarities), sample_size, replace=False)
            for idx in sample_idx:
                self.stats_keeper["similarities"].append(similarities[idx])
        else:
            for sim in similarities:
                self.stats_keeper["similarities"].append(sim)
        mask = similarities >= threshold
        matches = [(keys[i], float(similarities[i])) for i in np.where(mask)[0]]
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:limit]

    def _find_similar_iterative(self, query_vector, threshold, limit):
        """Classic iterative similarity for smaller memories"""
        results = []
        for key, vector in self.memory.items():
            sim = self.hdc.similarity(query_vector, vector)
            self.stats_keeper["similarities"].append(sim)

            if sim >= threshold:
                results.append((key, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def clear_cache(self):
        """Reset cache"""
        with self.lock:
            self.cache = {}

    def stats(self):
        """Get natural HDC statistics"""
        with self.lock:
            stats = {
                "memory_size": len(self.memory),
                "cache_size": len(self.cache),
                "cache_ratio": len(self.cache) / max(1, len(self.memory)),
                "cache_hit_ratio": self.stats_keeper["cache_hits"]
                / max(1, self.stats_keeper["lookups"]),
                "hd_dimension": self.hdc.dim,
            }
            similarities = list(self.stats_keeper["similarities"])
            if similarities:
                sims = np.array(similarities)
                stats.update(
                    {
                        "sim_mean": float(sims.mean()),
                        "sim_std": float(sims.std()),
                        "sim_min": float(sims.min()),
                        "sim_max": float(sims.max()),
                    }
                )

            return stats


class HDCDatasetEncoder:
    """Dataset encoding using HD-native patterns and memory management"""

    def __init__(self, database, config):
        self.database = database
        self.config = config
        self.hdc = HDCore(config)
        self.memory = HDMemory(self.hdc, config)
        self._subspaces = self.hdc.create_orthogonal_set(4)
        self.subspace_map = {
            "dataset": self._subspaces[0],
            "task": self._subspaces[1],
            "concept": self._subspaces[2],
            "tensor": self._subspaces[3],
        }

    def encode_sft_datasets(self):
        """Process datasets using HD-native batch operations"""
        datasets = self._fetch_sft_datasets()
        results = []
        dataset_batch = {f"dataset_{d['name']}": d for d in datasets}
        self.memory.batch_encode(dataset_batch)
        for dataset in datasets:
            dataset_key = f"dataset_{dataset['name']}"
            try:
                dataset_vec = self.memory.retrieve(dataset_key)
                subspace_vec = self.hdc.bind(dataset_vec, self.subspace_map["dataset"])
                dataset_id = self._store_hd_entity(
                    "dataset", dataset["name"], subspace_vec, dataset
                )
                task_count = self._process_elements(dataset, dataset_id, subspace_vec)

                results.append(
                    {
                        "dataset": dataset["name"],
                        "dataset_id": dataset_id,
                        "tasks_processed": task_count,
                        "status": "success",
                    }
                )
            except Exception as e:
                results.append(
                    {"dataset": dataset["name"], "error": str(e), "status": "failed"}
                )
        self.memory._flush_signatures()
        return {
            "processed_datasets": results,
            "system_stats": self.memory.stats(),
            "total_datasets": len(datasets),
            "success_rate": len([r for r in results if r["status"] == "success"])
            / len(datasets),
        }

    def _process_elements(self, dataset, dataset_id, dataset_vec):
        """Process tasks and concepts using HD relationships"""
        tasks = self._extract_tasks(dataset)
        task_batch = {f"task_{t['name']}": t for t in tasks}
        self.memory.batch_encode(task_batch)

        for task in tasks:
            task_key = f"task_{task['name']}"
            task_vec = self.memory.retrieve(task_key)
            task_subvec = self.hdc.bind(task_vec, self.subspace_map["task"])
            task_id = self._store_hd_entity("task", task["name"], task_subvec, task)
            self._store_hd_relation(
                dataset_id,
                task_id,
                "contains_task",
                self.hdc.similarity(dataset_vec, task_subvec),
            )

            self._process_concepts(task, task_id, task_subvec)

    def _process_concepts(self, task, task_id, task_vec):
        """Process concepts using HD semantic patterns"""
        concepts = self._extract_concepts(task)
        concept_batch = {f"concept_{c['name']}": c for c in concepts}
        self.memory.batch_encode(concept_batch)

        for concept in concepts:
            concept_key = f"concept_{concept['name']}"
            concept_vec = self.memory.retrieve(concept_key)
            concept_subvec = self.hdc.bind(concept_vec, self.subspace_map["concept"])
            concept_id = self._store_hd_entity(
                "concept", concept["name"], concept_subvec, concept
            )
            self._store_hd_relation(
                task_id,
                concept_id,
                "uses_concept",
                self.hdc.similarity(task_vec, concept_subvec),
            )

    def _extract_concepts(self, task):
        """HD-native concept extraction using memory features"""
        if not self.config.get("auto_extract_concepts", True):
            return []

        samples = task.get("samples", [])
        concept_vectors = []
        for sample in samples:
            text = self._get_sample_text(sample)
            if text:
                vec = self.memory.encode(f"sample_{hash(text)}", text)
                concept_vectors.append(vec)
        if concept_vectors:
            centroid = self.hdc.bundle(concept_vectors)
            return self._cluster_concepts(centroid, concept_vectors)

        return []

    def _cluster_concepts(self, centroid, vectors):
        """HD-native clustering using memory capabilities"""
        concepts = []
        threshold = self.hdc.get_adaptive_threshold()

        for vec in vectors:
            if self.hdc.similarity(centroid, vec) >= threshold:
                concept_name = f"concept_{hash(tuple(vec))}"
                concepts.append(
                    {
                        "name": concept_name,
                        "description": "HD-clustered concept",
                        "vector": vec,
                    }
                )

        return concepts[: self.config.get("max_concepts", 10)]

    def _store_hd_entity(self, etype, name, vector, metadata):
        """Store using HD memory's native format"""
        return self.memory.encode(
            f"{etype}_{name}", {"vector": vector, "metadata": metadata}
        )

    def _store_hd_relation(self, source, target, rel_type, weight):
        """Store relationship using memory's batch system"""
        self.memory.encode(
            f"rel_{source}_{target}",
            {"source": source, "target": target, "type": rel_type, "weight": weight},
        )

    def _store_hd_entity(self, entity_type: str, name: str, metadata: dict) -> str:
        """Store entities with full HD vector integration"""
        vector = self.memory.encode(f"{entity_type}_{name}", metadata)
        entity_id = str(uuid.uuid4())
        with self.database.conn as conn:
            conn.execute(
                """
                INSERT INTO hdc_entities 
                (entity_id, entity_type, name, hdc_key, vector_snapshot)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    entity_type,
                    name,
                    f"{entity_type}_{name}",
                    vector.tobytes(),  # Store raw bytes for verification
                ),
            )
        return entity_id

    def _store_hd_relation(
        self, source_id: str, target_id: str, rel_type: str, weight: float
    ):
        """Store relationships using HD binding patterns"""
        source_vec = self.memory.retrieve(source_id)
        target_vec = self.memory.retrieve(target_id)
        rel_vec = self.hdc.bind(source_vec, target_vec)
        rel_key = f"rel_{source_id}_{target_id}"
        self.memory.encode(
            rel_key, {"type": rel_type, "weight": weight, "vector": rel_vec}
        )

        with self.database.conn as conn:
            conn.execute(
                """
                INSERT INTO hdc_relations 
                (source_id, target_id, relation_type, similarity)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, rel_type, weight),
            )

    def find_similar_vectors(self, query_vector, **kwargs):
        """Unified similarity search using HD memory"""
        hd_results = self.memory.find_similar(query_vector, **kwargs)
        with self.database.conn as conn:
            cursor = conn.cursor()
            return [self._get_entity_details(r[0], cursor) for r in hd_results]

    def _get_entity_details(self, hd_key: str, cursor) -> dict:
        """Get database metadata for HD memory results"""
        cursor.execute(
            "SELECT entity_id, entity_type, name FROM hdc_entities WHERE hdc_key = ?",
            (hd_key,),
        )
        result = cursor.fetchone()
        return {
            "id": result[0],
            "type": result[1],
            "name": result[2],
            "vector": self.memory.retrieve(hd_key),
        }

    def encode_prompt(self, prompt_text: str) -> np.ndarray:
        """Leverage HD memory's text encoding directly"""
        return self.memory.encode(f"prompt_{self._text_hash(prompt_text)}", prompt_text)

    def encode_tensor_metrics(self, metrics: dict) -> np.ndarray:
        """Use HD memory's dictionary encoding"""
        return self.memory.encode(f"metrics_{self._metrics_hash(metrics)}", metrics)

    def match_tensor_to_prompt(self, prompt_vector: np.ndarray, **kwargs) -> list:
        """Subspace-optimized tensor matching"""
        return self.memory.find_similar(
            self.hdc.bind(prompt_vector, self.subspace_map["tensor"]),
            subspace=self.subspace_map["tensor"],
            **kwargs,
        )

    def _text_hash(self, text: str) -> int:
        """Consistent hashing for text entities"""
        return hash(text) % self.hdc.dim

    def _metrics_hash(self, metrics: dict) -> int:
        """Stable hash for metric dictionaries"""
        ordered = json.dumps(metrics, sort_keys=True)
        return hash(ordered) % self.hdc.dim


class HDCTensorNavigator:
    """Navigate tensor space using natural HDC properties"""

    def __init__(self, database, hd_core, hd_memory, config):
        self.database = database
        self.hdc = hd_core
        self.memory = hd_memory
        self.config = config

        # Use HDCore's native orthogonal set creation
        base_count = int(np.log2(self.hdc.dim))  # Natural capacity
        self.subspaces = dict(
            zip(
                ["performance", "architecture", "task", "metric"],
                self.hdc.create_orthogonal_set(base_count)[:4],
            )
        )

        # Memory-based metric mapping
        self._init_mapping()

    def _init_mapping(self):
        """Initialize mapping directly from HD memory"""
        # Check if we have stored mappings
        mapping_key = "metric_task_mapping"
        if mapping_key in self.memory.memory:
            self.metric_mapping = self.memory.retrieve(mapping_key)
        else:
            # Bootstrap from merge report data
            self.metric_mapping = self._build_mapping_from_reports()
            self.memory.encode(mapping_key, self.metric_mapping)

    def _build_mapping_from_reports(self):
        """Build initial mapping from merge reports"""
        reports = self._fetch_merge_reports()
        if not reports:
            # Create empty mapping that will learn over time
            return {}

        # Extract task-metric relationships
        mappings = {}
        for report in reports:
            task_vec = self.memory.encode(
                f"task_{hash(report['task'])}", report["task"]
            )

            for tensor in report.get("tensors", []):
                metrics = tensor.get("metrics", {})
                if not metrics:
                    continue

                # Encode metrics-to-task relationship
                metrics_vec = self.memory.encode(
                    f"metrics_{hash(str(metrics))}", metrics
                )
                performance = tensor.get("performance", 0.5)

                # Use tensor name as key
                tensor_name = tensor.get("name", "unknown")
                if tensor_name not in mappings:
                    # Initialize with task-performance binding
                    mappings[tensor_name] = self.hdc.bind(
                        task_vec,
                        self.hdc.bind(
                            metrics_vec, self.subspaces["performance"] * performance
                        ),
                    )
                else:
                    # Bundle with existing mapping
                    mappings[tensor_name] = self.hdc.bundle(
                        [
                            mappings[tensor_name],
                            self.hdc.bind(
                                task_vec,
                                self.hdc.bind(
                                    metrics_vec,
                                    self.subspaces["performance"] * performance,
                                ),
                            ),
                        ]
                    )

        return mappings

    def _fetch_merge_reports(self):
        """Get merge reports from database"""
        reports = []
        with self.database.conn as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT report_id, base_model_name, creation_timestamp, metrics_json FROM merge_reports"
            )

            for row in cursor.fetchall():
                report_id, model_name, timestamp, metrics = row
                if metrics:
                    try:
                        metrics_data = json.loads(metrics)
                        reports.append(
                            {
                                "report_id": report_id,
                                "model": model_name,
                                "timestamp": timestamp,
                                "metrics": metrics_data,
                                "task": f"optimize_{model_name}",  # Default task
                            }
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Error decoding metrics for report {report_id}: {e}"
                        )
                        pass

        return reports

    def update_mapping(self, task, metrics, performance):
        """Update mapping with new performance data"""
        # Encode task and metrics
        task_vec = self.memory.encode(f"task_{hash(task)}", task)
        metrics_vec = self.memory.encode(f"metrics_{hash(str(metrics))}", metrics)

        # Natural learning rate that scales with dimension
        # Lower dimensions need faster learning, higher dims can be more stable
        lr = self.config.get(
            "mapping_lr", np.sqrt(np.log2(self.hdc.dim) / self.hdc.dim)
        )

        # Create performance binding
        perf_binding = self.hdc.bind(
            task_vec,
            self.hdc.bind(metrics_vec, self.subspaces["performance"] * performance),
        )

        # Get metric key
        metric_key = f"metrics_{hash(str(metrics))}"

        # Update mapping
        if metric_key in self.metric_mapping:
            # Update existing mapping
            current = self.metric_mapping[metric_key]
            self.metric_mapping[metric_key] = self.hdc.bundle(
                [current, perf_binding], weights=[1 - lr, lr]
            )
        else:
            # New mapping
            self.metric_mapping[metric_key] = perf_binding

    def find_optimal_tensors(self, task, available_tensors, limit=None):
        """Find optimal tensors for a task using HDC similarity"""
        # Default to log2(dim) results - natural capacity
        limit = limit or int(np.log2(self.hdc.dim))

        # Encode task
        task_vec = self.memory.encode(f"task_{hash(task)}", task)

        # Get tensor signatures
        tensor_sigs = {}
        for tensor_id in available_tensors:
            sig_key = f"tensor_sig_{tensor_id}"
            if sig_key in self.memory.memory:
                tensor_sigs[tensor_id] = self.memory.retrieve(sig_key)

        # Find matches using native HDC similarity
        matches = []
        for tensor_id, sig in tensor_sigs.items():
            # Bind with task to see compatibility
            compatibility = self.hdc.similarity(
                task_vec, self.hdc.bind(sig, self.subspaces["task"])
            )
            matches.append((tensor_id, compatibility))

        # Sort by compatibility
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def predict_tensor_performance(self, task, tensor_metrics):
        """Predict tensor performance on a task"""
        # Encode task and metrics
        task_vec = self.memory.encode(f"task_{hash(task)}", task)
        metrics_vec = self.memory.encode(
            f"metrics_{hash(str(tensor_metrics))}", tensor_metrics
        )

        # Check for direct match in mappings
        metric_key = f"metrics_{hash(str(tensor_metrics))}"
        if metric_key in self.metric_mapping:
            # Extract performance through binding
            mapping_vec = self.metric_mapping[metric_key]
            similarity = self.hdc.similarity(
                task_vec,
                self.hdc.bind(
                    mapping_vec, self.hdc.permute(self.subspaces["performance"])
                ),
            )

            # Scale to [0,1] performance range
            return (similarity + 1) / 2

        # No direct match - use nearest neighbors
        nearest_metrics = self._find_nearest_metrics(metrics_vec)
        if not nearest_metrics:
            return 0.5  # Default neutral performance

        # Use weighted average of neighbor performances
        total_weight = sum(w for _, w in nearest_metrics)
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(
            self.predict_tensor_performance(task, m) * w for m, w in nearest_metrics
        )

        return weighted_sum / total_weight

    def _find_nearest_metrics(self, metrics_vec, limit=3):
        """Find nearest metric sets using HDC similarity"""
        metric_keys = [
            k for k in self.metric_mapping.keys() if k.startswith("metrics_")
        ]
        neighbors = []

        for key in metric_keys:
            mapping_vec = self.metric_mapping[key]
            sim = self.hdc.similarity(metrics_vec, mapping_vec)
            if sim > self.hdc.get_adaptive_threshold():
                # Extract original metrics dict from memory
                metric_dict = self.memory.retrieve(key)
                neighbors.append((metric_dict, sim))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:limit]


class HDCTensorSignature:
    """Generate tensor signatures using pure HDC patterns"""

    def __init__(self, database, hd_core, hd_memory, config):
        self.database = database
        self.hdc = hd_core
        self.memory = hd_memory
        self.config = config
        self._tensor_buffer = []
        self.buffer_size = int(np.sqrt(self.hdc.dim))  # Natural buffer size

    def generate_signature(self, tensor_id, metrics):
        """Create HDC signature from tensor metrics"""
        metrics_vec = self.memory.encode(f"metrics_{tensor_id}", metrics)
        sig_key = f"tensor_sig_{tensor_id}"
        self.memory.encode(sig_key, metrics_vec)
        self._buffer_signature(tensor_id, metrics_vec)
        return metrics_vec

    def generate_batch_signatures(self, tensor_metrics):
        """Process multiple tensors at once"""
        results = {}

        for tensor_id, metrics in tensor_metrics.items():
            metrics_key = f"metrics_{tensor_id}"
            sig_key = f"tensor_sig_{tensor_id}"
            metrics_vec = self.memory.encode(metrics_key, metrics)
            self.memory.encode(sig_key, metrics_vec)
            self._buffer_signature(tensor_id, metrics_vec)
            results[tensor_id] = metrics_vec
        return results

    def _buffer_signature(self, tensor_id, signature):
        """Buffer signature for batch database storage"""
        self._tensor_buffer.append((tensor_id, signature.tobytes(), time.time()))
        if len(self._tensor_buffer) >= self.buffer_size:
            self._flush_signatures()

    def _flush_signatures(self):
        """Store buffered signatures in database"""
        if not self._tensor_buffer:
            return

        with self.database.conn as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO tensor_signatures
                (tensor_id, signature, timestamp)
                VALUES (?, ?, ?)
                """,
                self._tensor_buffer,
            )

        self._tensor_buffer = []

    def compare_signatures(self, sig1, sig2):
        """Compare two signatures using HDC similarity"""
        similarity = float(self.hdc.similarity(sig1, sig2))
        hamming_sim = float(np.mean(sig1 == sig2))
        return {
            "similarity": similarity,
            "hamming": hamming_sim,
            "combined": (similarity + hamming_sim) / 2,
        }

    def retrieve_signature(self, tensor_id):
        """Get stored signature for a tensor"""
        sig_key = f"tensor_sig_{tensor_id}"
        if sig_key in self.memory.memory:
            return self.memory.retrieve(sig_key)

        with self.database.conn as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT signature FROM tensor_signatures WHERE tensor_id = ?",
                (tensor_id,),
            )

            result = cursor.fetchone()
            if result:
                sig_bytes = result[0]
                sig_vector = np.frombuffer(sig_bytes, dtype=np.int8)
                self.memory.encode(sig_key, sig_vector)
                return sig_vector

        # Not found
        return None

    def find_similar_tensors(self, query_signature, threshold=None, limit=None):
        """Find tensors with similar signatures"""
        if threshold is None:
            threshold = self.hdc.get_adaptive_threshold()

        limit = limit or int(np.log2(self.hdc.dim))
        sig_keys = [k for k in self.memory.memory.keys() if k.startswith("tensor_sig_")]
        matches = []
        for key in sig_keys:
            sig = self.memory.retrieve(key)
            sim = self.hdc.similarity(query_signature, sig)
            if sim >= threshold:
                tensor_id = key.replace("tensor_sig_", "")
                matches.append((tensor_id, sim))

        # Sort and limit
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]


class ModelProcessor:
    """Processes ANY model architecture because tensors are tensors, baby."""

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
        """Process ANY model's tensors - architecture is just details."""
        model_id = str(uuid.uuid4())
        model_configs = self._load_model_configs(model_path)
        self.database.store_model_data(model_id, model_name, model_configs)

        model = self.model_loader.load_model(model_path)
        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()
            for tensor_path, tensor in self._iter_model_tensors(model):
                self._process_tensor(cursor, model_id, tensor_path, tensor)

            conn.commit()

    def _iter_model_tensors(
        self, model: nn.Module
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate through ALL tensors because discrimination is bad."""
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only care about trainable tensors
                yield name, param.data.clone().cpu()

    def _process_tensor(
        self,
        cursor: sqlite3.Cursor,
        model_id: str,
        tensor_path: str,
        tensor: torch.Tensor,
    ) -> None:
        """Process a tensor without judgement about its role in life."""
        tensor_id = str(uuid.uuid4())
        related_tensors = self._get_related_tensors(tensor_path, tensor)
        normalized_tensor = self.tensor_analyzer._normalize_tensor(tensor)
        analysis_results = self.tensor_analyzer.analyze_tensor(
            normalized_tensor, related_tensors
        )
        self.database.store_tensor(cursor, tensor_id, model_id, tensor_path, tensor)
        self.database.store_tensor_metrics(
            cursor,
            tensor_id,
            analysis_results["single_metrics"],
            analysis_results.get("cross_metrics"),
        )

    def _get_related_tensors(
        self, tensor_path: str, tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Find tensors that might have meaningful relationships with this one."""
        # TODO: Add relationship detection, just return empty list fo
        return []

    def _load_model_configs(self, model_path: str) -> dict[str, str]:
        """Grab all the config files we might need later."""
        model_path = Path(model_path)
        configs = {}

        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "generation_config.json",
            "added_tokens.json",
        ]:
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


### -- ideas to integrate -- ###


# Standard Q-Learning Baseline (unchanged)
class QLearningAgent:
    def __init__(
        self,
        state_size=10,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_prob=0.1,
    ):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (
            reward
            + self.discount_factor * self.q_table[next_state, best_next_action]
            - self.q_table[state, action]
        )

    def train(self, episodes=100):
        performance_history = []
        for episode in range(episodes):
            state = random.randint(0, self.q_table.shape[0] - 1)
            action = self.choose_action(state)
            reward = random.uniform(0, 1)
            next_state = random.randint(0, self.q_table.shape[0] - 1)
            self.update_q_table(state, action, reward, next_state)
            performance_history.append(reward)
        return performance_history


# Optimized Swarm Agent
class SwarmAgent:
    def __init__(self, id, queen, state_size=10, action_size=4, hd_dim=1000):
        self.id = id
        self.queen = queen
        self.state_size = state_size
        self.action_size = action_size

        # Enhanced agent genetic traits
        self.traits = {
            "learning_rate": random.uniform(0.05, 0.5),
            "discount_factor": random.uniform(0.7, 0.99),
            "exploration_prob": random.uniform(0.05, 0.3),
            "eligibility_trace": random.uniform(0, 0.3),
            "reward_scaling": random.uniform(0.8, 1.2),
        }

        # HD memory for pattern recognition
        self.hd_memory = HDMemory(dim=hd_dim)
        self.state_vectors = {}  # Cache for encoded states

        # Sparse representation of Q-values for memory efficiency
        self.q_values = {}
        self.performance = 0.5
        self.memory = []
        self.experience_count = 0

    def encode_state(self, state):
        # Get or create HD vector for this state
        if state not in self.state_vectors:
            self.state_vectors[state] = self.hd_memory.encode(f"state_{state}", state)
        return self.state_vectors[state]

    def choose_action(self, state):
        state_key = f"s{state}"

        # Initialize state entry if needed
        if state_key not in self.q_values:
            self.q_values[state_key] = np.random.uniform(0, 0.1, size=self.action_size)

        # Dynamic exploration strategy based on experience
        current_exploration = max(
            0.05,
            self.traits["exploration_prob"]
            * (1.0 / (1.0 + 0.01 * self.experience_count)),
        )

        # Explore or exploit
        if random.uniform(0, 1) < current_exploration:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_values[state_key])

    def update_knowledge(self, state, action, reward, next_state, is_shared_exp=False):
        state_key = f"s{state}"
        next_state_key = f"s{next_state}"

        # Apply individual reward scaling
        scaled_reward = reward * self.traits["reward_scaling"]

        # Initialize if needed
        if state_key not in self.q_values:
            self.q_values[state_key] = np.random.uniform(0, 0.1, size=self.action_size)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.random.uniform(
                0, 0.1, size=self.action_size
            )

        # Encode states with HD vectors for pattern recognition
        state_vector = self.encode_state(state)
        next_state_vector = self.encode_state(next_state)

        # Similarity can influence learning
        state_similarity = self.hd_memory.similarity(state_vector, next_state_vector)
        similarity_factor = (
            1.0 + 0.2 * (state_similarity + 1.0) / 2.0
        )  # Range [1.0, 1.2]

        # Adaptive learning rate based on whether experience is shared
        effective_lr = self.traits["learning_rate"]
        if is_shared_exp:
            effective_lr *= 0.5  # Reduced learning from shared experiences

        # Q-learning update with eligibility traces influence
        best_next_value = np.max(self.q_values[next_state_key])
        td_error = (
            scaled_reward
            + self.traits["discount_factor"] * best_next_value
            - self.q_values[state_key][action]
        )

        # Apply eligibility trace effect to neighboring actions (simplified)
        for a in range(self.action_size):
            # Calculate action similarity - closer actions get more update
            action_similarity = 1.0 - abs(a - action) / self.action_size
            eligibility = self.traits["eligibility_trace"] * action_similarity

            if a == action:
                # Direct update for chosen action
                self.q_values[state_key][a] += (
                    effective_lr * td_error * similarity_factor
                )
            elif eligibility > 0.05:  # Skip tiny updates for efficiency
                # Neighboring actions get smaller updates
                self.q_values[state_key][a] += effective_lr * td_error * eligibility

        # Update memory for performance tracking
        if not is_shared_exp:  # Only track performance on own experiences
            self.memory.append(scaled_reward)
            if len(self.memory) > 10:  # Fixed small memory for efficiency
                self.memory.pop(0)

            # Update overall performance with recency bias
            self.performance = 0.3 * self.performance + 0.7 * np.mean(self.memory)
            self.experience_count += 1

            # Adaptive exploration decay based on performance
            if self.experience_count % 10 == 0:
                self.traits["exploration_prob"] = max(
                    0.05,  # Minimum exploration
                    self.traits["exploration_prob"] * (1.0 - 0.01 * self.performance),
                )

    def mutate_traits(self, mutation_rate=0.1):
        # Enhanced mutation with adaptive rates
        for trait in self.traits:
            if random.random() < mutation_rate:
                if trait == "learning_rate":
                    self.traits[trait] = max(
                        0.01, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1))
                    )
                elif trait == "discount_factor":
                    self.traits[trait] = max(
                        0.5, min(0.99, self.traits[trait] + random.uniform(-0.1, 0.1))
                    )
                elif trait == "exploration_prob":
                    self.traits[trait] = max(
                        0.01, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1))
                    )
                elif trait == "eligibility_trace":
                    self.traits[trait] = max(
                        0, min(0.5, self.traits[trait] + random.uniform(-0.1, 0.1))
                    )
                elif trait == "reward_scaling":
                    self.traits[trait] = max(
                        0.5, min(1.5, self.traits[trait] + random.uniform(-0.2, 0.2))
                    )


# Optimized Swarm Queen
class SwarmQueen:
    def __init__(self, num_agents=10, state_size=10, action_size=4):
        self.agents = [
            SwarmAgent(i, self, state_size, action_size) for i in range(num_agents)
        ]
        self.state_size = state_size
        self.action_size = action_size
        self.elite_size = max(1, num_agents // 5)  # Top 20% are elite
        self.generation = 0

        # Track best agent traits for convergence analysis
        self.best_traits_history = []

    def select_parents(self):
        # Tournament selection with diversity bonus
        sorted_agents = sorted(self.agents, key=lambda a: a.performance, reverse=True)

        # Calculate diversity metrics
        diversity_scores = []
        for agent in sorted_agents:
            # Measure trait distance from average
            avg_traits = {
                t: np.mean([a.traits[t] for a in sorted_agents[: self.elite_size]])
                for t in agent.traits
            }

            dist = sum(
                abs(agent.traits[t] - avg_traits[t]) / avg_traits[t]
                for t in agent.traits
            )

            diversity_scores.append(dist)

        # Normalize diversity scores
        if max(diversity_scores) > 0:
            diversity_scores = [d / max(diversity_scores) for d in diversity_scores]

        # Create weighted selection probabilities (performance + diversity)
        performance_ranks = [1.0 / (i + 1) for i in range(len(sorted_agents))]
        selection_weights = [
            0.7 * p + 0.3 * d for p, d in zip(performance_ranks, diversity_scores)
        ]

        # Normalize weights to probabilities
        total_weight = sum(selection_weights)
        selection_probs = [w / total_weight for w in selection_weights]

        # Select parents based on these probabilities
        parent_indices = np.random.choice(
            len(sorted_agents),
            size=min(10, len(sorted_agents)),
            p=selection_probs,
            replace=False,
        )

        return [sorted_agents[i] for i in parent_indices]

    def crossover(self, parent1, parent2):
        # Create child agent
        child = SwarmAgent(len(self.agents), self, self.state_size, self.action_size)

        # Inherit traits with intelligent crossover
        for trait in parent1.traits:
            # Interpolation rather than binary selection
            mix_ratio = random.random()
            child.traits[trait] = (
                mix_ratio * parent1.traits[trait]
                + (1 - mix_ratio) * parent2.traits[trait]
            )

            # Add small random variation
            child.traits[trait] += random.uniform(-0.05, 0.05)

            # Ensure bounds
            if trait == "learning_rate":
                child.traits[trait] = max(0.01, min(0.5, child.traits[trait]))
            elif trait == "discount_factor":
                child.traits[trait] = max(0.5, min(0.99, child.traits[trait]))
            elif trait == "exploration_prob":
                child.traits[trait] = max(0.01, min(0.5, child.traits[trait]))
            elif trait == "eligibility_trace":
                child.traits[trait] = max(0, min(0.5, child.traits[trait]))
            elif trait == "reward_scaling":
                child.traits[trait] = max(0.5, min(1.5, child.traits[trait]))

        # Knowledge transfer from parents - inherit some Q-values
        # This is computationally expensive but very effective for learning
        if random.random() < 0.5:  # 50% chance to transfer knowledge
            for state_key in set(parent1.q_values.keys()).intersection(
                parent2.q_values.keys()
            ):
                if len(child.q_values) < 20:  # Limit to prevent memory bloat
                    # Weighted average favoring the better performing parent
                    if parent1.performance > parent2.performance:
                        weight1, weight2 = 0.7, 0.3
                    else:
                        weight1, weight2 = 0.3, 0.7

                    child.q_values[state_key] = (
                        weight1 * parent1.q_values[state_key]
                        + weight2 * parent2.q_values[state_key]
                    )

        return child

    def evolve_population(self):
        # Track evolution progress
        self.generation += 1

        # Select parents
        parents = self.select_parents()

        # Preserve elites
        sorted_agents = sorted(self.agents, key=lambda a: a.performance, reverse=True)
        new_agents = []

        # Direct elitism - keep top performers unchanged
        elite_count = min(self.elite_size, len(sorted_agents))
        new_agents.extend(sorted_agents[:elite_count])

        # Track best agent traits
        if elite_count > 0:
            self.best_traits_history.append(sorted_agents[0].traits.copy())

        # Generate offspring for the rest of the population
        while len(new_agents) < len(self.agents):
            # Select two parents
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, k=2)

                # Create offspring
                child = self.crossover(parent1, parent2)

                # Mutation rate increases as generations progress to avoid premature convergence
                adaptive_mutation_rate = 0.1 + 0.05 * min(1.0, self.generation / 20)
                child.mutate_traits(mutation_rate=adaptive_mutation_rate)

                new_agents.append(child)
            else:
                # If not enough parents, create a random agent
                new_agent = SwarmAgent(
                    len(new_agents), self, self.state_size, self.action_size
                )
                new_agents.append(new_agent)

        # Update IDs and transfer population
        for i, agent in enumerate(new_agents):
            agent.id = i

        self.agents = new_agents

    def run_evaluation_cycle(self):
        # Create shared experience buffer
        shared_experiences = []

        # Evaluate all agents
        for agent in self.agents:
            agent_experiences = []

            # Each agent tries multiple tasks
            for _ in range(5):  # 5 evaluations per cycle for statistical stability
                state = random.randint(0, self.state_size - 1)
                action = agent.choose_action(state)

                # Enhanced reward function that creates better learning signal
                base_reward = random.uniform(0, 1)

                # Action-state matching component
                action_match = 1.0 - 0.3 * abs(
                    (action / self.action_size) - (state / self.state_size)
                )

                # Progressive difficulty based on agent performance
                difficulty_factor = 1.0 + 0.3 * (agent.performance)

                # Final reward calculation
                reward = (base_reward * action_match) / difficulty_factor

                # State transition model with some stochasticity
                next_state = (state + action + random.randint(-1, 1)) % self.state_size

                # Store experience
                agent_experiences.append((state, action, reward, next_state))

                # Agent learns from its own experience
                agent.update_knowledge(state, action, reward, next_state)

            # Add best experience to shared pool
            if agent_experiences:
                best_exp = max(
                    agent_experiences, key=lambda x: x[2]
                )  # Select by reward
                shared_experiences.append(best_exp)

        # Experience sharing among elite agents
        if shared_experiences:
            elite_agents = sorted(
                self.agents, key=lambda a: a.performance, reverse=True
            )[: self.elite_size]

            # Each elite agent learns from 2 random shared experiences
            for agent in elite_agents:
                for _ in range(min(2, len(shared_experiences))):
                    exp = random.choice(shared_experiences)
                    state, action, reward, next_state = exp
                    # Learn from shared experience with reduced weight
                    agent.update_knowledge(
                        state, action, reward, next_state, is_shared_exp=True
                    )

        # Return average and best performance
        performances = [agent.performance for agent in self.agents]
        return np.mean(performances), max(performances) if performances else 0

    def run_evolution_cycle(self, cycles_per_generation=3):
        # Run multiple evaluation cycles
        cycle_avg_performances = []
        cycle_best_performances = []

        for _ in range(cycles_per_generation):
            avg_perf, best_perf = self.run_evaluation_cycle()
            cycle_avg_performances.append(avg_perf)
            cycle_best_performances.append(best_perf)

        # Evolve population
        self.evolve_population()

        return np.mean(cycle_avg_performances), np.mean(cycle_best_performances)


# Optimized Evolutionary Reinforcement Learning Framework
class ERLFramework:
    def __init__(self, state_size=10, action_size=4, num_agents=10):
        self.swarm_queen = SwarmQueen(
            num_agents=num_agents, state_size=state_size, action_size=action_size
        )
        self.avg_performance_history = []
        self.best_performance_history = []
        self.population_size_history = []

    def train(self, iterations=50, cycles_per_generation=3):
        for i in range(iterations):
            # Adaptive cycles - more evaluation as training progresses
            adaptive_cycles = cycles_per_generation
            if i > iterations // 2:
                adaptive_cycles += 1  # One more cycle in latter half of training

            # Run evolution cycle
            avg_performance, best_performance = self.swarm_queen.run_evolution_cycle(
                adaptive_cycles
            )

            # Track histories
            self.avg_performance_history.append(avg_performance)
            self.best_performance_history.append(best_performance)
            self.population_size_history.append(len(self.swarm_queen.agents))

            # Dynamic population adjustments (every 10 iterations)
            if i > 0 and i % 10 == 0 and i < iterations - 10:
                # Check if performance has plateaued
                recent_best = self.best_performance_history[-5:]
                if (
                    max(recent_best) - min(recent_best) < 0.05
                ):  # Small variance indicates plateau
                    # Add new agents with traits similar to best agent but more diverse
                    best_agent = max(
                        self.swarm_queen.agents, key=lambda a: a.performance
                    )

                    # Add 2 new agents with traits influenced by best agent
                    for _ in range(2):
                        new_agent = SwarmAgent(
                            len(self.swarm_queen.agents),
                            self.swarm_queen,
                            self.swarm_queen.state_size,
                            self.swarm_queen.action_size,
                        )

                        # Copy and mutate best agent's traits with higher mutation
                        for trait in best_agent.traits:
                            mutation = random.uniform(-0.2, 0.2)
                            new_agent.traits[trait] = max(
                                0.01, min(0.99, best_agent.traits[trait] + mutation)
                            )

                        self.swarm_queen.agents.append(new_agent)

        return self.avg_performance_history, self.best_performance_history


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
    model_loader = ModelLoader(models_dir, database)  # Pass database to ModelLoader
    tensor_analyzer = TensorAnalyzer(device)
    model_processor = ModelProcessor(model_loader, tensor_analyzer, database)
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
        default="olm_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
