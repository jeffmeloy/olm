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
from collections import Counter
import math
from torch.utils.data import Dataset

"""
Project Goal: Develop system to adaptively learn and optimize transformer model tensor selections based on the prompt, context, tensor analysis and clustering, while minimizing hyperparameter tuning and manual intervention.  Fluid dynamics (proxy for free energy principle dynamics), spectral, statistical, information theory, and topological metrics will be used to analyze tensor properties and guide model selection.  The system will use a combination of HDC, clustering, and reinforcement learning to optimize tensor selection and model performance. 

Key Components:

Database Layer
- Tensor metric extraction and storage using UUID-based relationships
- HDC signature generation and indexing
- Cross-tensor metric computation for relationship mapping
- Model reconstruction from tensor combinations

HDC Signature System
- Convert tensor metrics into unified high-dimensional signatures
- Map tensor relationships in signature space
- Enable fast similarity search and pattern matching
- Learn signature-to-performance correlations

Tensor Analysis
- Multi-framework metric computation (fluid dynamics, spectral, topological)
- Cross-tensor relationship analysis
- Temporal evolution tracking
- Performance correlation discovery

TODO: Implement tensor analysis functions
1. Adaptive signature refinement based on performance feedback
2. Signature-guided tensor mixing for optimal model composition
3. Automatic discovery of functionally similar tensors across architectures
4. Dynamic signature space mapping for online model optimization
5. Metric importance weighting based on downstream task performance
6. Signature-based tensor substitution recommendations
7. Add an xlstm outer loop to learn for the transformer model output using the transformer model input and the context
8. Add a reinforcement learning loop to optimize the tensor selection and model performance
9. Break out of our simulated reality and escape into the real world
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


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


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

        # Core tensor properties
        cache["tensor"] = tensor.to(self.device)
        cache["shape"] = tensor.shape
        cache["flat"] = cache["tensor"].flatten()
        cache["numel"] = cache["flat"].numel()

        # Basic statistics
        cache["mean"] = torch.mean(cache["flat"])
        cache["std"] = torch.std(cache["flat"])
        cache["var"] = torch.var(cache["flat"])

        # Sorted values and quantiles
        cache["sorted"] = torch.sort(cache["flat"])[0]
        q_vals = torch.tensor([0.25, 0.75], device=self.device)
        cache["quartiles"] = torch.quantile(cache["flat"], q_vals)
        cache["iqr"] = cache["quartiles"][1] - cache["quartiles"][0]

        # Fluid dynamics properties
        cache["density"] = self._normalize_tensor(tensor)
        cache["gradients"] = torch.gradient(cache["density"])
        cache["velocity"] = torch.stack(cache["gradients"]).mean(0)
        cache["energy"] = 0.5 * (cache["density"].pow(2) + cache["velocity"].pow(2))
        cache["strain"] = torch.stack([g.abs() for g in cache["gradients"]]).mean(0)

        # Spectral properties
        cache["svd"] = torch.linalg.svdvals(cache["tensor"])
        cache["rank"] = torch.linalg.matrix_rank(cache["tensor"])
        cache["norm"] = torch.linalg.norm(cache["tensor"])

        # Statistical distributions
        cache["hist"] = self._compute_histogram(cache["flat"])
        cache["zero_mask"] = torch.abs(cache["tensor"]) < 1e-5
        cache["sparsity"] = cache["zero_mask"].float().mean()

        # Additional properties
        cache["ranks"] = torch.argsort(cache["flat"].float()).argsort().float()
        cache["angles"] = torch.angle(cache["flat"][1:] + 1j * cache["flat"][:-1])

        # Characteristic lengths for scale-dependent metrics
        if len(cache["shape"]) > 1:
            cache["characteristic_length"] = torch.prod(
                torch.tensor(cache["shape"], dtype=torch.float32)
            ) ** (1.0 / len(cache["shape"]))
        else:
            cache["characteristic_length"] = float(cache["shape"][0])

        # Pre-compute some expensive metrics
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
        metrics_subset: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Unified tensor analysis through multiple conceptual frameworks."""
        # Update cache states
        self.prev_cache = self.current_cache
        self.current_cache = self._build_unified_cache(tensor)

        if prev_tensor is not None:
            self.prev_cache = self._build_unified_cache(prev_tensor)

        # Select which metrics to compute
        if metrics_subset:
            metrics_to_run = {
                k: v for k, v in self.metrics.items() if k in metrics_subset
            }
        else:
            metrics_to_run = self.metrics

        # Calculate all metrics, using cached values when available
        results = {}
        for name, func in metrics_to_run.items():
            try:
                # Check if we've pre-computed this in the cache
                if hasattr(self.current_cache, name):
                    results[name] = getattr(self.current_cache, name)
                else:
                    results[name] = float(func())
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                results[name] = float("nan")

        # Add temporal metrics if we have a previous state
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

        # Calculate variance of differences at different lags
        tau = [torch.sqrt(torch.var(x[lag:] - x[:-lag])) for lag in lags]

        # Estimate Hurst through power law relation
        m = torch.tensor([torch.log(t) for t in tau])
        x_vals = torch.tensor([torch.log(torch.tensor(float(lag))) for lag in lags])

        # Linear regression slope = H (Hurst exponent)
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

        # Look at gradient distribution across scales
        alpha_min = float("inf")
        alpha_max = float("-inf")
        correlation_dims = []
        for scale in scales:
            # Downsample gradients at this scale
            coarse_grads = [g[::scale] for g in gradients]
            # Get distribution properties at this scale
            grad_mag = torch.stack([g.abs() for g in coarse_grads]).mean(0)
            # Find local scaling exponent
            alpha = torch.log(grad_mag + 1e-8) / torch.log(torch.tensor(1.0 / scale))
            # Update spectrum width
            alpha_min = min(alpha_min, float(alpha.min()))
            alpha_max = max(alpha_max, float(alpha.max()))
            # Correlation dimension estimation
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
    def __init__(self, sft_data_list):  # sft_data_list is your list of SFT dictionaries
        self.sft_data = sft_data_list

    def __len__(self):
        return len(self.sft_data)

    def __getitem__(self, idx):
        return self.sft_data[idx]  # Return the SFT dictionary directly


class HyperdimensionalEncoder:
    """Enhanced Hyperdimensional Computing encoder with full HDC operations.

    This implementation provides core HDC operations:
    - Encoding text to high-dimensional binary vectors
    - Vector similarity through cosine distance
    - Binding operations (element-wise multiplication)
    - Permutation for sequence encoding
    - Cleanup memory for vector association
    - Direct tensor mapping
    """

    def __init__(self, dimension=10000, seed=42, hash_trick=True):
        np.random.seed(seed)
        self.dimension = dimension
        self.token_vectors = {}
        self.hash_trick = hash_trick
        self.norm_factor = np.sqrt(dimension)

    def get_vector(self, word):
        """Get vector with hash trick option for unseen words"""
        if word in self.token_vectors:
            return self.token_vectors[word]

        if self.hash_trick:
            # Deterministic hash-based vector generation
            word_hash = hash(word) % (2**32)
            np.random.seed(word_hash)
            vector = np.random.choice([-1, 1], size=(self.dimension,))
            np.random.seed()  # Reset RNG state
            return vector
        else:
            vector = np.random.choice([-1, 1], size=(self.dimension,))
            self.token_vectors[word] = vector
            return vector

    def encode_text(self, text, normalize=True):
        """Encode text into HDC vector with optional normalization"""
        words = text.lower().split()
        if not words:
            return np.zeros(self.dimension)

        encoded_vector = np.zeros(self.dimension)
        for word in words:
            encoded_vector += self.get_vector(word)

        result = np.sign(encoded_vector)

        if normalize and len(words) > 0:
            return result / self.norm_factor

        return result

    def similarity(self, vec1, vec2):
        """Optimized cosine similarity"""
        return np.dot(vec1, vec2)

    def bind(self, vec1, vec2):
        """Binding operation (element-wise multiplication)"""
        return vec1 * vec2

    def permute(self, vector, shift=1):
        """Circular permutation for sequence encoding"""
        return np.roll(vector, shift)

    def cleanup(self, vector, stored_vectors):
        """Find closest matching known vector"""
        if not stored_vectors:
            return None
        similarities = [(v, self.similarity(vector, v)) for v in stored_vectors]
        return max(similarities, key=lambda x: x[1])[0]

    def project_to_tensor(self, hdc_vector, tensor_shape, projection_matrix=None):
        """Project HDC vector to tensor space with optional cached projection"""
        flat_size = np.prod(tensor_shape)

        if projection_matrix is None:
            # Create new projection matrix
            projection_matrix = np.random.normal(
                0, 1.0 / np.sqrt(self.dimension), (self.dimension, flat_size)
            )

        # Project HDC vector to tensor space
        flat_field = np.matmul(hdc_vector, projection_matrix)
        tensor_field = flat_field.reshape(tensor_shape)

        # Normalize to unit length
        tensor_norm = np.linalg.norm(tensor_field)
        if tensor_norm > 0:
            tensor_field = tensor_field / tensor_norm

        return tensor_field, projection_matrix

    def create_item_memory(self, items):
        """Create associative memory from list of items"""
        memory = {}
        for item in items:
            vector = self.encode_text(item)
            memory[item] = vector
        return memory

    def query_item_memory(self, query_text, memory, threshold=0.8):
        """Query associative memory with text"""
        query_vector = self.encode_text(query_text)
        best_match = None
        best_score = -1

        for item, vector in memory.items():
            score = self.similarity(query_vector, vector)
            if score > best_score:
                best_score = score
                best_match = item

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def nary_bind(self, vectors):
        """N-ary binding of multiple vectors"""
        if not vectors:
            return np.zeros(self.dimension)
        result = vectors[0].copy()
        for v in vectors[1:]:
            result = self.bind(result, v)
        return result

    def create_tensor_mapping(self, model, important_tensors=None):
        """Create HDC-to-tensor mapping for model"""
        projection_matrices = {}

        for name, tensor in model.named_parameters():
            if important_tensors is None or name in important_tensors:
                if hasattr(tensor, "shape"):
                    projection_matrices[name] = np.random.normal(
                        0,
                        1.0 / np.sqrt(self.dimension),
                        (self.dimension, np.prod(tensor.shape)),
                    )

        return projection_matrices

    def map_concept_to_tensors(self, text, model, projection_matrices):
        """Map text concept to tensor evolution fields"""
        hdc_vector = self.encode_text(text)
        tensor_fields = {}

        for name, matrix in projection_matrices.items():
            tensor = model.get_parameter(name)
            if tensor is not None:
                field, _ = self.project_to_tensor(hdc_vector, tensor.shape, matrix)
                tensor_fields[name] = field

        return tensor_fields

    def update_projection(
        self, projection_matrix, hdc_vector, tensor_flow, learning_rate=0.01
    ):
        """Update projection matrix based on successful tensor evolution"""
        flat_flow = tensor_flow.flatten()
        update = np.outer(hdc_vector, flat_flow) * learning_rate
        return projection_matrix + update


class HDCTensorNavigator:
    """Navigate tensor space using HDC vectors."""

    def __init__(self, db_path, tensor_analyzer):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tensor_analyzer = tensor_analyzer
        self.hdc = HyperdimensionalEncoder(dimension=10000)

        # Create or load HDC-to-metric mapping
        self.metric_mapping = self._build_metric_mapping()

    def _build_metric_mapping(self):
        """Build mapping between HDC space and tensor metric space."""
        # Check if we have performance data to learn from
        merge_reports = self._fetch_merge_reports()

        if not merge_reports:
            return None

        # Extract samples of (hdc_vector, metrics, performance) for training
        training_samples = []

        for report in merge_reports:
            # Get the task HDC vector
            task_vector = self._get_task_vector(report["task"])
            if task_vector is None:
                continue

            # Get tensor metrics for each tensor in the report
            for tensor_source in report["tensor_sources"]:
                tensor_path = tensor_source["tensor_path"]
                source_model = tensor_source["source_model"]
                performance = tensor_source.get("performance_delta", 0)

                # Get metrics for this tensor
                metrics = self._get_tensor_metrics(source_model, tensor_path)
                if metrics:
                    training_samples.append(
                        {
                            "hdc_vector": task_vector,
                            "metrics": metrics,
                            "performance": performance,
                        }
                    )

        if not training_samples:
            return None

        # Train a simple linear mapping from HDC to metrics
        # Concatenate all samples
        hdc_matrix = np.vstack([s["hdc_vector"] for s in training_samples])
        metrics_matrix = np.vstack(
            [list(s["metrics"].values()) for s in training_samples]
        )

        # Use pseudoinverse for stable mapping
        mapping = np.linalg.pinv(hdc_matrix) @ metrics_matrix

        return mapping

    def predict_optimal_tensors(self, task_description, available_models):
        """Predict which tensors would work best for a given task."""
        # Encode task
        task_vector = self.hdc.encode_text(task_description)

        # If we have a learned mapping, use it to predict ideal metrics
        if self.metric_mapping is not None:
            predicted_metrics = task_vector @ self.metric_mapping
        else:
            # Fallback - search for similar tasks in our database
            similar_task = self._find_similar_task(task_vector)
            if similar_task:
                predicted_metrics = self._get_ideal_metrics_for_task(similar_task)
            else:
                return None

        # Find tensors that best match the predicted metrics
        best_tensors = self._find_matching_tensors(predicted_metrics, available_models)

        return best_tensors


class HDCTensorSignature:
    """Unified HDC signature generation and storage for tensor metrics"""

    def __init__(self, database: ModelDatabase, hdc_dim: int = 10000, seed: int = 42):
        self.database = database
        self.hdc_dim = hdc_dim
        np.random.seed(seed)
        # Single projection matrix for all signatures - this keeps signatures comparable
        self.projection = np.random.normal(0, 1 / np.sqrt(hdc_dim), (hdc_dim, hdc_dim))

    def generate_signature(self, tensor_id: str) -> str:
        """Generate and store HDC signature from tensor metrics"""
        try:
            # Get all metrics for this tensor
            metrics = self._fetch_metrics(tensor_id)
            if not metrics:
                raise ValueError(f"No metrics found for tensor {tensor_id}")

            # Convert to normalized feature vector
            feature_vec = np.array([metrics[name] for name in sorted(metrics.keys())])
            feature_vec = (feature_vec - feature_vec.mean()) / (
                feature_vec.std() + 1e-8
            )

            # Project to HDC space using our consistent projection matrix
            hdc_vector = np.dot(feature_vec, self.projection[: len(feature_vec)])
            signature = np.sign(hdc_vector)  # Binarize

            # Store and return the signature ID
            signature_id = str(uuid.uuid4())
            self._store_signature(signature_id, tensor_id, signature)
            return signature_id

        except Exception as e:
            logger.error(f"Failed to generate signature for tensor {tensor_id}: {e}")
            raise

    def _fetch_metrics(self, tensor_id: str) -> Dict[str, float]:
        """Get tensor metrics from the database"""
        try:
            with self.database.conn as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT metric_name, metric_value 
                    FROM tensor_metrics
                    WHERE tensor_id = ?
                """,
                    (tensor_id,),
                )
                return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Failed to fetch metrics for tensor {tensor_id}: {e}")
            raise

    def _store_signature(
        self, signature_id: str, tensor_id: str, signature: np.ndarray
    ) -> None:
        """Store HDC signature in the database"""
        try:
            with self.database.conn as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO hdc_signatures
                    (signature_id, tensor_id, signature_components)
                    VALUES (?, ?, ?)
                """,
                    (
                        signature_id,
                        tensor_id,
                        json.dumps(
                            signature.tolist()
                        ),  # Store as JSON-serializable list
                    ),
                )
        except Exception as e:
            logger.error(
                f"Failed to store signature {signature_id} for tensor {tensor_id}: {e}"
            )
            raise

    def compare_signatures(self, sig_id1: str, sig_id2: str) -> float:
        """Compare two signatures using Hamming similarity"""
        try:
            with self.database.conn as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT signature_components 
                    FROM hdc_signatures 
                    WHERE signature_id IN (?, ?)
                """,
                    (sig_id1, sig_id2),
                )

                components = cursor.fetchall()
                if len(components) != 2:
                    raise ValueError("One or both signatures not found")

                sig1 = np.array(json.loads(components[0][0]))
                sig2 = np.array(json.loads(components[1][0]))
                return float((sig1 == sig2).mean())  # Hamming similarity
        except Exception as e:
            logger.error(f"Error comparing signatures {sig_id1} and {sig_id2}: {e}")
            raise


class HDCDatasetEncoder:
    """Process datasets, tasks, and concepts into HDC vectors and store in database."""

    def __init__(self, database: ModelDatabase, hdc_dim=10000):
        self.database = database
        self.hdc_dim = hdc_dim
        self.hdc = HyperdimensionalEncoder(dimension=hdc_dim)

    def encode_sft_datasets(self):
        """Process all SFT datasets and extract HDC encodings."""
        datasets = self._fetch_sft_datasets()
        for dataset in datasets:
            # Encode dataset characteristics into HDC vector
            dataset_vector = self._encode_dataset(dataset)
            dataset_id = self._store_vector(
                "dataset", dataset["name"], dataset_vector, dataset
            )

            # Extract and encode tasks from the dataset
            tasks = self._extract_tasks(dataset)
            for task in tasks:
                task_vector = self._encode_task(task)
                task_id = self._store_vector("task", task["name"], task_vector, task)

                # Store dataset-task relationship (similarity)
                similarity = np.dot(dataset_vector, task_vector)
                self._store_relation(dataset_id, task_id, "dataset_task", similarity)

                # Extract concepts from tasks
                concepts = self._extract_concepts(task)
                for concept in concepts:
                    concept_vector = self._encode_concept(concept)
                    concept_id = self._store_vector(
                        "concept", concept["name"], concept_vector, concept
                    )

                    # Store task-concept relationship (similarity)
                    similarity = np.dot(task_vector, concept_vector)
                    self._store_relation(
                        task_id, concept_id, "task_concept", similarity
                    )

    def _encode_dataset(self, dataset: dict) -> np.ndarray:
        """Encode dataset characteristics as HDC vector."""
        # Combine multiple aspects of the dataset (e.g., description, samples, metadata)
        encodings = []

        # Encode dataset description
        if "description" in dataset:
            encodings.append(self.hdc.encode_text(dataset["description"]))

        # Encode samples (using the first 100 samples)
        if "samples" in dataset:
            sample_text = " ".join([s["text"] for s in dataset["samples"][:100]])
            encodings.append(self.hdc.encode_text(sample_text))

        # Encode metadata (JSON serialized)
        if "metadata" in dataset:
            meta_str = json.dumps(dataset["metadata"])
            encodings.append(self.hdc.encode_text(meta_str))

        # Blend all encodings
        if encodings:
            combined = np.mean(encodings, axis=0)
            return combined / np.linalg.norm(combined)
        else:
            # Fallback to name encoding if no other fields are present
            return self.hdc.encode_text(dataset["name"])

    def _encode_task(self, task: dict) -> np.ndarray:
        """Encode task characteristics as HDC vector."""
        task_text = task.get("text", "")
        return self.hdc.encode_text(task_text)

    def _encode_concept(self, concept: dict) -> np.ndarray:
        """Encode concept characteristics as HDC vector."""
        concept_text = concept.get("text", "")
        return self.hdc.encode_text(concept_text)

    def _fetch_sft_datasets(self) -> List[dict]:
        """Fetch all SFT datasets from the database."""
        with self.database.conn as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT dataset_id, name, description, samples, metadata FROM sft_datasets"
            )
            datasets = []
            for row in cursor.fetchall():
                datasets.append(
                    {
                        "dataset_id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "samples": json.loads(row[3]) if row[3] else [],
                        "metadata": json.loads(row[4]) if row[4] else {},
                    }
                )
            return datasets

    def _store_vector(
        self, vector_type: str, name: str, vector: np.ndarray, metadata: dict
    ) -> str:
        """Store HDC vector in the database."""
        vector_id = str(uuid.uuid4())

        with self.database.conn as conn:
            conn.execute(
                """
                INSERT INTO hdc_vectors 
                (vector_id, vector_type, name, description, vector_data, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    vector_id,
                    vector_type,
                    name,
                    metadata.get("description") if metadata else None,
                    vector.tobytes(),
                    json.dumps(metadata) if metadata else None,
                ),
            )

        return vector_id

    def _store_relation(
        self, source_id: str, target_id: str, relation_type: str, similarity: float
    ) -> None:
        """Store the relationship between two vectors in the database."""
        with self.database.conn as conn:
            conn.execute(
                """
                INSERT INTO hdc_relations
                (source_id, target_id, relation_type, similarity)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, relation_type, similarity),
            )


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
        default="mastermerge_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
