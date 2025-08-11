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
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field


logging.basicConfig(level=logging.INFO, format="%(asctime)s > %(message)s")
logger = logging.getLogger(__name__)


@contextmanager
def gpu_compute():
    """Unified device management for GPU computation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        yield device
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()


class TensorAnalyzer:
    """
    A non-parametric, robust, and truly hyperparameter-free tensor analysis engine.

    This class computes a rich "fingerprint" for any tensor by measuring its intrinsic
    properties. All internal algorithmic parameters, such as analysis windows and pattern
    dimensions, are derived directly from the input tensor's own dimensions and complexity,
    ensuring the analysis scales naturally and consistently. Intended to be a pure, data-driven
    instrument, free of arbitrary constants and distributional assumptions.
    """

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.eps = torch.finfo(torch.float32).eps

    def _compute_effective_rank(self, tensor: torch.Tensor) -> float:
        """Computes the effective rank, a robust measure of intrinsic dimensionality."""
        if tensor.dim() < 2:
            return 1.0
        matrix = tensor if tensor.dim() == 2 else tensor.flatten(start_dim=1)
        try:
            S = torch.linalg.svdvals(matrix)
            S = S[S > self.eps]
            if S.numel() == 0:
                return 0.0
            s_squared = S.pow(2)
            return (s_squared.sum().pow(2) / s_squared.pow(2).sum()).item()
        except torch.linalg.LinAlgError:
            return 0.0

    @torch.no_grad()
    def analyze_tensor(self, tensor: torch.Tensor) -> dict[str, float]:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return self._null_signature()
        tensor = tensor.to(self.device, dtype=torch.float32)
        dims = torch.tensor(tensor.shape, dtype=torch.float32, device=self.device)
        characteristic_length = torch.pow(dims.prod(), 1.0 / dims.numel())
        effective_rank = self._compute_effective_rank(tensor)

        signature = {}
        signature.update(self._magnitude_analysis(tensor))
        signature.update(self._spectral_analysis(tensor, effective_rank))
        signature.update(self._temporal_complexity(tensor, characteristic_length))
        signature.update(self._concentration_analysis(tensor))
        signature.update(self._continuous_field_analysis(tensor))
        signature.update(
            self._geometric_analysis(tensor, effective_rank, characteristic_length)
        )

        return {k: v.item() for k, v in signature.items()}

    def _magnitude_analysis(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if tensor.numel() == 0:
            return self._null_magnitude_signature()
        abs_tensor = tensor.abs()
        l2_norm = torch.linalg.norm(tensor)
        abs_median = abs_tensor.median()
        abs_max = abs_tensor.max()

        nonzero_mask = abs_tensor > self.eps
        if nonzero_mask.any():
            robust_min = abs_tensor[nonzero_mask].min()
            dynamic_range = torch.log10(abs_max / robust_min.clamp(min=self.eps))
        else:
            dynamic_range = torch.tensor(0.0)
        return {
            "l2_norm": l2_norm,
            "abs_median": abs_median,
            "abs_max": abs_max,
            "dynamic_range": dynamic_range,
        }

    def _spectral_analysis(
        self, tensor: torch.Tensor, effective_rank: float
    ) -> dict[str, torch.Tensor]:
        if tensor.dim() < 2:
            return self._null_spectral_signature(effective_rank)
        matrix = tensor if tensor.dim() == 2 else tensor.flatten(start_dim=1)
        norm_factor = matrix.abs().max().clamp(min=self.eps)
        matrix_norm = matrix / norm_factor
        try:
            S = torch.linalg.svdvals(matrix_norm)
        except torch.linalg.LinAlgError:
            return self._null_spectral_signature(effective_rank)
        S = S[S > self.eps]
        if S.numel() == 0:
            return self._null_spectral_signature(effective_rank)

        p_s = S / S.sum()
        spectral_entropy = -(p_s * torch.log(p_s + self.eps)).sum()
        log_rank = torch.log(
            torch.arange(1, S.numel() + 1, device=self.device, dtype=torch.float32)
        )
        log_S = torch.log(S)
        cov_matrix = torch.cov(torch.stack([log_rank, log_S]))
        scaling_exponent = cov_matrix[0, 1] / (cov_matrix[0, 0] + self.eps)
        return {
            "effective_rank": torch.tensor(effective_rank),
            "spectral_entropy": spectral_entropy,
            "scaling_exponent": scaling_exponent,
        }

    def _temporal_complexity(
        self, tensor: torch.Tensor, characteristic_length: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        flat_tensor = tensor.flatten()
        n = flat_tensor.numel()
        median, mad = flat_tensor.median(), self._compute_mad(flat_tensor)
        if mad > self.eps:
            flat_tensor.sub_(median).div_(mad)

        # Bound derivation for Hurst exponent analysis
        min_lag = max(2, int(torch.sqrt(characteristic_length).item()))
        max_lag = min(
            n // 4, int(characteristic_length * torch.log2(characteristic_length + 1))
        )
        max_lag = max(min_lag, max_lag)
        if n < 20 or max_lag <= min_lag:
            return self._null_temporal_signature()

        lags = torch.arange(min_lag, max_lag + 1, device=self.device)
        rs_vals = []
        for lag in lags:
            diffs = flat_tensor[lag.item() :] - flat_tensor[: -lag.item()]
            if diffs.numel() < 2:
                continue
            cumsum_range = (diffs - diffs.median()).cumsum(0)
            R = cumsum_range.max() - cumsum_range.min()
            S = self._compute_mad(diffs)
            rs_vals.append(R / S.clamp(min=self.eps))

        if len(rs_vals) < 2:
            hurst_exponent = torch.tensor(0.5)
        else:
            log_lags = torch.log(lags[: len(rs_vals)].float())
            log_rs = torch.log(
                torch.tensor(rs_vals, device=self.device).clamp(min=self.eps)
            )
            cov_matrix = torch.cov(torch.stack([log_lags, log_rs]))
            hurst_exponent = cov_matrix[0, 1] / (cov_matrix[0, 0] + self.eps)

        lower_m = max(3, int(torch.log2(characteristic_length).item()))
        upper_m = 7  # Kept as a hard computational safety limit
        m = max(lower_m, min(upper_m, int(math.log(n))))

        if n < m:
            permutation_entropy = torch.tensor(0.0)
        else:
            unfolded = flat_tensor.unfold(0, m, 1)
            perms = torch.argsort(unfolded, dim=1)
            unique_perms, counts = torch.unique(perms, dim=0, return_counts=True)
            probs = counts.float() / perms.size(0)
            entropy = -(probs * torch.log(probs + self.eps)).sum()
            max_entropy = math.log(math.factorial(m))
            permutation_entropy = (
                entropy / max_entropy if max_entropy > 0 else torch.tensor(0.0)
            )
        return {
            "hurst_exponent": hurst_exponent,
            "permutation_entropy": permutation_entropy,
        }

    def _concentration_analysis(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        abs_flat = tensor.flatten().abs()
        sorted_abs, _ = torch.sort(abs_flat)
        n = sorted_abs.numel()
        if n < 2 or sorted_abs[-1] <= self.eps:
            return self._null_concentration_signature()
        cum_weights = torch.cumsum(sorted_abs, dim=0)
        total_weight = cum_weights[-1]
        gini_coefficient = (n + 1 - 2 * (cum_weights.sum() / total_weight)) / n
        midpoint_index = n // 2
        population_midpoint = midpoint_index / n
        wealth_at_midpoint = cum_weights[midpoint_index] / total_weight
        lorenz_asymmetry = 0.5 - (population_midpoint + wealth_at_midpoint)
        return {
            "gini_coefficient": gini_coefficient,
            "lorenz_asymmetry": lorenz_asymmetry,
        }

    def _continuous_field_analysis(
        self, tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        flat_tensor = tensor.flatten()
        n = flat_tensor.numel()
        if n < 4:
            return self._null_field_signature()
        median_centered = flat_tensor - flat_tensor.median()
        mad = self._compute_mad(flat_tensor)
        if mad <= self.eps:
            return self._null_field_signature()
        robust_variance_proxy = mad.pow(2)

        padded_size = 2 * n
        fft_val = torch.fft.fft(median_centered, n=padded_size)
        power_spectrum_full = fft_val * fft_val.conj()
        autocorr_full = torch.fft.ifft(power_spectrum_full).real
        autocorr_full /= robust_variance_proxy * n

        correlation_length = min(
            n // 2, int(torch.sqrt(torch.tensor(n, dtype=torch.float32)).item())
        )
        autocorrelations = autocorr_full[1 : correlation_length + 1].abs()
        if autocorrelations.numel() < 2:
            field_correlation_decay = torch.tensor(0.0)
        else:
            log_lags = torch.log(
                torch.arange(
                    1,
                    autocorrelations.numel() + 1,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            log_corrs = torch.log(autocorrelations.clamp(min=self.eps))
            cov_matrix = torch.cov(torch.stack([log_lags, log_corrs]))
            field_correlation_decay = cov_matrix[0, 1] / (cov_matrix[0, 0] + self.eps)

        power_spectrum = torch.abs(torch.fft.rfft(median_centered)).pow(2)
        if power_spectrum.numel() < 2:
            spectral_density_slope = torch.tensor(0.0)
        else:
            log_freq = torch.log(
                torch.arange(
                    1,
                    power_spectrum.numel() + 1,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            log_power = torch.log(power_spectrum.clamp(min=self.eps))
            cov_matrix = torch.cov(torch.stack([log_freq, log_power]))
            spectral_density_slope = cov_matrix[0, 1] / (cov_matrix[0, 0] + self.eps)
        return {
            "field_correlation_decay": field_correlation_decay,
            "spectral_density_slope": spectral_density_slope,
        }

    def _geometric_analysis(
        self,
        tensor: torch.Tensor,
        effective_rank: float,
        characteristic_length: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tensor.dim() < 2:
            return self._null_geometric_signature()
        matrix = tensor if tensor.dim() == 2 else tensor.reshape(tensor.size(0), -1)
        min_dim = max(3, int(characteristic_length.item()))
        if matrix.size(0) < min_dim or matrix.size(1) < min_dim:
            riemannian_curvature = torch.tensor(0.0)
        else:
            grad_y, grad_x = torch.gradient(matrix)
            g_xx = (grad_x * grad_x).median()
            g_yy = (grad_y * grad_y).median()
            g_xy = (grad_x * grad_y).median()
            det_g = g_xx * g_yy - g_xy.pow(2)
            riemannian_curvature = torch.log(det_g.clamp(min=self.eps))

        flat_tensor, n = tensor.flatten(), tensor.numel()
        log_n = torch.log(torch.tensor(n, dtype=torch.float32))
        sample_size = min(
            n, int(max(effective_rank, 1) * log_n.item() * characteristic_length.item())
        )
        sample_size = max(100, sample_size)

        if n > sample_size:
            sampled_tensor = flat_tensor[
                torch.randperm(n, device=self.device)[:sample_size]
            ]
        else:
            sampled_tensor = flat_tensor

        try:
            gram_matrix = torch.outer(sampled_tensor, sampled_tensor)
            gram_eigenvals = torch.linalg.eigvalsh(gram_matrix).real
            positive_eigenvals = gram_eigenvals[gram_eigenvals > self.eps]
            if positive_eigenvals.numel() > 1:
                information_curvature = -torch.log(positive_eigenvals).var()
            else:
                information_curvature = torch.tensor(0.0)
        except torch.linalg.LinAlgError:
            information_curvature = torch.tensor(0.0)
        return {
            "riemannian_curvature": riemannian_curvature,
            "information_curvature": information_curvature,
        }

    # Helper methods for null signatures
    def _null_signature(self) -> dict[str, float]:
        null_sigs = {
            **self._null_magnitude_signature(),
            **self._null_spectral_signature(),
            **self._null_temporal_signature(),
            **self._null_concentration_signature(),
            **self._null_field_signature(),
            **self._null_geometric_signature(),
        }
        return {k: v.item() for k, v in null_sigs.items()}

    def _null_magnitude_signature(self) -> dict[str, torch.Tensor]:
        keys = ["l2_norm", "abs_median", "abs_max", "dynamic_range"]
        return {k: torch.tensor(0.0, device=self.device) for k in keys}

    def _null_spectral_signature(self, rank=0.0) -> dict[str, torch.Tensor]:
        return {
            "effective_rank": torch.tensor(rank),
            "spectral_entropy": torch.tensor(0.0),
            "scaling_exponent": torch.tensor(0.0),
        }

    def _null_temporal_signature(self) -> dict[str, torch.Tensor]:
        return {
            "hurst_exponent": torch.tensor(0.5),
            "permutation_entropy": torch.tensor(0.0),
        }

    def _null_concentration_signature(self) -> dict[str, torch.Tensor]:
        keys = ["gini_coefficient", "lorenz_asymmetry"]
        return {k: torch.tensor(0.0, device=self.device) for k in keys}

    def _null_field_signature(self) -> dict[str, torch.Tensor]:
        keys = ["field_correlation_decay", "spectral_density_slope"]
        return {k: torch.tensor(0.0, device=self.device) for k in keys}

    def _null_geometric_signature(self) -> dict[str, torch.Tensor]:
        return {
            "riemannian_curvature": torch.tensor(0.0),
            "information_curvature": torch.tensor(0.0),
        }

    def _compute_mad(self, tensor: torch.Tensor) -> torch.Tensor:
        median = torch.median(tensor)
        abs_dev = torch.abs(tensor - median)
        return torch.median(abs_dev)


@dataclass
class OlmConfig:
    """Static configuration for OLM optimization - immutable during run."""

    # Paths, models, and datasets
    models_dir: str
    output_dir: str
    dataset_dir: str
    model_pool: list[str]
    dataset_config: dict[str, dict[str, any]]  # dataset name -> {mode, think}
    base_model_name: Optional[str]  # ignored if base_select is True
    # Strategy
    max_models: Optional[int]
    base_select: bool
    boundary_select: bool
    load_merge_report: bool
    samples: int
    direction: str
    improve_all: Optional[str]
    # Layer optimization flags
    layer_swap: bool
    markov: int
    # Tensor optimization flags
    tensor_swap: bool
    skip_norm: bool
    es_svd: bool
    es_norm: bool
    es_svd_generations: int
    es_norm_generations: int
    es_svd_sigma: float
    es_norm_sigma: float

    @classmethod
    def from_config_dict(cls, config: dict) -> "OlmConfig":
        """Create OlmConfig from YAML config dictionary."""
        return cls(
            # Required paths, models, and datasets
            models_dir=config["models_dir"],
            output_dir=config["output_dir"],
            dataset_dir=config["dataset_dir"],
            model_pool=config["models"],
            dataset_config=config["dataset"],
            base_model_name=config.get("base_model_name", None),
            # Overall strategy
            max_models=config.get("max_models", None),
            base_select=config.get("base_select", False),
            boundary_select=config.get("boundary_select", False),
            load_merge_report=config.get("load_merge_report", False),
            samples=config.get("samples", 200),
            direction=config.get("direction", "forward"),
            improve_all=config.get("improve_all", None),
            # Layer optimization
            layer_swap=config.get("layer_swap", False),
            markov=config.get("markov", 0),
            # Tensor optimization
            tensor_swap=config.get("tensor_swap", False),
            skip_norm=config.get("skip_norm", False),
            es_svd=config.get("es_svd", False),
            es_norm=config.get("es_norm", False),
            es_svd_generations=config.get("es_svd_generations", 1),
            es_norm_generations=config.get("es_norm_generations", 3),
            es_svd_sigma=config.get("es_svd_sigma", 0.1),
            es_norm_sigma=config.get("es_norm_sigma", 0.005),
        )

    def __post_init__(self):
        if self.max_models is None:
            self.max_models = len(self.model_pool)


@dataclass
class OlmRuntime:
    """Runtime state for OLM optimization - updated during execution."""

    working_model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    active_model_pool: list[str] = field(default_factory=list)
    datasets: dict[str, dict[str, any]] = field(default_factory=dict)
    merge_report: dict[str, any] = field(default_factory=dict)
    evaluation_cache: Optional[dict[str, any]] = None
    base_metrics: Optional[dict[str, float]] = None
    selected_base_model: Optional[str] = None
    selected_boundary_model: Optional[str] = None

    def update_base_model(self, model_name: str, metrics: dict[str, float]):
        """Update base model selection and metrics."""
        self.selected_base_model = model_name
        self.base_metrics = metrics

    def update_merge_report(self, report: dict[str, any]):
        """Update merge report with new optimization results."""
        self.merge_report.update(report)

    def set_working_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Set the active working model and tokenizer."""
        self.working_model = model
        self.tokenizer = tokenizer

    def load_datasets(self, config: OlmConfig):
        """Load and process datasets from config."""
        datasets = {}
        for dataset_name, dataset_config in config.dataset_config.items():
            with open(
                Path(config.dataset_dir) / dataset_name, "r", encoding="utf-8"
            ) as f:
                datasets[dataset_name] = {
                    "data": json.load(f),
                    "mode": dataset_config["mode"],
                    "think": dataset_config["think"],
                }
        self.datasets = datasets

    def create_evaluation_cache(self, samples: int):
        """Create evaluation cache from current tokenizer and datasets."""
        if self.tokenizer is None or not self.datasets:
            return

        try:
            cache = {}
            reasoning_mask = "<think>\\n\\n</think>\\n\\nCorrect answer:"
            for dataset_name, dataset in self.datasets.items():
                cache[dataset_name] = {}
                data_samples = dataset["data"][:samples]
                think = dataset.get("think", False)

                for conv_idx, conv in enumerate(data_samples):
                    context = get_context(conv)
                    if not think:
                        context += reasoning_mask
                    context_ids = self.tokenizer(context, return_tensors="pt")
                    expected_response = conv["conversation"][-1]["value"]
                    response_ids = self.tokenizer(
                        " " + expected_response, return_tensors="pt"
                    )
                    full_ids = torch.cat(
                        [context_ids.input_ids, response_ids.input_ids], dim=1
                    )
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
            self.evaluation_cache = cache
        except Exception as e:
            logger.info(f"Error creating tokenized dataset cache: {e}")
            self.evaluation_cache = None

    def evaluate_model_on_dataset(
        self,
        model: AutoModelForCausalLM,
        dataset_name: str,
        mode: str,
        samples: int = 200,
    ) -> float:
        """Evaluate a model on a specific dataset using current tokenizer and cache."""
        dataset = self.datasets[dataset_name]["data"]
        cache = None
        if self.evaluation_cache and dataset_name in self.evaluation_cache:
            cache = self.evaluation_cache[dataset_name]

        total_metric = 0.0
        max_conversations = min(len(dataset), samples)
        dataset = dataset[:max_conversations]
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model.device.type != device:
            model.to(device, dtype=torch.bfloat16)

        for idx, conversation in enumerate(dataset):
            conv_cache = cache.get(idx) if cache else None

            if mode == "diversity":
                metric = compute_response_diversity(
                    model, self.tokenizer, conversation, 4096, conv_cache
                )
            elif mode == "bigram_loss":
                metric = compute_bigram_loss(
                    model, self.tokenizer, conversation, conv_cache
                )
            elif mode == "exact_match":
                metric = compute_exact_match(
                    model, self.tokenizer, conversation, conv_cache
                )
            total_metric += metric

        model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return total_metric / len(dataset)

    def process_model(
        self,
        model_name: str,
        layer_metrics: dict,
        layer_idx: int,
        config: OlmConfig,
    ) -> dict:
        """Process a candidate model for optimization using runtime state."""
        improve_all = config.improve_all
        samples = config.samples

        if improve_all:
            prev_metrics = None
            last_idx = self.get_last_layer_idx()
            if last_idx is not None:
                best_model = self.merge_report["layers"][str(last_idx)]["best_model"]
                prev_metrics = {
                    dataset: scores[best_model]
                    for dataset, scores in self.merge_report["layers"][str(last_idx)][
                        "metrics"
                    ].items()
                }

        for dataset_name, dataset in self.datasets.items():
            if dataset_name not in layer_metrics:
                layer_metrics[dataset_name] = {}

            try:
                dataset_cache = None
                if self.evaluation_cache and dataset_name in self.evaluation_cache:
                    dataset_cache = self.evaluation_cache[dataset_name]

                metric = evaluate_model_on_dataset(
                    self.working_model,
                    self.tokenizer,
                    dataset["data"],
                    dataset["mode"],
                    samples,
                    cache=dataset_cache,
                )
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                metric = float("inf")

            layer_metrics[dataset_name][model_name] = metric
            logger.info(f"{model_name}: {dataset_name}: {metric}")

            should_skip = False
            if dataset["mode"] == improve_all or improve_all == "all":
                if metric > self.base_metrics[dataset_name] or (
                    prev_metrics
                    and dataset_name in prev_metrics
                    and metric > prev_metrics[dataset_name]
                ):
                    logger.info(
                        f"Layer degradation on {dataset_name}, skipping {model_name}"
                    )
                    should_skip = True

            elif improve_all == "base":
                if metric > self.base_metrics[dataset_name]:
                    should_skip = True

            if should_skip:
                logger.info(f"DEGRADATION DETECTED for {model_name} on {dataset_name}")
                for remaining in self.datasets:
                    if remaining not in layer_metrics:
                        layer_metrics[remaining] = {}
                    layer_metrics[remaining][model_name] = float("inf")
                return layer_metrics

        return layer_metrics

    def save_layer_state(
        self,
        results: dict,
        layer_idx: int,
        output_dir: str,
        best_model: str,
    ) -> None:
        """Save model state and merge report for current layer."""
        os.makedirs(output_dir, exist_ok=True)
        self.working_model.save_pretrained(output_dir)
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
        self.merge_report = report

    def save_tensor_state(
        self,
        tensor_metrics: dict,
        tensor_name: str,
        output_dir: str,
        best_model: str,
    ) -> None:
        """Save model state and merge report for tensor optimization."""
        os.makedirs(output_dir, exist_ok=True)
        self.working_model.save_pretrained(output_dir)
        report_path = os.path.join(output_dir, "merge_report.json")
        report = json.load(open(report_path)) if os.path.exists(report_path) else {}
        if "tensors" not in report:
            report["tensors"] = {}
        report["tensors"][tensor_name] = {
            "metrics": tensor_metrics,
            "best_model": best_model,
        }
        json.dump(report, open(report_path, "w"), indent=4)
        self.merge_report = report

    def apply_boundary_components(self, source_model: AutoModelForCausalLM) -> None:
        """Apply boundary components to the working model."""
        self.working_model.model.embed_tokens.load_state_dict(
            source_model.model.embed_tokens.state_dict()
        )
        self.working_model.model.norm.load_state_dict(
            source_model.model.norm.state_dict()
        )
        self.working_model.lm_head.load_state_dict(source_model.lm_head.state_dict())

    def select_component(
        self,
        component_type: str,
        models_dir: str,
        model_pool: list[str],
    ) -> dict:
        """Select best component (base or boundary) using runtime state."""
        logger.info(f"Evaluating {component_type} model candidate...")

        selected_datasets = {name: dataset for name, dataset in self.datasets.items()}
        if not selected_datasets:
            selected_datasets = self.datasets

        if component_type == "base":
            component_metrics = {dataset_name: {} for dataset_name in selected_datasets}
            for model_name in model_pool:
                logger.info(f"Evaluating model: {model_name}")
                try:
                    model, tokenizer = load_model(
                        Path(models_dir) / model_name.replace("/", "_"), "cpu"
                    )
                    for dataset_name, dataset in selected_datasets.items():
                        dataset_cache = (
                            self.evaluation_cache.get(dataset_name)
                            if self.evaluation_cache
                            else None
                        )
                        metric = evaluate_model_on_dataset(
                            model,
                            tokenizer,
                            dataset["data"],
                            dataset["mode"],
                            cache=dataset_cache,
                        )
                        component_metrics[dataset_name][model_name] = metric
                        logger.info(f"{model_name}: {dataset_name}: {metric}")
                    del model
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue

        elif component_type == "boundary":
            component_metrics = {dataset_name: {} for dataset_name in selected_datasets}
            original_states = {}

            if self.working_model:
                original_states = {
                    "embed_tokens": self.working_model.model.embed_tokens.state_dict(),
                    "norm": self.working_model.model.norm.state_dict(),
                    "lm_head": self.working_model.lm_head.state_dict(),
                }

            for model_name in model_pool:
                try:
                    if self.working_model:
                        source, _ = load_model(
                            f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                        )
                        self.apply_boundary_components(source)
                        test_model, test_tokenizer = self.working_model, self.tokenizer
                        del source
                    else:
                        test_model, test_tokenizer = load_model(
                            f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                        )

                    for dataset_name, dataset in selected_datasets.items():
                        dataset_cache = (
                            self.evaluation_cache.get(dataset_name)
                            if self.evaluation_cache
                            else None
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

                    if self.working_model and original_states:
                        for component_name, state in original_states.items():
                            getattr(
                                self.working_model.model
                                if component_name != "lm_head"
                                else self.working_model,
                                component_name,
                            ).load_state_dict(state)

                    if component_type == "base" and test_model != self.working_model:
                        del test_model

                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue

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
                    next(i for i, (m, _) in enumerate(sorted_models) if m == model_name)
                    + 1
                )
                ranks.append(rank)
            if ranks:
                model_ranks[model_name] = ranks

        if not model_ranks:
            raise RuntimeError(f"No valid models found for {component_type} selection")

        ordered_models = rank_models(model_ranks)
        ordered_metrics = {
            model_name: {
                ds: component_metrics[ds][model_name]
                for ds in component_metrics
                if model_name in component_metrics[ds]
            }
            for model_name in ordered_models
        }
        return ordered_metrics

    def layer_wise_optimization(
        self,
        model_pool: list[str],
        config: OlmConfig,
    ) -> AutoModelForCausalLM:
        """Perform layer-wise optimization using runtime state."""
        num_layers = self.working_model.config.num_hidden_layers
        direction = config.direction
        markov_order = config.markov
        models_dir = config.models_dir
        output_dir = config.output_dir

        processed_layers = set()
        if "layers" in self.merge_report:
            processed_layers = set(int(k) for k in self.merge_report["layers"].keys())
            if processed_layers:
                logger.info(
                    f"Found {len(processed_layers)} already processed layers in the merge report: {sorted(list(processed_layers))}"
                )

        for layer_idx in layer_sequence(num_layers, direction):
            if layer_idx in processed_layers:
                logger.info(f"Skipping layer {layer_idx}, already processed")
                continue

            logger.info(f"\nOptimizing layer {layer_idx} (markov={markov_order})")

            if markov_order == 0:
                layer_metrics = {dataset: {} for dataset in self.datasets}

                for model_name in model_pool:
                    try:
                        source, _ = load_model(
                            f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                        )
                        replace_layer(
                            self.working_model, get_layer(source, layer_idx), layer_idx
                        )
                        del source
                        torch.cuda.empty_cache()

                        layer_metrics = self.process_model(
                            model_name=model_name,
                            layer_metrics=layer_metrics,
                            layer_idx=layer_idx,
                            config=config,
                        )
                    except Exception as e:
                        logger.error(f"Layer optimization failed for {model_name}: {e}")
                        continue

                model_ranks = compute_layer_ranks(layer_metrics)
                if not model_ranks:
                    logger.error(f"No valid layer candidates for layer {layer_idx}")
                    continue

                best_model = rank_models(model_ranks)[0]

                try:
                    best_source, _ = load_model(
                        f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
                    )
                    self.working_model.model.layers[layer_idx].load_state_dict(
                        get_layer(best_source, layer_idx).state_dict()
                    )
                    self.save_layer_state(
                        layer_metrics, layer_idx, output_dir, best_model
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

                    self.save_model(output_dir)
                except Exception as e:
                    logger.error(f"Failed to apply best layer from {best_model}: {e}")
                    continue

            else:
                if direction == "forward":
                    window_start = max(0, layer_idx - markov_order)
                    window_end = layer_idx + 1
                else:
                    window_start = layer_idx
                    window_end = min(num_layers, layer_idx + markov_order + 1)

                window_range = range(window_start, window_end)
                window_size = len(window_range)

                original_states = {}
                for idx in window_range:
                    original_states[idx] = self.working_model.model.layers[
                        idx
                    ].state_dict()

                window_metrics = {dataset: {} for dataset in self.datasets}
                combo_to_models = {}

                from itertools import product

                for combo_idx, model_combo in enumerate(
                    product(model_pool, repeat=window_size)
                ):
                    combo_name = (
                        f"window_{window_start}-{window_end - 1}_combo_{combo_idx}"
                    )
                    combo_to_models[combo_name] = model_combo

                    try:
                        for i, model_name in enumerate(model_combo):
                            source, _ = load_model(
                                f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                            )
                            layer_in_window = window_start + i
                            replace_layer(
                                self.working_model,
                                get_layer(source, layer_in_window),
                                layer_in_window,
                            )
                            del source
                            torch.cuda.empty_cache()

                        window_metrics = self.process_model(
                            model_name=combo_name,
                            layer_metrics=window_metrics,
                            layer_idx=layer_idx,
                            config=config,
                        )

                    except Exception as e:
                        logger.error(
                            f"Window optimization failed for combo {combo_name}: {e}"
                        )
                        for dataset in self.datasets:
                            if dataset not in window_metrics:
                                window_metrics[dataset] = {}
                            window_metrics[dataset][combo_name] = float("inf")

                    finally:
                        for idx in window_range:
                            self.working_model.model.layers[idx].load_state_dict(
                                original_states[idx]
                            )

                model_ranks = compute_layer_ranks(window_metrics)
                if not model_ranks:
                    logger.error(f"No valid window candidates for layer {layer_idx}")
                    continue

                best_combo_name = rank_models(model_ranks)[0]
                best_combo = combo_to_models[best_combo_name]

                logger.info(f"Window evaluation complete for layer {layer_idx}:")
                logger.info(f"  Window range: {list(window_range)}")
                logger.info(f"  Combinations evaluated: {len(combo_to_models)}")

                ranked_combos = rank_models(model_ranks)[:3]
                for i, combo_name in enumerate(ranked_combos):
                    combo_models = combo_to_models[combo_name]
                    combo_metrics = {
                        ds: window_metrics[ds].get(combo_name, float("inf"))
                        for ds in self.datasets
                    }
                    avg_rank = sum(model_ranks[combo_name]) / len(
                        model_ranks[combo_name]
                    )
                    logger.info(
                        f"  Rank {i + 1}: {combo_models} - metrics: {combo_metrics} - avg_rank: {avg_rank:.2f}"
                    )

                best_metrics = {
                    ds: window_metrics[ds][best_combo_name] for ds in self.datasets
                }

                verification_passed = True
                for combo_name in combo_to_models:
                    if combo_name == best_combo_name:
                        continue
                    if any(combo_name in window_metrics[ds] for ds in self.datasets):
                        combo_metrics = {
                            ds: window_metrics[ds].get(combo_name, float("inf"))
                            for ds in self.datasets
                        }
                        if all(
                            combo_metrics[ds] < best_metrics[ds]
                            for ds in self.datasets
                            if combo_metrics[ds] != float("inf")
                        ):
                            logger.warning(
                                f"VERIFICATION FAILED: {combo_to_models[combo_name]} has better metrics than selected best: {combo_metrics} vs {best_metrics}"
                            )
                            verification_passed = False

                if verification_passed:
                    logger.info(
                        "  ✓ Verification passed: Best combo selection is correct"
                    )
                else:
                    logger.error(
                        "  ✗ Verification failed: Best combo selection may be incorrect"
                    )
                    logger.error(
                        f"  Selected: {best_combo} with metrics: {best_metrics}"
                    )

                if len(model_ranks) > 1:
                    logger.info(
                        f"  Ranking verification: {len(model_ranks)} combinations ranked"
                    )
                    for combo_name in ranked_combos[:3]:
                        rank_list = model_ranks[combo_name]
                        logger.info(
                            f"    {combo_to_models[combo_name]}: ranks {rank_list}, sum={sum(rank_list)}"
                        )

                try:
                    for i, model_name in enumerate(best_combo):
                        source, _ = load_model(
                            f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                        )
                        layer_in_window = window_start + i
                        replace_layer(
                            self.working_model,
                            get_layer(source, layer_in_window),
                            layer_in_window,
                        )
                        del source

                        if "layers" not in self.merge_report:
                            self.merge_report["layers"] = {}
                        self.merge_report["layers"][str(layer_in_window)] = {
                            "best_model": model_name,
                            "metrics": {
                                ds: window_metrics[ds][best_combo_name]
                                for ds in self.datasets
                            },
                        }

                    self.save_model(output_dir)

                    avg_rank = (
                        sum(model_ranks[best_combo_name])
                        / len(model_ranks[best_combo_name])
                        if isinstance(model_ranks[best_combo_name], list)
                        else model_ranks[best_combo_name]
                    )

                    logger.info(
                        f"Applied markov window {best_combo} for layers {list(window_range)} (avg rank: {avg_rank})"
                    )

                    for idx in window_range:
                        processed_layers.add(idx)

                except Exception as e:
                    logger.error(f"Failed to apply best window combination: {e}")
                    for idx in window_range:
                        self.working_model.model.layers[idx].load_state_dict(
                            original_states[idx]
                        )

        return self.working_model

    def tensor_wise_optimization(
        self,
        model_pool: list[str],
        config: OlmConfig,
    ) -> AutoModelForCausalLM:
        """Perform tensor-wise optimization using runtime state."""
        parameter_units = _get_target_parameter_units(self.working_model)
        unit_names = list(parameter_units.keys())
        models_dir = config.models_dir
        output_dir = config.output_dir
        if config.direction == "backward":
            unit_names.reverse()

        skip_norm = config.skip_norm
        working_params_dict = dict(self.working_model.named_parameters())
        processed_tensors = set(self.merge_report.get("tensors", {}).keys())

        for unit_name in tqdm(unit_names, desc="Continuous Optimization"):
            if skip_norm and any("norm" in part for part in unit_name.split(".")):
                continue

            unit_param_names = parameter_units[unit_name]
            weight_tensor_name = next(
                (p for p in unit_param_names if "weight" in p), None
            )

            if not weight_tensor_name or weight_tensor_name in processed_tensors:
                continue

            unit_metrics = {dataset_name: {} for dataset_name in self.datasets}
            orig_states = {
                name: param.data.clone()
                for name, param in working_params_dict.items()
                if name in unit_param_names
            }

            best_model = self.selected_base_model
            if config.tensor_swap:
                for model_name in model_pool:
                    try:
                        candidate, _ = load_model(
                            f"{models_dir}/{model_name.replace('/', '_')}", "cpu"
                        )
                        candidate_params = dict(candidate.named_parameters())

                        for param_name in unit_param_names:
                            if (
                                param_name in working_params_dict
                                and param_name in candidate_params
                            ):
                                working_params_dict[param_name].data.copy_(
                                    candidate_params[param_name].data
                                )

                        unit_metrics = self.process_model(
                            model_name=model_name,
                            layer_metrics=unit_metrics,
                            layer_idx=weight_tensor_name,
                            config=config,
                        )

                    except Exception as e:
                        for ds in unit_metrics:
                            unit_metrics[ds][model_name] = float("inf")
                        logger.warning(
                            f"Model {model_name} failed to improve metrics: {e}"
                        )
                    finally:
                        for name, data in orig_states.items():
                            if name in working_params_dict:
                                working_params_dict[name].data.copy_(data)
                        if "candidate" in locals():
                            del candidate
                            torch.cuda.empty_cache()

                model_ranks = compute_layer_ranks(unit_metrics)
                if model_ranks:
                    best_model = rank_models(model_ranks)[0]

            best_source, _ = load_model(
                f"{models_dir}/{best_model.replace('/', '_')}", "cpu"
            )
            best_params = dict(best_source.named_parameters())

            for param_name in unit_param_names:
                if param_name in working_params_dict and param_name in best_params:
                    working_params_dict[param_name].data.copy_(
                        best_params[param_name].data
                    )

            best_metrics = self.base_metrics.copy()
            for dataset_name, dataset in best_metrics.items():
                score = unit_metrics[dataset_name][best_model]
                if score == float("inf"):
                    score = self.base_metrics[dataset_name]
                best_metrics[dataset_name] = score
            updated_metrics = best_metrics.copy()

            if config.es_svd:
                updated_metrics = self.fine_tune_tensor_svd_es(
                    weight_tensor_name,
                    config,
                    best_metrics,
                )

            if config.es_norm:
                updated_metrics = self.fine_tune_tensor_norm_es(
                    weight_tensor_name,
                    config,
                    updated_metrics,
                )

            for dataset_name in self.datasets:
                unit_metrics[dataset_name][best_model] = updated_metrics[dataset_name]

            self.save_tensor_state(
                unit_metrics, weight_tensor_name, output_dir, best_model
            )
            del best_source
            torch.cuda.empty_cache()
            self.save_model(output_dir)

        return self.working_model

    @torch.no_grad()
    def fine_tune_tensor_svd_es(
        self,
        tensor_name: str,
        config: OlmConfig,
        baseline_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Fine-tune tensor using SVD evolutionary strategy."""
        params = dict(self.working_model.named_parameters())
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

        samples = config.samples
        generations = config.es_svd_generations
        initial_sigma = config.es_svd_sigma
        noise_clamp_val = 3.0
        epsilon = 1e-10

        original_tensor_dtype = original_model_tensor.dtype
        original_tensor_shape = original_model_tensor.shape
        compute_device = original_model_tensor.device
        current_best_flat = (
            original_model_tensor.data.to(compute_device, dtype=torch.float32)
            .flatten()
            .clone()
        )
        current_best_metrics = baseline_metrics.copy()

        U_svd, S_svd, V_svd = calculate_full_svd(
            current_best_flat.reshape(original_tensor_shape), epsilon
        )
        if U_svd is None or S_svd is None or V_svd is None or S_svd.numel() == 0:
            logger.warning(
                f"SVD failed for tensor {tensor_name} (shape {original_tensor_shape}). Skipping."
            )
            return baseline_metrics

        dim_s = S_svd.numel()
        target_s_indices = torch.arange(dim_s, device=compute_device)
        sigma_s = initial_sigma * max(S_svd.std().item(), epsilon)

        _lambda = max(4, int(4 + 3 * math.log(dim_s)))

        tensor_stats = original_model_tensor.data.float()
        logger.info(
            f"Tensor {tensor_name}, Baseline Metrics {baseline_metrics}, "
            f"Initial stats: mean={tensor_stats.mean().item()}, std={tensor_stats.std().item()}, "
            f"min={tensor_stats.min().item()}, max={tensor_stats.max().item()}"
        )

        logger.info(
            f"Optimizing {tensor_name} (SVD dim: {dim_s}) with pop: {_lambda}, for {generations} generations. Init sigma: {sigma_s}"
        )

        delta_s_batch = torch.empty(
            (_lambda, dim_s), device=compute_device, dtype=torch.float32
        )
        avg_rel_changes = torch.full(
            (_lambda,), float("inf"), device=compute_device, dtype=torch.float32
        )

        offspring_metrics_list = [{} for _ in range(_lambda)]

        for gen in range(generations):
            genstring = f"  Generation {gen + 1}/{generations}"
            logger.info(
                f"\n--- {genstring}, Tensor: {tensor_name}, Sigma: {sigma_s} ---"
            )

            delta_s_batch.normal_(0, 1)
            delta_s_batch.mul_(sigma_s)

            avg_rel_changes.fill_(float("inf"))
            for i in range(_lambda):
                current_eval = f"{genstring} Offspring {i + 1}/{_lambda}:"

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

                original_model_tensor.data.copy_(
                    (current_best_flat * noise_mult)
                    .reshape(original_tensor_shape)
                    .to(original_tensor_dtype)
                )
                logger.info(
                    f"{current_eval}Noise: mean={noise_mult.mean().item()}, std={noise_mult.std().item()}, "
                    f"min={noise_mult.min().item()}, max={noise_mult.max().item()}"
                )

                metrics, valid = {}, True
                for ds_name, ds_conf in self.datasets.items():
                    try:
                        eval_samples = min(samples, len(ds_conf["data"]))
                        if eval_samples > 0:
                            ds_cache = (
                                self.evaluation_cache.get(ds_name)
                                if self.evaluation_cache
                                else None
                            )
                            metrics[ds_name] = evaluate_model_on_dataset(
                                self.working_model,
                                self.tokenizer,
                                ds_conf["data"][:eval_samples],
                                ds_conf["mode"],
                                eval_samples,
                                cache=ds_cache,
                            )
                    except Exception as e:
                        valid = False
                        logger.warning(f"{current_eval} fail on {ds_name}: {e}")
                        return baseline_metrics

                if valid and metrics:
                    offspring_metrics_list[i] = metrics.copy()
                    logger.info(f"{current_eval} Metrics: {metrics}")

            valid_offspring = [i for i in range(_lambda) if offspring_metrics_list[i]]
            if not valid_offspring:
                logger.warning(f"{genstring} No valid offspring.")
                original_model_tensor.data.copy_(
                    current_best_flat.reshape(original_tensor_shape).to(
                        original_tensor_dtype
                    )
                )
                continue

            offspring_ranks = {}
            for ds_name in self.datasets.keys():
                scores = [
                    (i, offspring_metrics_list[i].get(ds_name, float("inf")))
                    for i in valid_offspring
                ]
                scores.sort(key=lambda x: x[1])
                for rank, (idx, _) in enumerate(scores, 1):
                    if idx not in offspring_ranks:
                        offspring_ranks[idx] = []
                    offspring_ranks[idx].append(rank)

            improved_indexes = [
                i
                for i in valid_offspring
                if any(
                    offspring_metrics_list[i][ds] < current_best_metrics[ds]
                    for ds in self.datasets
                )
            ]

            if not improved_indexes:
                logger.info(
                    f"{genstring} No improved offspring found. Keeping original."
                )
            else:
                improved_ranks = {str(i): offspring_ranks[i] for i in improved_indexes}

                best_idx_str = rank_models(improved_ranks)[0]
                best_idx = int(best_idx_str)

                current_best_metrics = offspring_metrics_list[best_idx].copy()
                logger.info(
                    f"{genstring} Selected offspring (idx={best_idx}) via Copeland. "
                    f"Metrics: {current_best_metrics}"
                )

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

                if gen < generations - 1:
                    U_new, S_new, V_new = calculate_full_svd(
                        current_best_flat.reshape(original_tensor_shape), epsilon
                    )
                    U_svd, S_svd, V_svd = U_new, S_new, V_new
                    sigma_s = initial_sigma * 0.8 * max(S_svd.std().item(), epsilon)
                    logger.info(f"{genstring} SVD updated. New sigma: {sigma_s}")

            original_model_tensor.data.copy_(
                current_best_flat.reshape(original_tensor_shape).to(
                    original_tensor_dtype
                )
            )

        logger.info(f"SVD ES for {tensor_name} done. Generations: {generations}")
        logger.info(f"Final metrics for {tensor_name}: {current_best_metrics}")
        return current_best_metrics

    @torch.no_grad()
    def fine_tune_tensor_norm_es(
        self,
        tensor_name: str,
        config: OlmConfig,
        baseline_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Fine-tune tensor using norm-based evolutionary strategy."""
        params = dict(self.working_model.named_parameters())
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

        samples = config.samples
        generations = config.es_norm_generations
        initial_sigma = config.es_norm_sigma
        epsilon = 1e-10

        original_tensor_dtype = original_model_tensor.dtype
        compute_device = original_model_tensor.device
        current_best_metrics = baseline_metrics.copy()

        magnitude_norms = torch.linalg.norm(
            original_model_tensor.data, dim=1, keepdim=True
        )
        directions = original_model_tensor.data / (magnitude_norms + epsilon)
        current_best_magnitudes = magnitude_norms.squeeze(1).clone()

        dim_s = current_best_magnitudes.numel()
        sigma_s = initial_sigma

        _lambda = max(4, int(4 + 3 * math.log(dim_s)))

        tensor_stats = original_model_tensor.data.float()
        logger.info(
            f"Tensor {tensor_name}, Baseline Metrics {baseline_metrics}, "
            f"Initial stats: mean={tensor_stats.mean().item()}, std={tensor_stats.std().item()}, "
            f"min={tensor_stats.min().item()}, max={tensor_stats.max().item()}"
        )

        logger.info(
            f"Optimizing {tensor_name} (Norm dim: {dim_s}) with pop: {_lambda}, for {generations} generations. Init sigma: {sigma_s}"
        )

        perturbations_batch = torch.empty(
            (_lambda, dim_s), device=compute_device, dtype=torch.float32
        )
        offspring_metrics_list = [{} for _ in range(_lambda)]

        for gen in range(generations):
            genstring = f"  Generation {gen + 1}/{generations}"
            logger.info(
                f"\n--- {genstring}, Tensor: {tensor_name}, Sigma: {sigma_s} ---"
            )

            log_magnitudes = torch.log(current_best_magnitudes + epsilon)
            perturbations_batch.normal_(0, 1)
            perturbations_batch.mul_(sigma_s)

            for i in range(_lambda):
                current_eval = f"{genstring} Offspring {i + 1}/{_lambda}:"

                candidate_log_mags = log_magnitudes + perturbations_batch[i]
                candidate_magnitudes = torch.exp(candidate_log_mags)

                reconstructed_tensor = directions * candidate_magnitudes.unsqueeze(1)
                original_model_tensor.data.copy_(
                    reconstructed_tensor.to(original_tensor_dtype)
                )

                metrics, valid = {}, True
                for ds_name, ds_conf in self.datasets.items():
                    try:
                        eval_samples = min(samples, len(ds_conf["data"]))
                        if eval_samples > 0:
                            ds_cache = (
                                self.evaluation_cache.get(ds_name)
                                if self.evaluation_cache
                                else None
                            )
                            metrics[ds_name] = evaluate_model_on_dataset(
                                self.working_model,
                                self.tokenizer,
                                ds_conf["data"][:eval_samples],
                                ds_conf["mode"],
                                eval_samples,
                                cache=ds_cache,
                            )
                    except Exception as e:
                        valid = False
                        logger.warning(f"{current_eval} fail on {ds_name}: {e}")
                        restored_tensor = (
                            directions * current_best_magnitudes.unsqueeze(1)
                        )
                        original_model_tensor.data.copy_(
                            restored_tensor.to(original_tensor_dtype)
                        )
                        return baseline_metrics

                if valid and metrics:
                    offspring_metrics_list[i] = metrics.copy()
                    logger.info(f"{current_eval} Metrics: {metrics}")

            valid_offspring = [i for i in range(_lambda) if offspring_metrics_list[i]]
            if not valid_offspring:
                logger.warning(f"{genstring} No valid offspring.")
                restored_tensor = directions * current_best_magnitudes.unsqueeze(1)
                original_model_tensor.data.copy_(
                    restored_tensor.to(original_tensor_dtype)
                )
                continue

            offspring_ranks = {}
            for ds_name in self.datasets.keys():
                scores = [
                    (i, offspring_metrics_list[i].get(ds_name, float("inf")))
                    for i in valid_offspring
                ]
                scores.sort(key=lambda x: x[1])
                for rank, (idx, _) in enumerate(scores, 1):
                    if idx not in offspring_ranks:
                        offspring_ranks[idx] = []
                    offspring_ranks[idx].append(rank)

            improved_indexes = [
                i
                for i in valid_offspring
                if any(
                    offspring_metrics_list[i][ds] < current_best_metrics[ds]
                    for ds in self.datasets
                )
            ]

            if not improved_indexes:
                logger.info(
                    f"{genstring} No improved offspring found. Keeping original."
                )
            else:
                improved_ranks = {str(i): offspring_ranks[i] for i in improved_indexes}
                best_idx_str = rank_models(improved_ranks)[0]
                best_idx = int(best_idx_str)

                winning_log_mags = log_magnitudes + perturbations_batch[best_idx]
                current_best_magnitudes = torch.exp(winning_log_mags)
                current_best_metrics = offspring_metrics_list[best_idx].copy()
                logger.info(
                    f"{genstring} Selected offspring (idx={best_idx}) via Copeland. "
                    f"Metrics: {current_best_metrics}"
                )

                sigma_s *= 0.9

            restored_tensor = directions * current_best_magnitudes.unsqueeze(1)
            original_model_tensor.data.copy_(restored_tensor.to(original_tensor_dtype))

        logger.info(f"Norm ES for {tensor_name} done. Generations: {generations}")
        logger.info(f"Final metrics for {tensor_name}: {current_best_metrics}")
        return current_best_metrics

    def load_from_merge_report(
        self,
        merge_report_path: str,
        models_dir: str,
    ) -> None:
        """Load model and state from existing merge report."""
        with open(merge_report_path) as f:
            merge_report = json.load(f)

        base_model_name = merge_report["base_model"]["name"]
        base_model_path = Path(models_dir) / base_model_name.replace("/", "_")
        logger.info(f"Loading base model {base_model_name}")
        model, tokenizer = load_model(str(base_model_path), "cpu")

        if "boundary_layers" in merge_report and merge_report["boundary_layers"].get(
            "name"
        ):
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
                        layer_source_path = Path(
                            models_dir
                        ) / layer_source_model.replace("/", "_")
                        layer_source, _ = load_model(str(layer_source_path), "cpu")
                        layer_idx = int(layer_idx)
                        replace_layer(
                            model, get_layer(layer_source, layer_idx), layer_idx
                        )
                        del layer_source
                    except Exception as e:
                        logger.error(
                            f"Error applying layer {layer_idx} from {layer_source_model}: {e}"
                        )
                        logger.info("Keeping base model layer")

        if "tensors" in merge_report:
            logger.info("Re-applying optimized tensors from merge report...")
            working_params = dict(model.named_parameters())
            parameter_units = _get_target_parameter_units(model)

            param_to_unit_map = {}
            for unit_name, param_list in parameter_units.items():
                for param_name in param_list:
                    param_to_unit_map[param_name] = unit_name

            for tensor_name, tensor_info in tqdm(
                merge_report["tensors"].items(), desc="Reconstructing Tensors"
            ):
                tensor_source_model = tensor_info.get("best_model")

                if not tensor_source_model:
                    continue

                unit_name = param_to_unit_map.get(tensor_name)
                if not unit_name:
                    logger.warning(
                        f"Could not find parameter unit for {tensor_name}, applying tensor individually."
                    )
                    unit_param_names = [tensor_name]
                else:
                    unit_param_names = parameter_units.get(unit_name, [tensor_name])

                try:
                    tensor_source_path = Path(models_dir) / tensor_source_model.replace(
                        "/", "_"
                    )
                    tensor_source, _ = load_model(str(tensor_source_path), "cpu")
                    source_params = dict(tensor_source.named_parameters())

                    for param_name in unit_param_names:
                        if param_name in working_params and param_name in source_params:
                            working_params[param_name].data.copy_(
                                source_params[param_name].data
                            )

                    del tensor_source
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        f"Error applying unit for tensor {tensor_name} from {tensor_source_model}: {e}"
                    )
                    logger.info(f"Keeping existing tensor for unit of {tensor_name}")

        self.working_model = model
        self.tokenizer = tokenizer
        self.merge_report = merge_report
        self.base_metrics = merge_report["base_model"]["metrics"]
        self.selected_base_model = merge_report["base_model"]["name"]

    def save_model(self, output_dir: str) -> None:
        """Save working model to output directory."""
        logger.info(f"Saving working model to {output_dir}")
        if self.working_model:
            self.working_model.save_pretrained(
                output_dir,
                safe_serialization=True,
                push_to_hub=False,
                exist_ok=True,
            )

        report_path = os.path.join(output_dir, "merge_report.json")
        with open(report_path, "w") as f:
            json.dump(self.merge_report, f, indent=4)

    def get_last_layer_idx(self) -> Optional[int]:
        """Get the index of the last optimized layer from merge report."""
        if not self.merge_report.get("layers"):
            return None
        return max(int(layer_num) for layer_num in self.merge_report["layers"].keys())

    def get_layer_metrics(self, layer_idx: int) -> dict:
        """Get metrics for a specific layer from merge report."""
        return self.merge_report["layers"][str(layer_idx)]["metrics"]


# --- Model and Dataset Handling ---


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

    return final_units


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
def compute_response_diversity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    max_length: int = 4096,
    cache: Optional[dict] = None,
) -> float:
    """Compute diversity of the response."""
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
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
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
    num_unique_words = len(set(words))
    diversity = max_length / max(num_unique_words, 1)

    return diversity


@torch.no_grad()
def compute_bigram_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    cache: Optional[dict] = None,
) -> float:
    """Autoregressive exact match model eval, with teacher forcing
    Lower scores are better: (correct + certain) < (correct + uncertain) < (incorrect = 1.0)"""
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
    shift_probs = torch.softmax(shift_logits, dim=-1)
    shift_labels = response_ids.input_ids[:, :-1]
    shift_correct = torch.gather(shift_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    shift_max, _ = torch.max(shift_probs, dim=-1)
    shift_wrong = shift_correct < shift_max
    shift_score = torch.where(shift_wrong, 1.0, 1.0 - shift_max)

    next_logits = outputs.logits[:, context_end_idx:-1, :]
    next_probs = torch.softmax(next_logits, dim=-1)
    next_labels = response_ids.input_ids[:, 1:]
    next_correct = torch.gather(next_probs, 2, next_labels.unsqueeze(-1)).squeeze(-1)
    next_max, _ = torch.max(next_probs, dim=-1)
    next_wrong = next_correct < next_max
    next_score = torch.where(next_wrong, 1.0, 1.0 - next_max)
    return (shift_score + next_score).mean().item()


@torch.no_grad()
def compute_exact_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation: dict,
    cache: Optional[dict] = None,
) -> float:
    """Autoregressive exact match model eval, without teacher forcing
    Lower scores are better: (correct + certain) < (correct + uncertain) < (incorrect = 1.0)"""

    if cache:
        context_ids = cache["context_ids"].to(model.device)
        expected_ids = cache["response_ids"].to(model.device)
    else:
        # prevents thinking output in reasoning models
        reasoning_mask = "<think>\\n\\n</think>\\n\\nCorrect answer:"
        context = get_context(conversation)
        context += reasoning_mask
        expected_response = f"{conversation['conversation'][-1]['value']}"
        context_ids = tokenizer(context, return_tensors="pt").to(model.device)
        expected_ids = tokenizer(" " + expected_response, return_tensors="pt").to(
            model.device
        )

    current_ids = context_ids.input_ids
    all_logits = []
    num_expected_tokens = len(expected_ids.input_ids[0])
    expected_tokens = expected_ids.input_ids[0]
    prompt_len = context_ids.input_ids.size(1)
    debug = False

    for i in range(num_expected_tokens):
        outputs = model(
            current_ids,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        next_token_logits = outputs.logits[:, -1:, :]
        all_logits.append(next_token_logits.detach())
        predicted_token_id = next_token_logits.argmax(dim=-1)
        current_ids = torch.cat([current_ids, predicted_token_id], dim=1)
        if debug:
            generated_ids = current_ids[:, prompt_len:]
            incremental_output = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            print(f"DEBUG incremental output: '{incremental_output}'")

        if predicted_token_id[0, 0] != expected_tokens[i]:
            return 1.0

    response_logits = torch.cat(all_logits, dim=1)
    probs = torch.softmax(response_logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    result = 1.0 - max_probs
    if debug:
        print(f"DEBUG 1-max_probs: {result}")
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

        if mode == "diversity":
            metric = compute_response_diversity(
                model, tokenizer, conversation, 4096, conv_cache
            )
        elif mode == "bigram_loss":
            metric = compute_bigram_loss(model, tokenizer, conversation, conv_cache)
        elif mode == "exact_match":
            metric = compute_exact_match(model, tokenizer, conversation, conv_cache)
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


# --- Ranking  ---


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


def rank_models(model_ranks: dict[str, list[int]]) -> list[str]:
    """Return models ordered best→worst by Copeland; break ties with Borda."""

    if len(model_ranks) == 1:
        return [next(iter(model_ranks))]

    models = list(model_ranks.keys())
    ranks_matrix = torch.tensor([model_ranks[m] for m in models])
    pairwise_wins = (ranks_matrix[:, None, :] < ranks_matrix[None, :, :]).sum(dim=2)
    copeland_scores = (pairwise_wins > pairwise_wins.T).sum(dim=1)
    scores = copeland_scores.tolist()
    for model, score in zip(models, scores):
        logger.info(f"Model: {model}, Copeland Score: {score}")

    groups: dict[int, list[str]] = {}
    for m, s in zip(models, scores):
        groups.setdefault(s, []).append(m)
    ordered = []
    for score in sorted(groups.keys(), reverse=True):  # high Copeland first
        group = groups[score]
        if len(group) > 1:
            group.sort(key=lambda m: sum(model_ranks[m]))  # Borda tie-break
        ordered.extend(group)

    return ordered


# --- SVD Operations  ---


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

    # Inverse damping to reduce noise impact for the largest singular values
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


# --- OLM Function ---


def olm(config: dict) -> str:
    """Optimal Layer Merging (OLM)"""
    olm_config = OlmConfig.from_config_dict(config)
    runtime = OlmRuntime()

    logger.info("Downloading required models...")
    for model_name in olm_config.model_pool:
        download_model(model_name, olm_config.models_dir)
    runtime.load_datasets(olm_config)
    merge_report_path = os.path.join(olm_config.output_dir, "merge_report.json")

    # reconstruct model from merge report or determine base model and rank models
    runtime.active_model_pool = olm_config.model_pool
    if olm_config.load_merge_report and os.path.exists(merge_report_path):
        logger.info("Loading working model from merge report...")
        runtime.load_from_merge_report(merge_report_path, olm_config.models_dir)
        runtime.save_model(olm_config.output_dir)
        if len(olm_config.model_pool) > olm_config.max_models:
            ordered_base_metrics = runtime.select_component(
                "base", olm_config.models_dir, olm_config.model_pool
            )
            runtime.active_model_pool = list(ordered_base_metrics.keys())[
                : olm_config.max_models
            ]
        base_path = Path(olm_config.models_dir) / runtime.selected_base_model.replace(
            "/", "_"
        )
    else:  # Initialize from scratch
        if olm_config.base_select:
            ordered_base_metrics = runtime.select_component(
                "base", olm_config.models_dir, olm_config.model_pool
            )
            runtime.active_model_pool = list(ordered_base_metrics.keys())
            base_model_name = next(iter(ordered_base_metrics))
            base_metrics = ordered_base_metrics[base_model_name]
        else:  # Use configured base model
            base_model_name = olm_config.base_model_name or olm_config.model_pool[0]
            logger.info(f"Using base model: {base_model_name}")
            if base_model_name not in olm_config.model_pool:
                olm_config.model_pool.insert(0, base_model_name)
                olm_config.max_models += 1
            if len(olm_config.model_pool) > olm_config.max_models:
                ordered_base_metrics = runtime.select_component(
                    "base", olm_config.models_dir, olm_config.model_pool
                )
                runtime.active_model_pool = list(ordered_base_metrics.keys())
            else:
                ordered_base_metrics = runtime.select_component(
                    "base", olm_config.models_dir, [base_model_name]
                )
            base_metrics = ordered_base_metrics[base_model_name]

        base_path = Path(olm_config.models_dir) / base_model_name.replace("/", "_")
        if not os.path.exists(base_path):
            download_model(base_model_name, olm_config.models_dir)
        model, tokenizer = load_model(str(base_path), "cpu")
        runtime.set_working_model(model, tokenizer)
        runtime.update_base_model(base_model_name, base_metrics)
        runtime.merge_report = {
            "base_model": {"name": base_model_name, "metrics": base_metrics}
        }

    logger.info(f"Using {len(runtime.active_model_pool)} models for optimization")
    logger.info(
        f"Copying {runtime.selected_base_model} files to {olm_config.output_dir}"
    )
    for file in os.listdir(base_path):
        try:
            if not file.startswith("."):
                shutil.copy2(
                    os.path.join(base_path, file),
                    os.path.join(olm_config.output_dir, file),
                )
        except Exception as e:
            logger.error(f"Error copying {file}: {e}")

    runtime.save_model(olm_config.output_dir)
    json.dump(runtime.merge_report, open(merge_report_path, "w"), indent=4)
    runtime.create_evaluation_cache(olm_config.samples)

    if olm_config.boundary_select:
        ordered_boundary_metrics = runtime.select_component(
            "boundary",
            olm_config.models_dir,
            runtime.active_model_pool,
        )
        boundary_model = next(iter(ordered_boundary_metrics))
        boundary_source, _ = load_model(
            f"{olm_config.models_dir}/{boundary_model.replace('/', '_')}", "cpu"
        )
        runtime.apply_boundary_components(boundary_source)
        del boundary_source

        runtime.merge_report["boundary_layers"] = {
            "name": boundary_model,
            "metrics": ordered_boundary_metrics[boundary_model],
        }
        runtime.save_model(olm_config.output_dir)

    if len(runtime.active_model_pool) > olm_config.max_models:
        runtime.active_model_pool = runtime.active_model_pool[: olm_config.max_models]

    if olm_config.layer_swap:
        runtime.layer_wise_optimization(runtime.active_model_pool, olm_config)

    if olm_config.tensor_swap:
        runtime.tensor_wise_optimization(runtime.active_model_pool, olm_config)

    return olm_config.output_dir


# --- Main Function ---


@torch.inference_mode()
def main(config_path: str) -> None:
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
