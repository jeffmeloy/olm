# olm.py Description

This script implements an Optimal Layer Merging (OLM) framework designed to create high-performance language models by strategically combining components from a pool of existing pre-trained models. It further refines these merged components using an evolutionary strategy ((1+λ)-ES) for fine-tuning at the tensor level without computing gradients or using backpropagation.

![OLM Architecture](olm.png)

## 1. Configuration and Setup

-   **Imports**: Imports necessary libraries including `torch`, `transformers`, `huggingface_hub`, `numpy`, `json`, `yaml`, `logging`, `argparse`, etc., for model handling, evaluation, data processing, and execution control.
-   **Logging**: Sets up basic logging to track the script's progress, key decisions, and any errors encountered.

## 2. Model and Dataset Handling

-   **`get_model_from_merge_report`**: Reconstructs a merged model from a JSON merge report. This involves loading a base model and selectively replacing its boundary components (embedding, final norm, LM head) and individual layers with components from other models as specified in the report.
-   **`download_model`**: Downloads a pre-trained model from the Hugging Face Hub if it's not already available locally in the specified `models_dir`.
-   **`load_model`**: Loads a pre-trained language model and its tokenizer from a local directory using `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, typically loading to CPU with `bfloat16` for memory efficiency.
-   **`get_datasets`**: Loads datasets from JSON files specified in the configuration. Each dataset includes its data and the designated evaluation `mode`.
-   **`get_evaluation_cache`**: Pre-computes and caches tokenized inputs (context, expected responses) for datasets to speed up repeated evaluations during optimization.

## 3. Evaluation Metrics

-   **`get_context`**: Formats the conversation history into a prompt string suitable for the model.
-   **`compute_response_quality`**: Computes a heuristic "quality" score for a generated response based on length and word repetition.
-   **`compute_uncertain_bigram`**: Evaluates model performance based on bigram probabilities, rewarding correct predictions and uncertainty.
-   **`compute_exact_uncertain_match`**: Calculates a score reflecting both the correctness of token predictions and the model's uncertainty.
-   **`evaluate_model_on_dataset`**: Evaluates a model's performance on a given dataset using one of the defined evaluation modes (e.g., "quality", "uncertain_bigram", "exact_uncertain").

## 4. Layer and Tensor Operations

-   **`get_layer`**: Retrieves a specific layer (e.g., a transformer block) from a model.
-   **`replace_layer`**: Replaces a layer in the target model with a layer from a source model.
-   **`layer_sequence`**: Generates a sequence of layer indices (either forward or backward) for iteration.
-   **`get_last_layer_idx`**: Gets the index of the last optimized layer from a merge report.
-   **`get_layer_metrics`**: Retrieves performance metrics for a specific layer from a merge report.
-   **`_get_target_parameter_units`**: Identifies functional units of parameters (e.g., all weights and biases related to `q_proj` in a layer) for tensor-wise optimization.
-   **`calculate_normalized_effective_rank` (NER)**: Computes the NER of a tensor, an auxiliary metric possibly indicating tensor complexity or redundancy.
-   **`calculate_full_svd`**: Computes the Singular Value Decomposition (U, S, Vh) of a tensor.
-   **`generate_noise_from_svd_coeffs`**: Generates a multiplicative noise tensor by perturbing singular values (S) obtained from SVD and reconstructing with U and Vh.

## 5. Merge Report, Ranking, and Selection

-   **`save_layer_state`**: Saves the current state of the working model and updates the merge report with metrics and the best model choice for the just-optimized layer.
-   **`save_tensor_state`**: Saves the model state and updates the merge report after tensor-wise optimization (including ES fine-tuning) for a parameter unit.
-   **`compute_model_ranks` / `compute_layer_ranks`**: Computes the rank of each candidate model (for a layer or tensor) based on its performance across multiple datasets, aggregating ranks to handle multi-objective optimization.
-   **`select_best_model`**: Selects the best candidate model (for a layer, tensor, base model, or boundary components) based on its aggregated rank.
-   **`select_component`**: A unified function to select the best base model, boundary layers, or layer-donor model based on rank aggregation across specified datasets and metrics.

## 6. Base Model Handling and Initialization

-   **`get_base_metrics`**: Evaluates the chosen base model on all datasets and saves these baseline metrics to the initial merge report. This involves copying base model files to the output directory.

## 7. Processing and Degradation Handling

-   **`process_model`**: Evaluates a candidate model (after a layer/tensor swap) on all datasets. It includes logic to detect performance degradation compared to established baselines (base model or previous best state) and can mark the candidate as unsuitable to save computation. Also computes NER for the evaluated component.

## 8. Boundary Layer Optimization

-   **`_apply_boundary_components`**: Helper function to apply `embed_tokens`, `model.norm`, and `lm_head` from a source model to the working model. The best source for these is determined by `select_component`.

## 9. Evolutionary Strategy ((1+λ)-ES) Fine-Tuning

-   **`fine_tune_tensor_1plambdaES`**: Implements a (1+λ)-Evolutionary Strategy to fine-tune a specific tensor (parameter unit) within the working model.
    -   **SVD-Based Perturbation**: Decomposes the tensor using SVD and generates `λ` offspring by perturbing its singular values.
    -   **Offspring Evaluation**: Each offspring (modified tensor applied to the model) is evaluated on the datasets.
    -   **Rank-Based Selection**: The best offspring is selected based on its aggregate rank across dataset metrics.
    -   **Iterative Refinement**: The base tensor is updated with the best perturbation, and the process (including SVD re-computation and sigma adaptation) repeats for a configured number of generations.
    -   **Memory Efficiency**: Operates with forward passes only, avoiding gradient computation, making it suitable for consumer hardware.

## 10. Optimization Functions

-   **`layer_wise_optimization`**: Implements layer-wise optimization. It iterates through each layer of the working model, tests layers from all models in the pool, selects the best-performing one using rank aggregation, and applies it.
-   **`tensor_wise_optimization`**: Implements tensor-wise (or parameter unit-wise) optimization. It iterates through identified parameter units, tests corresponding units from the model pool, selects the best one by rank, applies it, and then **further refines the newly applied unit using `fine_tune_tensor_1plambdaES`**.

## 11. Main OLM Function

-   **`olm`**: The main orchestrator function.
    -   Loads configuration and datasets.
    -   Handles model downloading and selection of the initial base model.
    -   Builds an evaluation cache.
    -   Initializes or loads a `merge_report.json`.
    -   Establishes base model metrics.
    -   Optionally performs boundary layer optimization.
    -   Optionally performs layer-wise optimization.
    -   Optionally performs tensor-wise optimization (which includes ES fine-tuning).
    -   Saves the final optimized model and the comprehensive merge report.

## 12. Main Execution

-   **`main`**: The script's entry point. Parses command-line arguments (path to config YAML, optional path to an existing merge report to resume/extend from), loads the configuration, and invokes the `olm` function.

## Key Concepts

-   **Layer Merging**: Strategically replacing entire layers in a base model with layers from a pool of candidate models.
-   **Tensor Merging (Parameter Unit Merging)**: Replacing specific groups of related weight tensors (functional units) within layers.
-   **(1+λ)-ES Fine-Tuning**: An evolutionary algorithm used to further optimize individual tensors/units by iteratively generating and evaluating SVD-based perturbations, selecting the best via rank aggregation. This is a gradient-free optimization method.
-   **Merge Report**: A JSON file that meticulously records the entire merging and optimization process: base model, chosen boundary layers, source of each replaced layer/tensor, and performance metrics at each stage.
-   **Evaluation Metrics & Cache**: Custom functions (quality, uncertainty-aware metrics) to assess model performance. An evaluation cache pre-processes data for faster repeated evaluations.
-   **Rank-Based Selection**: A robust method for choosing the "best" component (model, layer, tensor, ES offspring) by aggregating its performance ranks across multiple datasets/metrics.
-   **Degradation Handling**: Mechanisms to prevent applying a change that worsens performance on critical metrics compared to a baseline.
-   **Boundary Layers**: The initial embedding layer, final normalization layer, and the language model head, which are often optimized as a distinct set.
-   **SVD-Guided Perturbation**: Using Singular Value Decomposition to guide the generation of noise/perturbations for the ES, focusing on modifying singular values.

The `olm_config.yaml` file provides the configuration settings for the script, specifying the model pool, datasets for evaluation, evaluation modes, optimization strategies to employ (boundary, layer, tensor, ES), and parameters controlling the merging and ES processes.
