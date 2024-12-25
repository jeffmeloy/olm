# Optimized Layer Merging (OLM)

This is a transformer layer optimization framework that uses multi-modal evaluation to construct merged models with superior capabilities. Instead of naive weight averaging or random permutation testing, OLM systematically evaluates each layer's contribution across multiple cognitive domains.

## Core Methodology

The system runs parallel evaluation streams using three distinct modes:
- **Exact**: Pure computation validation using strict output matching
- **Bigram**: Sequence modeling coherence testing
- **Quality**: Extended generation stability analysis

![OLM Architecture](olm.png)

Each layer position gets optimized by testing candidate layers from the model pool against these evaluation modes. The system uses rank aggregation across datasets to handle different metric scales - a layer that's mediocre at everything loses to one that excels in any domain.

## Key Features

- Auto-selects optimal base model through comprehensive pre-evaluation
- Handles forward or backward layer traversal strategies
- Maintains tokenizer coherence through careful config preservation
- Detects and penalizes common merge failure modes:
  - Token distribution collapse (Chinese character detection)
  - Repetition loops (unique word ratio analysis)
  - Sequence modeling degradation (bigram perplexity)

## Usage

Basic config just needs:
```yaml
direction: "backward"  # or "forward"
models: [list of model paths]
dataset: 
  dataset1.json:
    mode: "exact"
  dataset2.json:
    mode: "bigram"
  dataset3.json:
    mode: "quality"
```

The system handles test dataset loading, model caching, and incremental result storage automatically. State gets preserved in merge_report.json for analysis or interrupted merge recovery.

## Implementation Notes

The code aims to combine the strengths of different pre-trained language models by selectively merging their layers. It starts with a base model and iteratively evaluates whether replacing a layer in the base model with a corresponding layer from another model improves performance on specific datasets and metrics.

**Key Steps:**

1. **Setup and Configuration:**
    *   Loads configuration from a YAML file (`olm_config.yaml` by default). This file specifies:
        *   `models_dir`: Directory to store downloaded models.
        *   `output_dir`: Directory to save the merged model and reports.
        *   `models`: A list of model names from the Hugging Face Hub to be considered for merging.
        *   `dataset`: A dictionary defining the datasets used for evaluation, along with the evaluation mode (e.g., "exact", "bigram", "quality").
        *   `base_model_name` (optional): The name of the initial base model. If not provided, it selects the best base model based on the 'quality' metric.
        *   `improve_all` (optional):  Indicates whether improvement should be enforced on a specific metric.
        *   `direction`:  Indicates the direction to iterate through the layers of the model (forward or backward).
    *   Downloads specified models from the Hugging Face Hub if they don't exist locally.
    *   Select a base model if not specified in config file.

2. **Base Model Evaluation:**
    *   Loads the chosen base model and its tokenizer.
    *   Evaluates the base model on each dataset using the specified evaluation modes/metrics and saves these as the baseline metrics. The metrics include:
        *   **Exact Match Loss:** Measures how well the model predicts the exact expected response.
        *   **Response Perplexity (Bigram Mode):**  Evaluates the fluency and coherence of the generated response using perplexity (lower is better).
        *   **Response Quality:** Scores the response based on length and uniqueness of words. It penalizes short or repetitive responses. Also penalizes if there is Chinese character in response.
    *   Creates a `merge_report.json` file to track the merging process, starting with the base model's metrics.

3. **Iterative Layer Merging:**
    *   Iterates through each layer of the model (either forward or backward, based on the config).
    *   For each layer:
        *   Iterates through each candidate model in the `models` list.
        *   Temporarily replaces the current layer in the working model with the corresponding layer from the candidate model.
        *   Evaluates the modified working model on all datasets using `process_model`.
        *   `process_model` function determines whether the layer replacement is accepted or rejected.
        *   Computes the rank of each candidate model based on its performance across datasets.
        *   Selects the best-performing candidate model (lowest average rank) for the current layer based on `model_ranks`.
        *   Permanently replaces the layer in the working model with the best candidate's layer.
        *   Saves the updated working model and updates the `merge_report.json` with the metrics and the name of the model from which the best layer was taken.

4. **Saving the Final Model:**
    *   After iterating through all layers, the final merged model is saved in the `output_dir`.
    *   The `merge_report.json` file contains a complete record of the merging process, including the base model metrics and the metrics for each layer, along with the source model for each merged layer.

**Helper Functions:**

*   `download_model`: Downloads a model from the Hugging Face Hub.
*   `load_model`: Loads a model and tokenizer from a local path.
*   `save_best_model`: Saves the best model for the current layer.
*   `save_layer_state`: Saves the model and updates the merge report for the current layer.
*   `get_context`: Extracts the conversation context from a dataset sample.
*   `compute_response_quality`: Calculates the quality score of a generated response.
*   `compute_response_perplexity`: Computes the perplexity of an expected response.
*   `compute_exact_match`: Calculates the average loss for predicting the exact response.
*   `evaluate_model_on_dataset`: Evaluates a model on a dataset using a specified mode.
*   `get_layer`: Retrieves a specific layer from a model.
*   `replace_layer`: Replaces a layer in a target model with a layer from a source model.
*   `layer_sequence`: Generates a sequence of layer indices based on the merging direction.
*   `get_last_layer_idx`: Gets the index of the most recently merged layer from the merge report.
*   `get_layer_metrics`: Retrieves the metrics for a specific layer from the merge report.
*   `compute_model_ranks`: Computes the ranks of a model across datasets.
*   `compute_layer_ranks`: Computes the average ranks of each model for a specific layer.
*   `get_datasets`: Loads the datasets from the configuration.
*   `get_base_metrics`: Establishes the baseline metrics for the base model.
*   `select_base_model`: Selects the best base model based on initial evaluations.
*   `process_model`: Evaluates a model with a replaced layer and checks for improvement against base and historical metrics.

**This code provides a framework for systematically combining the strengths of multiple language models at the layer level to create a new model that potentially performs better on a given set of tasks than any of the individual source models.**
