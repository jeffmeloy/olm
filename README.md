# Optimized Layer Merging (OLM)

Yo, this is a transformer layer optimization framework that uses multi-modal evaluation to construct merged models with superior capabilities. Instead of naive weight averaging or random permutation testing, OLM systematically evaluates each layer's contribution across multiple cognitive domains.

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

Layer optimization uses pure PyTorch operations for compatibility. The evaluation modes run lazy-loaded to minimize memory overhead during the merge process. Token generation gets batched when possible for throughput optimization.

The rank aggregation system means you can throw arbitrary evaluation datasets at it - as long as they're in the standard conversation format, the system will automatically incorporate their metrics into the layer selection process.

Fundamentally, this is empirical cognitive architecture optimization. We're essentially doing targeted brain surgery on transformer models, replacing components while maintaining functional coherence. The multi-modal evaluation ensures we don't accidentally lobotomize any critical capabilities during optimization.
