# Quick Evaluation Guide

## What was fixed

1. **Task list parsing**: Converts comma-separated string to list
2. **Clean output**: Shows only accuracy metrics, not full sample details
3. **Auto-save**: Results automatically saved to `results_summary.json`

## Usage

### Quick test (recommended first)
```bash
chmod +x test_clean_output.sh
./test_clean_output.sh
```

### Full evaluation
```bash
python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --tasks piqa,winogrande,arc_easy,arc_challenge,hellaswag \
    --evaluate_baseline \
    --num_fewshot 0 \
    --cache_dir ./cache \
    --output_dir ./logs/baseline
```

### Expected output (clean!)
```
================================================================================
EVALUATION RESULTS
================================================================================
  piqa                 | acc: 0.6289 | acc_norm: 0.6401
  winogrande           | acc: 0.5123 | acc_norm: 0.5045
  arc_easy             | acc: 0.4567 | acc_norm: 0.4612
  arc_challenge        | acc: 0.2345 | acc_norm: 0.2412
  hellaswag            | acc: 0.2889 | acc_norm: 0.2934
--------------------------------------------------------------------------------
  Average Accuracy     | 0.4243
================================================================================

Results saved to: ./logs/baseline/results_summary.json
```

## Compare results

```bash
python compare_results.py logs/baseline/results_summary.json \
                         logs/err0.01/results_summary.json \
                         logs/err0.025/results_summary.json
```

Output:
```
================================================================================
RESULTS COMPARISON
================================================================================

Task                  Baseline     err0.01  err0.025
--------------------------------------------------------------------------------
piqa                    0.6289   0.6289      0.6256(-0.03)
winogrande              0.5123   0.5123      0.5090(-0.03)
arc_easy                0.4567   0.4567      0.4534(-0.03)
arc_challenge           0.2345   0.2345      0.2312(-0.03)
hellaswag               0.2889   0.2889      0.2856(-0.03)
--------------------------------------------------------------------------------
Average                 0.4243   0.4243      0.4210(-0.03)
================================================================================
```

## Files created

- `test_clean_output.sh` - Quick test script
- `compare_results.py` - Compare baseline vs compressed
- `logs/*/results_summary.json` - Clean JSON results per run

## No more verbose output!

✅ Before: 1000+ lines of sample details
✅ After: Clean 10-line summary table
