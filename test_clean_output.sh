#!/bin/bash

echo "Testing clean output format..."
echo ""

python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --tasks piqa,arc_easy \
    --evaluate_baseline \
    --num_fewshot 0 \
    --limit 50 \
    --cache_dir ./cache \
    --output_dir ./logs/test_clean 2>&1 | grep -A 20 "EVALUATION RESULTS"

echo ""
echo "If you see a clean table above, the fix worked!"
