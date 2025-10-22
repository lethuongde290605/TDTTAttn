# Tucker-MPS Testing & Evaluation Guide

## 🚀 Quick Start

### 1. Test compression on single layer (Basic)
```bash
python test_tucker_mps.py --test-type full
```

### 2. Evaluate baseline model (No compression)
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --evaluate_baseline \
    --eval_ppl \
    --tasks "hellaswag,piqa"
```

### 3. Compress and evaluate with Tucker-MPS
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,piqa"
```

## 📊 Full Evaluation Pipeline

### Step 1: Evaluate Baseline
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --model_family opt \
    --evaluate_baseline \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa,arc_easy" \
    --num_fewshot 0
```

**Expected output:**
- WikiText2 PPL: ~27-30
- C4 PPL: ~20-25
- HellaSwag: ~0.30-0.35
- WinoGrande: ~0.50-0.55

### Step 2: Test Different Compression Levels

#### High Quality (Low Compression)
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa,arc_easy"
```
**Expected:**
- Compression ratio: ~0.60-0.65
- Space saved: ~35-40%
- PPL increase: <10%

#### Medium Compression
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.95 \
    --hooi_ranks 5 5 6 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa,arc_easy"
```
**Expected:**
- Compression ratio: ~0.45-0.55
- Space saved: ~45-55%
- PPL increase: ~10-20%

#### Aggressive Compression
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.90 \
    --hooi_ranks 4 4 5 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa,arc_easy"
```
**Expected:**
- Compression ratio: ~0.35-0.45
- Space saved: ~55-65%
- PPL increase: ~20-30%

## 🔍 Detailed Evaluation

### Evaluate on MMLU (Multiple choice)
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --tasks "mmlu" \
    --num_fewshot 5
```

### Evaluate on Code Tasks
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --tasks "humaneval,mbpp"
```

### Evaluate on Multiple Tasks at Once
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa,arc_easy,arc_challenge,boolq,openbookqa"
```

## 📈 Compare with EigenAttn

### Run EigenAttn
```bash
python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --multigpu \
    --error_budget 0.05 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa"
```

### Run Tucker-MPS
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,winogrande,piqa"
```

### Comparison Metrics

| Metric | EigenAttn | Tucker-MPS |
|--------|-----------|------------|
| Compression Ratio | ~0.40-0.50 | ~0.60-0.65 (eps=0.99) |
| WikiText2 PPL | ? | ? |
| C4 PPL | ? | ? |
| HellaSwag Acc | ? | ? |
| WinoGrande Acc | ? | ? |

## 🎛️ Parameter Tuning Guide

### MPS Epsilon (`--mps_eps`)
Controls energy retention in MPS decomposition:
- **0.99**: Keep 99% energy → Less compression, better quality
- **0.95**: Keep 95% energy → Moderate compression
- **0.90**: Keep 90% energy → Aggressive compression

### HOOI Ranks (`--hooi_ranks`)
Controls Tucker decomposition dimensions:
- **[6, 6, 8]**: Default, good balance
- **[5, 5, 6]**: More compression
- **[4, 4, 5]**: Aggressive compression
- **[7, 7, 9]**: Less compression, better quality

### Number of Samples (`--nsamples`)
Calibration samples for compression:
- **128**: Standard, good balance
- **256**: More stable compression
- **512**: Best quality, slower

## 🐛 Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce number of samples
--nsamples 64

# Evaluate on fewer samples
--limit 100
```

### Slow Evaluation
```bash
# Limit evaluation samples
--limit 500

# Evaluate on fewer tasks
--tasks "hellaswag,piqa"
```

### Check Compression Quality
```bash
# First test without evaluation
python test_tucker_mps.py --test-type full

# Then run full evaluation
python main_tucker_mps.py ...
```

## 📝 Output Logs

Logs are saved to: `./log/{model_family}/tucker_mps_eps{mps_eps}_ranks{hooi_ranks}.log`

Example:
```
./log/opt/tucker_mps_eps0.99_ranks[6, 6, 8].log
```

## 💾 Saved Checkpoints

Compressed models are saved to: `./checkpoints/{model_family}_tucker_mps_eps{mps_eps}/`

Example:
```
./checkpoints/opt_tucker_mps_eps0.99/model.pt
```

## 🔄 Complete Workflow

```bash
# 1. Evaluate baseline
python main_tucker_mps.py --model facebook/opt-125m --evaluate_baseline --eval_ppl --tasks "hellaswag,piqa"

# 2. Test compression (quick check)
python test_tucker_mps.py --test-type full

# 3. Full compression + evaluation
python main_tucker_mps.py --model facebook/opt-125m --mps_eps 0.99 --hooi_ranks 6 6 8 --nsamples 128 --eval_ppl --tasks "hellaswag,piqa,winogrande,arc_easy"

# 4. Compare different settings
for eps in 0.99 0.95 0.90; do
    python main_tucker_mps.py \
        --model facebook/opt-125m \
        --mps_eps $eps \
        --hooi_ranks 6 6 8 \
        --nsamples 128 \
        --eval_ppl \
        --tasks "hellaswag,piqa"
done
```

## 📊 Expected Runtime

On T4 GPU:
- **Baseline evaluation**: ~5-10 minutes
- **Compression (128 samples)**: ~10-15 minutes
- **Full evaluation (PPL + 4 tasks)**: ~20-30 minutes
- **Total (baseline + compress + eval)**: ~40-60 minutes

## 🎯 Recommended Starting Point

For first test:
```bash
python main_tucker_mps.py \
    --model facebook/opt-125m \
    --mps_eps 0.99 \
    --hooi_ranks 6 6 8 \
    --nsamples 128 \
    --eval_ppl \
    --tasks "hellaswag,piqa" \
    --num_fewshot 0
```

This gives you:
- ✅ Quick feedback (~30 min)
- ✅ Good compression quality
- ✅ Reliable metrics (PPL + 2 tasks)
- ✅ Easy to compare with EigenAttn
