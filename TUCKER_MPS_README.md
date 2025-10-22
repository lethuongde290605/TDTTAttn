# Tucker-MPS Compression for OPT Models

Đây là implementation của thuật toán nén Tucker-MPS cho các mô hình OPT, được thiết kế để thay thế cho EigenAttention compression.

## 📁 Cấu trúc Files

```
decompose/
  ├── tucker_mps_utils.py     # Các hàm tiện ích cho Tucker và MPS decomposition
  └── tucker_mps.py           # Main compression function cho OPT models

models/
  └── decompose_modules.py    # Chứa OPTTuckerMPSAttention và OPTTuckerMPSDecoderLayer

test_tucker_mps.py            # Script để test compression
```

## 🔧 Cách hoạt động

### 1. Tucker Decomposition (HOOI Algorithm)
Tucker decomposition phân tách tensor thành:
- **Core tensor**: Chứa thông tin cấu trúc chính
- **Factor matrices**: Ma trận biến đổi cho mỗi mode

```python
X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
```

### 2. MPS (Matrix Product States) Decomposition
MPS decomposition cho tensor 3D:
```
X → [B₁, G, B₂]
```
Với các tensor cores được kết nối tuần tự.

### 3. Combined Tucker-MPS
Thuật toán kết hợp:
1. Reshape weight matrix 768×768 → 64×64×144
2. Áp dụng MPS decomposition với threshold `mps_eps`
3. Áp dụng HOOI (Tucker) với ranks cho mỗi mode
4. Tensor contraction để tái tạo compressed weight
5. Reshape về 768×d (d < 768)

## 🚀 Sử dụng

### Test đơn giản compression function:

```bash
python test_tucker_mps.py --test-type compression-only
```

### Test full layer compression:

```bash
python test_tucker_mps.py --test-type full
```

### Tích hợp vào training pipeline:

```python
from decompose.tucker_mps import tucker_mps_compress

# Trong main script của bạn:
compressed_model = tucker_mps_compress(
    lm=language_model,
    args=args,
    dataloader=calib_dataloader,
    logger=logger,
    mps_eps=0.99,        # Threshold cho MPS (0.9-0.99)
    hooi_ranks=[6, 6, 8] # Ranks cho Tucker decomposition
)
```

## 🎛️ Hyperparameters

### `mps_eps` (MPS Epsilon Threshold)
- **Range**: 0.9 - 0.99
- **Higher values** (0.99): Giữ nhiều thông tin hơn, nén ít hơn
- **Lower values** (0.90): Nén mạnh hơn, có thể mất thông tin
- **Recommended**: 0.99 cho lần test đầu

### `hooi_ranks` (Tucker Decomposition Ranks)
- **Format**: `[r1, r2, r3]` - ranks cho 3 modes
- **Default**: `[6, 6, 8]`
- **Higher ranks**: Giữ nhiều thông tin hơn, nén ít hơn
- **Lower ranks**: Nén mạnh hơn
- **Examples**:
  - `[6, 6, 8]` - Moderate compression
  - `[5, 5, 6]` - Higher compression
  - `[4, 4, 5]` - Aggressive compression

## 📊 Expected Results

Với OPT-125m (hidden_size=768):

| Configuration | Compression Ratio | Space Saved | Reconstruction Error |
|--------------|------------------|-------------|---------------------|
| eps=0.99, ranks=[6,6,8] | ~0.40-0.50 | 50-60% | < 0.1 |
| eps=0.95, ranks=[5,5,6] | ~0.30-0.40 | 60-70% | < 0.15 |
| eps=0.90, ranks=[4,4,5] | ~0.25-0.35 | 65-75% | < 0.20 |

## 🔍 Classes Overview

### `OPTTuckerMPSAttention`
Thay thế cho `OPTAttention` với compressed weights:
- Nén K, Q, V projection matrices
- Xử lý compressed dimensions trong forward pass
- Tự động tạo projection layers khi cần

### `OPTTuckerMPSDecoderLayer`
Thay thế cho `OPTDecoderLayer`:
- Sử dụng `OPTTuckerMPSAttention`
- Giữ nguyên FFN và LayerNorm
- Compatible với OPT model architecture

## 🧪 Testing Workflow

### Step 1: Test compression function
```bash
python test_tucker_mps.py --test-type compression-only
```
Kiểm tra:
- ✓ Compression function chạy không lỗi
- ✓ Output shape đúng
- ✓ Reconstruction error reasonable

### Step 2: Test single layer
```bash
python test_tucker_mps.py --test-type full
```
Kiểm tra:
- ✓ Layer initialization thành công
- ✓ Forward pass không lỗi
- ✓ Output shape khớp với input
- ✓ Compression ratio và error trong ngưỡng chấp nhận

### Step 3: Test multiple configurations
Script sẽ tự động test với các configuration khác nhau:
- eps = [0.99, 0.95, 0.90]
- ranks = [[6,6,8], [5,5,6], [4,4,5]]

### Step 4: Integrate vào main pipeline
Sau khi test thành công, thêm vào main training script.

## 🐛 Troubleshooting

### Error: "Input tensor must have shape (768, 768)"
→ Check weight matrix shape. Tucker-MPS hiện tại chỉ support 768×768 (OPT-125m, OPT-350m, etc.)
→ Có thể modify `combined_mps_hooi_compression()` để support sizes khác

### Error: Shape mismatch trong forward pass
→ Check compressed dimensions compatibility
→ Verify projection layers được tạo đúng

### High reconstruction error (> 0.2)
→ Tăng `mps_eps` (ví dụ từ 0.90 → 0.95)
→ Tăng `hooi_ranks` (ví dụ từ [4,4,5] → [6,6,8])

### Out of memory
→ Giảm batch size trong test
→ Test trên GPU với memory lớn hơn
→ Compress từng layer một thay vì toàn bộ model

## 📝 So sánh với EigenAttention

| Aspect | EigenAttention | Tucker-MPS |
|--------|---------------|------------|
| Decomposition | SVD per head | Tucker + MPS combined |
| Adaptivity | Dynamic rank selection | Fixed ranks |
| Complexity | Per-head basis vectors | Global tensor decomposition |
| Memory | Lower during decompose | Higher during decompose |
| Compression | Head-wise | Global structure |

## 🔜 Next Steps

1. **Test trên calibration data thực**
   - Sử dụng dataset giống EigenAttn (C4, WikiText, etc.)
   - Compare reconstruction error

2. **Evaluate downstream tasks**
   - Test perplexity trên validation set
   - Run lm-eval benchmarks

3. **Optimize hyperparameters**
   - Grid search cho best eps và ranks
   - Balance giữa compression và accuracy

4. **Extend sang models khác**
   - Llama (hidden_size khác 768)
   - MPT
   - Flexible size support

## 📚 References

- Tucker Decomposition: Kolda & Bader, "Tensor Decompositions and Applications", 2009
- MPS: Matrix Product States from quantum physics
- HOOI: Higher-Order Orthogonal Iteration algorithm

## 🤝 Contribution

Nếu muốn extend hoặc improve:
1. Test với model sizes khác (OPT-1.3B, OPT-2.7B)
2. Implement dynamic rank selection tương tự EigenAttn
3. Add support cho Llama, GPT-2, etc.
4. Optimize memory usage trong decomposition

---

**Author**: Based on EigenAttention architecture
**Date**: October 2025
