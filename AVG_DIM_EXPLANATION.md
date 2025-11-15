# Giải thích tham số `avg_dim` trong EigenAttn

## Tóm tắt ngắn gọn

**`avg_dim`** (hay `avg_dim_features`) là số lượng **forward passes tối thiểu** trước khi **lấy trung bình** (average) các activation features để tạo calibration data cho SVD.

## Chi tiết cơ chế hoạt động

### 1. Vị trí sử dụng

```python
# Trong main_eigen_attn.py
args.eigen_attn_params = {
    "threshold": 1.0,
    "avg_dim_features": args.avg_dim,  # <-- Truyền vào đây
    "error_budget": args.error_budget,
}
```

### 2. Workflow thu thập features

Trong `get_kqv_opt()`, `get_kqv_llama()`, `get_kqv_mpt()`:

```python
@torch.no_grad()
def get_kqv_opt(layer, fp_inps, args):
    avg_k = []
    k = {}  # Dictionary lưu tạm activations
    
    avg_dim = args.eigen_attn_params['avg_dim_features']
    
    def forward_hook_k(m, x, y, name):
        # Thu thập activation từ k_proj
        if name in k.keys():
            k[name] += [y.view(-1, y_size[-1])]  # Append vào list
        else:
            k[name] = [y.view(-1, y_size[-1])]   # Khởi tạo list
        
        dim = len(k[name])  # Đếm số lượng activations đã thu
        
        # ✅ KHI ĐỦ avg_dim mẫu → Lấy trung bình và append vào avg_k
        if dim >= avg_dim:
            Y = torch.stack(k.pop(name)).mean(dim=0)  # Average!
            avg_k.append(Y)
    
    # Chạy forward qua nsamples (ví dụ 128 samples)
    for j in range(args.nsamples):
        layer(fp_inps[j].unsqueeze(0))
    
    # Stack tất cả averaged features
    avg_k = torch.stack(avg_k)
    return avg_k, avg_q, avg_v
```

## 3. Ví dụ cụ thể

### Case 1: `avg_dim = 16`, `nsamples = 128`

```
Forward pass 1-15:   k["layer.k_proj"] = [feat1, feat2, ..., feat15]
                     (Chưa đủ 16 → giữ lại)

Forward pass 16:     k["layer.k_proj"] = [feat1, ..., feat16]
                     ✅ Đủ 16 mẫu!
                     → avg_k.append(mean([feat1, ..., feat16]))
                     → k["layer.k_proj"] bị pop (xóa)

Forward pass 17-32:  k["layer.k_proj"] = [feat17, ..., feat32]
                     ✅ Đủ 16 mẫu!
                     → avg_k.append(mean([feat17, ..., feat32]))

... (tiếp tục)

Kết quả: avg_k chứa 128/16 = 8 averaged features
```

### Case 2: `avg_dim = 1`, `nsamples = 128`

```
Mỗi forward pass:    ✅ Đủ 1 mẫu ngay
                     → avg_k.append(feat_i)  (không có averaging thực sự)

Kết quả: avg_k chứa 128 features (không average, giữ nguyên mỗi sample)
```

### Case 3: `avg_dim = 512`, `nsamples = 128`

```
Forward pass 1-128:  k["layer.k_proj"] = [feat1, ..., feat128]
                     ❌ Chưa đủ 512 mẫu!
                     → Không bao giờ append vào avg_k

Kết quả: avg_k = [] (EMPTY!)
⚠️ Lỗi khi stack: "cannot stack empty list"
```

## 4. Ý nghĩa và ảnh hưởng

### Mục đích của averaging:

1. **Giảm noise**: Average nhiều samples → features ổn định hơn
2. **Giảm memory**: Thay vì lưu 128 features, chỉ lưu 128/avg_dim features
3. **Tăng tính đại diện**: Mỗi averaged feature đại diện cho một "batch" samples

### Ảnh hưởng đến chất lượng SVD:

| avg_dim | Số features cho SVD | Noise level | Memory | Quality |
|---------|---------------------|-------------|---------|---------|
| 1       | = nsamples (128)    | Cao         | Cao     | Tốt (nhiều data) |
| 16      | nsamples/16 (8)     | Trung bình  | Thấp    | Tốt (balanced) |
| 64      | nsamples/64 (2)     | Thấp        | Rất thấp| Kém (ít data) |
| 512     | 0 (nếu nsamples=128)| N/A         | N/A     | ❌ Lỗi |

### Giá trị khuyến nghị:

```python
# Với nsamples = 128
--avg_dim 1     # No averaging, giữ tất cả samples
--avg_dim 16    # Moderate averaging, 8 averaged features
--avg_dim 32    # More averaging, 4 averaged features
--avg_dim 64    # Heavy averaging, 2 averaged features

# ⚠️ QUAN TRỌNG: avg_dim PHẢI ≤ nsamples
```

## 5. Trade-offs

### avg_dim nhỏ (ví dụ 1):
✅ **Ưu điểm**:
- Nhiều features → SVD có nhiều data
- Rank selection chính xác hơn
- Basis vectors chi tiết hơn

❌ **Nhược điểm**:
- Memory cao
- Noise nhiều hơn
- Có thể overfit với calibration data

### avg_dim lớn (ví dụ 64):
✅ **Ưu điểm**:
- Memory thấp
- Features ổn định (ít noise)
- Generalize tốt hơn

❌ **Nhược điểm**:
- Ít features → SVD có ít data
- Có thể mất thông tin chi tiết
- Rank selection kém chính xác

## 6. Code flow đầy đủ

```
1. main_eigen_attn.py
   └─> args.avg_dim (from command line)
   └─> args.eigen_attn_params['avg_dim_features'] = args.avg_dim

2. eigenattn() function
   └─> Gọi get_kqv_opt(layer, inps, args)

3. get_kqv_opt()
   └─> avg_dim = args.eigen_attn_params['avg_dim_features']
   └─> Forward hooks thu thập activations
   └─> Mỗi khi len(k[name]) >= avg_dim:
       └─> Y = stack(k.pop(name)).mean(dim=0)  # AVERAGE!
       └─> avg_k.append(Y)
   └─> Return avg_k, avg_q, avg_v

4. decompose_opt_layer()
   └─> feat_k, feat_q, feat_v = get_kqv_opt(...)
   └─> basis_kq, eval_kq = generate_basis_vectors_per_head(feat_k, ...)
       └─> SVD(feat_k.t())  # <-- Dùng averaged features
```

## 7. Debugging tips

### Kiểm tra số features thực tế:

```python
# Thêm vào get_kqv_opt()
print(f"[DEBUG] avg_dim={avg_dim}, nsamples={args.nsamples}")
print(f"[DEBUG] Expected features: {args.nsamples // avg_dim}")
print(f"[DEBUG] Actual avg_k.shape: {avg_k.shape}")
```

### Lỗi thường gặp:

**Lỗi 1: "cannot stack empty sequence"**
```
Nguyên nhân: avg_dim > nsamples
Giải pháp: Giảm avg_dim hoặc tăng nsamples
```

**Lỗi 2: SVD rank quá thấp**
```
Nguyên nhân: avg_dim quá lớn → quá ít features
Giải pháp: Giảm avg_dim để có nhiều features hơn
```

**Lỗi 3: Out of memory**
```
Nguyên nhân: avg_dim = 1 → quá nhiều features
Giải pháp: Tăng avg_dim để giảm số features
```

## 8. Ví dụ chạy

```bash
# Baseline: Không averaging (128 features)
python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --nsamples 128 \
    --avg_dim 1 \
    --error_budget 0.025

# Moderate averaging (8 features)
python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --nsamples 128 \
    --avg_dim 16 \
    --error_budget 0.025

# Heavy averaging (4 features) - tiết kiệm memory
python main_eigen_attn.py \
    --model facebook/opt-125m \
    --net opt-125m \
    --nsamples 128 \
    --avg_dim 32 \
    --error_budget 0.025
```

## 9. Kết luận

**`avg_dim`** điều khiển **batch size của averaging** khi thu thập calibration features:

- **Nhỏ** → nhiều features chi tiết → tốn memory → chính xác hơn
- **Lớn** → ít features tổng hợp → tiết kiệm memory → tổng quát hơn
- **Phải < nsamples** để tránh lỗi!

**Giá trị khuyến nghị cho paper**: `--avg_dim 16` với `--nsamples 128` (tạo 8 averaged features)
