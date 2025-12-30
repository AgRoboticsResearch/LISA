# LISA Setup Guide: RTX 5090 (Blackwell) & CUDA 12.9

**Summary:** This guide summarizes the complex setup of LISA on a modern RTX 5090 (Blackwell architecture) using Python 3.12, PyTorch 2.7.0, and CUDA 12.8/12.9. 


---

## 1. Environment & System Variables
To ensure the RTX 5090 and DeepSpeed function correctly, you must align the shell paths to the CUDA 12.9 toolkit and handle version mismatches between PyTorch and the system compiler.

```bash
# Point to the correct CUDA 12.9 Toolkit
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Bridge the gap: PyTorch expects cu128, but we compiled for 12.9
export BNB_CUDA_VERSION=128

# Skip DeepSpeed version guardrails for minor mismatches
export DS_SKIP_CUDA_CHECK=1

```

---

## 2. Manual Source Code Patches

LISA uses a customized version of LLaVA/MPT that is incompatible with modern `transformers` (v4.36+). You must manually apply these code changes.

### A. Resolve 'llava' Registration Collision

**File:** `LISA/model/llava/model/language_model/llava_llama.py`

**Explanation:** Hugging Face added official LLaVA support, creating a name conflict. We tell the registry to allow overwriting the default implementation.

```python
# Locate line ~166 and update to:
AutoConfig.register("llava", LlavaConfig, exist_ok=True)

```

### B. Restore Missing `_expand_mask` and `_make_causal_mask` Functions

**File:** `LISA/model/llava/model/language_model/mpt/hf_prefixlm_converter.py`

**Explanation:** Internal masking functions were removed in newer transformers. Manually inject _expand_mask and _make_causal_mask back into the scope of the MPT adapter.

```python
# --- ADD THIS BLOCK AFTER IMPORTS (around line 35) ---

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, device=device, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# Map the local functions to the names the MPT code expects
_expand_mask_bloom = _expand_mask
_make_causal_mask_bloom = _make_causal_mask
_expand_mask_opt = _expand_mask
_make_causal_mask_opt = _make_causal_mask

# --- END OF PATCH ---

```

---

## 3. Compiling `bitsandbytes` for Blackwell

The RTX 5090 uses **Compute Capability 12.0 (sm_120)**. Standard `pip` binaries do not support this yet. You must build it from source to enable 4-bit/8-bit quantization.

### Steps to Build:

```bash
# 1. Clean the environment
git clone https://github.com/bitsandbytes-foundation/bitsandbytes
cd bitsandbytes
rm -rf build && mkdir build && cd build

# 2. Configure specifically for sm_120 (Blackwell)
cmake .. \
  -DCOMPUTE_BACKEND=cuda \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc

# 3. Compile and Install
make -j$(nproc)
pip install -e ..

# 4. Critical Symlink
# Since PyTorch 2.7.0 nightlies look for 'cuda128.so', link our 'cuda129.so' to it.
cd ../bitsandbytes
ln -sf libbitsandbytes_cuda129.so libbitsandbytes_cuda128.so

```

## 4. Installing Flash Attention

For RTX 5090, use the following pre-built wheel for Flash Attention:

```bash
pip install https://github.com/loscrossos/lib_flashattention/releases/download/v2.7.4.post1_crossos00/flash_attn-2.7.4.post1+cu129torch2.7.0-cp312-cp312-linux_x86_64.whl
```

---


## 5. Deployment Script (`train.sh`)

Use this script to launch your training with all the environment fixes applied automatically.

```bash
#!/bin/bash

# Environment Overrides
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=128
export DS_SKIP_CUDA_CHECK=1

# Launch LISA
deepspeed --master_port=24999 train_ds.py \
  --version="/mnt/nas/fruit_dataset/wyn/202507/describe_anything_checkpoint/checkpoints/LLaVA-Lightning-7B-delta-v1-1" \
  --dataset_dir="/mnt/nas/fruit_dataset/wyn/cs_dataset/LISA/dataset" \
  --vision_pretrained="/mnt/nas/fruit_dataset/wyn/202507/describe_anything_checkpoint/checkpoints/sam-vit-huge/sam_vit_h_4b8939.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"

```
