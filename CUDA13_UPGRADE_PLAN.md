# CUDA 13 Upgrade Plan - OpenClio

## Summary

This document provides a concrete plan to upgrade OpenClio to support CUDA 13.

## Current State

- **PyTorch**: No version specified (uses latest)
- **vLLM**: No version specified  
- **FAISS**: Using `faiss-cpu` (CPU-only)
- **Numba**: No version specified
- **CUDA Code**: Only `torch.cuda.manual_seed()` - compatible with CUDA 13

## Required Changes

### 1. Update Dependencies

#### pyproject.toml Changes

**Note:** `faiss-gpu` is not available on PyPI. We keep `faiss-cpu` as default and document GPU installation via conda.

```toml
dependencies = [
    "umap-learn",
    "concave_hull",
    "tqdm",
    "dataclasses",
    "vllm>=0.6.0",  # Check latest version with CUDA 13 support
    "pandas",
    "sentence_transformers",
    "numpy",
    "scikit-learn",
    "scipy",
    "torch>=2.5.0",  # PyTorch 2.5+ supports CUDA 13
    "cloudpickle",
    "setuptools",
    "pyarrow",
    "faiss-cpu",  # Default: CPU version (GPU via conda - see FAISS_GPU_INSTALLATION.md)
    "cryptography",
    "numba>=0.60.0"  # Latest version with CUDA 13 support
]

[project.optional-dependencies]
gpu = [
    # Note: faiss-gpu is not available via pip, install via conda:
    # conda install -c pytorch faiss-gpu
]
```

#### setup.py Changes

```python
install_requires = [
    "concave_hull", 
    "tqdm", 
    "umap-learn", 
    "numba>=0.60.0", 
    "cryptography", 
    "dataclasses", 
    "vllm>=0.6.0",
    "pandas", 
    "sentence_transformers", 
    "numpy", 
    "scikit-learn", 
    "scipy", 
    "torch>=2.5.0", 
    "cloudpickle", 
    "setuptools", 
    "pyarrow", 
    "faiss-cpu"  # Default: CPU version (GPU via conda)
]
```

### 2. Optional: Add GPU Device Selection for FAISS

The current `faissKMeans.py` will automatically use GPU if `faiss-gpu` is installed. However, you may want to add explicit GPU device selection:

**Optional Enhancement to faissKMeans.py:**

```python
def __init__(self, n_clusters=8, max_iter=25, n_init=1,
             tol=1e-4, approximate=False, M=32, ef=256,
             random_state=None, n_jobs=-1, verbose=False,
             use_gpu=None):  # Add use_gpu parameter
    # ... existing code ...
    self.use_gpu = use_gpu if use_gpu is not None else faiss.get_num_gpus() > 0
    self.gpu_id = 0 if self.use_gpu else None
```

Then in `fit()` method, add GPU resource if available:

```python
def fit(self, X, y=None):
    X = self._as_float32(X)
    n_samples, d = X.shape

    # Set GPU resources if available
    if self.use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        # ... rest of code ...
```

**Note:** This is optional - FAISS will use GPU automatically if available.

### 3. Installation Instructions

#### Prerequisites

1. **Install CUDA 13 Toolkit:**
   ```bash
   # Download from NVIDIA website
   # Verify installation:
   nvcc --version  # Should show CUDA 13.x
   nvidia-smi      # Should show compatible driver
   ```

2. **Install PyTorch with CUDA 13:**
   ```bash
   # For CUDA 13.0:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   
   # For CUDA 13.1:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131
   ```

3. **Install FAISS GPU (via Conda - not available via pip):**
   ```bash
   # FAISS GPU is not available on PyPI, use conda instead:
   conda install -c pytorch faiss-gpu
   
   # See FAISS_GPU_INSTALLATION.md for detailed instructions
   ```

4. **Install vLLM:**
   ```bash
   pip install vllm --upgrade
   # vLLM may require building from source for CUDA 13
   # Check: https://github.com/vllm-project/vllm
   ```

5. **Install OpenClio:**
   ```bash
   pip install -e .
   # or
   pip install git+https://github.com/Phylliida/OpenClio.git
   ```

### 4. Verification Steps

After installation, verify CUDA 13 support:

```python
# Test PyTorch CUDA
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

# Test FAISS GPU
import faiss
print(f"FAISS GPU count: {faiss.get_num_gpus()}")

# Test vLLM (requires model)
# import vllm
# llm = vllm.LLM(model="Qwen/Qwen3-8B")
```

### 5. Testing Checklist

- [ ] CUDA 13 toolkit installed and verified
- [ ] PyTorch detects CUDA 13
- [ ] FAISS GPU installation successful
- [ ] vLLM works with CUDA 13
- [ ] OpenClio example runs successfully
- [ ] GPU utilization visible during clustering
- [ ] LLM inference uses GPU

## Potential Issues

### Issue 1: FAISS GPU Installation
**Problem:** `faiss-gpu` may not have pre-built wheels for CUDA 13  
**Solution:** 
- Use conda: `conda install -c pytorch faiss-gpu`
- Build from source if needed
- Check FAISS GitHub for CUDA 13 support status

### Issue 2: vLLM CUDA 13 Compatibility
**Problem:** vLLM may require specific CUDA toolkit version  
**Solution:**
- Check vLLM GitHub issues/releases for CUDA 13 support
- May need to build from source
- Consider using vLLM's Docker image with CUDA 13

### Issue 3: Version Conflicts
**Problem:** Dependency version conflicts  
**Solution:**
- Use virtual environment
- Install PyTorch first, then other dependencies
- Check compatibility matrices

## Alternative: Keep CPU Fallback

If GPU support is problematic, you can make GPU optional:

```toml
[project.optional-dependencies]
gpu = [
    "faiss-gpu",
    "torch>=2.5.0",  # With CUDA support
]
```

Then install with: `pip install openclio[gpu]`

## Next Steps

1. **Update dependency files** (pyproject.toml, setup.py)
2. **Test installation** in clean environment
3. **Verify CUDA 13 compatibility** of each dependency
4. **Update README** with CUDA 13 requirements
5. **Test end-to-end** with example data
6. **Document any issues** encountered

## References

- PyTorch CUDA Installation: https://pytorch.org/get-started/locally/
- FAISS Installation: https://github.com/facebookresearch/faiss
- vLLM Installation: https://github.com/vllm-project/vllm
- CUDA 13 Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

