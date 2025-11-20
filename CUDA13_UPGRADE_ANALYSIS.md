# CUDA 13 Upgrade Analysis for OpenClio

## Current State Analysis

### CUDA-Related Dependencies

1. **PyTorch (`torch`)**
   - Current: No version specified (uses latest)
   - Usage: `torch.cuda.manual_seed()` in `openclio.py:935`
   - Status: ✅ Code is CUDA-compatible

2. **vLLM**
   - Current: No version specified
   - Usage: Primary LLM inference engine
   - Status: ⚠️ Requires CUDA support

3. **FAISS**
   - Current: `faiss-cpu` (CPU-only version)
   - Usage: K-means clustering in `faissKMeans.py`
   - Status: ❌ Currently CPU-only, should upgrade to GPU version

4. **Numba**
   - Current: No version specified
   - Usage: JIT compilation (may use CUDA)
   - Status: ⚠️ Should verify CUDA 13 compatibility

### Code Analysis

**CUDA Usage Found:**
- `openclio/openclio.py:935`: `torch.cuda.manual_seed(seed)` - Standard PyTorch CUDA call, compatible with CUDA 13

**No Direct CUDA Code:**
- The codebase uses high-level libraries (PyTorch, vLLM, FAISS) and doesn't directly call CUDA APIs
- No deprecated CUDA API usage found

## Upgrade Requirements for CUDA 13

### 1. PyTorch Upgrade

**Recommended:** PyTorch 2.5+ (supports CUDA 13.0+)

```python
# Install PyTorch with CUDA 13 support
# For CUDA 13.0:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# For CUDA 13.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131
```

**Action Required:**
- Update `pyproject.toml` and `setup.py` to specify PyTorch version with CUDA 13 support
- Consider making CUDA version configurable

### 2. vLLM Upgrade

**Status:** vLLM is actively developed and should support CUDA 13 in recent versions

**Action Required:**
- Check latest vLLM version that supports CUDA 13
- Update dependency version constraint
- Test with CUDA 13 installation

**Installation:**
```bash
# vLLM typically requires building from source or using pre-built wheels
# Check vLLM GitHub releases for CUDA 13 support
pip install vllm --upgrade
```

### 3. FAISS Upgrade (Critical)

**Current:** `faiss-cpu` (CPU-only)

**Required:** `faiss-gpu` (GPU-enabled with CUDA support)

**Action Required:**
- Replace `faiss-cpu` with `faiss-gpu` in dependencies
- Update `pyproject.toml` and `setup.py`
- Consider making GPU/CPU version optional based on availability

**Installation:**
```bash
# FAISS GPU with CUDA 13 support
pip install faiss-gpu

# Or specify CUDA version if available:
# pip install faiss-gpu-cu13
```

**Code Changes:**
- `faissKMeans.py` should work without changes (FAISS API is the same)
- May need to add GPU device selection logic

### 4. Numba Upgrade

**Status:** Numba should support CUDA 13 in recent versions

**Action Required:**
- Update to latest numba version
- Verify CUDA 13 compatibility

## Recommended Changes

### Option 1: Full GPU Support (Recommended)

Update dependencies to use GPU versions:

**pyproject.toml:**
```toml
dependencies = [
    # ... other dependencies ...
    "torch>=2.5.0",  # Specify minimum version with CUDA 13 support
    "vllm>=0.6.0",   # Check latest version with CUDA 13 support
    "faiss-gpu",     # Change from faiss-cpu to faiss-gpu
    "numba>=0.60.0", # Latest version with CUDA 13 support
    # ... rest of dependencies ...
]
```

**setup.py:**
```python
install_requires = [
    # ... other dependencies ...
    "torch>=2.5.0",
    "vllm>=0.6.0",
    "faiss-gpu",  # Changed from faiss-cpu
    "numba>=0.60.0",
    # ... rest of dependencies ...
]
```

### Option 2: Optional GPU Support

Make GPU support optional with fallback to CPU:

**pyproject.toml:**
```toml
dependencies = [
    # ... other dependencies ...
    "torch>=2.5.0",
    "vllm>=0.6.0",
    "faiss-cpu",  # Keep CPU as default
    "numba>=0.60.0",
    # ... rest of dependencies ...
]

[project.optional-dependencies]
gpu = [
    "faiss-gpu",  # Optional GPU support
]
```

## Testing Checklist

After upgrading:

1. ✅ Verify CUDA 13 installation:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

2. ✅ Test PyTorch CUDA functionality:
   ```python
   import torch
   assert torch.cuda.is_available()
   x = torch.randn(10, 10).cuda()
   ```

3. ✅ Test FAISS GPU:
   ```python
   import faiss
   # Test GPU index creation
   ```

4. ✅ Test vLLM with CUDA 13:
   ```python
   import vllm
   # Test LLM initialization
   ```

5. ✅ Run OpenClio tests/examples:
   - Test with example data
   - Verify clustering works with GPU
   - Verify LLM inference works

## Potential Issues & Solutions

### Issue 1: FAISS GPU Installation
- **Problem:** `faiss-gpu` may not have pre-built wheels for CUDA 13
- **Solution:** May need to build from source or use conda

### Issue 2: vLLM CUDA 13 Compatibility
- **Problem:** vLLM may require specific CUDA toolkit version
- **Solution:** Check vLLM documentation for CUDA 13 support status

### Issue 3: Mixed CPU/GPU Dependencies
- **Problem:** Some dependencies may still use CPU
- **Solution:** Ensure all GPU-capable libraries use GPU versions

## Next Steps

1. **Check Current Versions:**
   ```bash
   pip list | grep -E "torch|vllm|faiss|numba"
   ```

2. **Verify CUDA 13 Installation:**
   ```bash
   nvcc --version  # Should show CUDA 13.x
   ```

3. **Update Dependencies:**
   - Modify `pyproject.toml` and `setup.py`
   - Test installation in clean environment

4. **Update Documentation:**
   - Add CUDA 13 requirements to README
   - Document installation steps

5. **Test Thoroughly:**
   - Run example code
   - Verify GPU utilization
   - Check performance improvements

## Additional Notes

- CUDA 13 introduces new features but maintains backward compatibility
- The codebase doesn't use low-level CUDA APIs, so migration should be straightforward
- Consider adding GPU availability checks in code
- May want to add environment variable for GPU/CPU selection

